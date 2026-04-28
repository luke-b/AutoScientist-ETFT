"""
train.py — Top-level fine-tuning entrypoint for AutoScientist-ETFT.

Trains an LLM on the ETFT corpus (𝒟_Gen + 𝒟_Rationale) using HuggingFace
PEFT/LoRA.  Adjust hyperparameters via config.yaml or CLI flags.

Usage:
    python train.py --dataset ./data --model meta-llama/Meta-Llama-3-8B
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_dataset(data_root: Path) -> tuple[list[dict], list[dict]]:
    """
    Merge 𝒟_Gen and 𝒟_Rationale into a single list of training examples.

    Returns
    -------
    tuple[list[dict], list[dict]]
        A (examples, file_infos) pair where *file_infos* records each loaded
        file's path, record count, and SHA-256 checksum for the corpus manifest.
    """
    examples: list[dict] = []
    file_infos: list[dict] = []

    d_gen_dir = data_root / "d_gen"
    d_rat_dir = data_root / "d_rationale"

    for jsonl_file in sorted(d_gen_dir.glob("*.jsonl")):
        file_records = _load_jsonl(jsonl_file)
        examples.extend(file_records)
        checksum = _sha256(jsonl_file)
        file_infos.append({"path": str(jsonl_file), "records": len(file_records), "sha256": checksum})
        logger.info("Loaded %d examples from %s", len(file_records), jsonl_file)

    for jsonl_file in sorted(d_rat_dir.glob("*.jsonl")):
        file_records = _load_jsonl(jsonl_file)
        examples.extend(file_records)
        checksum = _sha256(jsonl_file)
        file_infos.append({"path": str(jsonl_file), "records": len(file_records), "sha256": checksum})
        logger.info("Loaded %d examples from %s (rationale)", len(file_records), jsonl_file)

    return examples, file_infos


def build_width_dataset(
    data_root: Path,
    generation_index: int,
) -> tuple[list[dict], list[dict]]:
    """
    Load the Width training dataset for a specific generation index.

    The Width dataset lives under ``data_root/d_width/gen_{generation_index}/``.
    These files are produced by ``WidthDatasetBuilder`` and contain
    architecturally diverse lateral solutions for a single generational jump.

    Parameters
    ----------
    data_root:
        Root data directory.
    generation_index:
        The evolutionary generation index ``n`` to load Width data for.

    Returns
    -------
    tuple[list[dict], list[dict]]
        A (examples, file_infos) pair.
    """
    examples: list[dict] = []
    file_infos: list[dict] = []

    width_dir = data_root / "d_width" / f"gen_{generation_index}"
    if not width_dir.exists():
        raise FileNotFoundError(
            f"Width dataset directory not found: {width_dir}. "
            f"Run WidthDatasetBuilder for generation {generation_index} first."
        )

    for jsonl_file in sorted(width_dir.glob("*.jsonl")):
        file_records = _load_jsonl(jsonl_file)
        examples.extend(file_records)
        checksum = _sha256(jsonl_file)
        file_infos.append({
            "path": str(jsonl_file),
            "records": len(file_records),
            "sha256": checksum,
            "generation_index": generation_index,
        })
        logger.info(
            "Loaded %d Width examples from %s (gen_%d)",
            len(file_records), jsonl_file, generation_index,
        )

    return examples, file_infos


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of the file at *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_corpus_manifest(
    output_dir: Path,
    data_root: Path,
    file_infos: list[dict],
    model_name: str,
    cfg: dict,
    training_mode: str = "depth",
    generation_index: int | None = None,
) -> Path:
    """
    Write ``corpus_manifest.json`` into *output_dir*.

    Records dataset files, record counts, checksums, the model name, the
    training mode (``"depth"`` or ``"width"``), and the full config snapshot
    so each checkpoint is traceable to its training data.

    Parameters
    ----------
    output_dir:
        Directory to write the manifest into.
    data_root:
        Root data directory used during training.
    file_infos:
        Per-file records from ``build_dataset`` or ``build_width_dataset``.
    model_name:
        Base model name or checkpoint path.
    cfg:
        Full runtime config dict snapshotted into the manifest.
    training_mode:
        ``"depth"`` for the full-trajectory Depth Historian adapter or
        ``"width"`` for a single-generation Width Specialist adapter.
    generation_index:
        For ``training_mode="width"``, the generation index this adapter covers.

    Returns
    -------
    Path
        Path to the written manifest file.
    """
    from datetime import datetime, timezone

    manifest = {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "model_name": model_name,
        "data_root": str(data_root),
        "training_mode": training_mode,
        "generation_index": generation_index,
        "total_records": sum(fi["records"] for fi in file_infos),
        "files": file_infos,
        "config_snapshot": cfg,
    }
    manifest_path = output_dir / "corpus_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Corpus manifest written to %s", manifest_path)
    return manifest_path


# ---------------------------------------------------------------------------
# Core training logic (requires [finetune] extras)
# ---------------------------------------------------------------------------


def _load_training_examples(
    data_root: Path,
    training_mode: str,
    generation_index: int | None,
) -> tuple[list[dict], list[dict]]:
    """
    Dispatch to the correct dataset loader based on *training_mode*.

    Parameters
    ----------
    data_root:
        Root data directory.
    training_mode:
        ``"depth"`` → full corpus (𝒟_Gen + 𝒟_Rationale).
        ``"width"`` → Width dataset for *generation_index*.
    generation_index:
        Required when *training_mode* is ``"width"``.

    Returns
    -------
    tuple[list[dict], list[dict]]
        (examples, file_infos)
    """
    if training_mode == "width":
        if generation_index is None:
            raise ValueError("training_mode='width' requires generation_index to be set.")
        return build_width_dataset(data_root, generation_index)
    # default: "depth" — full trajectory corpus
    return build_dataset(data_root)


def train(
    cfg: dict,
    model_name: str,
    data_root: Path,
    output_dir: Path,
    training_mode: str = "depth",
    generation_index: int | None = None,
) -> None:
    """
    Fine-tune an LLM on the ETFT corpus using HuggingFace PEFT/LoRA.

    Parameters
    ----------
    cfg:
        Runtime configuration dict (from ``config.yaml``).
    model_name:
        HuggingFace model name or local path for the base model.
    data_root:
        Root data directory used to load the training corpus.
    output_dir:
        Directory to save the trained LoRA checkpoint.
    training_mode:
        ``"depth"`` — train on the full trajectory corpus (𝒟_Gen + 𝒟_Rationale).
        ``"width"`` — train on the Width dataset for a specific generation
        (requires ``generation_index`` to be set).
    generation_index:
        When ``training_mode="width"``, the generation index whose Width dataset
        is loaded from ``data_root/d_width/gen_{generation_index}/``.
    """
    try:
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Fine-tuning dependencies not installed.  "
            "Run: pip install 'autoscientist-etft[finetune]'"
        ) from exc

    train_cfg = cfg.get("training", {})
    val_split: float = float(train_cfg.get("val_split", 0.1))

    logger.info("Loading tokenizer and base model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")

    lora_config = LoraConfig(
        r=train_cfg.get("lora_r", 16),
        lora_alpha=train_cfg.get("lora_alpha", 32),
        lora_dropout=train_cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    raw_examples, file_infos = _load_training_examples(
        data_root, training_mode, generation_index
    )
    if not raw_examples:
        raise SystemExit(
            f"No training examples found under {data_root} "
            f"(mode={training_mode}, generation={generation_index}). "
            "Generate the corpus first."
        )

    # Write corpus manifest before training so the checkpoint is traceable
    write_corpus_manifest(
        output_dir, data_root, file_infos, model_name, cfg,
        training_mode=training_mode, generation_index=generation_index,
    )

    def tokenize(example: dict) -> dict:
        prompt = example.get("prompt", "")
        completion = example.get("completion", "")
        text = f"{prompt}{completion}"
        return tokenizer(text, truncation=True, max_length=2048)

    hf_dataset = Dataset.from_list(raw_examples).map(tokenize, remove_columns=raw_examples[0].keys())

    # Validation split
    if val_split > 0.0 and len(hf_dataset) >= 2:
        splits = hf_dataset.train_test_split(test_size=val_split, seed=42)
        train_dataset = splits["train"]
        eval_dataset = splits["test"]
        logger.info(
            "Dataset split: %d train / %d validation examples.",
            len(train_dataset), len(eval_dataset),
        )
    else:
        train_dataset = hf_dataset
        eval_dataset = None
        logger.info("No validation split (val_split=%.2f or too few examples).", val_split)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        fp16=train_cfg.get("fp16", True),
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        load_best_model_at_end=eval_dataset is not None,
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training …")
    trainer.train()
    logger.info("Training complete. Saving to %s", output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def eval_only(cfg: dict, model_name: str, data_root: Path, checkpoint_dir: Path) -> dict:
    """
    Evaluate a saved checkpoint against the held-out test split without training.

    Loads the corpus from *data_root*, applies the same tokenisation pipeline
    and validation split, then runs the HuggingFace ``Trainer.evaluate()``
    on the eval dataset.

    Returns
    -------
    dict
        HuggingFace evaluation metrics dict (e.g. ``{"eval_loss": 2.31, ...}``).
    """
    try:
        from datasets import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Fine-tuning dependencies not installed.  "
            "Run: pip install 'autoscientist-etft[finetune]'"
        ) from exc

    train_cfg = cfg.get("training", {})
    val_split: float = float(train_cfg.get("val_split", 0.1))

    logger.info("Loading checkpoint from %s for evaluation …", checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint_dir), torch_dtype="auto")

    raw_examples, _ = build_dataset(data_root)
    if not raw_examples:
        raise SystemExit(f"No examples found under {data_root}.")

    def tokenize(example: dict) -> dict:
        text = example.get("prompt", "") + example.get("completion", "")
        return tokenizer(text, truncation=True, max_length=2048)

    hf_dataset = Dataset.from_list(raw_examples).map(tokenize, remove_columns=raw_examples[0].keys())

    if val_split > 0.0 and len(hf_dataset) >= 2:
        splits = hf_dataset.train_test_split(test_size=val_split, seed=42)
        eval_dataset = splits["test"]
    else:
        eval_dataset = hf_dataset

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size",
                                                  train_cfg.get("per_device_train_batch_size", 4)),
        report_to="none",
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    metrics = trainer.evaluate()
    logger.info("Evaluation metrics: %s", metrics)
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune an LLM on the ETFT corpus (𝒟_Gen + 𝒟_Rationale) or on a "
            "Width dataset for a specific generation (Orthogonal Calibration)."
        )
    )
    parser.add_argument(
        "--dataset",
        default="./data",
        help="Path to the data root directory (default: ./data)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Base model name or path (overrides config.yaml training.base_model)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Checkpoint output directory (overrides config.yaml training.output_dir)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        default=False,
        help=(
            "Evaluate a saved checkpoint against the held-out validation split "
            "without training.  Requires --output-dir to point at a saved checkpoint."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["depth", "width"],
        default="depth",
        help=(
            "Training mode for Orthogonal Calibration:\n"
            "  depth  — train the Depth Historian on the full trajectory corpus "
            "(𝒟_Gen + 𝒟_Rationale). This is the default and produces a LoRA "
            "adapter that understands the macro-vector of innovation.\n"
            "  width  — train a Width Specialist on the lateral Width dataset for "
            "a specific generation. Requires --generation to be set."
        ),
    )
    parser.add_argument(
        "--generation",
        type=int,
        default=None,
        help=(
            "Generation index n for --mode width. "
            "Loads data/d_width/gen_{n}/ and saves the adapter to "
            "checkpoints/width_adapters/gen_{n}/ (unless --output-dir overrides)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    model_name = args.model or cfg["training"]["base_model"]
    data_root = Path(args.dataset)

    # Determine output directory based on training mode
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.mode == "width" and args.generation is not None:
        oc_cfg = cfg.get("orthogonal_calibration", {})
        width_adapter_root = oc_cfg.get("width", {}).get(
            "adapter_output_dir",
            cfg.get("training", {}).get("output_dir", "./checkpoints") + "/width_adapters",
        )
        output_dir = Path(width_adapter_root) / f"gen_{args.generation}"
    else:
        oc_cfg = cfg.get("orthogonal_calibration", {})
        depth_adapter_dir = oc_cfg.get("depth", {}).get(
            "adapter_output_dir",
            cfg.get("training", {}).get("output_dir", "./checkpoints"),
        )
        output_dir = Path(depth_adapter_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        eval_only(cfg, model_name, data_root, output_dir)
    else:
        train(
            cfg, model_name, data_root, output_dir,
            training_mode=args.mode,
            generation_index=args.generation,
        )


if __name__ == "__main__":
    main()
