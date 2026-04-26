"""
train.py — Top-level fine-tuning entrypoint for AutoScientist-ETFT.

Trains an LLM on the ETFT corpus (𝒟_Gen + 𝒟_Rationale) using HuggingFace
PEFT/LoRA.  Adjust hyperparameters via config.yaml or CLI flags.

Usage:
    python train.py --dataset ./data --model meta-llama/Meta-Llama-3-8B
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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


def build_dataset(data_root: Path) -> list[dict]:
    """Merge 𝒟_Gen and 𝒟_Rationale into a single list of training examples."""
    examples: list[dict] = []

    d_gen_dir = data_root / "d_gen"
    d_rat_dir = data_root / "d_rationale"

    for jsonl_file in sorted(d_gen_dir.glob("*.jsonl")):
        examples.extend(_load_jsonl(jsonl_file))
        logger.info("Loaded %d examples from %s", len(examples), jsonl_file)

    for jsonl_file in sorted(d_rat_dir.glob("*.jsonl")):
        examples.extend(_load_jsonl(jsonl_file))
        logger.info("Loaded %d examples from %s (rationale)", len(examples), jsonl_file)

    return examples


# ---------------------------------------------------------------------------
# Core training logic (requires [finetune] extras)
# ---------------------------------------------------------------------------

def train(cfg: dict, model_name: str, data_root: Path, output_dir: Path) -> None:
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

    raw_examples = build_dataset(data_root)
    if not raw_examples:
        raise SystemExit(f"No training examples found under {data_root}. Generate the corpus first.")

    def tokenize(example: dict) -> dict:
        prompt = example.get("prompt", "")
        completion = example.get("completion", "")
        text = f"{prompt}{completion}"
        return tokenizer(text, truncation=True, max_length=2048)

    hf_dataset = Dataset.from_list(raw_examples).map(tokenize, remove_columns=raw_examples[0].keys())

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        fp16=train_cfg.get("fp16", True),
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training …")
    trainer.train()
    logger.info("Training complete. Saving to %s", output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune an LLM on the ETFT corpus (𝒟_Gen + 𝒟_Rationale)."
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    model_name = args.model or cfg["training"]["base_model"]
    output_dir = Path(args.output_dir or cfg["training"]["output_dir"])
    data_root = Path(args.dataset)

    output_dir.mkdir(parents=True, exist_ok=True)

    train(cfg, model_name, data_root, output_dir)


if __name__ == "__main__":
    main()
