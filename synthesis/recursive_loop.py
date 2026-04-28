"""
synthesis/recursive_loop.py — Recursive SOTA+2 Generator for Orthogonal
Calibration (Benda, 2026 §3).

``RecursiveSOTAGenerator`` implements the auto-regressive SOTA+x discovery loop:

  k=1 (SOTA+1):
    1. Call DynamicLoRARouter with current Width LoRA Lₙ
    2. Validate the result in the sandbox (syntax + subprocess execution)
    3. If valid → promote to new SOTA, append to trajectory

  k=2 (SOTA+2):
    4. Train a new Width LoRA Lₙ₊₁ on-the-fly using the validated SOTA+1
       as the single-jump training example
    5. Register Lₙ₊₁ in AdapterLibrary
    6. Hot-swap Lₙ₊₁ and re-run synthesis → SOTA+2

The loop terminates when ``max_depth`` is reached OR the sandbox validation
fails.

Each iteration produces a ``RecursiveResult`` persisted to
``data/candidates/recursive/``.

Usage
-----
    from synthesis.recursive_loop import RecursiveSOTAGenerator

    generator = RecursiveSOTAGenerator(cfg)
    results = generator.generate(
        sota_code=current_sota,
        generation_index=n,
        max_depth=2,
        output_dir=Path("data/candidates/recursive"),
    )
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class RecursiveResult(BaseModel):
    """
    Outcome of one iteration of the Recursive SOTA+x loop.

    Attributes
    ----------
    depth:
        Iteration depth (1 = SOTA+1, 2 = SOTA+2, …).
    candidate_id:
        Unique identifier for this candidate.
    code:
        Generated Python source code.
    rationale:
        Explanation of the architectural choices.
    adapter_id:
        Generation index of the Width LoRA used.
    sandbox_passed:
        Whether the generated code passed sandbox validation.
    sandbox_metrics:
        Static metrics extracted by ``extract_features``.
    passed_gate:
        Whether the Objective Calibration gate was open for this depth.
    metadata:
        Additional provenance information.
    """

    depth: int
    candidate_id: str
    code: str
    rationale: str = ""
    adapter_id: int | None = None
    sandbox_passed: bool = False
    sandbox_metrics: dict[str, Any] = Field(default_factory=dict)
    passed_gate: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# RecursiveSOTAGenerator
# ---------------------------------------------------------------------------


class RecursiveSOTAGenerator:
    """
    Auto-regressive SOTA+2 generator via iterative LoRA training + hot-swap.

    Parameters
    ----------
    cfg:
        Full runtime config dict.
    adapter_library:
        Optional ``AdapterLibrary`` instance.  Created from cfg if not provided.
    """

    def __init__(
        self,
        cfg: dict | None = None,
        adapter_library: Any = None,
    ) -> None:
        self._cfg = cfg or {}
        oc_cfg = self._cfg.get("orthogonal_calibration", {})
        rec_cfg = oc_cfg.get("recursive", {})

        self._max_depth: int = int(rec_cfg.get("max_sota_plus_x", 2))

        if adapter_library is not None:
            self._library = adapter_library
        else:
            from synthesis.lora_routing.adapter_library import AdapterLibrary
            lr_cfg = self._cfg.get("lora_routing", {})
            index_path = lr_cfg.get(
                "adapter_index_path",
                self._cfg.get("training", {}).get("output_dir", "./checkpoints")
                + "/adapter_index.json",
            )
            self._library = AdapterLibrary(index_path=Path(index_path))

    # ------------------------------------------------------------------
    def generate(
        self,
        sota_code: str,
        generation_index: int,
        max_depth: int | None = None,
        trajectory_steps: list | None = None,
        output_dir: Path | None = None,
    ) -> list[RecursiveResult]:
        """
        Run the Recursive SOTA+x discovery loop.

        Parameters
        ----------
        sota_code:
            Source code of the starting SOTA (SOTA+0).
        generation_index:
            Generation index of the current SOTA (used to select Width LoRA Lₙ).
        max_depth:
            Maximum number of successive SOTA+x hypotheses to attempt.
            Defaults to ``cfg.orthogonal_calibration.recursive.max_sota_plus_x``.
        trajectory_steps:
            Ordered ``TrajectoryStep`` list used by the depth analysis phase.
        output_dir:
            Directory to persist ``RecursiveResult`` JSON files.  Defaults to
            ``./data/candidates/recursive/``.

        Returns
        -------
        list[RecursiveResult]
            One ``RecursiveResult`` per depth level attempted (including failed).
        """
        effective_depth = max_depth if max_depth is not None else self._max_depth
        out_dir = output_dir or Path(
            self._cfg.get("data", {}).get("root", "./data")
        ) / "candidates" / "recursive"
        out_dir.mkdir(parents=True, exist_ok=True)

        results: list[RecursiveResult] = []
        current_sota = sota_code
        current_gen = generation_index
        steps = list(trajectory_steps or [])

        for depth in range(1, effective_depth + 1):
            logger.info(
                "RecursiveSOTAGenerator: depth=%d (SOTA+%d, gen=%d) …",
                depth, depth, current_gen,
            )

            # ── Step 1: Generate SOTA+k via DynamicLoRARouter ─────────────
            result = self._run_router(
                sota_code=current_sota,
                generation_index=current_gen,
                depth=depth,
                trajectory_steps=steps,
            )

            # ── Step 2: Sandbox validation ────────────────────────────────
            result = self._sandbox_validate(result, current_sota)

            # ── Step 3: Objective Calibration gate ────────────────────────
            result = self._objective_calibration_gate(
                result=result,
                historical_output=current_sota,
            )

            results.append(result)
            self._persist(result, out_dir)

            if not result.sandbox_passed:
                logger.warning(
                    "RecursiveSOTAGenerator: SOTA+%d failed sandbox — halting loop.",
                    depth,
                )
                break

            if not result.passed_gate:
                logger.warning(
                    "RecursiveSOTAGenerator: SOTA+%d did not pass Objective "
                    "Calibration gate — halting loop.",
                    depth,
                )
                break

            # ── Step 4: Promote and extend trajectory ─────────────────────
            current_sota = result.code
            current_gen = current_gen + 1

            # Append the new SOTA as a trajectory step
            steps = self._extend_trajectory(steps, current_sota, current_gen)

            # ── Step 5: On-the-fly Width LoRA training for next depth ─────
            if depth < effective_depth:
                self._train_next_lora(
                    before_code=results[-1].metadata.get("predecessor_code", sota_code)
                    if results else sota_code,
                    after_code=current_sota,
                    next_gen=current_gen,
                )

        logger.info(
            "RecursiveSOTAGenerator: completed %d depth(s), %d result(s) accepted.",
            len(results),
            sum(1 for r in results if r.sandbox_passed and r.passed_gate),
        )
        return results

    # ------------------------------------------------------------------
    def _run_router(
        self,
        sota_code: str,
        generation_index: int,
        depth: int,
        trajectory_steps: list,
    ) -> RecursiveResult:
        """Run the DynamicLoRARouter and build a RecursiveResult."""
        try:
            from synthesis.lora_routing.router import DynamicLoRARouter
            router = DynamicLoRARouter(cfg=self._cfg, adapter_library=self._library)
            synthesis = router.route(
                sota_code=sota_code,
                trajectory_steps=trajectory_steps,
                generation_index=generation_index,
            )
            code = synthesis.code
            rationale = synthesis.rationale
            adapter_path = synthesis.adapter_used
            adapter_id = generation_index if synthesis.success else None
        except Exception as exc:
            logger.warning(
                "RecursiveSOTAGenerator: router failed at depth=%d: %s", depth, exc
            )
            code = ""
            rationale = ""
            adapter_path = None
            adapter_id = None

        return RecursiveResult(
            depth=depth,
            candidate_id=str(uuid.uuid4())[:8],
            code=code,
            rationale=rationale,
            adapter_id=adapter_id,
            sandbox_passed=False,
            passed_gate=False,
            metadata={
                "predecessor_code": sota_code,
                "adapter_path": adapter_path,
                "generation_index": generation_index,
            },
        )

    # ------------------------------------------------------------------
    def _sandbox_validate(self, result: RecursiveResult, predecessor_code: str) -> RecursiveResult:
        """Run syntax + subprocess validation and update result flags."""
        if not result.code:
            return result.model_copy(update={"sandbox_passed": False})

        try:
            from etft.sandbox import DockerSandbox
            sandbox = DockerSandbox(self._cfg)
            sb_result = sandbox.run_script(result.code, timeout=30)
            passed = sb_result.returncode == 0
            if not passed:
                logger.warning(
                    "RecursiveSOTAGenerator: sandbox failed at depth=%d "
                    "(returncode=%d): %s",
                    result.depth, sb_result.returncode, sb_result.stderr[:200],
                )
        except Exception as exc:
            logger.warning(
                "RecursiveSOTAGenerator: sandbox error at depth=%d: %s",
                result.depth, exc,
            )
            passed = False

        # Compute static metrics regardless of sandbox outcome
        try:
            from corpus.performance_estimator.feature_extractor import extract_features
            metrics = extract_features(result.code)
        except Exception:
            metrics = {}

        return result.model_copy(update={
            "sandbox_passed": passed,
            "sandbox_metrics": metrics,
        })

    # ------------------------------------------------------------------
    def _objective_calibration_gate(
        self,
        result: RecursiveResult,
        historical_output: str,
    ) -> RecursiveResult:
        """Run Objective Calibration and update the passed_gate flag."""
        if not result.code:
            return result.model_copy(update={"passed_gate": False})

        try:
            from calibration.objective_calibration import ObjectiveCalibration
            oc = ObjectiveCalibration(self._cfg)
            cal_result = oc.validate(
                input_code=result.metadata.get("predecessor_code", ""),
                historical_output=historical_output,
                generated_outputs=[result.code],
            )
            passed = cal_result.is_creative_innovator
            gate_meta = {
                "diversity_score": cal_result.architectural_diversity_score,
                "performance_delta": cal_result.performance_delta,
                "gate_diagnostic": cal_result.gate_diagnostic,
            }
        except Exception as exc:
            logger.warning(
                "RecursiveSOTAGenerator: ObjectiveCalibration error at depth=%d: %s",
                result.depth, exc,
            )
            passed = True  # fail open — do not block the loop on calibration errors
            gate_meta = {"gate_diagnostic": f"calibration_error: {exc}"}

        updated_metadata = {**result.metadata, **gate_meta}
        return result.model_copy(update={
            "passed_gate": passed,
            "metadata": updated_metadata,
        })

    # ------------------------------------------------------------------
    @staticmethod
    def _extend_trajectory(
        steps: list,
        new_code: str,
        new_gen: int,
    ) -> list:
        """Append a new TrajectoryStep representing the promoted SOTA."""
        try:
            from corpus.regression_pipeline.schemas import TrajectoryStep
            new_step = TrajectoryStep(
                step_index=new_gen,
                algorithm_id=f"recursive_g{new_gen}",
                algorithm_family="recursive_synthesis",
                code=new_code,
                fitness_score=1.0 + new_gen * 0.1,
                metadata={"source": "recursive_sota_generator", "generation": new_gen},
            )
            return steps + [new_step]
        except Exception as exc:
            logger.debug("RecursiveSOTAGenerator: could not extend trajectory: %s", exc)
            return steps

    # ------------------------------------------------------------------
    def _train_next_lora(
        self,
        before_code: str,
        after_code: str,
        next_gen: int,
    ) -> None:
        """
        Train a new Width LoRA Lₙ₊₁ on-the-fly from a single SOTA+k jump.

        Writes a temporary JSONL dataset to ``data/d_width/gen_{next_gen}/``
        and invokes ``train.train()`` in Width mode.  On completion the
        adapter is registered in ``self._library``.
        """
        import tempfile

        oc_cfg = self._cfg.get("orthogonal_calibration", {})
        width_cfg = oc_cfg.get("width", {})
        adapter_root = Path(
            width_cfg.get(
                "adapter_output_dir",
                self._cfg.get("training", {}).get("output_dir", "./checkpoints")
                + "/width_adapters",
            )
        )
        adapter_dir = adapter_root / f"gen_{next_gen}"

        data_root = Path(self._cfg.get("data", {}).get("root", "./data"))
        width_dir = data_root / "d_width" / f"gen_{next_gen}"
        width_dir.mkdir(parents=True, exist_ok=True)

        # Write a minimal single-jump Width training example
        example = {
            "prompt": (
                f"# Orthogonal Width Training — Generation {next_gen}\n"
                f"# Task: Given the predecessor algorithm below, propose a NOVEL, "
                f"architecturally diverse successor.\n\n"
                f"{before_code}"
            ),
            "completion": after_code,
        }
        jsonl_path = width_dir / f"width_gen_{next_gen}.jsonl"
        with open(jsonl_path, "w") as fh:
            fh.write(json.dumps(example) + "\n")

        logger.info(
            "RecursiveSOTAGenerator: training Width LoRA for gen_%d …", next_gen
        )

        try:
            from train import train as _train
            train_cfg = self._cfg.get("training", {})
            model_name = train_cfg.get("base_model", "meta-llama/Meta-Llama-3-8B")
            _train(
                cfg=self._cfg,
                model_name=model_name,
                data_root=data_root,
                output_dir=adapter_dir,
                training_mode="width",
                generation_index=next_gen,
            )
            logger.info(
                "RecursiveSOTAGenerator: Width LoRA gen_%d trained → %s",
                next_gen, adapter_dir,
            )
        except SystemExit as exc:
            logger.warning(
                "RecursiveSOTAGenerator: Width LoRA training skipped "
                "(finetune deps not installed): %s",
                exc,
            )
        except Exception as exc:
            logger.warning(
                "RecursiveSOTAGenerator: Width LoRA training failed for gen_%d: %s",
                next_gen, exc,
            )
            return

        # Register the adapter (even if training was skipped — path is still valid
        # if the caller pre-populated the directory)
        if adapter_dir.exists():
            self._library.register(
                generation_index=next_gen,
                adapter_path=adapter_dir,
                metadata={"source": "recursive_on_the_fly", "generation": next_gen},
            )

    # ------------------------------------------------------------------
    @staticmethod
    def _persist(result: RecursiveResult, out_dir: Path) -> None:
        """Save a RecursiveResult JSON file to *out_dir*."""
        try:
            out_path = out_dir / f"recursive_depth{result.depth}_{result.candidate_id}.json"
            out_path.write_text(result.model_dump_json(indent=2))
            logger.info("RecursiveSOTAGenerator: result saved to %s", out_path)
        except Exception as exc:
            logger.debug("RecursiveSOTAGenerator: could not persist result: %s", exc)
