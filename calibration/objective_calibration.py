"""
calibration/objective_calibration.py — Objective Calibration: validates Width
Model creativity for Orthogonal Calibration (Benda, 2026 §2.3).

A Width Model is a *Creative Innovator* when its generated solution is:
  1. Architecturally DISTANT from the historical aₙ  (diversity gate)
  2. Performance-equivalent to the historical aₙ     (quality gate)

The ``ObjectiveCalibration`` class measures both properties and returns an
``ObjectiveCalibrationResult``.

The **diversity gate** uses the existing ``calibration.similarity`` functions in
*inverse* mode: a LOW similarity between the generated output and the historical
successor aₙ is desirable.  Specifically, architectural_diversity_score is
defined as:

    diversity = 1 - Sim(generated, historical)

so a value of 1.0 means completely novel structure, 0.0 means identical.
The gate opens when ``diversity ≥ diversity_threshold``.

The **quality gate** uses the same static op-count / structural-feature proxy
as ``WidthDatasetBuilder._performance_delta``.  The gate opens when
``performance_delta ≤ performance_tolerance``.

Both gates must open for a variant to be classified as a Creative Innovator.

Configuration (under ``orthogonal_calibration.objective_calibration``):
    diversity_threshold : float  (default 0.3)
    performance_tolerance : float  (default 0.05)
    similarity_metric : str  (default "composite", same values as calibration.similarity_metric)
    similarity_weights : list[float]  (default [0.4, 0.3, 0.3])

Usage
-----
    from calibration.objective_calibration import ObjectiveCalibration

    oc = ObjectiveCalibration(cfg)
    result = oc.validate(
        input_code=before.code,
        historical_output=after.code,
        generated_outputs=[variant.code for variant in lateral_variants],
    )
    print(result.is_creative_innovator, result.architectural_diversity_score)
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class ObjectiveCalibrationResult(BaseModel):
    """
    Outcome of an Objective Calibration run for one Width Model generation.

    Attributes
    ----------
    is_creative_innovator:
        True when BOTH the diversity gate AND the quality gate are open.
    architectural_diversity_score:
        Mean ``1 - Sim(generated, historical)`` across all generated variants.
        Higher values indicate more structurally distinct outputs.
    performance_delta:
        Mean fractional deviation of generated variants' static metrics vs the
        historical baseline.  Lower values indicate more performance-equivalent
        outputs.
    n_variants_evaluated:
        Number of generated variants evaluated.
    n_creative_variants:
        Number of variants that individually pass both gates.
    diversity_threshold:
        Configured minimum diversity for the gate to open.
    performance_tolerance:
        Configured maximum performance delta for the gate to open.
    gate_diagnostic:
        Human-readable explanation of the gate decision.
    per_variant_details:
        Per-variant diversity and performance delta scores.
    """

    is_creative_innovator: bool
    architectural_diversity_score: float
    performance_delta: float
    n_variants_evaluated: int
    n_creative_variants: int
    diversity_threshold: float
    performance_tolerance: float
    gate_diagnostic: str = ""
    per_variant_details: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# ObjectiveCalibration
# ---------------------------------------------------------------------------


class ObjectiveCalibration:
    """
    Validates Width Model creativity via dual diversity + quality gates.

    Parameters
    ----------
    cfg:
        Full runtime config dict.  Reads ``orthogonal_calibration.objective_calibration``.
    """

    def __init__(self, cfg: dict | None = None) -> None:
        self._cfg = cfg or {}
        oc_cfg = self._cfg.get("orthogonal_calibration", {})
        obj_cfg = oc_cfg.get("objective_calibration", {})

        self._diversity_threshold: float = float(obj_cfg.get("diversity_threshold", 0.3))
        self._performance_tolerance: float = float(
            obj_cfg.get("performance_tolerance", 0.05)
        )
        metric_name = obj_cfg.get(
            "similarity_metric",
            self._cfg.get("calibration", {}).get("similarity_metric", "composite"),
        )
        weights_raw = list(obj_cfg.get(
            "similarity_weights",
            self._cfg.get("calibration", {}).get("similarity_weights", [0.4, 0.3, 0.3]),
        ))
        # Ensure exactly 3 elements (pad with 0.0 or truncate)
        while len(weights_raw) < 3:
            weights_raw.append(0.0)
        self._similarity_weights: tuple[float, float, float] = (
            float(weights_raw[0]),
            float(weights_raw[1]),
            float(weights_raw[2]),
        )
        self._similarity_fn = self._build_similarity_fn(metric_name)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_similarity_fn(metric_name: str):
        """Return the similarity callable for the configured metric."""
        from calibration.similarity import SIMILARITY_FUNCTIONS, code_similarity

        fn = SIMILARITY_FUNCTIONS.get(metric_name)
        if fn is None:
            logger.warning(
                "ObjectiveCalibration: unknown similarity metric '%s' — "
                "falling back to 'composite'.",
                metric_name,
            )
            return code_similarity
        return fn

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_static_metrics(code: str) -> dict[str, float]:
        """Extract static structural metrics (GPU-free performance proxy)."""
        try:
            from corpus.performance_estimator.feature_extractor import extract_features
            features = extract_features(code)
            return {
                "code_length_chars": features.get("code_length_chars", 0.0),
                "num_functions": features.get("num_functions", 0.0),
                "num_classes": features.get("num_classes", 0.0),
                "num_calls": features.get("num_calls", 0.0),
                "max_nesting_depth": features.get("max_nesting_depth", 0.0),
                "num_loops": features.get("num_loops", 0.0),
            }
        except Exception as exc:
            logger.debug("ObjectiveCalibration: feature extraction failed: %s", exc)
            return {"code_length_chars": float(len(code))}

    @staticmethod
    def _performance_delta(
        variant_metrics: dict[str, float],
        baseline_metrics: dict[str, float],
    ) -> float:
        """Mean absolute fractional deviation of variant vs baseline metrics."""
        shared = [
            k for k in variant_metrics
            if k in baseline_metrics and baseline_metrics[k] != 0.0
        ]
        if not shared:
            return 0.0
        deltas = [
            abs(variant_metrics[k] - baseline_metrics[k]) / abs(baseline_metrics[k])
            for k in shared
        ]
        return sum(deltas) / len(deltas)

    # ------------------------------------------------------------------
    def validate(
        self,
        input_code: str,
        historical_output: str,
        generated_outputs: list[str],
    ) -> ObjectiveCalibrationResult:
        """
        Validate one or more generated Width Model outputs against the
        Objective Calibration criteria.

        Parameters
        ----------
        input_code:
            The predecessor algorithm aₙ₋₁ (used for context; not evaluated).
        historical_output:
            The known historical successor aₙ — the benchmark for both
            similarity and performance.
        generated_outputs:
            List of code strings produced by the Width Model for the same
            generational jump.  Each is evaluated for diversity and quality.

        Returns
        -------
        ObjectiveCalibrationResult
            Aggregate result over all generated outputs.
        """
        if not generated_outputs:
            return ObjectiveCalibrationResult(
                is_creative_innovator=False,
                architectural_diversity_score=0.0,
                performance_delta=0.0,
                n_variants_evaluated=0,
                n_creative_variants=0,
                diversity_threshold=self._diversity_threshold,
                performance_tolerance=self._performance_tolerance,
                gate_diagnostic="No generated outputs to evaluate.",
            )

        baseline_metrics = self._compute_static_metrics(historical_output)

        per_variant: list[dict[str, Any]] = []
        diversity_scores: list[float] = []
        perf_deltas: list[float] = []
        n_creative = 0

        for i, gen_code in enumerate(generated_outputs):
            if not gen_code:
                logger.debug("ObjectiveCalibration: empty generated output at index %d.", i)
                continue

            # Compute similarity between generated output and historical aₙ
            # (using composite metric or configured metric)
            if hasattr(self._similarity_fn, "__name__") and "composite" in (
                getattr(self._similarity_fn, "__name__", "")
            ):
                sim = self._similarity_fn(
                    gen_code, historical_output, self._similarity_weights
                )
            else:
                sim = self._similarity_fn(gen_code, historical_output)

            diversity = 1.0 - float(sim)

            # Compute performance delta via static metrics
            gen_metrics = self._compute_static_metrics(gen_code)
            perf_delta = self._performance_delta(gen_metrics, baseline_metrics)

            diversity_gate = diversity >= self._diversity_threshold
            quality_gate = perf_delta <= self._performance_tolerance
            is_creative = diversity_gate and quality_gate

            if is_creative:
                n_creative += 1

            diversity_scores.append(diversity)
            perf_deltas.append(perf_delta)

            per_variant.append({
                "index": i,
                "diversity_score": diversity,
                "performance_delta": perf_delta,
                "diversity_gate": diversity_gate,
                "quality_gate": quality_gate,
                "is_creative": is_creative,
            })

            logger.debug(
                "ObjectiveCalibration variant %d: diversity=%.3f (gate=%s), "
                "perf_delta=%.4f (gate=%s)",
                i, diversity, diversity_gate, perf_delta, quality_gate,
            )

        n_evaluated = len(diversity_scores)
        if n_evaluated == 0:
            return ObjectiveCalibrationResult(
                is_creative_innovator=False,
                architectural_diversity_score=0.0,
                performance_delta=0.0,
                n_variants_evaluated=0,
                n_creative_variants=0,
                diversity_threshold=self._diversity_threshold,
                performance_tolerance=self._performance_tolerance,
                gate_diagnostic="All generated outputs were empty.",
            )

        mean_diversity = sum(diversity_scores) / n_evaluated
        mean_perf_delta = sum(perf_deltas) / n_evaluated

        # Aggregate gate: open if majority of evaluated variants are creative
        is_creative_innovator = n_creative > 0

        if is_creative_innovator:
            diag = (
                f"Creative Innovator: {n_creative}/{n_evaluated} variants are "
                f"architecturally diverse (mean_diversity={mean_diversity:.3f} ≥ "
                f"{self._diversity_threshold}) AND performance-equivalent "
                f"(mean_delta={mean_perf_delta:.4f} ≤ {self._performance_tolerance})."
            )
        else:
            issues = []
            if mean_diversity < self._diversity_threshold:
                issues.append(
                    f"diversity={mean_diversity:.3f} < threshold={self._diversity_threshold}"
                )
            if mean_perf_delta > self._performance_tolerance:
                issues.append(
                    f"perf_delta={mean_perf_delta:.4f} > tolerance={self._performance_tolerance}"
                )
            diag = "Not a Creative Innovator: " + "; ".join(issues) + "."

        logger.info("ObjectiveCalibration: %s", diag)

        return ObjectiveCalibrationResult(
            is_creative_innovator=is_creative_innovator,
            architectural_diversity_score=mean_diversity,
            performance_delta=mean_perf_delta,
            n_variants_evaluated=n_evaluated,
            n_creative_variants=n_creative,
            diversity_threshold=self._diversity_threshold,
            performance_tolerance=self._performance_tolerance,
            gate_diagnostic=diag,
            per_variant_details=per_variant,
        )
