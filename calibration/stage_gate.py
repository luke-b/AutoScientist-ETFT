"""
calibration/stage_gate.py — Hard algorithmic stage-gate for SOTA+x synthesis.

The StageGate enforces that the model's Confidence Level C (mean reconstruction
similarity over a replayed trajectory) exceeds a configurable threshold before
speculative SOTA+x synthesis is permitted.

When the gate is closed the caller receives a ``StageGateResult`` with
``ready=False`` and a diagnostic explaining the shortfall, so the pipeline
can either abort or request additional calibration.

Usage
-----
    gate = StageGate(threshold=0.65)
    result = gate.evaluate(confidence_level=0.72, n_steps=5)
    if result.ready:
        # proceed to SOTA+1 synthesis
        ...
    else:
        logger.warning(result.diagnostic)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from corpus.regression_pipeline.schemas import ReconstructionResult

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 0.65


@dataclass(frozen=True)
class StageGateResult:
    """Outcome of a single stage-gate evaluation."""

    ready: bool
    confidence_level: Optional[float]
    threshold: float
    n_steps: int
    diagnostic: str

    def __str__(self) -> str:
        status = "OPEN" if self.ready else "CLOSED"
        c_str = f"{self.confidence_level:.3f}" if self.confidence_level is not None else "N/A"
        return (
            f"StageGate [{status}] "
            f"C={c_str} / threshold={self.threshold:.3f} "
            f"(n={self.n_steps} steps)"
        )


class StageGate:
    """
    Evaluates whether the Calibration Engine's Confidence Level (C) meets
    the minimum threshold required to unlock SOTA+x synthesis.

    Parameters
    ----------
    threshold:
        Minimum C value in [0, 1] required for ``ready=True``.
        Defaults to 0.65 (empirically conservative).
    min_steps:
        Minimum number of trajectory steps that must have been replayed
        before the gate can open.  Prevents the gate from opening on a
        trivially short (1-step) trajectory.
    min_step_similarity:
        Per-step minimum floor.  Even if the mean C ≥ threshold, the gate
        is closed when *any* individual step similarity falls below this
        value.  Set to 0.0 to disable.  Defaults to 0.35.
    recency_weighting:
        When *True*, later trajectory steps receive linearly higher weight
        when computing C (step n is weighted n / Σ(1..n)).  When *False*
        (default) an unweighted mean is used.
    max_agent_failure_rate:
        Fraction of attempted steps that may fail (agent returned
        ``success=False``) before the gate closes with an ``"agent_failure"``
        diagnostic rather than a low-C diagnostic.  Defaults to 0.5.
    """

    def __init__(
        self,
        threshold: float = _DEFAULT_THRESHOLD,
        min_steps: int = 2,
        min_step_similarity: float = 0.35,
        recency_weighting: bool = False,
        max_agent_failure_rate: float = 0.5,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1]; got {threshold!r}")
        if not 0.0 <= min_step_similarity <= 1.0:
            raise ValueError(
                f"min_step_similarity must be in [0, 1]; got {min_step_similarity!r}"
            )
        if not 0.0 <= max_agent_failure_rate <= 1.0:
            raise ValueError(
                f"max_agent_failure_rate must be in [0, 1]; got {max_agent_failure_rate!r}"
            )
        self.threshold = threshold
        self.min_steps = min_steps
        self.min_step_similarity = min_step_similarity
        self.recency_weighting = recency_weighting
        self.max_agent_failure_rate = max_agent_failure_rate

    # ------------------------------------------------------------------
    def _compute_confidence(self, results: list[ReconstructionResult]) -> float:
        """
        Compute C from a list of ``ReconstructionResult`` objects with valid scores.

        Applies recency weighting when ``self.recency_weighting`` is True:
        step *k* (1-indexed) receives weight *k* / Σ(1..n).
        """
        scores = [r.similarity_score for r in results if r.similarity_score is not None]
        if not scores:
            return 0.0
        if not self.recency_weighting:
            return sum(scores) / len(scores)
        n = len(scores)
        total_weight = n * (n + 1) / 2  # Σ(1..n)
        return sum((i + 1) * s for i, s in enumerate(scores)) / total_weight

    # ------------------------------------------------------------------
    def evaluate(
        self,
        confidence_level: Optional[float] = None,
        n_steps: int = 0,
        reconstruction_results: Optional[list[ReconstructionResult]] = None,
        n_steps_attempted: int = 0,
        n_agent_failures: int = 0,
    ) -> StageGateResult:
        """
        Determine whether C is sufficient to open the gate.

        Parameters
        ----------
        confidence_level:
            The pre-computed C from ``ReplaySession.confidence_level``.
            When *reconstruction_results* is provided this value is ignored
            and C is recomputed internally (enabling recency weighting).
        n_steps:
            Number of trajectory steps that were replayed (used to
            enforce the ``min_steps`` guard).
        reconstruction_results:
            Optional list of per-step results used for per-step floor checks
            and recency-weighted C computation.
        n_steps_attempted:
            Total steps attempted (including failures); used to compute the
            agent failure rate.
        n_agent_failures:
            Number of agent calls that returned ``success=False``.

        Returns
        -------
        StageGateResult
        """
        results = reconstruction_results or []

        # ------------------------------------------------------------------
        # 1. Agent failure rate gate
        # ------------------------------------------------------------------
        if n_steps_attempted > 0:
            failure_rate = n_agent_failures / n_steps_attempted
            if failure_rate >= self.max_agent_failure_rate:
                diagnostic = (
                    f"Agent failure rate {failure_rate:.0%} "
                    f"({n_agent_failures}/{n_steps_attempted} steps) "
                    f"≥ max_agent_failure_rate={self.max_agent_failure_rate:.0%}. "
                    "Too many reconstruction attempts failed — calibration result "
                    "is unreliable. SOTA+x synthesis is BLOCKED."
                )
                result = StageGateResult(
                    ready=False,
                    confidence_level=confidence_level,
                    threshold=self.threshold,
                    n_steps=n_steps,
                    diagnostic=diagnostic,
                )
                logger.warning("StageGate CLOSED (agent_failure) — %s", diagnostic)
                return result

        # ------------------------------------------------------------------
        # 2. Minimum replayed-step depth
        # ------------------------------------------------------------------
        if n_steps < self.min_steps:
            diagnostic = (
                f"Insufficient trajectory depth: {n_steps} step(s) replayed "
                f"(minimum {self.min_steps} required). "
                "Provide a longer trajectory or lower min_steps."
            )
            result = StageGateResult(
                ready=False,
                confidence_level=confidence_level,
                threshold=self.threshold,
                n_steps=n_steps,
                diagnostic=diagnostic,
            )
            logger.warning("StageGate CLOSED — %s", diagnostic)
            return result

        # ------------------------------------------------------------------
        # 3. (Re-)compute C — honours recency_weighting when results supplied
        # ------------------------------------------------------------------
        if results:
            c = self._compute_confidence(results)
        else:
            c = confidence_level if confidence_level is not None else 0.0

        # ------------------------------------------------------------------
        # 4. Per-step minimum floor
        # ------------------------------------------------------------------
        if self.min_step_similarity > 0.0 and results:
            failed_floors = [
                r for r in results
                if r.similarity_score is not None
                and r.similarity_score < self.min_step_similarity
            ]
            if failed_floors:
                worst = min(
                    r.similarity_score for r in failed_floors
                    if r.similarity_score is not None
                )
                diagnostic = (
                    f"Per-step minimum floor violated: "
                    f"{len(failed_floors)} step(s) have similarity < "
                    f"{self.min_step_similarity:.2f} "
                    f"(worst = {worst:.3f}). "
                    "Mean C may satisfy the threshold but individual steps show "
                    "catastrophic reconstruction failure. SOTA+x synthesis is BLOCKED."
                )
                result = StageGateResult(
                    ready=False,
                    confidence_level=c,
                    threshold=self.threshold,
                    n_steps=n_steps,
                    diagnostic=diagnostic,
                )
                logger.warning("StageGate CLOSED (floor) — %s", diagnostic)
                return result

        # ------------------------------------------------------------------
        # 5. Mean-C gate
        # ------------------------------------------------------------------
        ready = c >= self.threshold

        agent_note = ""
        if n_agent_failures > 0:
            agent_note = (
                f" Note: {n_agent_failures} agent failure(s) excluded from C computation."
            )

        if ready:
            diagnostic = (
                f"Confidence Level C={c:.3f} ≥ threshold={self.threshold:.3f}. "
                "Model is Ready for Action — SOTA+x synthesis unlocked."
                + agent_note
            )
            logger.info("StageGate OPEN — %s", diagnostic)
        else:
            shortfall = self.threshold - c
            diagnostic = (
                f"Confidence Level C={c:.3f} < threshold={self.threshold:.3f} "
                f"(shortfall: {shortfall:.3f}). "
                "Model has not demonstrated functional parity with historical SOTA. "
                "SOTA+x synthesis is BLOCKED until C ≥ threshold."
                + agent_note
            )
            logger.warning("StageGate CLOSED — %s", diagnostic)

        return StageGateResult(
            ready=ready,
            confidence_level=c,
            threshold=self.threshold,
            n_steps=n_steps,
            diagnostic=diagnostic,
        )
