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

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 0.65


@dataclass(frozen=True)
class StageGateResult:
    """Outcome of a single stage-gate evaluation."""

    ready: bool
    confidence_level: float
    threshold: float
    n_steps: int
    diagnostic: str

    def __str__(self) -> str:
        status = "OPEN" if self.ready else "CLOSED"
        return (
            f"StageGate [{status}] "
            f"C={self.confidence_level:.3f} / threshold={self.threshold:.3f} "
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
    """

    def __init__(
        self,
        threshold: float = _DEFAULT_THRESHOLD,
        min_steps: int = 2,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1]; got {threshold!r}")
        self.threshold = threshold
        self.min_steps = min_steps

    # ------------------------------------------------------------------
    def evaluate(self, confidence_level: float, n_steps: int) -> StageGateResult:
        """
        Determine whether C is sufficient to open the gate.

        Parameters
        ----------
        confidence_level:
            The computed C from ``ReplaySession.confidence_level``.
        n_steps:
            Number of trajectory steps that were replayed (used to
            enforce the ``min_steps`` guard).

        Returns
        -------
        StageGateResult
        """
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

        ready = confidence_level >= self.threshold
        if ready:
            diagnostic = (
                f"Confidence Level C={confidence_level:.3f} ≥ threshold={self.threshold:.3f}. "
                "Model is Ready for Action — SOTA+x synthesis unlocked."
            )
            logger.info("StageGate OPEN — %s", diagnostic)
        else:
            shortfall = self.threshold - confidence_level
            diagnostic = (
                f"Confidence Level C={confidence_level:.3f} < threshold={self.threshold:.3f} "
                f"(shortfall: {shortfall:.3f}). "
                "Model has not demonstrated functional parity with historical SOTA. "
                "SOTA+x synthesis is BLOCKED until C ≥ threshold."
            )
            logger.warning("StageGate CLOSED — %s", diagnostic)

        return StageGateResult(
            ready=ready,
            confidence_level=confidence_level,
            threshold=self.threshold,
            n_steps=n_steps,
            diagnostic=diagnostic,
        )
