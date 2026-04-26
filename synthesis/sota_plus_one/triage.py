"""
synthesis/sota_plus_one/triage.py — Applies the Probabilistic Heuristic Filter
to a SOTAPlusOneCandidate before GPU submission.
"""

from __future__ import annotations

import logging
from pathlib import Path

from corpus.performance_estimator.filter_model import ProbabilisticFilter
from corpus.regression_pipeline.schemas import SOTAPlusOneCandidate

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = Path("checkpoints/perf_filter.joblib")


class TriageFilter:
    """Scores a SOTA+1 candidate and decides whether to pass or reject it."""

    def __init__(self, cfg: dict | None = None, model_path: Path | None = None) -> None:
        path = model_path or _DEFAULT_MODEL_PATH
        self._filter = ProbabilisticFilter(model_path=path)
        synth_cfg = (cfg or {}).get("synthesis", {}).get("triage", {})
        self.risk_threshold: float = float(synth_cfg.get("risk_threshold", 0.7))

    # ------------------------------------------------------------------
    def evaluate(self, candidate: SOTAPlusOneCandidate) -> SOTAPlusOneCandidate:
        """
        Score *candidate* and return an updated copy with risk_score and
        triage_passed set.
        """
        risk = self._filter.predict_failure_probability(candidate.code)
        passed = risk <= self.risk_threshold

        if passed:
            logger.info(
                "Candidate %s passed triage (risk=%.3f ≤ %.3f).",
                candidate.candidate_id, risk, self.risk_threshold,
            )
        else:
            logger.warning(
                "Candidate %s REJECTED by triage (risk=%.3f > %.3f).",
                candidate.candidate_id, risk, self.risk_threshold,
            )

        return candidate.model_copy(update={"risk_score": risk, "triage_passed": passed})
