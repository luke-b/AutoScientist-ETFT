"""
feedback/rl_loop/feedback_router.py — Routes failed experiments and triage
rejections to the correct negative dataset and constructs in-context RL
reward signals.
"""

from __future__ import annotations

import logging
from pathlib import Path

from corpus.regression_pipeline.schemas import ExperimentResult, FailureRecord, SOTAPlusOneCandidate
from feedback.rl_loop.negative_data_collector import NegativeDataCollector
from feedback.rl_loop.reward_signal import (
    format_for_context,
    from_experiment_failure,
    from_triage_rejection,
)

logger = logging.getLogger(__name__)


class FeedbackRouter:
    """
    Central hub that converts failures into FailureRecords, appends them to
    the negative datasets, and exposes reward signals for in-context RL.
    """

    def __init__(self, cfg: dict | None = None, data_root: Path | None = None) -> None:
        root = data_root or Path((cfg or {}).get("data", {}).get("root", "./data"))
        self._collector = NegativeDataCollector(root)
        self._reward_log: list[FailureRecord] = []

    # ------------------------------------------------------------------
    def route_experiment_failure(
        self, result: ExperimentResult, also_rationale: bool = False
    ) -> FailureRecord:
        """Process a failed micro-experiment."""
        record = from_experiment_failure(result)
        self._collector.collect(record, also_rationale=also_rationale)
        self._reward_log.append(record)
        logger.info(
            "Routed experiment failure %s → d_perf (reward=%.2f)",
            record.experiment_id, record.reward_signal,
        )
        return record

    # ------------------------------------------------------------------
    def route_triage_failure(
        self, candidate: SOTAPlusOneCandidate, also_rationale: bool = True
    ) -> FailureRecord:
        """Process a triage-rejected SOTA+1 candidate."""
        record = from_triage_rejection(candidate)
        self._collector.collect(record, also_rationale=also_rationale)
        self._reward_log.append(record)
        logger.info(
            "Routed triage rejection %s → d_perf + d_rationale (reward=%.2f)",
            record.candidate_id, record.reward_signal,
        )
        return record

    # ------------------------------------------------------------------
    def build_rl_context_prefix(self) -> str:
        """
        Return a formatted string of all accumulated negative reward signals
        suitable for prepending to an LLM prompt (in-context RL injection).
        """
        if not self._reward_log:
            return ""
        lines = ["=== IN-CONTEXT RL FEEDBACK (negative signals) ==="]
        for record in self._reward_log:
            lines.append(format_for_context(record))
        lines.append("=== END FEEDBACK ===\n")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    @property
    def reward_log(self) -> list[FailureRecord]:
        """All accumulated FailureRecords since instantiation."""
        return list(self._reward_log)
