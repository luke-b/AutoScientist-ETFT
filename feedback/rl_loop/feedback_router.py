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
    from_physical_eval_failure,
    from_triage_rejection,
)

logger = logging.getLogger(__name__)


class FeedbackRouter:
    """
    Central hub that converts failures into FailureRecords, appends them to
    the negative datasets, and exposes reward signals for in-context RL.
    """

    def __init__(
        self,
        cfg: dict | None = None,
        data_root: Path | None = None,
        load_history: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        cfg:
            Full runtime configuration dict.
        data_root:
            Root directory for data sub-directories.  Defaults to
            ``cfg['data']['root']`` or ``./data``.
        load_history:
            When *True* the router pre-populates its in-memory reward log from
            ``d_perf/failure_records.jsonl`` so that in-context RL feedback
            accumulated across previous ARL runs is available immediately.
        """
        root = data_root or Path((cfg or {}).get("data", {}).get("root", "./data"))
        self._collector = NegativeDataCollector(root)
        self._reward_log: list[FailureRecord] = []

        if load_history:
            self._load_history()

    # ------------------------------------------------------------------
    def _load_history(self) -> None:
        """Pre-populate the reward log from persisted failure records on disk."""
        count = 0
        for record in self._collector.iter_records():
            self._reward_log.append(record)
            count += 1
        if count:
            logger.info(
                "FeedbackRouter: loaded %d historical FailureRecords for in-context RL.",
                count,
            )

    # ------------------------------------------------------------------
    @classmethod
    def load_from_disk(
        cls,
        data_root: Path,
        cfg: dict | None = None,
    ) -> FeedbackRouter:
        """
        Convenience factory that creates a :class:`FeedbackRouter` with the
        full historical failure log pre-loaded from *data_root*.

        Parameters
        ----------
        data_root:
            Directory containing the ``d_perf/`` sub-directory.
        cfg:
            Optional runtime configuration dict.

        Returns
        -------
        FeedbackRouter
            A router whose :attr:`reward_log` already contains all
            previously persisted :class:`FailureRecord` objects.
        """
        return cls(cfg=cfg, data_root=data_root, load_history=True)

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
    def route_physical_eval_failure(
        self,
        candidate: SOTAPlusOneCandidate,
        failure_reason: str,
        also_rationale: bool = True,
    ) -> FailureRecord:
        """
        Process a candidate that failed physical GPU/cluster evaluation.

        Physical evaluation failures carry a stronger negative reward (-2.0)
        than micro-experiment failures (-1.0) to reflect the higher cost of
        committed cluster resources.

        Parameters
        ----------
        candidate:
            The SOTA+1 candidate that failed physical evaluation.
        failure_reason:
            Human-readable description of the failure (e.g. "OOM on H100",
            "divergent training loss after 500 steps").
        also_rationale:
            When True (default), also write a rationale training example to
            𝒟_Rationale so the fine-tuned model learns from this failure.

        Returns
        -------
        FailureRecord
            The persisted failure record (reward_signal = -2.0).
        """
        record = from_physical_eval_failure(candidate, failure_reason)
        self._collector.collect(record, also_rationale=also_rationale)
        self._reward_log.append(record)
        logger.warning(
            "Routed physical eval failure %s → d_perf + d_rationale (reward=%.2f): %s",
            record.candidate_id, record.reward_signal, failure_reason,
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
