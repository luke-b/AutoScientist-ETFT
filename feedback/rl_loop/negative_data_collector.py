"""
feedback/rl_loop/negative_data_collector.py — Appends high-value failure
records to 𝒟_Perf (and optionally 𝒟_Rationale) to continuously harden the
Probabilistic Heuristic Filter.

Self-hardening by design: every failed SOTA+1 hypothesis, whether caught by
triage or discovered during physical evaluation, automatically populates the
negative dataset with zero manual annotation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from corpus.regression_pipeline.schemas import FailureRecord, PerfSample

logger = logging.getLogger(__name__)


class NegativeDataCollector:
    """Writes FailureRecord objects to the appropriate JSONL datasets."""

    def __init__(self, data_root: Path) -> None:
        self._d_perf_dir = data_root / "d_perf"
        self._d_rationale_dir = data_root / "d_rationale"
        self._d_perf_dir.mkdir(parents=True, exist_ok=True)
        self._d_rationale_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def collect(self, record: FailureRecord, also_rationale: bool = False) -> None:
        """
        Persist *record* to 𝒟_Perf (always) and optionally 𝒟_Rationale.

        Parameters
        ----------
        record:
            The FailureRecord to persist.
        also_rationale:
            If True, also write a rationale training example that teaches the
            LLM why this candidate failed.
        """
        self._append_d_perf(record)
        if also_rationale:
            self._append_d_rationale(record)

    # ------------------------------------------------------------------
    def _append_d_perf(self, record: FailureRecord) -> None:

        features = record.features or {}
        sample = PerfSample(
            algorithm_id=record.candidate_id or record.experiment_id or "unknown",
            features=features,
            label=1,  # failure
            failure_reason=record.failure_reason,
            metadata=record.metadata,
        )
        out_path = self._d_perf_dir / "feedback_failures.jsonl"
        with open(out_path, "a") as f:
            f.write(sample.model_dump_json() + "\n")
        logger.debug("Appended failure to %s", out_path)

    # ------------------------------------------------------------------
    def _append_d_rationale(self, record: FailureRecord) -> None:
        example = {
            "prompt": (
                f"# Candidate: {record.candidate_id or record.experiment_id}\n"
                f"# Source: {record.source}\n"
                f"# Task: Explain why this candidate failed and how to avoid it.\n"
            ),
            "completion": (
                f"This candidate failed because: {record.failure_reason}. "
                f"To avoid this failure, consider addressing the root cause "
                f"before submitting similar candidates."
            ),
        }
        out_path = self._d_rationale_dir / "feedback_rationale.jsonl"
        with open(out_path, "a") as f:
            f.write(json.dumps(example) + "\n")
        logger.debug("Appended rationale to %s", out_path)
