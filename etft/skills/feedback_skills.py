"""
etft/skills/feedback_skills.py — Skills wrapping the FeedbackRouter for
routing experiment failures and triage rejections to RL datasets.

All skills are pure Python — no LLM calls.
"""

from __future__ import annotations

from typing import Any

from etft.skills.base import Skill


class RouteExperimentFailureSkill(Skill):
    """Route a failed experiment result to the negative dataset."""

    name = "route_experiment_failure"
    description = (
        "Process a failed micro-experiment result: generate a FailureRecord, "
        "append it to the negative dataset (𝒟_Perf), and record a reward signal."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "result": {
                "type": "object",
                "description": "ExperimentResult dict (from run_experiment).",
            },
            "also_rationale": {
                "type": "boolean",
                "description": "Whether to also write to the rationale dataset.",
                "default": False,
            },
        },
        "required": ["result"],
    }

    def __init__(self, cfg: dict | None = None, data_root: str | None = None) -> None:
        self._cfg = cfg
        self._data_root = data_root

    def execute(self, **kwargs: Any) -> dict:
        from pathlib import Path

        from corpus.regression_pipeline.schemas import ExperimentResult
        from feedback.rl_loop.feedback_router import FeedbackRouter

        raw: dict = kwargs["result"]
        also_rationale: bool = bool(kwargs.get("also_rationale", False))

        result = ExperimentResult(
            experiment_id=raw.get("experiment_id", ""),
            script=raw.get("script", ""),
            success=bool(raw.get("success", False)),
            metrics={k: float(v) for k, v in raw.get("metrics", {}).items()},
            error_message=raw.get("error_message"),
        )

        data_root = Path(self._data_root) if self._data_root else None
        router = FeedbackRouter(cfg=self._cfg, data_root=data_root)
        record = router.route_experiment_failure(result, also_rationale=also_rationale)
        return {
            "experiment_id": record.experiment_id,
            "reward_signal": record.reward_signal,
        }


class RouteTriageFailureSkill(Skill):
    """Route a triage-rejected SOTA+1 candidate to the negative dataset."""

    name = "route_triage_failure"
    description = (
        "Process a triage-rejected SOTA+1 candidate: generate a FailureRecord, "
        "append it to 𝒟_Perf and 𝒟_Rationale, and record a reward signal."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "candidate": {
                "type": "object",
                "description": "SOTAPlusOneCandidate dict with candidate_id, trajectory_id, code, rationale.",
            },
            "also_rationale": {
                "type": "boolean",
                "description": "Whether to write to the rationale dataset.",
                "default": True,
            },
        },
        "required": ["candidate"],
    }

    def __init__(self, cfg: dict | None = None, data_root: str | None = None) -> None:
        self._cfg = cfg
        self._data_root = data_root

    def execute(self, **kwargs: Any) -> dict:
        from pathlib import Path

        from corpus.regression_pipeline.schemas import SOTAPlusOneCandidate
        from feedback.rl_loop.feedback_router import FeedbackRouter

        raw: dict = kwargs["candidate"]
        also_rationale: bool = bool(kwargs.get("also_rationale", True))

        candidate = SOTAPlusOneCandidate(
            candidate_id=raw.get("candidate_id", ""),
            trajectory_id=raw.get("trajectory_id", ""),
            code=raw.get("code", ""),
            rationale=raw.get("rationale", ""),
        )

        data_root = Path(self._data_root) if self._data_root else None
        router = FeedbackRouter(cfg=self._cfg, data_root=data_root)
        record = router.route_triage_failure(candidate, also_rationale=also_rationale)
        return {
            "candidate_id": record.candidate_id,
            "reward_signal": record.reward_signal,
        }


class BuildRLContextSkill(Skill):
    """Build an in-context RL feedback prefix from accumulated failure records."""

    name = "build_rl_context"
    description = (
        "Build a formatted string of accumulated negative reward signals "
        "suitable for prepending to an LLM prompt (in-context RL injection)."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "failure_records": {
                "type": "array",
                "description": "List of FailureRecord dicts to format as RL context.",
                "items": {"type": "object"},
            },
        },
        "required": ["failure_records"],
    }

    def execute(self, **kwargs: Any) -> dict:
        from feedback.rl_loop.reward_signal import format_for_context

        records_raw: list[dict] = kwargs["failure_records"]

        if not records_raw:
            return {"rl_context": ""}

        lines = ["=== IN-CONTEXT RL FEEDBACK (negative signals) ==="]
        for r in records_raw:
            # Build a minimal fake FailureRecord-like object for format_for_context
            class _FakeRecord:
                pass

            rec = _FakeRecord()
            for k, v in r.items():
                setattr(rec, k, v)
            lines.append(format_for_context(rec))  # type: ignore[arg-type]
        lines.append("=== END FEEDBACK ===\n")

        return {"rl_context": "\n".join(lines)}
