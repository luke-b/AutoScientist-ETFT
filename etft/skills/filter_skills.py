"""
etft/skills/filter_skills.py — Skills wrapping the Probabilistic Heuristic
Filter for training and candidate triage.

All skills are pure Python — no LLM calls.
"""

from __future__ import annotations

from typing import Any

from etft.skills.base import Skill


class TrainPerformanceFilterSkill(Skill):
    """Train the RandomForest performance filter on 𝒟_Perf data."""

    name = "train_performance_filter"
    description = (
        "Train the Probabilistic Heuristic Filter (RandomForest) on 𝒟_Perf JSONL "
        "data and save the model to disk. Returns the output model path."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "data_dir": {
                "type": "string",
                "description": "Directory containing 𝒟_Perf JSONL files.",
            },
            "output_path": {
                "type": "string",
                "description": "Path to save the trained model (.joblib).",
                "default": "./checkpoints/perf_filter.joblib",
            },
        },
        "required": ["data_dir"],
    }

    def __init__(self, cfg: dict | None = None) -> None:
        self._cfg = cfg

    def execute(self, **kwargs: Any) -> dict:
        from pathlib import Path

        from corpus.performance_estimator.train_filter import train

        data_dir = Path(kwargs["data_dir"])
        output_path = Path(kwargs.get("output_path", "./checkpoints/perf_filter.joblib"))

        train(data_dir, output_path, self._cfg)
        return {"output_path": str(output_path), "success": True}


class TriageCandidateSkill(Skill):
    """Score a SOTA+1 candidate with the Probabilistic Filter."""

    name = "triage_candidate"
    description = (
        "Score a SOTA+1 candidate using the trained Probabilistic Heuristic Filter. "
        "Returns risk_score and triage_passed (risk_score ≤ threshold)."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "candidate": {
                "type": "object",
                "description": "SOTAPlusOneCandidate dict with candidate_id, trajectory_id, code, rationale.",
            },
            "model_path": {
                "type": "string",
                "description": "Path to trained filter model.",
                "default": "./checkpoints/perf_filter.joblib",
            },
        },
        "required": ["candidate"],
    }

    def __init__(self, cfg: dict | None = None) -> None:
        self._cfg = cfg

    def execute(self, **kwargs: Any) -> dict:
        from pathlib import Path

        from corpus.regression_pipeline.schemas import SOTAPlusOneCandidate
        from synthesis.sota_plus_one.triage import TriageFilter

        raw: dict = kwargs["candidate"]
        model_path = Path(kwargs.get("model_path", "./checkpoints/perf_filter.joblib"))

        candidate = SOTAPlusOneCandidate(
            candidate_id=raw.get("candidate_id", ""),
            trajectory_id=raw.get("trajectory_id", ""),
            code=raw.get("code", ""),
            rationale=raw.get("rationale", ""),
        )

        triage = TriageFilter(cfg=self._cfg, model_path=model_path)
        result = triage.evaluate(candidate)
        return {
            "candidate_id": result.candidate_id,
            "risk_score": result.risk_score,
            "triage_passed": result.triage_passed,
        }
