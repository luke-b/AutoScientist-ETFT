"""
etft/skills/experiment_skills.py — Skills wrapping experiment execution and
metrics collection.

All skills are pure Python — no LLM calls.
"""

from __future__ import annotations

from typing import Any

from etft.skills.base import Skill


class RunExperimentSkill(Skill):
    """Run a micro-experiment script and return results including parsed metrics."""

    name = "run_experiment"
    description = (
        "Execute a Python micro-experiment script in a sandboxed subprocess. "
        "Enforces timeout, import whitelist, and script size limits. "
        "Returns experiment_id, success, metrics dict, and stdout/stderr."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "script": {
                "type": "string",
                "description": "Self-contained Python experiment script.",
            },
        },
        "required": ["script"],
    }

    def __init__(self, cfg: dict | None = None) -> None:
        self._cfg = cfg

    def execute(self, **kwargs: Any) -> dict:
        from agents.empirical.runner import ExperimentRunner

        script: str = kwargs["script"]
        runner = ExperimentRunner(self._cfg)
        result = runner.run(script)
        return {
            "experiment_id": result.experiment_id,
            "success": result.success,
            "metrics": result.metrics,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
            "error_message": result.error_message or "",
        }


class CollectExperimentMetricsSkill(Skill):
    """Aggregate multiple ExperimentResult dicts into summary statistics."""

    name = "collect_metrics"
    description = (
        "Aggregate a list of experiment result dicts into summary statistics "
        "(mean, stdev, min, max per metric, success rate)."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "description": "List of experiment result dicts from run_experiment.",
                "items": {"type": "object"},
            },
        },
        "required": ["results"],
    }

    def execute(self, **kwargs: Any) -> dict:
        from agents.empirical.metrics_collector import MetricsCollector
        from corpus.regression_pipeline.schemas import ExperimentResult

        raw_results: list[dict] = kwargs["results"]
        collector = MetricsCollector()

        for r in raw_results:
            collector.add(
                ExperimentResult(
                    experiment_id=r.get("experiment_id", ""),
                    script=r.get("script", ""),
                    success=bool(r.get("success", False)),
                    metrics={k: float(v) for k, v in r.get("metrics", {}).items()},
                    error_message=r.get("error_message"),
                )
            )

        return collector.to_dict()
