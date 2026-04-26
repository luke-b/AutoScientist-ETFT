"""
etft/skills/analysis_skills.py — Skills wrapping trajectory delta analysis,
Pareto ranking, and bottleneck report generation.

All skills are pure Python — no LLM calls.
"""

from __future__ import annotations

from typing import Any

from etft.skills.base import Skill


class ComputeTrajectoryDeltasSkill(Skill):
    """Compute code and performance deltas between adjacent trajectory steps."""

    name = "compute_trajectory_deltas"
    description = (
        "Given a list of trajectory steps (each with step_index, algorithm_family, "
        "code, fitness_score), compute the unified diffs and fitness deltas between "
        "adjacent pairs. Returns a list of StepDelta dicts."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "description": "List of trajectory step dicts.",
                "items": {
                    "type": "object",
                    "properties": {
                        "step_index": {"type": "integer"},
                        "algorithm_id": {"type": "string"},
                        "algorithm_family": {"type": "string"},
                        "code": {"type": "string"},
                        "fitness_score": {"type": "number"},
                    },
                    "required": ["step_index", "algorithm_family", "code", "fitness_score"],
                },
            },
            "min_delta_lines": {
                "type": "integer",
                "description": "Minimum changed lines to keep a delta (noise filter).",
                "default": 5,
            },
        },
        "required": ["steps"],
    }

    def execute(self, **kwargs: Any) -> dict:
        from dataclasses import asdict

        from analysis.pareto_delta.delta_calculator import DeltaCalculator
        from corpus.regression_pipeline.schemas import TrajectoryStep

        raw_steps: list[dict] = kwargs["steps"]
        min_delta_lines: int = int(kwargs.get("min_delta_lines", 5))

        steps = [
            TrajectoryStep(
                step_index=s["step_index"],
                algorithm_id=s.get("algorithm_id", f"step_{s['step_index']}"),
                algorithm_family=s["algorithm_family"],
                code=s["code"],
                fitness_score=float(s["fitness_score"]),
            )
            for s in raw_steps
        ]

        calculator = DeltaCalculator(min_delta_lines=min_delta_lines)
        deltas = calculator.compute_trajectory(steps)
        return {"deltas": [asdict(d) for d in deltas]}


class RankParetoSkill(Skill):
    """Rank sub-components by cumulative fitness contribution (Pareto 80/20)."""

    name = "rank_pareto"
    description = (
        "Given a list of StepDelta dicts, rank the architectural sub-components "
        "by their cumulative contribution to fitness improvement. Returns the "
        "full ranked list and the minimal Pareto set."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "deltas": {
                "type": "array",
                "description": "List of StepDelta dicts from compute_trajectory_deltas.",
                "items": {"type": "object"},
            },
            "pareto_threshold": {
                "type": "number",
                "description": "Cumulative fraction threshold (default 0.80).",
                "default": 0.80,
            },
        },
        "required": ["deltas"],
    }

    def execute(self, **kwargs: Any) -> dict:
        from dataclasses import asdict

        from analysis.pareto_delta.delta_calculator import ComponentDelta, StepDelta
        from analysis.pareto_delta.pareto_ranker import ParetoRanker

        raw_deltas: list[dict] = kwargs["deltas"]
        pareto_threshold: float = float(kwargs.get("pareto_threshold", 0.80))

        deltas = []
        for d in raw_deltas:
            component_deltas = [
                ComponentDelta(**cd) for cd in d.get("component_deltas", [])
            ]
            deltas.append(
                StepDelta(
                    trajectory_id=d["trajectory_id"],
                    step_index_before=d["step_index_before"],
                    step_index_after=d["step_index_after"],
                    fitness_delta=float(d["fitness_delta"]),
                    unified_diff=d.get("unified_diff", ""),
                    lines_added=d.get("lines_added", 0),
                    lines_removed=d.get("lines_removed", 0),
                    component_deltas=component_deltas,
                )
            )

        ranker = ParetoRanker(pareto_threshold=pareto_threshold)
        all_ranks = ranker.rank(deltas)
        pareto_set = ranker.pareto_set(deltas)
        return {
            "all_ranks": [asdict(r) for r in all_ranks],
            "pareto_set": [asdict(r) for r in pareto_set],
        }


class GenerateBottleneckReportSkill(Skill):
    """Generate a full bottleneck report from a trajectory's steps."""

    name = "generate_bottleneck_report"
    description = (
        "End-to-end bottleneck analysis: given trajectory steps, compute "
        "deltas, rank components, and return a structured bottleneck report."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "description": "List of trajectory step dicts.",
                "items": {"type": "object"},
            },
            "pareto_threshold": {
                "type": "number",
                "description": "Cumulative fraction threshold (default 0.80).",
                "default": 0.80,
            },
            "min_delta_lines": {
                "type": "integer",
                "description": "Noise filter for delta lines.",
                "default": 5,
            },
        },
        "required": ["steps"],
    }

    def execute(self, **kwargs: Any) -> dict:
        from analysis.pareto_delta.bottleneck_reporter import generate_report
        from corpus.regression_pipeline.schemas import TrajectoryStep

        raw_steps: list[dict] = kwargs["steps"]
        pareto_threshold: float = float(kwargs.get("pareto_threshold", 0.80))
        min_delta_lines: int = int(kwargs.get("min_delta_lines", 5))

        steps = [
            TrajectoryStep(
                step_index=s["step_index"],
                algorithm_id=s.get("algorithm_id", f"step_{s['step_index']}"),
                algorithm_family=s["algorithm_family"],
                code=s["code"],
                fitness_score=float(s["fitness_score"]),
            )
            for s in raw_steps
        ]

        return generate_report(
            steps,
            pareto_threshold=pareto_threshold,
            min_delta_lines=min_delta_lines,
        )
