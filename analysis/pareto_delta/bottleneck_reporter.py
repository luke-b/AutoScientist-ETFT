"""
analysis/pareto_delta/bottleneck_reporter.py — Produces a structured report
of the top bottlenecks and innovation vectors identified by ParetoRanker.

Output is a dict suitable for JSON serialisation or passing directly to the
literature and synthesis agents.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from analysis.pareto_delta.delta_calculator import DeltaCalculator, StepDelta
from analysis.pareto_delta.pareto_ranker import ComponentRank, ParetoRanker
from corpus.regression_pipeline.schemas import TrajectoryStep


def generate_report(
    steps: list[TrajectoryStep],
    pareto_threshold: float = 0.80,
    min_delta_lines: int = 5,
) -> dict:
    """
    End-to-end: compute deltas → rank → build report dict.

    Parameters
    ----------
    steps:
        All TrajectorySteps for one trajectory (any order; sorted internally).
    pareto_threshold:
        Cumulative fraction threshold for the Pareto set.
    min_delta_lines:
        Noise filter passed to DeltaCalculator.

    Returns
    -------
    dict with keys:
        trajectory_id, total_steps, total_fitness_gain, pareto_threshold,
        pareto_set (list of component dicts), all_ranks (full list),
        bottleneck_summary (str)
    """
    if len(steps) < 2:
        return {"error": "Need at least 2 trajectory steps to generate a report."}

    calculator = DeltaCalculator(min_delta_lines=min_delta_lines)
    ranker = ParetoRanker(pareto_threshold=pareto_threshold)

    sorted_steps = sorted(steps, key=lambda s: s.step_index)
    deltas: list[StepDelta] = calculator.compute_trajectory(sorted_steps)

    total_fitness_gain = sum(max(d.fitness_delta, 0.0) for d in deltas)
    pareto_set: list[ComponentRank] = ranker.pareto_set(deltas)
    all_ranks: list[ComponentRank] = ranker.rank(deltas)

    trajectory_id = sorted_steps[0].algorithm_family if sorted_steps else "unknown"

    bottleneck_names = [r.component for r in pareto_set]
    if bottleneck_names:
        summary = (
            f"The top {len(bottleneck_names)} component(s) — "
            f"{', '.join(bottleneck_names)} — "
            f"account for ≥{pareto_threshold*100:.0f}% of fitness gains "
            f"(total Δℱ = {total_fitness_gain:.4f})."
        )
    else:
        summary = "No significant component-level bottlenecks identified."

    return {
        "trajectory_id": trajectory_id,
        "total_steps": len(sorted_steps),
        "total_fitness_gain": total_fitness_gain,
        "pareto_threshold": pareto_threshold,
        "pareto_set": [asdict(r) for r in pareto_set],
        "all_ranks": [asdict(r) for r in all_ranks],
        "bottleneck_summary": summary,
    }


def save_report(report: dict, output_path: Path) -> None:
    """Write report dict to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
