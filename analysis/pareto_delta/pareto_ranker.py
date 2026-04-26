"""
analysis/pareto_delta/pareto_ranker.py — Ranks sub-component changes by their
cumulative contribution to fitness improvement (Retrospective 80/20 Δ Analysis).

The ranker aggregates StepDelta objects across a full trajectory, normalises
each component's contribution by total fitness gain, and returns the minimal
set of components that account for ≥ pareto_threshold of the gain.
"""

from __future__ import annotations

from dataclasses import dataclass

from analysis.pareto_delta.delta_calculator import StepDelta


@dataclass
class ComponentRank:
    """Ranked sub-component with its cumulative contribution metrics."""

    rank: int
    component: str
    total_fitness_contribution: float
    cumulative_fraction: float
    occurrence_count: int
    total_lines_changed: int


class ParetoRanker:
    """
    Identifies the components responsible for the top *pareto_threshold*
    fraction of cumulative fitness gain across a trajectory.
    """

    def __init__(self, pareto_threshold: float = 0.80) -> None:
        if not (0.0 < pareto_threshold <= 1.0):
            raise ValueError("pareto_threshold must be in (0, 1].")
        self.pareto_threshold = pareto_threshold

    # ------------------------------------------------------------------
    def rank(self, deltas: list[StepDelta]) -> list[ComponentRank]:
        """
        Compute a Pareto-ranked list of components.

        Parameters
        ----------
        deltas:
            Ordered list of StepDelta objects from DeltaCalculator.

        Returns
        -------
        list[ComponentRank]
            All components, sorted descending by contribution.
            The first N where cumulative_fraction ≥ pareto_threshold are the
            "Pareto set" driving 80% of the gains.
        """
        if not deltas:
            return []

        total_fitness = sum(max(d.fitness_delta, 0.0) for d in deltas)
        if total_fitness == 0.0:
            total_fitness = 1.0  # avoid div-by-zero for flat trajectories

        # Aggregate contribution per component
        contrib: dict[str, float] = {}
        occurrences: dict[str, int] = {}
        lines_changed: dict[str, int] = {}

        for delta in deltas:
            if delta.fitness_delta <= 0 or not delta.component_deltas:
                continue
            # Distribute this delta's fitness gain proportionally across components
            total_component_lines = sum(cd.total_change for cd in delta.component_deltas) or 1
            for cd in delta.component_deltas:
                weight = cd.total_change / total_component_lines
                contrib[cd.component] = contrib.get(cd.component, 0.0) + (
                    delta.fitness_delta * weight
                )
                occurrences[cd.component] = occurrences.get(cd.component, 0) + 1
                lines_changed[cd.component] = (
                    lines_changed.get(cd.component, 0) + cd.total_change
                )

        # Sort descending by contribution
        sorted_components = sorted(contrib.items(), key=lambda x: x[1], reverse=True)

        ranks: list[ComponentRank] = []
        cumulative = 0.0
        for rank_idx, (comp, contribution) in enumerate(sorted_components, start=1):
            cumulative += contribution / total_fitness
            ranks.append(
                ComponentRank(
                    rank=rank_idx,
                    component=comp,
                    total_fitness_contribution=contribution,
                    cumulative_fraction=min(cumulative, 1.0),
                    occurrence_count=occurrences[comp],
                    total_lines_changed=lines_changed[comp],
                )
            )

        return ranks

    # ------------------------------------------------------------------
    def pareto_set(self, deltas: list[StepDelta]) -> list[ComponentRank]:
        """Return only the minimal set of components exceeding pareto_threshold."""
        all_ranks = self.rank(deltas)
        result = []
        for r in all_ranks:
            result.append(r)
            if r.cumulative_fraction >= self.pareto_threshold:
                break
        return result
