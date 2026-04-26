"""
analysis/pareto_delta/delta_calculator.py — Computes code and performance
deltas between adjacent pairs (aᵢ₋₁, aᵢ) in an evolutionary trajectory.

Deltas are expressed as:
  - line-level unified diffs (structural change)
  - fitness delta (Δℱ = ℱ(aᵢ) − ℱ(aᵢ₋₁))
  - per-component change counts derived from the diff hunks
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from corpus.regression_pipeline.schemas import TrajectoryStep

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ComponentDelta:
    """Change summary for a single architectural sub-component."""

    component: str
    lines_added: int
    lines_removed: int
    lines_changed: int

    @property
    def total_change(self) -> int:
        return self.lines_added + self.lines_removed + self.lines_changed


@dataclass
class StepDelta:
    """Full delta between two adjacent trajectory steps."""

    trajectory_id: str
    step_index_before: int
    step_index_after: int
    fitness_delta: float
    unified_diff: str
    lines_added: int
    lines_removed: int
    component_deltas: list[ComponentDelta] = field(default_factory=list)

    @property
    def total_lines_changed(self) -> int:
        return self.lines_added + self.lines_removed


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------


# Heuristic: map common Python constructs to component labels
_COMPONENT_PATTERNS: list[tuple[str, str]] = [
    (r"^\+.*class\s+\w", "class_definition"),
    (r"^\+.*def\s+\w", "function_definition"),
    (r"^\+.*import\s+", "import"),
    (r"^\+.*for\s+\w", "loop"),
    (r"^\+.*nn\.", "neural_network_layer"),
    (r"^\+.*optim\.", "optimiser"),
    (r"^\+.*BatchNorm", "batch_normalisation"),
    (r"^\+.*Dropout", "dropout"),
    (r"^\+.*attention|Attention", "attention_mechanism"),
    (r"^\+.*residual|skip.connect", "residual_connection"),
]


class DeltaCalculator:
    """Computes StepDelta objects from adjacent TrajectoryStep pairs."""

    def __init__(self, min_delta_lines: int = 5) -> None:
        self.min_delta_lines = min_delta_lines

    # ------------------------------------------------------------------
    def compute(self, before: TrajectoryStep, after: TrajectoryStep) -> StepDelta | None:
        """
        Compute the delta between *before* and *after*.

        Returns None if the diff is below min_delta_lines (noise filter).
        """
        diff_lines = list(
            difflib.unified_diff(
                before.code.splitlines(keepends=True),
                after.code.splitlines(keepends=True),
                fromfile=f"step_{before.step_index}",
                tofile=f"step_{after.step_index}",
            )
        )
        added = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))

        if added + removed < self.min_delta_lines:
            return None

        component_deltas = self._extract_component_deltas(diff_lines)

        return StepDelta(
            trajectory_id=before.algorithm_family,
            step_index_before=before.step_index,
            step_index_after=after.step_index,
            fitness_delta=after.fitness_score - before.fitness_score,
            unified_diff="".join(diff_lines),
            lines_added=added,
            lines_removed=removed,
            component_deltas=component_deltas,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_component_deltas(diff_lines: list[str]) -> list[ComponentDelta]:
        counts: dict[str, int] = {}
        for line in diff_lines:
            if not line.startswith("+") or line.startswith("+++"):
                continue
            for pattern, label in _COMPONENT_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    counts[label] = counts.get(label, 0) + 1

        return [
            ComponentDelta(
                component=comp,
                lines_added=count,
                lines_removed=0,
                lines_changed=0,
            )
            for comp, count in counts.items()
        ]

    # ------------------------------------------------------------------
    def compute_trajectory(
        self, steps: list[TrajectoryStep]
    ) -> list[StepDelta]:
        """Compute deltas for all adjacent pairs in a sorted trajectory."""
        sorted_steps = sorted(steps, key=lambda s: s.step_index)
        deltas = []
        for i in range(1, len(sorted_steps)):
            d = self.compute(sorted_steps[i - 1], sorted_steps[i])
            if d is not None:
                deltas.append(d)
        return deltas
