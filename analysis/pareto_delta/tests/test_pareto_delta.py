"""
tests/test_pareto_delta.py — Unit tests for analysis/pareto_delta.
"""

from __future__ import annotations

import pytest

from analysis.pareto_delta.bottleneck_reporter import generate_report
from analysis.pareto_delta.delta_calculator import DeltaCalculator
from analysis.pareto_delta.pareto_ranker import ParetoRanker
from corpus.regression_pipeline.schemas import TrajectoryStep

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_step(index: int, code: str, fitness: float) -> TrajectoryStep:
    return TrajectoryStep(
        step_index=index,
        algorithm_id=f"algo_{index}",
        algorithm_family="test_family",
        code=code,
        fitness_score=fitness,
    )


CODE_V0 = """\
def model(x):
    return x
"""

CODE_V1 = """\
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, 3)

    def forward(self, x):
        return self.conv(x)
"""

CODE_V2 = """\
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 64, 3)
        self.bn = nn.BatchNorm2d(64)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.drop(x)
"""


# ---------------------------------------------------------------------------
# DeltaCalculator
# ---------------------------------------------------------------------------


class TestDeltaCalculator:
    def setup_method(self):
        self.calc = DeltaCalculator(min_delta_lines=1)

    def test_compute_returns_delta(self):
        before = _make_step(0, CODE_V0, 0.5)
        after = _make_step(1, CODE_V1, 0.8)
        delta = self.calc.compute(before, after)
        assert delta is not None
        assert delta.lines_added > 0
        assert delta.fitness_delta == pytest.approx(0.3)

    def test_compute_below_threshold_returns_none(self):
        calc = DeltaCalculator(min_delta_lines=1000)
        before = _make_step(0, "x = 1\n", 0.5)
        after = _make_step(1, "x = 2\n", 0.6)
        # Only 1 line changed, well below 1000
        delta = calc.compute(before, after)
        assert delta is None

    def test_compute_trajectory(self):
        steps = [
            _make_step(0, CODE_V0, 0.5),
            _make_step(1, CODE_V1, 0.8),
            _make_step(2, CODE_V2, 0.95),
        ]
        deltas = self.calc.compute_trajectory(steps)
        assert len(deltas) == 2

    def test_component_detection(self):
        before = _make_step(0, CODE_V0, 0.5)
        after = _make_step(1, CODE_V2, 0.9)
        delta = self.calc.compute(before, after)
        assert delta is not None
        components = [cd.component for cd in delta.component_deltas]
        # BatchNorm and Dropout should be detected
        assert any("batch" in c.lower() or "dropout" in c.lower() for c in components)


# ---------------------------------------------------------------------------
# ParetoRanker
# ---------------------------------------------------------------------------


class TestParetoRanker:
    def setup_method(self):
        self.calc = DeltaCalculator(min_delta_lines=1)
        self.ranker = ParetoRanker(pareto_threshold=0.80)

    def test_rank_returns_sorted_descending(self):
        steps = [
            _make_step(0, CODE_V0, 0.5),
            _make_step(1, CODE_V1, 0.8),
            _make_step(2, CODE_V2, 0.95),
        ]
        deltas = self.calc.compute_trajectory(steps)
        ranks = self.ranker.rank(deltas)
        if ranks:
            contributions = [r.total_fitness_contribution for r in ranks]
            assert contributions == sorted(contributions, reverse=True)

    def test_pareto_set_cumulative_fraction(self):
        steps = [
            _make_step(0, CODE_V0, 0.5),
            _make_step(1, CODE_V2, 0.9),
        ]
        deltas = self.calc.compute_trajectory(steps)
        pareto = self.ranker.pareto_set(deltas)
        if pareto:
            assert pareto[-1].cumulative_fraction >= 0.80

    def test_ranker_invalid_threshold(self):
        with pytest.raises(ValueError):
            ParetoRanker(pareto_threshold=1.5)

    def test_rank_empty_deltas(self):
        assert self.ranker.rank([]) == []


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------


def test_generate_report_structure():
    steps = [
        _make_step(0, CODE_V0, 0.5),
        _make_step(1, CODE_V1, 0.8),
        _make_step(2, CODE_V2, 0.95),
    ]
    report = generate_report(steps, pareto_threshold=0.80)
    assert "trajectory_id" in report
    assert "total_fitness_gain" in report
    assert "pareto_set" in report
    assert "bottleneck_summary" in report
    assert report["total_steps"] == 3


def test_generate_report_insufficient_steps():
    steps = [_make_step(0, CODE_V0, 0.5)]
    report = generate_report(steps)
    assert "error" in report
