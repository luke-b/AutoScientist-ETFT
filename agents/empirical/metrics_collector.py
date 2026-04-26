"""
agents/empirical/metrics_collector.py — Aggregates ExperimentResult objects
and provides summary statistics across a batch of micro-experiments.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from corpus.regression_pipeline.schemas import ExperimentResult


@dataclass
class MetricsSummary:
    """Summary statistics for a named metric across multiple experiments."""

    metric_name: str
    values: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        return statistics.mean(self.values) if self.values else 0.0

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.values) if len(self.values) > 1 else 0.0

    @property
    def minimum(self) -> float:
        return min(self.values) if self.values else 0.0

    @property
    def maximum(self) -> float:
        return max(self.values) if self.values else 0.0

    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "count": self.count,
            "mean": self.mean,
            "stdev": self.stdev,
            "min": self.minimum,
            "max": self.maximum,
        }


class MetricsCollector:
    """Accumulates ExperimentResults and produces aggregate summaries."""

    def __init__(self) -> None:
        self._results: list[ExperimentResult] = []

    # ------------------------------------------------------------------
    def add(self, result: ExperimentResult) -> None:
        """Record one experiment result."""
        self._results.append(result)

    # ------------------------------------------------------------------
    def summarize(self) -> dict[str, MetricsSummary]:
        """
        Return a dict mapping metric name → MetricsSummary across all
        successful experiments.
        """
        summaries: dict[str, MetricsSummary] = {}
        for result in self._results:
            if not result.success:
                continue
            for name, value in result.metrics.items():
                if name not in summaries:
                    summaries[name] = MetricsSummary(metric_name=name)
                summaries[name].values.append(value)
        return summaries

    # ------------------------------------------------------------------
    @property
    def success_rate(self) -> float:
        if not self._results:
            return 0.0
        return sum(1 for r in self._results if r.success) / len(self._results)

    # ------------------------------------------------------------------
    @property
    def failures(self) -> list[ExperimentResult]:
        return [r for r in self._results if not r.success]

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        summaries = self.summarize()
        return {
            "total_experiments": len(self._results),
            "success_rate": self.success_rate,
            "failure_count": len(self.failures),
            "metrics": {k: v.to_dict() for k, v in summaries.items()},
        }
