"""
tests/test_empirical_agent.py — Unit tests for agents/empirical.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agents.empirical.experiment_designer import ExperimentDesigner, _extract_code
from agents.empirical.metrics_collector import MetricsCollector, MetricsSummary
from agents.empirical.runner import ExperimentRunner, _parse_metrics
from corpus.regression_pipeline.schemas import ExperimentResult, ResearchBrief


# ---------------------------------------------------------------------------
# ExperimentDesigner
# ---------------------------------------------------------------------------


def _make_brief() -> ResearchBrief:
    return ResearchBrief(
        bottleneck="batch_normalisation",
        query="batch norm CNNs",
        synthesis="Batch norm improves convergence.",
        hypotheses=["Add BN after conv layers.", "Use GN instead of BN."],
    )


def test_extract_code_with_fence():
    raw = "```python\nprint('hello')\n```"
    assert _extract_code(raw) == "print('hello')"


def test_extract_code_plain():
    raw = "print('hello')"
    assert _extract_code(raw) == "print('hello')"


def test_designer_calls_proxy():
    designer = ExperimentDesigner.__new__(ExperimentDesigner)
    designer._llm = MagicMock()
    designer._llm.complete.return_value = "```python\nprint('METRIC: acc=0.9')\n```"
    designer.max_script_size = 65536

    script = designer.design(_make_brief(), hypothesis_index=0)
    assert "METRIC" in script
    designer._llm.complete.assert_called_once()


def test_designer_no_hypotheses_raises():
    designer = ExperimentDesigner.__new__(ExperimentDesigner)
    designer._llm = MagicMock()
    designer.max_script_size = 65536
    empty_brief = ResearchBrief(
        bottleneck="x", query="q", synthesis="s", hypotheses=[]
    )
    with pytest.raises(ValueError):
        designer.design(empty_brief)


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------


class TestExperimentRunner:
    def setup_method(self):
        self.runner = ExperimentRunner()

    def test_simple_script_passes(self):
        script = "print('METRIC: loss=0.42')\n"
        result = self.runner.run(script)
        assert result.success
        assert result.metrics.get("loss") == pytest.approx(0.42)

    def test_syntax_error_fails_at_runtime(self):
        result = self.runner.run("def foo(\n")
        assert not result.success

    def test_runtime_error_fails(self):
        result = self.runner.run("raise RuntimeError('boom')\n")
        assert not result.success
        assert "1" in result.error_message  # exit code 1

    def test_disallowed_import_blocked(self):
        result = self.runner.run("import requests\n")
        assert not result.success
        assert "requests" in result.error_message

    def test_allowed_import_passes(self):
        result = self.runner.run(
            "import numpy as np\nprint('METRIC: val=1.0')\n"
        )
        assert result.success

    def test_oversized_script_rejected(self):
        runner = ExperimentRunner({"agents": {"empirical": {"max_script_size_bytes": 10}}})
        result = runner.run("x = 1" * 100)
        assert not result.success
        assert "max size" in result.error_message


# ---------------------------------------------------------------------------
# _parse_metrics
# ---------------------------------------------------------------------------


def test_parse_metrics_single():
    assert _parse_metrics("METRIC: accuracy=0.95") == {"accuracy": pytest.approx(0.95)}


def test_parse_metrics_multiple():
    stdout = "METRIC: loss=0.1\nsome other line\nMETRIC: acc=0.9"
    metrics = _parse_metrics(stdout)
    assert metrics["loss"] == pytest.approx(0.1)
    assert metrics["acc"] == pytest.approx(0.9)


def test_parse_metrics_invalid_value():
    metrics = _parse_metrics("METRIC: val=NaN_text")
    assert "val" not in metrics


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------


class TestMetricsCollector:
    def setup_method(self):
        self.collector = MetricsCollector()

    def _result(self, success: bool, metrics: dict | None = None) -> ExperimentResult:
        return ExperimentResult(
            experiment_id="e1",
            script="pass",
            success=success,
            metrics=metrics or {},
        )

    def test_success_rate_all_pass(self):
        for _ in range(3):
            self.collector.add(self._result(True, {"acc": 0.9}))
        assert self.collector.success_rate == pytest.approx(1.0)

    def test_success_rate_mixed(self):
        self.collector.add(self._result(True, {"acc": 0.9}))
        self.collector.add(self._result(False))
        assert self.collector.success_rate == pytest.approx(0.5)

    def test_summarize_mean(self):
        self.collector.add(self._result(True, {"acc": 0.8}))
        self.collector.add(self._result(True, {"acc": 1.0}))
        summary = self.collector.summarize()
        assert summary["acc"].mean == pytest.approx(0.9)

    def test_failures_filtered(self):
        self.collector.add(self._result(False))
        assert len(self.collector.failures) == 1

    def test_to_dict_structure(self):
        self.collector.add(self._result(True, {"loss": 0.3}))
        d = self.collector.to_dict()
        assert "total_experiments" in d
        assert "success_rate" in d
        assert "metrics" in d
