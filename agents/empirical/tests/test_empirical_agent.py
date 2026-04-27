"""
tests/test_empirical_agent.py — Unit tests for agents/empirical.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agents.empirical.experiment_designer import ExperimentDesigner, _extract_code
from agents.empirical.metrics_collector import MetricsCollector
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
    from etft.skills.base import AgentResult

    designer = ExperimentDesigner.__new__(ExperimentDesigner)
    designer._agent = MagicMock()
    designer._agent.run_task.return_value = AgentResult(
        output="```python\nprint('METRIC: acc=0.9')\n```",
        steps=[],
        skills_invoked=[],
        success=True,
    )
    designer.max_script_size = 65536

    script = designer.design(_make_brief(), hypothesis_index=0)
    assert "METRIC" in script
    designer._agent.run_task.assert_called_once()


def test_designer_rl_context_forwarded_to_agent():
    """rl_context string is present in the context passed to AgentClient."""
    from etft.skills.base import AgentResult

    designer = ExperimentDesigner.__new__(ExperimentDesigner)
    designer._agent = MagicMock()
    designer._agent.run_task.return_value = AgentResult(
        output="```python\nprint('METRIC: acc=0.9')\n```",
        steps=[],
        skills_invoked=[],
        success=True,
    )
    designer.max_script_size = 65536

    rl_signal = "=== IN-CONTEXT RL FEEDBACK (negative signals) ===\n[NEGATIVE FEEDBACK | source=micro_experiment reward=-1.00]\nReason: Script crashed\n=== END FEEDBACK ==="
    designer.design(_make_brief(), hypothesis_index=0, rl_context=rl_signal)

    # context is passed as a keyword argument named 'context'
    call_kwargs_full = designer._agent.run_task.call_args.kwargs
    assert "rl_feedback" in call_kwargs_full.get("context", {}), (
        "rl_context must be forwarded as 'rl_feedback' in the agent context dict"
    )
    assert rl_signal in call_kwargs_full["context"]["rl_feedback"]


def test_designer_empty_rl_context_not_injected():
    """When rl_context is empty string, 'rl_feedback' key must NOT appear in context."""
    from etft.skills.base import AgentResult

    designer = ExperimentDesigner.__new__(ExperimentDesigner)
    designer._agent = MagicMock()
    designer._agent.run_task.return_value = AgentResult(
        output="```python\nprint('METRIC: v=1')\n```",
        steps=[],
        skills_invoked=[],
        success=True,
    )
    designer.max_script_size = 65536

    designer.design(_make_brief(), hypothesis_index=0, rl_context="")

    call_kwargs_full = designer._agent.run_task.call_args.kwargs
    assert "rl_feedback" not in call_kwargs_full.get("context", {})


def test_designer_no_hypotheses_raises():
    designer = ExperimentDesigner.__new__(ExperimentDesigner)
    designer._agent = MagicMock()
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
# Sandbox security hardening
# ---------------------------------------------------------------------------


class TestSandboxSecurityHardening:
    """Tests for bypass patterns that must be caught by _check_imports."""

    def setup_method(self):
        self.runner = ExperimentRunner()

    def test_exec_call_blocked(self):
        result = self.runner.run("exec('import os')\n")
        assert not result.success
        assert "exec" in result.error_message

    def test_eval_call_blocked(self):
        result = self.runner.run("x = eval('1+1')\n")
        assert not result.success
        assert "eval" in result.error_message

    def test_dunder_import_call_blocked(self):
        result = self.runner.run("os = __import__('os')\n")
        assert not result.success
        assert "__import__" in result.error_message

    def test_compile_call_blocked(self):
        result = self.runner.run("compile('x=1', '<str>', 'exec')\n")
        assert not result.success
        assert "compile" in result.error_message

    def test_importlib_import_module_blocked(self):
        result = self.runner.run(
            "import importlib\nimportlib.import_module('os')\n"
        )
        assert not result.success
        assert "importlib" in result.error_message

    def test_getattr_exec_blocked(self):
        # Use builtins indirectly without importing to trigger the getattr check
        result = self.runner.run("getattr(__builtins__, 'exec')('pass')\n")
        assert not result.success
        assert "exec" in result.error_message

    def test_allowed_call_passes(self):
        result = self.runner.run(
            "import numpy as np\nprint('METRIC: val=1.0')\n"
        )
        assert result.success

    def test_custom_blocked_builtin_from_config(self):
        cfg = {"agents": {"empirical": {"blocked_builtins": ["vars"]}}}
        runner = ExperimentRunner(cfg)
        result = runner.run("vars()\n")
        assert not result.success
        assert "vars" in result.error_message


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
