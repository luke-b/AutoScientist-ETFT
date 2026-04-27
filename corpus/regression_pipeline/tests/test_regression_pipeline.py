"""
tests/test_regression_pipeline.py — Unit tests for corpus/regression_pipeline.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from corpus.regression_pipeline.cicd_validator import CICDValidator
from corpus.regression_pipeline.dataset_builder import DatasetBuilder
from corpus.regression_pipeline.fitness_evaluator import FitnessEvaluator
from corpus.regression_pipeline.regression_agent import RegressionAgent, _extract_code
from corpus.regression_pipeline.schemas import (
    RationaleRecord,
    TrajectoryPair,
    TrajectoryStep,
    ValidationStatus,
)

# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------


def test_trajectory_pair_fitness_delta():
    before = TrajectoryStep(
        step_index=0, algorithm_id="a0", algorithm_family="test", code="x=1", fitness_score=0.5
    )
    after = TrajectoryStep(
        step_index=1, algorithm_id="a1", algorithm_family="test", code="x=2", fitness_score=0.8
    )
    pair = TrajectoryPair(trajectory_id="t1", step_before=before, step_after=after)
    assert abs(pair.fitness_delta - 0.3) < 1e-9


def test_trajectory_pair_to_training_example():
    before = TrajectoryStep(
        step_index=0, algorithm_id="a0", algorithm_family="cnn", code="pass", fitness_score=0.5
    )
    after = TrajectoryStep(
        step_index=1, algorithm_id="a1", algorithm_family="cnn", code="return 1", fitness_score=0.9
    )
    pair = TrajectoryPair(trajectory_id="t1", step_before=before, step_after=after)
    ex = pair.to_training_example()
    assert "prompt" in ex and "completion" in ex
    assert "cnn" in ex["prompt"]
    assert ex["completion"] == "return 1"


def test_rationale_record_to_training_example():
    rec = RationaleRecord(
        trajectory_id="t1",
        step_index_before=0,
        step_index_after=1,
        delta_summary="Added batch normalisation.",
        changed_components=["batch_norm"],
        performance_impact=0.4,
    )
    ex = rec.to_training_example()
    assert "prompt" in ex and "completion" in ex
    assert "Added batch normalisation" in ex["completion"]


# ---------------------------------------------------------------------------
# CICDValidator
# ---------------------------------------------------------------------------


class TestCICDValidator:
    def setup_method(self):
        self.validator = CICDValidator()

    def test_valid_code_passes(self):
        result = self.validator.validate("x = 1 + 1\n")
        assert result.passed

    def test_syntax_error_fails(self):
        result = self.validator.validate("def foo(\n")
        assert result.status == ValidationStatus.FAIL_SYNTAX

    def test_runtime_error_fails(self):
        result = self.validator.validate("raise RuntimeError('boom')\n")
        assert result.status == ValidationStatus.FAIL_RUNTIME

    def test_timeout(self, monkeypatch):
        import subprocess

        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=1)

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = self.validator.validate("while True: pass\n")
        assert result.status == ValidationStatus.FAIL_TIMEOUT

    def test_oversized_code_fails(self):
        validator = CICDValidator({"corpus": {"regression_pipeline": {"max_code_size_bytes": 10}}})
        result = validator.validate("x = " + "1" * 20)
        assert result.status == ValidationStatus.FAIL_SYNTAX

    def test_oom_detection(self, monkeypatch):
        import subprocess

        fake_result = MagicMock()
        fake_result.returncode = 1
        fake_result.stdout = ""
        fake_result.stderr = "CUDA out of memory"
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: fake_result)
        result = self.validator.validate("x = 1\n")
        assert result.status == ValidationStatus.FAIL_OOM


# ---------------------------------------------------------------------------
# RegressionAgent
# ---------------------------------------------------------------------------


def test_extract_code_with_fence():
    raw = "Here is the code:\n```python\nreturn 42\n```\n"
    assert _extract_code(raw) == "return 42"


def test_extract_code_without_fence():
    raw = "return 42"
    assert _extract_code(raw) == "return 42"


def test_regression_agent_calls_llm():
    from etft.skills.base import AgentResult

    agent = RegressionAgent.__new__(RegressionAgent)
    agent._agent = MagicMock()
    agent._agent.run_task.return_value = AgentResult(
        output="```python\ndef simple(): pass\n```",
        steps=[],
        skills_invoked=[],
        success=True,
    )
    result = agent.regress("def complex(): pass", "cnn", 0.9)
    assert "simple" in result
    agent._agent.run_task.assert_called_once()


# ---------------------------------------------------------------------------
# DatasetBuilder
# ---------------------------------------------------------------------------


def test_dataset_builder_writes_jsonl(tmp_path):
    from etft.skills.base import AgentResult

    steps = [
        TrajectoryStep(
            step_index=i,
            algorithm_id=f"a{i}",
            algorithm_family="test",
            code=f"x = {i}\n",
            fitness_score=float(i),
        )
        for i in range(3)
    ]

    builder = DatasetBuilder.__new__(DatasetBuilder)
    builder._cfg = {}

    mock_validator = MagicMock()
    mock_validator.validate.return_value = MagicMock(passed=True)
    builder._validator = mock_validator

    mock_agent = MagicMock()
    mock_agent.run_task.return_value = AgentResult(
        output='{"delta_summary": "added x", "changed_components": ["x"], "performance_impact": 0.1}',
        steps=[],
        skills_invoked=[],
        success=True,
    )
    builder._agent = mock_agent

    d_gen_path, d_rationale_path = builder.build_from_trajectory(steps, tmp_path, trajectory_id="tid1")

    assert d_gen_path.exists()
    assert d_rationale_path.exists()

    lines = d_gen_path.read_text().strip().split("\n")
    assert len(lines) == 2  # 3 steps → 2 pairs


# ---------------------------------------------------------------------------
# FitnessEvaluator
# ---------------------------------------------------------------------------


class TestFitnessEvaluator:
    def test_metric_line_parsed(self):
        """Script that prints a METRIC line → evaluator returns that value."""
        evaluator = FitnessEvaluator()
        script = "print('METRIC: accuracy=0.75')\n"
        # We need to inject the script as the "code" arg; the evaluator runs it.
        score = evaluator.evaluate(script, current_fitness=1.0)
        assert abs(score - 0.75) < 1e-9

    def test_primary_metric_preferred(self):
        """When the primary metric name is present, it is used over others."""
        evaluator = FitnessEvaluator(primary_metric="fitness")
        script = "print('METRIC: accuracy=0.5\\nMETRIC: fitness=0.9')\n"
        score = evaluator.evaluate(script, current_fitness=1.0)
        assert abs(score - 0.9) < 1e-9

    def test_fallback_on_failure(self):
        """A failing script returns current_fitness * fallback_factor."""
        evaluator = FitnessEvaluator(fallback_factor=0.85)
        script = "raise RuntimeError('boom')\n"
        score = evaluator.evaluate(script, current_fitness=1.0)
        assert abs(score - 0.85) < 1e-9

    def test_fallback_when_no_metrics(self):
        """A script that succeeds but prints no METRIC lines also falls back."""
        evaluator = FitnessEvaluator(fallback_factor=0.85)
        script = "x = 1\n"
        score = evaluator.evaluate(script, current_fitness=1.0)
        assert abs(score - 0.85) < 1e-9

    def test_uses_first_metric_when_primary_absent(self):
        """When primary metric is absent, first available metric is used."""
        evaluator = FitnessEvaluator(primary_metric="fitness")
        script = "print('METRIC: loss=0.3')\n"
        score = evaluator.evaluate(script, current_fitness=1.0)
        assert abs(score - 0.3) < 1e-9

    def test_custom_fallback_factor(self):
        """Custom fallback_factor is respected."""
        evaluator = FitnessEvaluator(fallback_factor=0.5)
        script = "raise ValueError('x')\n"
        score = evaluator.evaluate(script, current_fitness=0.8)
        assert abs(score - 0.4) < 1e-9  # 0.8 * 0.5

    def test_runner_mocked(self):
        """Evaluator delegates to ExperimentRunner (mock path)."""
        from corpus.regression_pipeline.schemas import ExperimentResult

        evaluator = FitnessEvaluator.__new__(FitnessEvaluator)
        evaluator._runner = MagicMock()
        evaluator._primary_metric = "acc"
        evaluator._fallback_factor = 0.85
        evaluator._runner.run.return_value = ExperimentResult(
            experiment_id="e1",
            script="",
            success=True,
            metrics={"acc": 0.99},
        )
        score = evaluator.evaluate("some code", current_fitness=1.0)
        assert abs(score - 0.99) < 1e-9
        evaluator._runner.run.assert_called_once_with("some code")
