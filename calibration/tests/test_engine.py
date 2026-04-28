"""
calibration/tests/test_engine.py — Integration tests for CalibrationEngine.

Mocks AgentClient and verifies that CalibrationEngine produces well-formed
CalibrationRecord objects and persists them correctly.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from calibration.engine import CalibrationEngine
from corpus.regression_pipeline.schemas import TrajectoryStep


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _AgentResult:
    output: str
    success: bool = True


class MockAgentClient:
    def __init__(self, responses: list[tuple[str, bool]] | None = None) -> None:
        self._responses = list(responses or [("```python\nx = 1\n```", True)])
        self._call_count = 0

    def run_task(self, task: str, system: str = "", **kwargs: Any) -> _AgentResult:
        idx = min(self._call_count, len(self._responses) - 1)
        output, success = self._responses[idx]
        self._call_count += 1
        return _AgentResult(output=output, success=success)


def _make_step(step_index: int, code: str, split: str = "train") -> TrajectoryStep:
    return TrajectoryStep(
        step_index=step_index,
        algorithm_id=f"alg_v{step_index}",
        algorithm_family="test_family",
        code=code,
        fitness_score=float(step_index),
        split=split,
    )


_CODE_STEP_0 = "def model(x):\n    return x\n"
_CODE_STEP_1 = "def model(x):\n    return x * 2\n"
_CODE_STEP_2 = "def model(x):\n    return x * 2 + 1\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(cfg_overrides: dict | None = None, mock_responses=None) -> CalibrationEngine:
    """Build a CalibrationEngine with a patched AgentClient."""
    cfg = {
        "calibration": {
            "confidence_threshold": 0.65,
            "min_replay_steps": 2,
            "min_step_similarity": 0.0,  # disabled for simpler tests
            "max_code_chars": 8000,
            "similarity_metric": "composite",
            "similarity_weights": [0.4, 0.3, 0.3],
            "recency_weighting": False,
            "max_agent_failure_rate": 0.5,
            "blind_mode": False,
            "held_out_frac": 0.0,
            "max_step_gap": None,
        }
    }
    if cfg_overrides:
        cfg["calibration"].update(cfg_overrides)
    engine = CalibrationEngine(cfg)
    # Replace the real AgentClient with the mock
    engine._agent = MockAgentClient(mock_responses)
    # Re-create the ReplaySession factory closure will pick it up via engine._agent
    return engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCalibrationEngineRecord:
    def test_run_produces_calibration_record(self):
        engine = _make_engine()
        steps = [
            _make_step(0, _CODE_STEP_0),
            _make_step(1, _CODE_STEP_1),
            _make_step(2, _CODE_STEP_2),
        ]
        record = engine.run(trajectory_id="t1", steps=steps)
        assert record.trajectory_id == "t1"
        assert record.n_steps_replayed == 2
        assert record.confidence_level is not None
        assert 0.0 <= record.confidence_level <= 1.0
        assert isinstance(record.gate_passed, bool)
        assert record.gate_status == "evaluated"

    def test_record_metadata_contains_weights(self):
        engine = _make_engine()
        steps = [_make_step(0, _CODE_STEP_0), _make_step(1, _CODE_STEP_1)]
        record = engine.run(trajectory_id="t2", steps=steps)
        assert "similarity_weights" in record.metadata
        assert record.metadata["similarity_weights"] == pytest.approx([0.4, 0.3, 0.3])

    def test_perfect_prediction_opens_gate(self):
        # Agent echoes back ground-truth code → C = 1.0
        responses = [
            (f"```python\n{_CODE_STEP_1}\n```", True),
            (f"```python\n{_CODE_STEP_2}\n```", True),
        ]
        engine = _make_engine(mock_responses=responses)
        steps = [
            _make_step(0, _CODE_STEP_0),
            _make_step(1, _CODE_STEP_1),
            _make_step(2, _CODE_STEP_2),
        ]
        record = engine.run(trajectory_id="perfect", steps=steps)
        assert record.gate_passed is True
        assert record.confidence_level == pytest.approx(1.0, abs=1e-6)

    def test_poor_prediction_closes_gate(self):
        # Agent returns completely unrelated code → C ≈ 0 < 0.65
        responses = [("```python\nprint('hello')\n```", True)]
        engine = _make_engine(mock_responses=responses)
        steps = [
            _make_step(0, _CODE_STEP_0),
            _make_step(1, "import torch\nimport numpy as np\nclass ComplexNet: pass\n"),
        ]
        record = engine.run(trajectory_id="poor", steps=steps)
        assert record.gate_passed is False


class TestCalibrationEnginePersistence:
    def test_record_persisted_to_json(self):
        engine = _make_engine()
        steps = [_make_step(0, _CODE_STEP_0), _make_step(1, _CODE_STEP_1)]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            record = engine.run(trajectory_id="persist_test", steps=steps, output_dir=out_dir)
            out_file = out_dir / "calibration_persist_test.json"
            assert out_file.exists()
            data = json.loads(out_file.read_text())
            assert data["trajectory_id"] == "persist_test"
            assert data["gate_passed"] == record.gate_passed

    def test_no_file_when_output_dir_is_none(self):
        engine = _make_engine()
        steps = [_make_step(0, _CODE_STEP_0), _make_step(1, _CODE_STEP_1)]
        # Should not raise even without output_dir
        record = engine.run(trajectory_id="no_persist", steps=steps, output_dir=None)
        assert record is not None


class TestCalibrationEngineAgentFailure:
    """B2: agent failures tracked in CalibrationRecord and gate status."""

    def test_n_agent_failures_in_record(self):
        responses = [("", False), (f"```python\n{_CODE_STEP_2}\n```", True)]
        engine = _make_engine(mock_responses=responses)
        steps = [
            _make_step(0, _CODE_STEP_0),
            _make_step(1, _CODE_STEP_1),
            _make_step(2, _CODE_STEP_2),
        ]
        record = engine.run(trajectory_id="fail_test", steps=steps)
        assert record.n_agent_failures == 1
        assert record.n_steps_attempted == 2

    def test_all_failures_gives_agent_failure_gate_status(self):
        # 100% failure rate → agent_failure gate status
        responses = [("", False)]
        engine = _make_engine(
            cfg_overrides={"max_agent_failure_rate": 0.5},
            mock_responses=responses,
        )
        steps = [_make_step(0, _CODE_STEP_0), _make_step(1, _CODE_STEP_1)]
        record = engine.run(trajectory_id="all_fail", steps=steps)
        assert record.gate_passed is False
        assert record.gate_status == "agent_failure"


class TestCalibrationEngineConfigValidation:
    def test_invalid_weights_sum_raises(self):
        cfg = {
            "calibration": {
                "similarity_weights": [0.4, 0.4, 0.4],  # sums to 1.2
            }
        }
        with pytest.raises(ValueError, match="sum to 1.0"):
            CalibrationEngine(cfg)

    def test_invalid_weights_count_raises(self):
        cfg = {
            "calibration": {
                "similarity_weights": [0.5, 0.5],  # only 2 values
            }
        }
        with pytest.raises(ValueError, match="exactly 3 elements"):
            CalibrationEngine(cfg)
