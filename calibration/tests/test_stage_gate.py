"""
calibration/tests/test_stage_gate.py — Unit tests for StageGate.
"""

import pytest
from calibration.stage_gate import StageGate


class TestStageGate:
    def test_gate_opens_above_threshold(self):
        gate = StageGate(threshold=0.65, min_steps=2)
        result = gate.evaluate(confidence_level=0.70, n_steps=3)
        assert result.ready is True

    def test_gate_stays_closed_below_threshold(self):
        gate = StageGate(threshold=0.65, min_steps=2)
        result = gate.evaluate(confidence_level=0.50, n_steps=3)
        assert result.ready is False

    def test_gate_opens_exactly_at_threshold(self):
        gate = StageGate(threshold=0.65, min_steps=2)
        result = gate.evaluate(confidence_level=0.65, n_steps=3)
        assert result.ready is True

    def test_gate_closed_for_insufficient_steps(self):
        gate = StageGate(threshold=0.65, min_steps=3)
        result = gate.evaluate(confidence_level=0.90, n_steps=2)
        assert result.ready is False
        assert "Insufficient trajectory depth" in result.diagnostic

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            StageGate(threshold=1.5)

    def test_result_contains_diagnostic(self):
        gate = StageGate(threshold=0.65, min_steps=2)
        result = gate.evaluate(confidence_level=0.40, n_steps=5)
        assert "shortfall" in result.diagnostic.lower() or "blocked" in result.diagnostic.lower()

    def test_result_str_includes_status(self):
        gate = StageGate(threshold=0.65, min_steps=2)
        result = gate.evaluate(confidence_level=0.80, n_steps=4)
        assert "OPEN" in str(result)
