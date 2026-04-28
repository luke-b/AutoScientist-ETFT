"""
calibration/tests/test_stage_gate.py — Unit tests for StageGate.
"""

import pytest

from calibration.stage_gate import StageGate
from corpus.regression_pipeline.schemas import ReconstructionResult


def _make_result(
    sim: float | None,
    before: int = 0,
    after: int = 1,
    agent_success: bool = True,
) -> ReconstructionResult:
    """Helper: build a minimal ReconstructionResult for gate tests."""
    return ReconstructionResult(
        trajectory_id="t1",
        step_index_before=before,
        step_index_after=after,
        predicted_code="x = 1",
        true_code="x = 1",
        similarity_score=sim,
        agent_success=agent_success,
    )


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

    def test_invalid_min_step_similarity_raises(self):
        with pytest.raises(ValueError):
            StageGate(min_step_similarity=1.5)

    def test_result_contains_diagnostic(self):
        gate = StageGate(threshold=0.65, min_steps=2)
        result = gate.evaluate(confidence_level=0.40, n_steps=5)
        assert "shortfall" in result.diagnostic.lower() or "blocked" in result.diagnostic.lower()

    def test_result_str_includes_status(self):
        gate = StageGate(threshold=0.65, min_steps=2)
        result = gate.evaluate(confidence_level=0.80, n_steps=4)
        assert "OPEN" in str(result)

    # ------------------------------------------------------------------
    # A4: Per-step minimum floor
    # ------------------------------------------------------------------

    def test_per_step_floor_closes_gate(self):
        """Gate must close when any step falls below min_step_similarity."""
        gate = StageGate(threshold=0.65, min_steps=2, min_step_similarity=0.35)
        results = [
            _make_result(0.90, 0, 1),
            _make_result(0.88, 1, 2),
            _make_result(0.20, 2, 3),  # below floor
        ]
        gate_result = gate.evaluate(
            confidence_level=0.66,
            n_steps=3,
            reconstruction_results=results,
        )
        assert gate_result.ready is False
        assert "floor" in gate_result.diagnostic.lower() or "minimum" in gate_result.diagnostic.lower()

    def test_per_step_floor_passes_when_all_above(self):
        gate = StageGate(threshold=0.65, min_steps=2, min_step_similarity=0.35)
        results = [
            _make_result(0.80, 0, 1),
            _make_result(0.75, 1, 2),
            _make_result(0.70, 2, 3),
        ]
        gate_result = gate.evaluate(
            confidence_level=0.75,
            n_steps=3,
            reconstruction_results=results,
        )
        assert gate_result.ready is True

    def test_per_step_floor_disabled_at_zero(self):
        """Setting min_step_similarity=0.0 must disable the floor check."""
        gate = StageGate(threshold=0.0, min_steps=2, min_step_similarity=0.0)
        results = [
            _make_result(0.80, 0, 1),
            _make_result(0.00, 1, 2),  # would fail at 0.35
        ]
        gate_result = gate.evaluate(
            confidence_level=0.70,
            n_steps=2,
            reconstruction_results=results,
        )
        # floor disabled: gate should open based on mean-C only (mean = 0.4 ≥ 0.0)
        assert gate_result.ready is True

    # ------------------------------------------------------------------
    # A4: Recency weighting
    # ------------------------------------------------------------------

    def test_recency_weighting_favours_later_steps(self):
        """
        With recency weighting, good later steps should produce a higher C
        than bad later steps compared to the unweighted mean.
        """
        good_late = [
            _make_result(0.50, 0, 1),
            _make_result(0.80, 1, 2),  # late, good
        ]
        bad_late = [
            _make_result(0.80, 0, 1),
            _make_result(0.50, 1, 2),  # late, bad
        ]
        gate_weighted = StageGate(
            threshold=0.0, min_steps=2, recency_weighting=True
        )
        c_good = gate_weighted._compute_confidence(good_late)
        c_bad = gate_weighted._compute_confidence(bad_late)
        assert c_good > c_bad

    # ------------------------------------------------------------------
    # B2: Agent failure rate gate
    # ------------------------------------------------------------------

    def test_agent_failure_rate_closes_gate(self):
        """Gate must close with 'agent_failure' diagnostic when failure rate ≥ threshold."""
        gate = StageGate(threshold=0.65, min_steps=1, max_agent_failure_rate=0.5)
        result = gate.evaluate(
            confidence_level=0.70,
            n_steps=1,
            n_steps_attempted=4,
            n_agent_failures=2,  # exactly 50% => closes
        )
        assert result.ready is False
        assert "agent failure" in result.diagnostic.lower()

    def test_agent_failure_below_threshold_adds_note(self):
        """Gate should open but include a note when failures are below threshold."""
        gate = StageGate(threshold=0.65, min_steps=2, max_agent_failure_rate=0.5)
        result = gate.evaluate(
            confidence_level=0.80,
            n_steps=3,
            n_steps_attempted=4,
            n_agent_failures=1,  # 25% < 50%
        )
        assert result.ready is True
        assert "failure" in result.diagnostic.lower()

    def test_none_similarity_excluded_from_c(self):
        """
        Steps with similarity_score=None (agent failure) must not contribute
        to the computed C.
        """
        gate = StageGate(threshold=0.65, min_steps=1, min_step_similarity=0.0)
        results = [
            _make_result(0.80, 0, 1),
            _make_result(None, 1, 2, agent_success=False),  # excluded
        ]
        c = gate._compute_confidence(results)
        assert c == pytest.approx(0.80)
