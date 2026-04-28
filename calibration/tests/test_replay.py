"""
calibration/tests/test_replay.py — Integration tests for ReplaySession.

Uses a ``MockAgentClient`` that returns deterministic responses so the
agent network is never hit during testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from calibration.replay import ReplaySession
from calibration.similarity import code_similarity
from corpus.regression_pipeline.schemas import TrajectoryStep


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _AgentResult:
    output: str
    success: bool = True


class MockAgentClient:
    """
    A deterministic mock of ``AgentClient``.

    Parameters
    ----------
    responses:
        Ordered list of ``(output, success)`` tuples returned by successive
        ``run_task`` calls.  If the list is exhausted, subsequent calls return
        the last entry.
    """

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CODE_STEP_0 = "def model(x):\n    return x\n"
_CODE_STEP_1 = "def model(x):\n    return x * 2\n"
_CODE_STEP_2 = "def model(x):\n    return x * 2 + 1\n"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReplaySessionBasic:
    def test_two_steps_produces_one_result(self):
        session = ReplaySession(agent=MockAgentClient(), similarity_fn=code_similarity)
        steps = [_make_step(0, _CODE_STEP_0), _make_step(1, _CODE_STEP_1)]
        results = session.run(steps)
        assert len(results) == 1

    def test_three_steps_produces_two_results(self):
        session = ReplaySession(agent=MockAgentClient(), similarity_fn=code_similarity)
        steps = [
            _make_step(0, _CODE_STEP_0),
            _make_step(1, _CODE_STEP_1),
            _make_step(2, _CODE_STEP_2),
        ]
        results = session.run(steps)
        assert len(results) == 2

    def test_confidence_level_is_mean_of_scores(self):
        # Perfect prediction: agent echoes back the ground-truth code
        agent = MockAgentClient([
            (f"```python\n{_CODE_STEP_1}\n```", True),
            (f"```python\n{_CODE_STEP_2}\n```", True),
        ])
        session = ReplaySession(agent=agent, similarity_fn=code_similarity)
        steps = [
            _make_step(0, _CODE_STEP_0),
            _make_step(1, _CODE_STEP_1),
            _make_step(2, _CODE_STEP_2),
        ]
        session.run(steps)
        # Both scores should be 1.0 → C should be 1.0
        assert session.confidence_level == pytest.approx(1.0, abs=1e-6)

    def test_confidence_level_zero_when_no_steps(self):
        session = ReplaySession(agent=MockAgentClient(), similarity_fn=code_similarity)
        assert session.confidence_level == pytest.approx(0.0)

    def test_single_step_produces_no_results(self):
        session = ReplaySession(agent=MockAgentClient(), similarity_fn=code_similarity)
        results = session.run([_make_step(0, _CODE_STEP_0)])
        assert results == []


class TestReplaySessionAgentFailure:
    """B2: agent failures must not contaminate the confidence level."""

    def test_failed_step_excluded_from_confidence(self):
        # Step 0→1: agent fails; step 1→2: agent returns perfect code
        agent = MockAgentClient([
            ("", False),  # failure for pair 0→1
            (f"```python\n{_CODE_STEP_2}\n```", True),  # perfect for pair 1→2
        ])
        session = ReplaySession(agent=agent, similarity_fn=code_similarity)
        steps = [
            _make_step(0, _CODE_STEP_0),
            _make_step(1, _CODE_STEP_1),
            _make_step(2, _CODE_STEP_2),
        ]
        session.run(steps)

        # Only one successful step → results has 1 entry
        assert len(session.results) == 1
        assert session.n_agent_failures == 1
        assert session.n_steps_attempted == 2
        # C = score of the one successful step (≈ 1.0)
        assert session.confidence_level == pytest.approx(1.0, abs=1e-6)

    def test_all_steps_failed_gives_zero_confidence(self):
        agent = MockAgentClient([("", False)])
        session = ReplaySession(agent=agent, similarity_fn=code_similarity)
        steps = [_make_step(0, _CODE_STEP_0), _make_step(1, _CODE_STEP_1)]
        session.run(steps)
        assert session.n_agent_failures == 1
        assert session.confidence_level == pytest.approx(0.0)


class TestReplaySessionTruncation:
    """B1: code truncation warning and prompt length cap."""

    def test_truncation_does_not_error(self):
        long_code = "x = 1\n" * 5000  # >> 8000 chars
        session = ReplaySession(
            agent=MockAgentClient(),
            similarity_fn=code_similarity,
            max_code_chars=200,
        )
        steps = [_make_step(0, long_code), _make_step(1, _CODE_STEP_1)]
        results = session.run(steps)
        assert len(results) == 1
        assert results[0].metadata.get("truncated") is True

    def test_no_truncation_flag_for_short_code(self):
        session = ReplaySession(
            agent=MockAgentClient(),
            similarity_fn=code_similarity,
            max_code_chars=8000,
        )
        steps = [_make_step(0, _CODE_STEP_0), _make_step(1, _CODE_STEP_1)]
        session.run(steps)
        assert session.results[0].metadata.get("truncated") is False


class TestReplaySessionGapHandling:
    """C3: non-contiguous step gap handling."""

    def test_gap_stored_in_metadata(self):
        session = ReplaySession(agent=MockAgentClient(), similarity_fn=code_similarity)
        steps = [_make_step(0, _CODE_STEP_0), _make_step(5, _CODE_STEP_1)]
        session.run(steps)
        assert session.results[0].metadata.get("step_gap") == 5

    def test_max_step_gap_skips_large_gaps(self):
        session = ReplaySession(
            agent=MockAgentClient(),
            similarity_fn=code_similarity,
            max_step_gap=2,
        )
        steps = [_make_step(0, _CODE_STEP_0), _make_step(10, _CODE_STEP_1)]
        results = session.run(steps)
        # gap=10 > max_step_gap=2 → skipped
        assert results == []

    def test_max_step_gap_none_allows_all_gaps(self):
        session = ReplaySession(
            agent=MockAgentClient(),
            similarity_fn=code_similarity,
            max_step_gap=None,
        )
        steps = [_make_step(0, _CODE_STEP_0), _make_step(100, _CODE_STEP_1)]
        results = session.run(steps)
        assert len(results) == 1


class TestReplaySessionHeldOutSplit:
    """A1: only 'calibration'-split steps should be replayed when present."""

    def test_only_calibration_steps_replayed(self):
        agent = MockAgentClient()
        session = ReplaySession(agent=agent, similarity_fn=code_similarity)
        steps = [
            _make_step(0, _CODE_STEP_0, split="train"),
            _make_step(1, _CODE_STEP_1, split="train"),
            _make_step(2, _CODE_STEP_2, split="calibration"),
            _make_step(3, "def model(x):\n    return x * 3\n", split="calibration"),
        ]
        results = session.run(steps)
        # Only the two "calibration" steps form one pair → 1 result
        assert len(results) == 1
        assert results[0].step_index_before == 2
        assert results[0].step_index_after == 3

    def test_all_steps_used_when_no_calibration_split(self):
        session = ReplaySession(agent=MockAgentClient(), similarity_fn=code_similarity)
        steps = [
            _make_step(0, _CODE_STEP_0, split="train"),
            _make_step(1, _CODE_STEP_1, split="train"),
            _make_step(2, _CODE_STEP_2, split="train"),
        ]
        results = session.run(steps)
        assert len(results) == 2


class TestReplaySessionBlindMode:
    """A3: blind mode must not include causal hints in the prompt."""

    def test_blind_mode_uses_ordinal_descriptions(self):
        captured_prompts = []

        class CapturingAgent:
            def run_task(self, task, system="", **kwargs):
                captured_prompts.append(task)
                return _AgentResult(output="```python\nx = 1\n```", success=True)

        session = ReplaySession(
            agent=CapturingAgent(),
            similarity_fn=code_similarity,
            blind_mode=True,
        )
        step_before = _make_step(0, _CODE_STEP_0)
        step_after = _make_step(1, _CODE_STEP_1)
        step_after.metadata["delta_loss"] = 0.03
        step_after.metadata["delta_memory_mb"] = 5.0
        step_after.metadata["causal_explanation"] = "Added batch norm."
        session.run([step_before, step_after])

        assert captured_prompts, "Agent was not called"
        prompt = captured_prompts[0]
        # Blind mode: exact deltas and causal hint must be absent
        assert "0.03" not in prompt
        assert "5.0" not in prompt
        assert "Added batch norm" not in prompt
        # Ordinal descriptions must be present
        assert any(
            word in prompt.lower()
            for word in ("small", "moderate", "large", "negligible", "very large")
        )
