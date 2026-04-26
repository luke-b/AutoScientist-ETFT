"""
tests/test_rl_loop.py — Unit tests for feedback/rl_loop.
"""

from __future__ import annotations

import json

import pytest

from corpus.regression_pipeline.schemas import (
    ExperimentResult,
    FailureRecord,
    SOTAPlusOneCandidate,
)
from feedback.rl_loop.feedback_router import FeedbackRouter
from feedback.rl_loop.negative_data_collector import NegativeDataCollector
from feedback.rl_loop.reward_signal import (
    format_for_context,
    from_experiment_failure,
    from_triage_rejection,
)

# ---------------------------------------------------------------------------
# reward_signal
# ---------------------------------------------------------------------------


def _failed_experiment() -> ExperimentResult:
    return ExperimentResult(
        experiment_id="exp_001",
        script="raise RuntimeError('boom')",
        success=False,
        error_message="Exit code 1",
        stderr="RuntimeError: boom",
    )


def _rejected_candidate(risk: float = 0.85) -> SOTAPlusOneCandidate:
    return SOTAPlusOneCandidate(
        candidate_id="cand_001",
        trajectory_id="traj_1",
        code="x = 1",
        rationale="test",
        risk_score=risk,
        triage_passed=False,
    )


def test_from_experiment_failure_signal():
    record = from_experiment_failure(_failed_experiment())
    assert record.source == "micro_experiment"
    assert record.reward_signal == pytest.approx(-1.0)
    assert record.experiment_id == "exp_001"
    assert "boom" in record.failure_reason or "Exit code" in record.failure_reason


def test_from_triage_rejection_signal():
    record = from_triage_rejection(_rejected_candidate(risk=0.9))
    assert record.source == "triage"
    assert record.reward_signal < -1.0  # graded: -(1 + risk)
    assert record.candidate_id == "cand_001"


def test_format_for_context_contains_reward():
    record = FailureRecord(
        source="triage",
        candidate_id="x",
        failure_reason="Too risky.",
        reward_signal=-1.5,
    )
    text = format_for_context(record)
    assert "NEGATIVE FEEDBACK" in text
    assert "-1.50" in text
    assert "Too risky" in text


# ---------------------------------------------------------------------------
# NegativeDataCollector
# ---------------------------------------------------------------------------


def test_collector_appends_d_perf(tmp_path):
    collector = NegativeDataCollector(tmp_path)
    record = from_experiment_failure(_failed_experiment())
    collector.collect(record)

    out = tmp_path / "d_perf" / "feedback_failures.jsonl"
    assert out.exists()
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["label"] == 1


def test_collector_appends_d_rationale(tmp_path):
    collector = NegativeDataCollector(tmp_path)
    record = from_triage_rejection(_rejected_candidate())
    collector.collect(record, also_rationale=True)

    out_r = tmp_path / "d_rationale" / "feedback_rationale.jsonl"
    assert out_r.exists()
    lines = out_r.read_text().strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert "prompt" in data and "completion" in data


def test_collector_multiple_appends(tmp_path):
    collector = NegativeDataCollector(tmp_path)
    for _ in range(3):
        collector.collect(from_experiment_failure(_failed_experiment()))
    out = tmp_path / "d_perf" / "feedback_failures.jsonl"
    assert len(out.read_text().strip().split("\n")) == 3


# ---------------------------------------------------------------------------
# FeedbackRouter
# ---------------------------------------------------------------------------


def test_router_experiment_failure(tmp_path):
    router = FeedbackRouter(data_root=tmp_path)
    record = router.route_experiment_failure(_failed_experiment())
    assert record.source == "micro_experiment"
    assert len(router.reward_log) == 1


def test_router_triage_failure(tmp_path):
    router = FeedbackRouter(data_root=tmp_path)
    record = router.route_triage_failure(_rejected_candidate())
    assert record.source == "triage"
    assert len(router.reward_log) == 1


def test_router_rl_context_prefix(tmp_path):
    router = FeedbackRouter(data_root=tmp_path)
    router.route_experiment_failure(_failed_experiment())
    router.route_triage_failure(_rejected_candidate())
    prefix = router.build_rl_context_prefix()
    assert "IN-CONTEXT RL FEEDBACK" in prefix
    assert prefix.count("NEGATIVE FEEDBACK") == 2


def test_router_empty_context_prefix(tmp_path):
    router = FeedbackRouter(data_root=tmp_path)
    assert router.build_rl_context_prefix() == ""
