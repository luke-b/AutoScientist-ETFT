"""
feedback/rl_loop/reward_signal.py — Constructs in-context RL reward signals
from failure events.

Negative reward signals are injected back into the empirical agent's context
window so that subsequent experiment designs avoid repeating the same mistakes.
"""

from __future__ import annotations

from corpus.regression_pipeline.schemas import ExperimentResult, FailureRecord, SOTAPlusOneCandidate

# ---------------------------------------------------------------------------
# Reward construction
# ---------------------------------------------------------------------------


def from_experiment_failure(result: ExperimentResult) -> FailureRecord:
    """
    Convert a failed ExperimentResult into a FailureRecord with a negative
    reward signal.

    The magnitude is fixed at -1.0 for all failures (binary feedback).
    """
    return FailureRecord(
        source="micro_experiment",
        experiment_id=result.experiment_id,
        failure_reason=result.error_message or result.stderr or "unknown",
        reward_signal=-1.0,
        metadata={
            "stdout": result.stdout[-500:] if result.stdout else "",
            "stderr": result.stderr[-500:] if result.stderr else "",
        },
    )


def from_triage_rejection(candidate: SOTAPlusOneCandidate) -> FailureRecord:
    """
    Convert a triage-rejected SOTA+1 candidate into a FailureRecord.

    Risk score is encoded as a graded negative reward: higher risk → stronger
    negative signal (in [-2, -1]).
    """
    reward = -(1.0 + candidate.risk_score)  # range [-2, -1]
    return FailureRecord(
        source="triage",
        candidate_id=candidate.candidate_id,
        failure_reason=f"Triage rejected: P(fail)={candidate.risk_score:.3f}",
        features={},
        reward_signal=reward,
        metadata={"risk_score": candidate.risk_score},
    )


def format_for_context(record: FailureRecord) -> str:
    """
    Format a FailureRecord as a plain-text negative reward signal suitable
    for prepending to an LLM context window (in-context RL).
    """
    return (
        f"[NEGATIVE FEEDBACK | source={record.source} reward={record.reward_signal:.2f}]\n"
        f"Reason: {record.failure_reason}\n"
    )
