"""
tests/test_synthesis.py — Unit tests for synthesis/sota_plus_one.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from corpus.regression_pipeline.schemas import ResearchBrief, SOTAPlusOneCandidate
from synthesis.sota_plus_one.synthesizer import SOTAPlusOneSynthesizer, _extract_code_and_rationale
from synthesis.sota_plus_one.triage import TriageFilter


# ---------------------------------------------------------------------------
# _extract_code_and_rationale
# ---------------------------------------------------------------------------


def test_extract_with_both_sections():
    text = (
        "IMPLEMENTATION:\n```python\ndef foo(): pass\n```\n\n"
        "RATIONALE:\nAdded dropout for regularisation."
    )
    code, rationale = _extract_code_and_rationale(text)
    assert "def foo" in code
    assert "dropout" in rationale


def test_extract_no_code_block():
    text = "No code here."
    code, rationale = _extract_code_and_rationale(text)
    assert code == ""
    assert rationale == text


# ---------------------------------------------------------------------------
# SOTAPlusOneSynthesizer
# ---------------------------------------------------------------------------


def _make_brief() -> ResearchBrief:
    return ResearchBrief(
        bottleneck="dropout",
        query="dropout regularisation",
        synthesis="Dropout reduces overfitting.",
        hypotheses=["Add Dropout(0.3) after linear layers."],
    )


def test_synthesizer_generates_candidates():
    synth = SOTAPlusOneSynthesizer.__new__(SOTAPlusOneSynthesizer)
    synth.max_candidates = 2
    synth._llm = MagicMock()
    synth._llm.complete.return_value = (
        "IMPLEMENTATION:\n```python\ndef model(): pass\n```\n\nRATIONALE:\nAdded dropout."
    )

    candidates = synth.generate(
        sota_code="def old_model(): pass",
        trajectory_id="t1",
        bottleneck="dropout",
        brief=_make_brief(),
    )

    assert len(candidates) == 2
    assert all(c.triage_passed for c in candidates)  # default risk = 0.0
    assert synth._llm.complete.call_count == 2


def test_synthesizer_skips_empty_code():
    synth = SOTAPlusOneSynthesizer.__new__(SOTAPlusOneSynthesizer)
    synth.max_candidates = 1
    synth._llm = MagicMock()
    synth._llm.complete.return_value = "No code here."

    candidates = synth.generate("def x(): pass", "t1", "x", _make_brief())
    assert candidates == []


# ---------------------------------------------------------------------------
# TriageFilter
# ---------------------------------------------------------------------------


def test_triage_pass_when_no_model(tmp_path):
    filt = TriageFilter(cfg=None, model_path=tmp_path / "nonexistent.joblib")
    candidate = SOTAPlusOneCandidate(
        candidate_id="c1",
        trajectory_id="t1",
        code="x = 1\n",
        rationale="test",
    )
    result = filt.evaluate(candidate)
    assert result.triage_passed
    assert result.risk_score == pytest.approx(0.0)


def test_triage_reject_above_threshold(tmp_path):
    filt = TriageFilter(
        cfg={"synthesis": {"triage": {"risk_threshold": 0.5}}},
        model_path=tmp_path / "nonexistent.joblib",
    )
    # Patch the internal filter to return high risk
    filt._filter = MagicMock()
    filt._filter.predict_failure_probability.return_value = 0.9

    candidate = SOTAPlusOneCandidate(
        candidate_id="c2", trajectory_id="t1", code="x = 1\n", rationale="test"
    )
    result = filt.evaluate(candidate)
    assert not result.triage_passed
    assert result.risk_score == pytest.approx(0.9)


def test_triage_pass_below_threshold(tmp_path):
    filt = TriageFilter(
        cfg={"synthesis": {"triage": {"risk_threshold": 0.7}}},
        model_path=tmp_path / "nonexistent.joblib",
    )
    filt._filter = MagicMock()
    filt._filter.predict_failure_probability.return_value = 0.3

    candidate = SOTAPlusOneCandidate(
        candidate_id="c3", trajectory_id="t1", code="x = 1\n", rationale="test"
    )
    result = filt.evaluate(candidate)
    assert result.triage_passed
