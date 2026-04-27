"""
tests/test_pipeline.py — Integration tests for pipeline.py orchestrator.

All LLM and arXiv calls are mocked so tests run offline in CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agents.literature.searcher import PaperRecord
from corpus.regression_pipeline.schemas import ResearchBrief
from etft.skills.base import LLMResponse

_SIMPLE_ALGO = "x = 1\nprint('METRIC: accuracy=0.5')\n"

_DUMMY_PAPERS = [
    PaperRecord(
        arxiv_id="arxiv:0001",
        title="Test Paper",
        authors=["Alice"],
        abstract="Abstract.",
        url="https://arxiv.org/abs/0001",
        published="2026-01-01",
    )
]


def _make_llm_response(content: str) -> LLMResponse:
    return LLMResponse(content=content, tool_calls=[], finish_reason="stop")


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


def test_pipeline_state_load_save(tmp_path):
    """State is created fresh and round-trips correctly."""
    from pipeline import _load_state, _save_state

    state = _load_state(tmp_path)
    assert state["completed_stages"] == []
    assert state["trajectory_ids"] == []

    state["completed_stages"].append("corpus")
    _save_state(state, tmp_path)

    reloaded = _load_state(tmp_path)
    assert "corpus" in reloaded["completed_stages"]


def test_pipeline_state_corrupted_file(tmp_path):
    """Corrupted state file falls back to a clean state dict."""
    from pipeline import _load_state

    (tmp_path / "pipeline_state.json").write_text("not valid json{{")
    state = _load_state(tmp_path)
    assert state["completed_stages"] == []


# ---------------------------------------------------------------------------
# Single-stage corpus
# ---------------------------------------------------------------------------


def test_pipeline_corpus_stage(tmp_path):
    """Running --stage corpus populates d_gen and records state."""
    from pipeline import run_pipeline

    predecessor_response = _make_llm_response(f"```python\n{_SIMPLE_ALGO}```")

    with patch("etft.llm.LLMClient.complete_with_tools", return_value=predecessor_response):
        result = run_pipeline(
            cfg={},
            target="test_family",
            seed_code=_SIMPLE_ALGO,
            seed_fitness=1.0,
            depth=2,
            data_root=tmp_path,
            stage="corpus",
        )

    state = result["state"]
    assert "corpus" in state["completed_stages"]
    assert (tmp_path / "d_gen").exists()


# ---------------------------------------------------------------------------
# Resume skips completed stages
# ---------------------------------------------------------------------------


def test_pipeline_resume_skips_corpus(tmp_path):
    """--resume flag skips 'corpus' if it's already in completed_stages."""
    from pipeline import _load_state, _save_state, run_pipeline

    # Pre-mark corpus as done
    state = _load_state(tmp_path)
    state["completed_stages"].append("corpus")
    _save_state(state, tmp_path)

    call_count = {"corpus": 0}
    original_corpus = None

    import pipeline as _pl
    original = _pl._run_corpus_stage

    def _patched_corpus(*args, **kwargs):
        call_count["corpus"] += 1
        return original(*args, **kwargs)

    with (
        patch.object(_pl, "_run_corpus_stage", side_effect=_patched_corpus),
    ):
        run_pipeline(
            cfg={},
            target="test_family",
            seed_code=_SIMPLE_ALGO,
            data_root=tmp_path,
            stage="corpus",
            resume=True,
        )

    assert call_count["corpus"] == 0, "corpus stage should have been skipped"


# ---------------------------------------------------------------------------
# Invalid stage raises ValueError
# ---------------------------------------------------------------------------


def test_pipeline_invalid_stage_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown stage"):
        from pipeline import run_pipeline
        run_pipeline(cfg={}, target="x", seed_code="pass\n", data_root=tmp_path, stage="invalid")


# ---------------------------------------------------------------------------
# ARL stage integration (mocked)
# ---------------------------------------------------------------------------


def test_pipeline_arl_stage(tmp_path):
    """Running --stage arl records arl_runs in state and returns summary."""
    from pipeline import run_pipeline

    synthesis_response = _make_llm_response(
        "SYNTHESIS:\nFindings.\n\nHYPOTHESES:\n- H1."
    )
    experiment_response = _make_llm_response(f"```python\n{_SIMPLE_ALGO}```")

    call_count = [0]

    def _mock_complete(messages, tools=None):
        call_count[0] += 1
        if call_count[0] <= 2:
            return synthesis_response
        return experiment_response

    with (
        patch("etft.llm.LLMClient.complete_with_tools", side_effect=_mock_complete),
        patch("agents.literature.searcher.LiteratureSearcher.search", return_value=_DUMMY_PAPERS),
        patch("agents.literature.retriever.httpx.get", side_effect=Exception("no network")),
    ):
        result = run_pipeline(
            cfg={},
            target="batch_normalisation",
            seed_code=_SIMPLE_ALGO,
            data_root=tmp_path,
            stage="arl",
        )

    state = result["state"]
    assert "arl" in state["completed_stages"]
    assert len(state["arl_runs"]) == 1
    assert state["arl_runs"][0]["bottleneck"] == "batch_normalisation"


# ---------------------------------------------------------------------------
# Log events are written for each stage
# ---------------------------------------------------------------------------


def test_pipeline_events_logged(tmp_path):
    """State events include a timestamp and stage for each completed stage."""
    from pipeline import run_pipeline

    predecessor_response = _make_llm_response(f"```python\n{_SIMPLE_ALGO}```")

    with patch("etft.llm.LLMClient.complete_with_tools", return_value=predecessor_response):
        result = run_pipeline(
            cfg={},
            target="test_events",
            seed_code=_SIMPLE_ALGO,
            seed_fitness=1.0,
            depth=2,
            data_root=tmp_path,
            stage="corpus",
        )

    events = result["state"].get("events", [])
    assert len(events) >= 1
    ev = events[0]
    assert "timestamp" in ev
    assert ev["stage"] == "corpus"


# ---------------------------------------------------------------------------
# State file is persisted to disk
# ---------------------------------------------------------------------------


def test_pipeline_state_written_to_disk(tmp_path):
    """pipeline_state.json is written after a stage completes."""
    from pipeline import run_pipeline

    predecessor_response = _make_llm_response(f"```python\n{_SIMPLE_ALGO}```")

    with patch("etft.llm.LLMClient.complete_with_tools", return_value=predecessor_response):
        run_pipeline(
            cfg={},
            target="state_test",
            seed_code=_SIMPLE_ALGO,
            depth=2,
            data_root=tmp_path,
            stage="corpus",
        )

    state_file = tmp_path / "pipeline_state.json"
    assert state_file.exists(), "pipeline_state.json was not written to disk"
    on_disk = json.loads(state_file.read_text())
    assert "corpus" in on_disk["completed_stages"]
