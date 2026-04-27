"""
tests/test_e2e.py — End-to-end integration test for the full ETFT pipeline.

Covers the full lifecycle:
    run_regression_pipeline → run_arl → generate (SOTA+1) → triage → FeedbackRouter

All LLM calls and external network calls are mocked so the test runs
offline in CI without a proxy container or arXiv access.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agents.literature.searcher import PaperRecord
from corpus.regression_pipeline.schemas import (
    ResearchBrief,
    SOTAPlusOneCandidate,
)
from etft.skills.base import LLMResponse

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SIMPLE_ALGO = "x = 1\nprint('METRIC: accuracy=0.5')\n"
_IMPROVED_ALGO = "x = 2\nprint('METRIC: accuracy=0.75')\n"

_DUMMY_PAPERS = [
    PaperRecord(
        arxiv_id="arxiv:1234",
        title="Efficient Batch Norm",
        authors=["Alice"],
        abstract="We show batch norm improves convergence.",
        url="https://arxiv.org/abs/1234",
        published="2026-01-01",
    ),
    PaperRecord(
        arxiv_id="arxiv:5678",
        title="Group Normalization",
        authors=["Bob"],
        abstract="Group normalisation as an alternative to batch norm.",
        url="https://arxiv.org/abs/5678",
        published="2026-02-01",
    ),
]


def _make_llm_response(content: str) -> LLMResponse:
    return LLMResponse(content=content, tool_calls=[], finish_reason="stop")


# ---------------------------------------------------------------------------
# Test 1: Regression pipeline writes 𝒟_Gen JSONL
# ---------------------------------------------------------------------------


def test_regression_pipeline_writes_d_gen(tmp_path: Path) -> None:
    """run_regression_pipeline with mocked LLM produces a valid 𝒟_Gen JSONL."""
    from corpus.regression_pipeline.run import run_regression_pipeline

    predecessor_response = _make_llm_response(f"```python\n{_SIMPLE_ALGO}```")

    with patch("etft.llm.LLMClient.complete_with_tools", return_value=predecessor_response):
        run_regression_pipeline(
            cfg={},
            algorithm_family="test_cnn",
            seed_code=_IMPROVED_ALGO,
            seed_fitness=1.0,
            depth=2,
            output_dir=tmp_path,
        )

    d_gen_files = list((tmp_path / "d_gen").glob("*.jsonl"))
    assert d_gen_files, "Expected at least one 𝒟_Gen JSONL file."
    lines = d_gen_files[0].read_text().strip().splitlines()
    assert lines, "𝒟_Gen file must not be empty."
    record = json.loads(lines[0])
    assert "prompt" in record and "completion" in record


# ---------------------------------------------------------------------------
# Test 2: ARL pipeline runs end-to-end (mocked LLM + arXiv)
# ---------------------------------------------------------------------------


def test_arl_pipeline_end_to_end(tmp_path: Path) -> None:
    """run_arl with mocked LLM + arXiv search completes and writes a summary JSON."""
    from agents.run_arl import run_arl

    synthesis_response = _make_llm_response(
        "SYNTHESIS:\nBatch norm helps convergence.\n\n"
        "HYPOTHESES:\n- Add BN after conv layers.\n- Try GN instead."
    )
    experiment_response = _make_llm_response(
        f"```python\n{_SIMPLE_ALGO}```"
    )

    call_count = [0]

    def _mock_complete(messages, tools=None):
        call_count[0] += 1
        # First two calls → synthesis; subsequent → experiment design
        if call_count[0] <= 2:
            return synthesis_response
        return experiment_response

    with (
        patch("etft.llm.LLMClient.complete_with_tools", side_effect=_mock_complete),
        patch(
            "agents.literature.searcher.LiteratureSearcher.search",
            return_value=_DUMMY_PAPERS,
        ),
        patch(
            "agents.literature.retriever.httpx.get",
            side_effect=Exception("network disabled in test"),
        ),
    ):
        summary = run_arl(cfg={}, bottleneck="batch_normalisation", output_dir=tmp_path)

    assert summary["bottleneck"] == "batch_normalisation"
    assert "research_brief" in summary
    assert "experiment_metrics" in summary

    out_file = tmp_path / "arl_batch_normalisation.json"
    assert out_file.exists(), "ARL summary JSON was not written."


# ---------------------------------------------------------------------------
# Test 3: SOTA+1 generation + triage produces a candidate file
# ---------------------------------------------------------------------------


def test_sota_plus_one_generation_and_triage(tmp_path: Path) -> None:
    """generate() with mocked LLM writes a triage-passed candidate JSON."""
    from synthesis.sota_plus_one.generate import generate

    code_response = _make_llm_response(
        f"IMPLEMENTATION:\n```python\n{_IMPROVED_ALGO}```\n\n"
        "RATIONALE:\nAdded dropout for regularisation."
    )

    with patch("etft.llm.LLMClient.complete_with_tools", return_value=code_response):
        brief = ResearchBrief(
            bottleneck="dropout",
            query="dropout regularisation",
            synthesis="Dropout reduces overfitting.",
            hypotheses=["Add Dropout(0.3)."],
        )
        results = generate(
            cfg={"synthesis": {"sota_plus_one": {"max_candidates": 1}}},
            trajectory_id="traj_test",
            sota_code=_SIMPLE_ALGO,
            bottleneck="dropout",
            brief=brief,
            empirical_summary=None,
            output_dir=tmp_path / "candidates",
        )

    assert results, "Expected at least one candidate result."
    # The triage filter defaults to pass-through when no model is trained
    passed = [r for r in results if r["triage_passed"]]
    assert passed, "Expected at least one candidate to pass triage."

    candidate_files = list((tmp_path / "candidates").glob("*.json"))
    assert candidate_files, "Expected a candidate JSON file to be written."
    data = json.loads(candidate_files[0].read_text())
    assert "code" in data and "rationale" in data


# ---------------------------------------------------------------------------
# Test 4: Failed triage populates 𝒟_Perf via FeedbackRouter
# ---------------------------------------------------------------------------


def test_triage_failure_routes_to_d_perf(tmp_path: Path) -> None:
    """A triage-rejected candidate is persisted in 𝒟_Perf by FeedbackRouter."""
    from feedback.rl_loop.feedback_router import FeedbackRouter

    candidate = SOTAPlusOneCandidate(
        candidate_id="cand_e2e",
        trajectory_id="traj_e2e",
        code=_SIMPLE_ALGO,
        rationale="Test rejection.",
        risk_score=0.95,
        triage_passed=False,
    )

    router = FeedbackRouter(data_root=tmp_path)
    record = router.route_triage_failure(candidate, also_rationale=True)

    assert record.source == "triage"
    assert record.reward_signal < 0

    d_perf_file = tmp_path / "d_perf" / "feedback_failures.jsonl"
    assert d_perf_file.exists(), "𝒟_Perf JSONL was not created."
    lines = d_perf_file.read_text().strip().splitlines()
    assert lines
    entry = json.loads(lines[0])
    assert entry["label"] == 1

    # RL context should include the rejection
    prefix = router.build_rl_context_prefix()
    assert "IN-CONTEXT RL FEEDBACK" in prefix


# ---------------------------------------------------------------------------
# Test 5: FitnessEvaluator integrates into the regression run
# ---------------------------------------------------------------------------


def test_fitness_evaluator_in_regression_pipeline(tmp_path: Path) -> None:
    """Regression pipeline uses FitnessEvaluator; fitness score is measured, not hardcoded."""
    from corpus.regression_pipeline.fitness_evaluator import FitnessEvaluator

    evaluator = FitnessEvaluator()
    # Script with a METRIC line
    score = evaluator.evaluate(_SIMPLE_ALGO, current_fitness=1.0)
    assert score == pytest.approx(0.5)

    # Script that fails → fallback
    score_fail = evaluator.evaluate("raise RuntimeError('fail')\n", current_fitness=1.0)
    assert score_fail == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Test 6: Vector store caches and serves chunks
# ---------------------------------------------------------------------------


def test_vector_store_cache_hit(tmp_path: Path) -> None:
    """LiteratureRetriever skips fetch for already-cached papers."""
    from agents.literature.retriever import Chunk, LiteratureRetriever
    from agents.literature.vector_store import InMemoryVectorStore

    store = InMemoryVectorStore()
    # Pre-populate with a chunk for paper arxiv:1234
    store.upsert([Chunk(paper_id="arxiv:1234", title="T", text="cached text", chunk_index=0)])

    retriever = LiteratureRetriever(cfg={}, vector_store=store)

    fetch_count = [0]
    original_fetch = retriever._fetch_text

    def counting_fetch(paper):
        fetch_count[0] += 1
        return original_fetch(paper)

    retriever._fetch_text = counting_fetch  # type: ignore[method-assign]

    # Only paper arxiv:1234 in the list — should be a cache hit
    with patch(
        "agents.literature.retriever.httpx.get",
        side_effect=Exception("should not be called"),
    ):
        chunks = retriever.retrieve_and_chunk([_DUMMY_PAPERS[0]])

    # Cache hit → no network fetch attempted
    assert fetch_count[0] == 0
    assert chunks == []  # retrieve_and_chunk returns newly fetched chunks only


def test_vector_store_upsert_and_query() -> None:
    """InMemoryVectorStore upserts and returns chunks."""
    from agents.literature.retriever import Chunk
    from agents.literature.vector_store import InMemoryVectorStore

    store = InMemoryVectorStore()
    chunk = Chunk(paper_id="p1", title="T1", text="some text", chunk_index=0)
    store.upsert([chunk])

    assert store.has_paper("p1")
    assert not store.has_paper("p2")

    results = store.query("some text", n_results=5)
    assert len(results) == 1
    assert results[0].text == "some text"
