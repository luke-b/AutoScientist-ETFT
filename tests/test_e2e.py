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


def test_arl_summary_contains_rl_context_prefix(tmp_path: Path) -> None:
    """run_arl summary JSON includes 'rl_context_prefix' key."""
    from agents.run_arl import run_arl

    synthesis_response = _make_llm_response(
        "SYNTHESIS:\nSome finding.\n\nHYPOTHESES:\n- Hypothesis A."
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
        summary = run_arl(cfg={}, bottleneck="dropout", output_dir=tmp_path)

    assert "rl_context_prefix" in summary, (
        "run_arl() summary must include 'rl_context_prefix' key for downstream callers"
    )
    assert isinstance(summary["rl_context_prefix"], str)


def test_arl_rl_prefix_grows_with_failures(tmp_path: Path) -> None:
    """
    After a failing experiment, the RL prefix passed to the next design call
    must contain negative feedback and grow in length.
    """
    from agents.run_arl import run_arl

    synthesis_response = _make_llm_response(
        "SYNTHESIS:\nFinding.\n\nHYPOTHESES:\n- H1.\n- H2."
    )
    # First experiment script succeeds; second fails at runtime
    _fail_script = "raise RuntimeError('deliberate failure')\n"
    responses = [
        _make_llm_response(f"```python\n{_fail_script}```"),   # H1 → fail
        _make_llm_response(f"```python\n{_SIMPLE_ALGO}```"),   # H2 → pass
    ]
    synth_count = [0]
    exp_count = [0]

    def _mock_complete(messages, tools=None):
        # Check if this looks like a synthesis call (contains "HYPOTHESES" instructions)
        msg_text = " ".join(m.get("content", "") or "" for m in messages if isinstance(m, dict))
        if "Synthesise" in msg_text or synth_count[0] < 2:
            synth_count[0] += 1
            return synthesis_response
        idx = min(exp_count[0], len(responses) - 1)
        exp_count[0] += 1
        return responses[idx]

    # Capture which rl_context values were passed to designer.design()
    rl_contexts_seen: list[str] = []
    import agents.empirical.experiment_designer as _ed_mod
    original_design = _ed_mod.ExperimentDesigner.design

    def _patched_design(self, brief, hypothesis_index=0, rl_context=""):
        rl_contexts_seen.append(rl_context)
        return original_design(self, brief, hypothesis_index=hypothesis_index, rl_context=rl_context)

    with (
        patch("etft.llm.LLMClient.complete_with_tools", side_effect=_mock_complete),
        patch("agents.literature.searcher.LiteratureSearcher.search", return_value=_DUMMY_PAPERS),
        patch("agents.literature.retriever.httpx.get", side_effect=Exception("no network")),
        patch.object(_ed_mod.ExperimentDesigner, "design", _patched_design),
    ):
        run_arl(cfg={}, bottleneck="test_bottleneck", output_dir=tmp_path)

    assert len(rl_contexts_seen) >= 2, "Expected design() to be called at least twice"
    # First call: no prior failures → empty RL context
    assert rl_contexts_seen[0] == "", f"First design call should have empty RL context, got: {rl_contexts_seen[0]!r}"
    # Second call: first experiment failed → RL context must be non-empty and include a failure
    assert rl_contexts_seen[1] != "", (
        "Second design call should receive non-empty RL context after the first experiment failed"
    )
    # The RL prefix must convey *some* negative information (reward signal < 0)
    assert "reward=-" in rl_contexts_seen[1] or "Reason:" in rl_contexts_seen[1], (
        f"RL context should contain failure information, got: {rl_contexts_seen[1]!r}"
    )


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


def test_vector_store_cache_hit(any_vector_store) -> None:
    """LiteratureRetriever skips fetch for already-cached papers (both backends)."""
    from agents.literature.retriever import Chunk, LiteratureRetriever

    store = any_vector_store
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


def test_vector_store_upsert_and_query(any_vector_store) -> None:
    """VectorStore upserts and returns relevant chunks (both backends)."""
    from agents.literature.retriever import Chunk

    store = any_vector_store
    chunk = Chunk(paper_id="p1", title="T1", text="some text", chunk_index=0)
    store.upsert([chunk])

    assert store.has_paper("p1")
    assert not store.has_paper("p2")

    results = store.query("some text", n_results=5)
    assert len(results) == 1
    assert results[0].text == "some text"


def test_vector_store_idempotent_upsert(any_vector_store) -> None:
    """Upserting the same chunk twice does not create duplicates (both backends)."""
    from agents.literature.retriever import Chunk

    store = any_vector_store
    chunk = Chunk(paper_id="p_dup", title="Dup", text="duplicate text", chunk_index=0)
    store.upsert([chunk])
    store.upsert([chunk])  # second upsert of the exact same chunk

    results = store.query("duplicate text", n_results=10)
    assert len(results) == 1  # still only one entry


def test_chroma_vector_store_persistence(tmp_path: Path) -> None:
    """ChromaVectorStore persists chunks so a new instance sees previous data."""
    pytest.importorskip("chromadb")
    from agents.literature.retriever import Chunk
    from agents.literature.vector_store import ChromaVectorStore, _OfflineEmbeddingFunction

    persist_dir = str(tmp_path / "chroma_persist")
    collection_name = "test_persistence"
    ef = _OfflineEmbeddingFunction()

    # First instance: upsert a chunk
    store1 = ChromaVectorStore(persist_dir=persist_dir, collection_name=collection_name, embedding_function=ef)
    chunk = Chunk(
        paper_id="p_persist",
        title="Persistence Test",
        text="persistence check text",
        chunk_index=0,
    )
    store1.upsert([chunk])
    assert store1.has_paper("p_persist")

    # Second instance pointing at the same directory must see the stored chunk
    store2 = ChromaVectorStore(persist_dir=persist_dir, collection_name=collection_name, embedding_function=ef)
    assert store2.has_paper("p_persist"), "ChromaVectorStore did not persist data across instances."

    results = store2.query("persistence check", n_results=5)
    assert len(results) == 1
    assert results[0].paper_id == "p_persist"
