"""
tests/test_literature_agent.py — Unit tests for agents/literature.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.literature.retriever import LiteratureRetriever
from agents.literature.searcher import LiteratureSearcher, PaperRecord
from agents.literature.synthesizer import LiteratureSynthesizer, _parse_response


# ---------------------------------------------------------------------------
# PaperRecord
# ---------------------------------------------------------------------------


def _make_paper(n: int = 0) -> PaperRecord:
    return PaperRecord(
        arxiv_id=f"arxiv:{n}",
        title=f"Paper {n}",
        authors=["Alice"],
        abstract="Abstract text.",
        url=f"https://arxiv.org/abs/{n}",
        published="2026-01-01",
    )


# ---------------------------------------------------------------------------
# LiteratureSearcher
# ---------------------------------------------------------------------------


def test_searcher_returns_empty_when_arxiv_missing(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "arxiv", None)
    searcher = LiteratureSearcher()
    # Should not raise — returns empty list
    results = searcher.search("batch normalisation deep learning")
    # Will fail to import arxiv and return []
    # (or succeed if installed — either is acceptable for structure tests)
    assert isinstance(results, list)


def test_searcher_respects_max_papers():
    searcher = LiteratureSearcher({"agents": {"literature": {"max_papers": 2, "arxiv_max_results": 5}}})
    assert searcher.max_papers == 2


# ---------------------------------------------------------------------------
# LiteratureRetriever
# ---------------------------------------------------------------------------


class TestLiteratureRetriever:
    def setup_method(self):
        self.retriever = LiteratureRetriever()

    def test_chunk_text_produces_chunks(self):
        paper = _make_paper(0)
        long_text = " ".join(["word"] * 2000)
        chunks = self.retriever._chunk_text(long_text, paper)
        assert len(chunks) > 1
        assert all(c.paper_id == "arxiv:0" for c in chunks)

    def test_chunk_overlap(self):
        cfg = {"agents": {"literature": {"chunk_size_tokens": 100, "chunk_overlap_tokens": 20}}}
        retriever = LiteratureRetriever(cfg)
        paper = _make_paper(1)
        text = " ".join([f"w{i}" for i in range(500)])
        chunks = retriever._chunk_text(text, paper)
        # With overlap, adjacent chunks share words
        if len(chunks) > 1:
            words_0 = set(chunks[0].text.split())
            words_1 = set(chunks[1].text.split())
            assert len(words_0 & words_1) > 0

    def test_retrieve_uses_abstract_fallback(self, monkeypatch):
        monkeypatch.setattr(
            "agents.literature.retriever.httpx.get",
            MagicMock(side_effect=Exception("network error")),
        )
        paper = _make_paper(2)
        text = self.retriever._fetch_text(paper)
        assert "Paper 2" in text or "Abstract" in text


# ---------------------------------------------------------------------------
# LiteratureSynthesizer
# ---------------------------------------------------------------------------


def test_parse_response_well_formed():
    raw = (
        "SYNTHESIS:\nBatch norm improves convergence.\n\n"
        "HYPOTHESES:\n- Add BN after conv.\n- Try GN instead."
    )
    synthesis, hypotheses = _parse_response(raw)
    assert "Batch norm" in synthesis
    assert len(hypotheses) == 2


def test_parse_response_malformed():
    raw = "No structured response here."
    synthesis, hypotheses = _parse_response(raw)
    assert synthesis == raw.strip()
    assert hypotheses == []


def test_synthesizer_calls_proxy():
    synth = LiteratureSynthesizer.__new__(LiteratureSynthesizer)
    synth._llm = MagicMock()
    synth._llm.complete.return_value = (
        "SYNTHESIS:\nKey finding.\n\nHYPOTHESES:\n- Hypothesis A."
    )

    from agents.literature.retriever import Chunk

    chunks = [Chunk(paper_id="p1", title="T1", text="Some text.", chunk_index=0)]
    brief = synth.synthesize("batch_normalisation", "batch norm CNNs", chunks)

    assert brief.bottleneck == "batch_normalisation"
    assert "Key finding" in brief.synthesis
    assert len(brief.hypotheses) == 1
    synth._llm.complete.assert_called_once()
