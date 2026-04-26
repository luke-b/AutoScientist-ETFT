"""
agents/literature/retriever.py — Fetches full paper content and chunks it
into token-sized passages for retrieval-augmented generation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import httpx

from agents.literature.searcher import PaperRecord

logger = logging.getLogger(__name__)

_WORDS_PER_TOKEN = 0.75  # rough approximation


@dataclass
class Chunk:
    """A passage extracted from a paper."""

    paper_id: str
    title: str
    text: str
    chunk_index: int


class LiteratureRetriever:
    """Downloads paper PDFs/HTML and chunks the text for RAG."""

    def __init__(self, cfg: dict | None = None) -> None:
        lit_cfg = (cfg or {}).get("agents", {}).get("literature", {})
        self.chunk_size_tokens: int = int(lit_cfg.get("chunk_size_tokens", 512))
        self.chunk_overlap_tokens: int = int(lit_cfg.get("chunk_overlap_tokens", 64))

    # ------------------------------------------------------------------
    def retrieve_and_chunk(self, papers: list[PaperRecord]) -> list[Chunk]:
        """
        For each paper, attempt to fetch the abstract HTML page from arXiv
        (full-text PDF parsing is out-of-scope; we use abstract + title).
        Returns a flat list of Chunk objects.
        """
        chunks: list[Chunk] = []
        for paper in papers:
            text = self._fetch_text(paper)
            paper_chunks = self._chunk_text(text, paper)
            chunks.extend(paper_chunks)
        logger.info("Retrieved %d chunks from %d papers.", len(chunks), len(papers))
        return chunks

    # ------------------------------------------------------------------
    def _fetch_text(self, paper: PaperRecord) -> str:
        """Return the best available text for the paper (abstract fallback)."""
        # Try the arXiv HTML abstract page
        abs_url = paper.url.replace("http://", "https://")
        try:
            resp = httpx.get(abs_url, timeout=10, follow_redirects=True)
            if resp.status_code == 200:
                raw = resp.text
                # Strip HTML tags
                clean = re.sub(r"<[^>]+>", " ", raw)
                clean = re.sub(r"\s+", " ", clean).strip()
                return clean[:8000]  # cap at ~8 KB
        except Exception as exc:
            logger.debug("Failed to fetch %s: %s", abs_url, exc)

        # Fallback: abstract only
        return f"{paper.title}\n\n{paper.abstract}"

    # ------------------------------------------------------------------
    def _chunk_text(self, text: str, paper: PaperRecord) -> list[Chunk]:
        words = text.split()
        chunk_words = max(1, int(self.chunk_size_tokens / _WORDS_PER_TOKEN))
        overlap_words = max(0, int(self.chunk_overlap_tokens / _WORDS_PER_TOKEN))
        step = max(1, chunk_words - overlap_words)

        chunks: list[Chunk] = []
        for i, start in enumerate(range(0, len(words), step)):
            chunk_text = " ".join(words[start : start + chunk_words])
            if chunk_text:
                chunks.append(
                    Chunk(
                        paper_id=paper.arxiv_id,
                        title=paper.title,
                        text=chunk_text,
                        chunk_index=i,
                    )
                )
        return chunks
