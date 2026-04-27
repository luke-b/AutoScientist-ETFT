"""
agents/literature/retriever.py — Fetches full paper content and chunks it
into token-sized passages for retrieval-augmented generation.

Retrieved chunks are persisted in a VectorStore so that repeat ARL runs
avoid redundant network fetches for already-seen papers.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

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

    def __init__(self, cfg: dict | None = None, vector_store=None) -> None:
        lit_cfg = (cfg or {}).get("agents", {}).get("literature", {})
        self.chunk_size_tokens: int = int(lit_cfg.get("chunk_size_tokens", 512))
        self.chunk_overlap_tokens: int = int(lit_cfg.get("chunk_overlap_tokens", 64))
        self.request_delay: float = float(lit_cfg.get("request_delay_seconds", 0.4))
        self._vector_store = vector_store  # optional VectorStore instance

    # ------------------------------------------------------------------
    def retrieve_and_chunk(self, papers: list[PaperRecord]) -> list[Chunk]:
        """
        For each paper, check the vector store cache first.  Only fetch
        papers that are not yet cached, then upsert new chunks.

        Returns a flat list of Chunk objects (cached + newly fetched).
        """
        all_chunks: list[Chunk] = []
        papers_to_fetch: list[PaperRecord] = []

        for paper in papers:
            if self._vector_store is not None and self._vector_store.has_paper(paper.arxiv_id):
                logger.debug("Cache hit for paper %s — skipping fetch.", paper.arxiv_id)
            else:
                papers_to_fetch.append(paper)

        for i, paper in enumerate(papers_to_fetch):
            text = self._fetch_text(paper)
            new_chunks = self._chunk_text(text, paper)
            all_chunks.extend(new_chunks)
            # Persist to vector store
            if self._vector_store is not None:
                self._vector_store.upsert(new_chunks)
            # Rate-limit delay between fetches
            if self.request_delay > 0 and i < len(papers_to_fetch) - 1:
                time.sleep(self.request_delay)

        logger.info(
            "Fetched %d new chunks from %d papers (%d cached).",
            len(all_chunks),
            len(papers_to_fetch),
            len(papers) - len(papers_to_fetch),
        )
        return all_chunks

    # ------------------------------------------------------------------
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        reraise=False,
    )
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
        except (httpx.HTTPError, httpx.TimeoutException):
            raise  # let tenacity retry
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
