"""
agents/literature/searcher.py — Queries arXiv (and optionally Semantic Scholar)
for papers relevant to a given bottleneck component.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass
class PaperRecord:
    """Minimal paper metadata."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    published: str


class LiteratureSearcher:
    """Searches arXiv for papers about a specific bottleneck topic."""

    def __init__(self, cfg: dict | None = None) -> None:
        lit_cfg = (cfg or {}).get("agents", {}).get("literature", {})
        self.max_results: int = int(lit_cfg.get("arxiv_max_results", 20))
        self.max_papers: int = int(lit_cfg.get("max_papers", 10))
        self.request_delay: float = float(lit_cfg.get("request_delay_seconds", 0.4))

    # ------------------------------------------------------------------
    def search(self, query: str) -> list[PaperRecord]:
        """
        Search arXiv for *query* and return up to ``max_papers`` results.

        Retries up to 3 times with exponential back-off on transient errors.
        Returns an empty list (with a warning) if the arxiv package is not
        installed or the network is unavailable after all retries.
        """
        try:
            import arxiv
        except ImportError:
            logger.warning("arxiv package not installed — returning empty results.")
            return []

        try:
            return self._search_with_retry(query, arxiv)
        except Exception as exc:
            logger.error("arXiv search failed after retries: %s", exc)
            return []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _search_with_retry(self, query: str, arxiv_module) -> list[PaperRecord]:  # type: ignore[no-untyped-def]
        client = arxiv_module.Client()
        search = arxiv_module.Search(
            query=query,
            max_results=self.max_results,
            sort_by=arxiv_module.SortCriterion.Relevance,
        )
        papers = []
        for result in client.results(search):
            papers.append(
                PaperRecord(
                    arxiv_id=result.entry_id,
                    title=result.title,
                    authors=[a.name for a in result.authors],
                    abstract=result.summary,
                    url=result.entry_id,
                    published=str(result.published.date()) if result.published else "",
                )
            )
            if len(papers) >= self.max_papers:
                break
            # Respect arXiv rate limits between individual result fetches
            if self.request_delay > 0:
                time.sleep(self.request_delay)
        logger.info("Found %d papers for query: %r", len(papers), query)
        return papers
