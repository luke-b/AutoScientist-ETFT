"""
agents/literature/searcher.py — Queries arXiv (and optionally Semantic Scholar)
for papers relevant to a given bottleneck component.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

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

    # ------------------------------------------------------------------
    def search(self, query: str) -> list[PaperRecord]:
        """
        Search arXiv for *query* and return up to ``max_papers`` results.

        Returns an empty list (with a warning) if the arxiv package is not
        installed or the network is unavailable.
        """
        try:
            import arxiv
        except ImportError:
            logger.warning("arxiv package not installed — returning empty results.")
            return []

        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.Relevance,
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
            logger.info("Found %d papers for query: %r", len(papers), query)
            return papers

        except Exception as exc:
            logger.error("arXiv search failed: %s", exc)
            return []
