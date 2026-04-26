"""
etft/skills/literature_skills.py — Skills wrapping arXiv search and paper
chunk retrieval for literature-augmented reasoning.

All skills are pure Python — no LLM calls.
"""

from __future__ import annotations

from typing import Any

from etft.skills.base import Skill


class SearchArxivSkill(Skill):
    """Search arXiv for papers relevant to a bottleneck query."""

    name = "search_arxiv"
    description = (
        "Search arXiv for papers matching a query string. "
        "Returns a list of paper records (title, abstract, URL, authors)."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string for arXiv.",
            },
            "max_papers": {
                "type": "integer",
                "description": "Maximum number of papers to return.",
                "default": 10,
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: dict | None = None) -> None:
        self._cfg = cfg

    def execute(self, **kwargs: Any) -> dict:
        from dataclasses import asdict

        from agents.literature.searcher import LiteratureSearcher

        query: str = kwargs["query"]
        max_papers: int = int(kwargs.get("max_papers", 10))

        effective_cfg = dict(self._cfg or {})
        lit_cfg = effective_cfg.setdefault("agents", {}).setdefault("literature", {})
        lit_cfg.setdefault("max_papers", max_papers)

        searcher = LiteratureSearcher(effective_cfg)
        papers = searcher.search(query)
        return {"papers": [asdict(p) for p in papers]}


class FetchAndChunkPapersSkill(Skill):
    """Fetch paper content and chunk it into token-sized passages."""

    name = "fetch_and_chunk_papers"
    description = (
        "Given a list of paper records, fetch their content and split it "
        "into token-sized chunks suitable for retrieval-augmented prompting."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "papers": {
                "type": "array",
                "description": "List of paper record dicts (from search_arxiv).",
                "items": {"type": "object"},
            },
        },
        "required": ["papers"],
    }

    def __init__(self, cfg: dict | None = None) -> None:
        self._cfg = cfg

    def execute(self, **kwargs: Any) -> dict:
        from dataclasses import asdict

        from agents.literature.retriever import LiteratureRetriever
        from agents.literature.searcher import PaperRecord

        raw_papers: list[dict] = kwargs["papers"]
        papers = [PaperRecord(**p) for p in raw_papers]

        retriever = LiteratureRetriever(self._cfg)
        chunks = retriever.retrieve_and_chunk(papers)
        return {"chunks": [asdict(c) for c in chunks]}
