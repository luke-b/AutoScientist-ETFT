"""
agents/literature/synthesizer.py — Uses the coding-agent proxy to distill
retrieved literature chunks into a structured ResearchBrief.
"""

from __future__ import annotations

import logging

from agents.literature.retriever import Chunk
from corpus.regression_pipeline.schemas import ResearchBrief
from etft.llm import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a world-class ML research assistant. "
    "Synthesise the provided literature excerpts into a concise research brief. "
    "Focus on actionable findings relevant to the stated bottleneck."
)

_TEMPLATE = """\
Bottleneck: {bottleneck}
Research query: {query}

=== LITERATURE EXCERPTS ===
{excerpts}

=== TASK ===
1. Write a 3–5 sentence synthesis of the key findings.
2. List 2–5 concrete hypotheses for addressing the bottleneck.

Respond in this exact format:
SYNTHESIS:
<your synthesis here>

HYPOTHESES:
- <hypothesis 1>
- <hypothesis 2>
...
"""


def _parse_response(text: str) -> tuple[str, list[str]]:
    synthesis = ""
    hypotheses: list[str] = []

    parts = text.split("HYPOTHESES:")
    if len(parts) == 2:
        synthesis_part = parts[0].replace("SYNTHESIS:", "").strip()
        synthesis = synthesis_part
        for line in parts[1].strip().splitlines():
            line = line.strip().lstrip("-").strip()
            if line:
                hypotheses.append(line)
    else:
        synthesis = text.strip()

    return synthesis, hypotheses


class LiteratureSynthesizer:
    """Calls the coding-agent proxy to synthesise literature into a ResearchBrief."""

    def __init__(self, cfg: dict | None = None) -> None:
        self._llm = LLMClient(cfg)

    # ------------------------------------------------------------------
    def synthesize(
        self,
        bottleneck: str,
        query: str,
        chunks: list[Chunk],
        papers: list | None = None,
    ) -> ResearchBrief:
        """
        Produce a ResearchBrief from *chunks* for the given *bottleneck*.

        Parameters
        ----------
        bottleneck:
            The architectural bottleneck identified by ParetoRanker.
        query:
            The search query used to retrieve the chunks.
        chunks:
            Text passages from retrieved papers.
        papers:
            Optional list of PaperRecord objects for metadata.
        """
        # Build excerpts block (cap at ~4000 chars to stay in context)
        excerpt_texts = [f"[{c.title}]\n{c.text}" for c in chunks]
        combined = "\n\n---\n\n".join(excerpt_texts)
        if len(combined) > 4000:
            combined = combined[:4000] + "\n... [truncated]"

        prompt = _TEMPLATE.format(bottleneck=bottleneck, query=query, excerpts=combined)
        logger.info("Synthesising %d chunks for bottleneck=%r", len(chunks), bottleneck)
        raw = self._llm.complete(prompt, system=_SYSTEM)

        synthesis, hypotheses = _parse_response(raw)

        paper_dicts = []
        if papers:
            for p in papers:
                paper_dicts.append(
                    {"arxiv_id": p.arxiv_id, "title": p.title, "url": p.url}
                    if hasattr(p, "arxiv_id")
                    else (p if isinstance(p, dict) else str(p))
                )

        return ResearchBrief(
            bottleneck=bottleneck,
            query=query,
            papers=paper_dicts,
            synthesis=synthesis,
            hypotheses=hypotheses,
        )
