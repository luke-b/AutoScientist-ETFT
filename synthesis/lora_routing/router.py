"""
synthesis/lora_routing/router.py — Dynamic LoRA Routing Engine for
Inference-Time Orchestration (Benda, 2026 §2).

``DynamicLoRARouter`` implements the three-phase inference flow described in
the Inference-Time Orchestration paper:

  Phase 1 — Depth Analysis (base Foundation Model only)
      Feeds the full trajectory context into the unaltered base model and
      extracts a structural ``DepthAnalysis`` (bottleneck, summary, epoch).

  Phase 2 — Agentic RAG Retrieval
      Uses the bottleneck from Phase 1 to drive an arXiv search via the
      existing ``LiteratureSearcher`` / ``LiteratureRetriever`` agents.

  Phase 3 — Width Synthesis (Width LoRA hot-swapped)
      Loads the generational Width LoRA for the recommended epoch, then
      runs generation with the full context: trajectory + papers + depth analysis.

When ``lora_routing.enabled=false`` in ``config.yaml`` (the default), the
synthesizer falls back to the existing proxy path and this router is never called.

Usage
-----
    from synthesis.lora_routing.router import DynamicLoRARouter

    router = DynamicLoRARouter(cfg)
    result = router.route(
        sota_code=current_sota,
        trajectory_steps=steps,
        generation_index=n,
    )
    print(result.code)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from synthesis.lora_routing.adapter_library import AdapterLibrary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DepthAnalysis:
    """Structured output from Phase 1 depth analysis."""

    bottleneck: str
    structural_summary: str
    recommended_epoch: int
    raw_response: str = ""


@dataclass
class SynthesisResult:
    """Output of the full three-phase routing pipeline."""

    code: str
    rationale: str
    adapter_used: str | None
    depth_analysis: DepthAnalysis | None = None
    papers_retrieved: int = 0
    success: bool = True
    error: str = ""


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_DEPTH_ANALYSIS_SYSTEM = (
    "You are an expert ML researcher analysing the historical progression of an algorithm family. "
    "Your task is to identify the key structural bottleneck in the current SOTA and recommend "
    "which evolutionary generation's Width model should be used for synthesis. "
    "Respond in the following structured format:\n"
    "BOTTLENECK: <one-line description of the main architectural bottleneck>\n"
    "STRUCTURAL_SUMMARY: <2-3 sentences describing the macro-innovation vector>\n"
    "RECOMMENDED_EPOCH: <integer generation index n>\n"
)

_DEPTH_ANALYSIS_TEMPLATE = """\
=== EVOLUTIONARY TRAJECTORY ===
{trajectory_summary}

=== CURRENT SOTA (aₙ) ===
{sota_code}

Analyse the trajectory above. Identify the main structural bottleneck and
recommend the best generational Width model epoch for lateral synthesis.
"""

_WIDTH_SYNTHESIS_SYSTEM = (
    "You are an elite ML researcher combining historical trajectory context, "
    "recent literature, and depth analysis to propose a SOTA+1 algorithm. "
    "Generate a Python implementation that is structurally NOVEL and DIVERSE "
    "while addressing the identified bottleneck. "
    "Respond with:\n"
    "1. A Python implementation (```python ... ``` block).\n"
    "2. A RATIONALE section explaining the structural novelty.\n"
)

_WIDTH_SYNTHESIS_TEMPLATE = """\
=== DEPTH ANALYSIS ===
Bottleneck: {bottleneck}
Structural Summary: {structural_summary}

=== RETRIEVED LITERATURE ===
{literature_summary}

=== CURRENT SOTA CODE ===
{sota_code}

Propose SOTA+1: an architecturally NOVEL Python algorithm that addresses the bottleneck
by incorporating the literature findings. Be structurally diverse — do not simply extend
the current SOTA linearly.

IMPLEMENTATION:
```python
<code here>
```

RATIONALE:
<explanation here>
"""


def _extract_code(text: str) -> str:
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return re.sub(r"```[a-z]*", "", text).strip()


def _extract_rationale(text: str) -> str:
    if "RATIONALE:" in text:
        return text.split("RATIONALE:", 1)[1].strip()
    cleaned = re.sub(r"```python.*?```", "", text, flags=re.DOTALL).strip()
    return cleaned or ""


def _parse_depth_analysis(response: str, fallback_epoch: int) -> DepthAnalysis:
    """Parse the structured depth analysis response."""
    bottleneck = ""
    summary = ""
    epoch = fallback_epoch

    for line in response.splitlines():
        if line.startswith("BOTTLENECK:"):
            bottleneck = line.split(":", 1)[1].strip()
        elif line.startswith("STRUCTURAL_SUMMARY:"):
            summary = line.split(":", 1)[1].strip()
        elif line.startswith("RECOMMENDED_EPOCH:"):
            try:
                epoch = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass

    return DepthAnalysis(
        bottleneck=bottleneck or "unknown bottleneck",
        structural_summary=summary or response[:300],
        recommended_epoch=epoch,
        raw_response=response,
    )


def _build_trajectory_summary(trajectory_steps: list[Any], max_steps: int = 5) -> str:
    """Build a compact trajectory summary string for the depth analysis prompt."""
    if not trajectory_steps:
        return "(no trajectory steps provided)"
    steps = trajectory_steps[-max_steps:]
    lines = []
    for s in steps:
        sid = getattr(s, "algorithm_id", "?")
        idx = getattr(s, "step_index", "?")
        fitness = getattr(s, "fitness_score", "?")
        lines.append(f"  step {idx}: {sid}  (fitness={fitness})")
    return "\n".join(lines)


def _format_papers(chunks: list[Any], papers: list[Any], max_chars: int = 1500) -> str:
    """Format retrieved literature for the synthesis prompt."""
    if not papers and not chunks:
        return "No literature retrieved."
    lines: list[str] = []
    for p in papers[:5]:
        title = getattr(p, "title", str(p))
        abstract = getattr(p, "abstract", "")
        lines.append(f"- {title}: {abstract[:200]}")
    combined = "\n".join(lines)
    return combined[:max_chars] if combined else "No literature retrieved."


# ---------------------------------------------------------------------------
# DynamicLoRARouter
# ---------------------------------------------------------------------------


class DynamicLoRARouter:
    """
    Dynamic LoRA Routing Engine — orchestrates three-phase inference for
    Orthogonal Calibration synthesis.

    Parameters
    ----------
    cfg:
        Full runtime config dict.
    adapter_library:
        Optional ``AdapterLibrary`` instance.  When *None* the library is
        instantiated from ``cfg.lora_routing.adapter_index_path``.
    hot_swap:
        Optional ``LoRAHotSwap`` instance.  When *None* a new instance is
        created from *cfg*.
    """

    def __init__(
        self,
        cfg: dict | None = None,
        adapter_library: AdapterLibrary | None = None,
        hot_swap: Any = None,
    ) -> None:
        self._cfg = cfg or {}
        lr_cfg = self._cfg.get("lora_routing", {})

        # Adapter library
        if adapter_library is not None:
            self._library = adapter_library
        else:
            index_path = lr_cfg.get(
                "adapter_index_path",
                self._cfg.get("training", {}).get("output_dir", "./checkpoints")
                + "/adapter_index.json",
            )
            self._library = AdapterLibrary(index_path=Path(index_path))

        # Hot-swap engine (lazy-loaded to avoid importing torch at startup)
        self._hot_swap_instance = hot_swap
        self._foundation_loaded = False

        # LLM proxy client for depth analysis (uses existing proxy path)
        from etft.llm import LLMClient
        self._llm = LLMClient(self._cfg)

        # Literature agents (reuse existing infrastructure)
        self._lit_cfg = self._cfg.get("agents", {}).get("literature", {})

    # ------------------------------------------------------------------
    def _get_hot_swap(self) -> Any:
        """Lazily create and return the LoRAHotSwap instance."""
        if self._hot_swap_instance is None:
            from synthesis.lora_routing.hot_swap import LoRAHotSwap
            self._hot_swap_instance = LoRAHotSwap(self._cfg)
        return self._hot_swap_instance

    def _ensure_foundation_loaded(self) -> None:
        """Load the Foundation Model if it hasn't been loaded yet."""
        if not self._foundation_loaded:
            self._get_hot_swap().load_foundation()
            self._foundation_loaded = True

    # ------------------------------------------------------------------
    def _depth_analysis(
        self,
        sota_code: str,
        trajectory_steps: list[Any],
        generation_index: int,
    ) -> DepthAnalysis:
        """
        Phase 1: Run depth analysis using the base Foundation Model (no LoRA).

        Feeds the full trajectory context into the unaltered model to extract a
        structured ``DepthAnalysis``.
        """
        trajectory_summary = _build_trajectory_summary(trajectory_steps)
        sota_snippet = sota_code[:3000]

        prompt = _DEPTH_ANALYSIS_TEMPLATE.format(
            trajectory_summary=trajectory_summary,
            sota_code=sota_snippet,
        )

        logger.info("DynamicLoRARouter Phase 1: depth analysis …")

        if self._hot_swap_instance is not None or self._foundation_loaded:
            # Use in-process generation via the Foundation Model (no adapter)
            self._ensure_foundation_loaded()
            hot_swap = self._get_hot_swap()
            if hot_swap.is_adapter_loaded():
                hot_swap.remove_adapter()
            raw_response = hot_swap.generate(
                _DEPTH_ANALYSIS_SYSTEM + "\n\n" + prompt
            )
        else:
            # Fall back to proxy for depth analysis
            raw_response = self._llm.complete(prompt, system=_DEPTH_ANALYSIS_SYSTEM)

        analysis = _parse_depth_analysis(raw_response, fallback_epoch=generation_index)
        logger.info(
            "DynamicLoRARouter Phase 1: bottleneck='%s', epoch=%d",
            analysis.bottleneck, analysis.recommended_epoch,
        )
        return analysis

    # ------------------------------------------------------------------
    def _retrieve_papers(self, depth_analysis: DepthAnalysis) -> tuple[list, list]:
        """
        Phase 2: Agentic RAG retrieval driven by the depth analysis bottleneck.

        Returns
        -------
        tuple[list[Paper], list[Chunk]]
        """
        query = f"{depth_analysis.bottleneck} deep learning improvement techniques"
        logger.info("DynamicLoRARouter Phase 2: literature retrieval for '%s' …", query)

        try:
            from agents.literature.retriever import LiteratureRetriever
            from agents.literature.searcher import LiteratureSearcher
            from agents.literature.vector_store import build_vector_store

            searcher = LiteratureSearcher(self._cfg)
            papers = searcher.search(query)

            vector_store = build_vector_store(self._cfg)
            retriever = LiteratureRetriever(self._cfg, vector_store=vector_store)
            chunks = retriever.retrieve_and_chunk(papers)

            logger.info(
                "DynamicLoRARouter Phase 2: %d papers, %d chunks retrieved.",
                len(papers), len(chunks),
            )
            return papers, chunks
        except Exception as exc:
            logger.warning(
                "DynamicLoRARouter Phase 2: literature retrieval failed: %s — "
                "proceeding without literature context.",
                exc,
            )
            return [], []

    # ------------------------------------------------------------------
    def _width_synthesis(
        self,
        sota_code: str,
        depth_analysis: DepthAnalysis,
        papers: list,
        adapter_path: str | None,
    ) -> tuple[str, str]:
        """
        Phase 3: Width Synthesis with hot-swapped LoRA.

        Returns
        -------
        tuple[str, str]
            (code, rationale)
        """
        literature_summary = _format_papers([], papers)
        prompt = _WIDTH_SYNTHESIS_TEMPLATE.format(
            bottleneck=depth_analysis.bottleneck,
            structural_summary=depth_analysis.structural_summary,
            literature_summary=literature_summary,
            sota_code=sota_code[:3000],
        )

        full_prompt = _WIDTH_SYNTHESIS_SYSTEM + "\n\n" + prompt

        logger.info(
            "DynamicLoRARouter Phase 3: Width synthesis "
            "(adapter=%s) …", adapter_path or "none",
        )

        if adapter_path and self._foundation_loaded:
            # In-process generation with hot-swapped Width LoRA
            hot_swap = self._get_hot_swap()
            with hot_swap.adapter(adapter_path):
                raw_output = hot_swap.generate(full_prompt)
        else:
            # Fall back to proxy (no LoRA hot-swap)
            raw_output = self._llm.complete(prompt, system=_WIDTH_SYNTHESIS_SYSTEM)

        code = _extract_code(raw_output)
        rationale = _extract_rationale(raw_output)
        return code, rationale

    # ------------------------------------------------------------------
    def route(
        self,
        sota_code: str,
        trajectory_steps: list[Any] | None = None,
        generation_index: int = 0,
    ) -> SynthesisResult:
        """
        Execute the full three-phase Dynamic LoRA Routing pipeline.

        Parameters
        ----------
        sota_code:
            Source code of the current SOTA algorithm.
        trajectory_steps:
            Ordered list of ``TrajectoryStep`` objects representing the
            evolutionary history.
        generation_index:
            Hint for which generation's Width LoRA to use when the depth
            analysis does not produce a clear recommendation.

        Returns
        -------
        SynthesisResult
            Contains the generated code, rationale, adapter path used, and
            metadata from all three phases.
        """
        steps = trajectory_steps or []

        try:
            # ── Phase 1: Depth Analysis ──────────────────────────────────
            depth_analysis = self._depth_analysis(sota_code, steps, generation_index)

            # ── Phase 2: Agentic RAG Retrieval ───────────────────────────
            papers, _ = self._retrieve_papers(depth_analysis)

            # ── Phase 3: Width Synthesis ──────────────────────────────────
            # Resolve the adapter path from the library (if available)
            epoch = depth_analysis.recommended_epoch
            adapter_path: str | None = None
            try:
                record = self._library.get(epoch)
                adapter_path = record.adapter_path
                logger.info(
                    "DynamicLoRARouter: resolved Width LoRA for gen_%d at %s",
                    epoch, adapter_path,
                )
            except KeyError:
                logger.warning(
                    "DynamicLoRARouter: no Width LoRA registered for gen_%d — "
                    "falling back to proxy generation.",
                    epoch,
                )

            code, rationale = self._width_synthesis(
                sota_code=sota_code,
                depth_analysis=depth_analysis,
                papers=papers,
                adapter_path=adapter_path,
            )

            return SynthesisResult(
                code=code,
                rationale=rationale,
                adapter_used=adapter_path,
                depth_analysis=depth_analysis,
                papers_retrieved=len(papers),
                success=bool(code),
                error="" if code else "No code block produced by Width synthesis.",
            )

        except Exception as exc:
            logger.exception("DynamicLoRARouter.route failed: %s", exc)
            return SynthesisResult(
                code="",
                rationale="",
                adapter_used=None,
                success=False,
                error=str(exc),
            )
