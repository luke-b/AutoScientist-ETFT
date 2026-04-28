"""
etft/skills/synthesis_skill.py — EvolutionaryResearchSynthesisSkill.

Single agentic tool that orchestrates the full SOTA+1 discovery loop:
  1. Literature search + retrieval + synthesis → ResearchBrief
  2. Pareto delta analysis (if trajectory_steps supplied) → focused hypotheses
  3. Micro-experiment design + execution + failure routing → empirical evidence
  4. SOTA+1 synthesis + triage → ranked candidate list

All LLM work is delegated to the existing domain agents/modules.  This skill
itself is pure Python orchestration — no direct LLM calls.
"""

from __future__ import annotations

import logging
from typing import Any

from etft.skills.base import Skill

logger = logging.getLogger(__name__)


class EvolutionaryResearchSynthesisSkill(Skill):
    """
    Orchestrate the full SOTA+1 discovery pipeline as a single agent tool call.

    This is the top-level skill wiring together literature search, empirical
    micro-experiments, and synthesis into a ranked list of SOTA+1 candidates.
    """

    name = "run_evolutionary_synthesis"
    description = (
        "Run the full SOTA+1 discovery loop for a given bottleneck component. "
        "Searches literature, runs micro-experiments, and returns ranked candidates. "
        "Inputs: sota_code (str), bottleneck (str), trajectory_steps (list, optional), "
        "max_hypotheses (int, optional). "
        "Returns: candidates, brief, experiment_metrics, pareto_bottlenecks, rl_context_length."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "sota_code": {
                "type": "string",
                "description": "Source code of the current SOTA algorithm.",
            },
            "bottleneck": {
                "type": "string",
                "description": "Architectural bottleneck to investigate (e.g. 'batch_normalisation').",
            },
            "trajectory_steps": {
                "type": "array",
                "description": (
                    "Optional list of trajectory step dicts (step_index, algorithm_family, "
                    "code, fitness_score) used for Pareto delta analysis."
                ),
                "items": {"type": "object"},
                "default": [],
            },
            "max_hypotheses": {
                "type": "integer",
                "description": "Maximum number of hypotheses to test. Defaults to all from brief.",
                "default": 5,
            },
            "trajectory_id": {
                "type": "string",
                "description": "Identifier for the source trajectory (default: 'skill_run').",
                "default": "skill_run",
            },
        },
        "required": ["sota_code", "bottleneck"],
    }

    def __init__(self, cfg: dict | None = None, data_root: str | None = None) -> None:
        self._cfg = cfg or {}
        self._data_root = data_root

    # ------------------------------------------------------------------
    def execute(self, **kwargs: Any) -> dict:
        """
        Run the full synthesis pipeline and return a structured result dict.

        Returns
        -------
        dict
            ``{"candidates": [...], "brief": {...}, "experiment_metrics": {...},
               "pareto_bottlenecks": [...], "rl_context_length": int}``
        """
        from pathlib import Path

        from agents.empirical.experiment_designer import ExperimentDesigner
        from agents.empirical.metrics_collector import MetricsCollector
        from agents.empirical.runner import ExperimentRunner
        from agents.literature.retriever import LiteratureRetriever
        from agents.literature.searcher import LiteratureSearcher
        from agents.literature.synthesizer import LiteratureSynthesizer
        from agents.literature.vector_store import build_vector_store
        from feedback.rl_loop.feedback_router import FeedbackRouter
        from synthesis.sota_plus_one.synthesizer import SOTAPlusOneSynthesizer
        from synthesis.sota_plus_one.triage import TriageFilter

        sota_code: str = kwargs["sota_code"]
        bottleneck: str = kwargs["bottleneck"]
        raw_steps: list[dict] = kwargs.get("trajectory_steps") or []
        max_hypotheses: int = int(kwargs.get("max_hypotheses", 5))
        trajectory_id: str = kwargs.get("trajectory_id", "skill_run")

        cfg = self._cfg
        data_root = Path(self._data_root) if self._data_root else None

        # ------------------------------------------------------------------
        # Phase 1: Literature synthesis
        # ------------------------------------------------------------------
        query = f"{bottleneck} deep learning improvement techniques"
        searcher = LiteratureSearcher(cfg)
        papers = searcher.search(query)
        logger.info("EvolutionaryResearchSynthesisSkill: retrieved %d papers", len(papers))

        vector_store = build_vector_store(cfg)
        retriever = LiteratureRetriever(cfg, vector_store=vector_store)
        chunks = retriever.retrieve_and_chunk(papers)
        logger.info("EvolutionaryResearchSynthesisSkill: produced %d chunks", len(chunks))

        synthesizer = LiteratureSynthesizer(cfg, vector_store=vector_store)
        brief = synthesizer.synthesize(bottleneck, query, chunks, papers)
        logger.info(
            "EvolutionaryResearchSynthesisSkill: brief with %d hypotheses",
            len(brief.hypotheses),
        )

        # ------------------------------------------------------------------
        # Phase 1b: Optional Pareto delta analysis to refocus hypotheses
        # ------------------------------------------------------------------
        pareto_bottlenecks: list[str] = []
        if raw_steps and len(raw_steps) >= 2:
            pareto_bottlenecks = self._run_pareto_analysis(raw_steps, cfg)
            if pareto_bottlenecks:
                logger.info(
                    "EvolutionaryResearchSynthesisSkill: Pareto bottlenecks: %s",
                    pareto_bottlenecks,
                )

        # ------------------------------------------------------------------
        # Phase 2: Micro-experiments
        # ------------------------------------------------------------------
        designer = ExperimentDesigner(cfg)
        runner = ExperimentRunner(cfg)
        collector = MetricsCollector()
        router = FeedbackRouter(cfg, data_root=data_root, load_history=False)

        hypotheses_to_test = brief.hypotheses[:max_hypotheses]
        for i, hypothesis in enumerate(hypotheses_to_test):
            logger.info(
                "EvolutionaryResearchSynthesisSkill: experiment %d/%d: %r",
                i + 1, len(hypotheses_to_test), hypothesis,
            )
            rl_prefix = router.build_rl_context_prefix()

            try:
                script = designer.design(brief, hypothesis_index=i, rl_context=rl_prefix)
            except Exception as exc:
                logger.warning("Experiment design failed for hypothesis %d: %s", i, exc)
                continue

            result = runner.run(script)
            collector.add(result)

            if not result.success:
                try:
                    router.route_experiment_failure(result)
                except Exception as exc:
                    logger.debug("Feedback routing skipped: %s", exc)

        experiment_metrics = collector.to_dict()
        final_rl_prefix = router.build_rl_context_prefix()

        # ------------------------------------------------------------------
        # Phase 3: SOTA+1 synthesis + triage
        # ------------------------------------------------------------------
        sota_synthesizer = SOTAPlusOneSynthesizer(cfg)
        triage = TriageFilter(cfg)

        raw_candidates = sota_synthesizer.generate(
            sota_code=sota_code,
            trajectory_id=trajectory_id,
            bottleneck=bottleneck,
            brief=brief,
            empirical_summary=experiment_metrics,
            rl_context=final_rl_prefix,
        )

        candidates: list[dict] = []
        for candidate in raw_candidates:
            scored = triage.evaluate(candidate)
            candidates.append(scored.model_dump())

        logger.info(
            "EvolutionaryResearchSynthesisSkill: %d candidates (%d passed triage)",
            len(candidates),
            sum(1 for c in candidates if c.get("triage_passed")),
        )

        return {
            "candidates": candidates,
            "brief": brief.model_dump(),
            "experiment_metrics": experiment_metrics,
            "pareto_bottlenecks": pareto_bottlenecks,
            "rl_context_length": len(final_rl_prefix),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _run_pareto_analysis(raw_steps: list[dict], cfg: dict) -> list[str]:
        """Return top Pareto bottleneck component names from trajectory steps."""
        try:
            from analysis.pareto_delta.bottleneck_reporter import generate_report
            from corpus.regression_pipeline.schemas import TrajectoryStep

            steps = [
                TrajectoryStep(
                    step_index=s["step_index"],
                    algorithm_id=s.get("algorithm_id", f"step_{s['step_index']}"),
                    algorithm_family=s.get("algorithm_family", "unknown"),
                    code=s["code"],
                    fitness_score=float(s["fitness_score"]),
                )
                for s in raw_steps
            ]
            analysis_cfg = cfg.get("analysis", {}).get("pareto_delta", {})
            report = generate_report(
                steps,
                pareto_threshold=float(analysis_cfg.get("pareto_threshold", 0.80)),
                min_delta_lines=int(analysis_cfg.get("min_delta_lines", 5)),
            )
            return [entry["component"] for entry in report.get("pareto_set", [])]
        except Exception as exc:
            logger.debug("Pareto analysis skipped: %s", exc)
            return []
