"""
agents/run_arl.py — CLI entrypoint for the Agentic Research Loop (ARL).

Orchestrates the full pipeline:
  literature search → retrieval → synthesis → experiment design →
  experiment execution → metrics collection → feedback routing

Usage:
    python -m agents.run_arl --bottleneck batch_normalisation
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from agents.empirical.experiment_designer import ExperimentDesigner
from agents.empirical.metrics_collector import MetricsCollector
from agents.empirical.runner import ExperimentRunner
from agents.literature.retriever import LiteratureRetriever
from agents.literature.searcher import LiteratureSearcher
from agents.literature.synthesizer import LiteratureSynthesizer
from etft.config import load_config

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core ARL loop
# ---------------------------------------------------------------------------


def run_arl(cfg: dict, bottleneck: str, output_dir: Path) -> dict:
    """
    Execute the Agentic Research Loop for *bottleneck*.

    Returns
    -------
    dict
        Summary containing the research brief, experiment results, and metrics.
    """
    logger.info("=== Agentic Research Loop | bottleneck=%r ===", bottleneck)

    # --- Phase 1: Literature synthesis ---
    query = f"{bottleneck} deep learning improvement techniques"
    searcher = LiteratureSearcher(cfg)
    papers = searcher.search(query)
    logger.info("Retrieved %d papers.", len(papers))

    retriever = LiteratureRetriever(cfg)
    chunks = retriever.retrieve_and_chunk(papers)
    logger.info("Produced %d text chunks.", len(chunks))

    synthesizer = LiteratureSynthesizer(cfg)
    brief = synthesizer.synthesize(bottleneck, query, chunks, papers)
    logger.info("Research brief: %d hypotheses.", len(brief.hypotheses))

    # --- Phase 2: Empirical micro-experiments ---
    designer = ExperimentDesigner(cfg)
    runner = ExperimentRunner(cfg)
    collector = MetricsCollector()

    for i, hypothesis in enumerate(brief.hypotheses):
        logger.info("Designing experiment %d / %d: %r", i + 1, len(brief.hypotheses), hypothesis)
        try:
            script = designer.design(brief, hypothesis_index=i)
        except Exception as exc:
            logger.error("Experiment design failed for hypothesis %d: %s", i, exc)
            continue

        result = runner.run(script)
        collector.add(result)
        if result.success:
            logger.info("Experiment %d succeeded: %s", i, result.metrics)
        else:
            logger.warning("Experiment %d failed: %s", i, result.error_message)

    metrics_summary = collector.to_dict()
    logger.info("ARL complete. Success rate: %.0f%%", metrics_summary["success_rate"] * 100)

    # --- Persist results ---
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "bottleneck": bottleneck,
        "research_brief": brief.model_dump(),
        "experiment_metrics": metrics_summary,
    }
    out_path = output_dir / f"arl_{bottleneck}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("ARL summary written to %s", out_path)

    # --- Route failures for RL feedback ---
    _route_failures(collector, cfg, output_dir)

    return summary


def _route_failures(collector: MetricsCollector, cfg: dict, output_dir: Path) -> None:
    """Forward failed experiments to the feedback router."""
    failures = collector.failures
    if not failures:
        return
    try:
        from feedback.rl_loop.feedback_router import FeedbackRouter
        router = FeedbackRouter(cfg, data_root=output_dir.parent)
        for failure in failures:
            router.route_experiment_failure(failure)
    except Exception as exc:
        logger.warning("Feedback routing skipped: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Agentic Research Loop.")
    parser.add_argument(
        "--bottleneck",
        required=True,
        help="Architectural bottleneck component, e.g. 'batch_normalisation'.",
    )
    parser.add_argument("--output-dir", default="./data/arl_results", help="Output directory.")
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_arl(cfg, args.bottleneck, Path(args.output_dir))


if __name__ == "__main__":
    main()
