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
from agents.literature.vector_store import build_vector_store
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

    The FeedbackRouter is created once per run.  After each failed
    micro-experiment its negative reward signal is immediately appended to
    the router's internal log and the updated RL context prefix is forwarded
    to the *next* experiment design call — closing the organic negative data
    loop within a single ARL campaign.

    Returns
    -------
    dict
        Summary containing the research brief, experiment results, metrics,
        and the final RL context prefix produced during the run.
    """
    from etft.run_logger import RunLogger
    from feedback.rl_loop.feedback_router import FeedbackRouter

    logger.info("=== Agentic Research Loop | bottleneck=%r ===", bottleneck)

    run_log = RunLogger(output_dir / "runs")
    router = FeedbackRouter(cfg, data_root=output_dir.parent)

    # --- Build persistent vector store ---
    vector_store = build_vector_store(cfg)

    # --- Phase 1: Literature synthesis ---
    query = f"{bottleneck} deep learning improvement techniques"
    searcher = LiteratureSearcher(cfg)
    papers = searcher.search(query)
    logger.info("Retrieved %d papers.", len(papers))

    retriever = LiteratureRetriever(cfg, vector_store=vector_store)
    chunks = retriever.retrieve_and_chunk(papers)
    logger.info("Produced %d text chunks.", len(chunks))

    synthesizer = LiteratureSynthesizer(cfg, vector_store=vector_store)
    brief = synthesizer.synthesize(bottleneck, query, chunks, papers)
    logger.info("Research brief: %d hypotheses.", len(brief.hypotheses))

    # --- Phase 2: Empirical micro-experiments ---
    designer = ExperimentDesigner(cfg)
    runner = ExperimentRunner(cfg)
    collector = MetricsCollector()

    for i, hypothesis in enumerate(brief.hypotheses):
        logger.info("Designing experiment %d / %d: %r", i + 1, len(brief.hypotheses), hypothesis)

        # Retrieve the accumulated RL feedback from all prior failures in this run
        rl_prefix = router.build_rl_context_prefix()

        try:
            script = designer.design(brief, hypothesis_index=i, rl_context=rl_prefix)
        except Exception as exc:
            logger.error("Experiment design failed for hypothesis %d: %s", i, exc)
            continue

        result = runner.run(script)
        collector.add(result)

        if result.success:
            logger.info("Experiment %d succeeded: %s", i, result.metrics)
            run_log.log_experiment(
                hypothesis=hypothesis,
                result=result,
                rl_prefix_length=len(rl_prefix),
            )
        else:
            logger.warning("Experiment %d failed: %s", i, result.error_message)
            # Route failure immediately so the next design call sees it
            try:
                router.route_experiment_failure(result)
            except Exception as exc:
                logger.warning("Feedback routing failed for experiment %d: %s", i, exc)
            run_log.log_experiment(
                hypothesis=hypothesis,
                result=result,
                rl_prefix_length=len(rl_prefix),
            )

    metrics_summary = collector.to_dict()
    logger.info("ARL complete. Success rate: %.0f%%", metrics_summary["success_rate"] * 100)

    # Collect the final RL prefix (includes all failures from this run)
    final_rl_prefix = router.build_rl_context_prefix()

    # --- Persist results ---
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "bottleneck": bottleneck,
        "research_brief": brief.model_dump(),
        "experiment_metrics": metrics_summary,
        "rl_context_prefix": final_rl_prefix,
    }
    out_path = output_dir / f"arl_{bottleneck}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("ARL summary written to %s", out_path)

    run_log.log_run_summary(bottleneck=bottleneck, metrics=metrics_summary)

    return summary


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
