"""
run.py — CLI entrypoint for the AI-Assisted Top-Down Regression Pipeline.

Usage:
    python -m corpus.regression_pipeline.run --target <algorithm_family> \\
        [--seed-code path/to/sota.py] [--depth 5] [--output-dir ./data]
"""

from __future__ import annotations

import argparse
import logging
import uuid
from pathlib import Path

from dotenv import load_dotenv

from corpus.regression_pipeline.cicd_validator import CICDValidator
from corpus.regression_pipeline.dataset_builder import DatasetBuilder
from corpus.regression_pipeline.regression_agent import RegressionAgent
from corpus.regression_pipeline.schemas import TrajectoryStep
from etft.config import load_config

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core pipeline logic
# ---------------------------------------------------------------------------


def run_regression_pipeline(
    cfg: dict,
    algorithm_family: str,
    seed_code: str,
    seed_fitness: float,
    depth: int,
    output_dir: Path,
) -> None:
    """
    Build an evolutionary trajectory by iteratively de-optimising *seed_code*.

    Parameters
    ----------
    cfg:
        Full config dict.
    algorithm_family:
        Human-readable label for the algorithm family.
    seed_code:
        Source code of the current SOTA (top of the trajectory).
    seed_fitness:
        Fitness score of the seed SOTA.
    depth:
        Maximum number of regression steps.
    output_dir:
        Root data directory (sub-dirs d_gen / d_rationale created automatically).
    """
    trajectory_id = str(uuid.uuid4())
    logger.info("Starting regression pipeline | family=%s depth=%d tid=%s", algorithm_family, depth, trajectory_id)

    agent = RegressionAgent(cfg)
    validator = CICDValidator(cfg)

    steps: list[TrajectoryStep] = [
        TrajectoryStep(
            step_index=depth,
            algorithm_id=f"{algorithm_family}_step_{depth}",
            algorithm_family=algorithm_family,
            code=seed_code,
            fitness_score=seed_fitness,
        )
    ]

    current_code = seed_code
    current_fitness = seed_fitness

    for step in range(depth - 1, -1, -1):
        logger.info("Regression step %d / %d …", depth - step, depth)
        predecessor_code = agent.regress(current_code, algorithm_family, current_fitness)

        result = validator.validate(predecessor_code)
        if not result.passed:
            logger.warning("Validation failed at step %d (%s) — stopping early.", step, result.status)
            break

        estimated_fitness = current_fitness * 0.85  # placeholder delta; real eval replaces this
        steps.append(
            TrajectoryStep(
                step_index=step,
                algorithm_id=f"{algorithm_family}_step_{step}",
                algorithm_family=algorithm_family,
                code=predecessor_code,
                fitness_score=estimated_fitness,
                metadata={"validation": result.model_dump()},
            )
        )
        current_code = predecessor_code
        current_fitness = estimated_fitness

    if len(steps) < 2:
        logger.error("Not enough valid steps to build a trajectory (need ≥ 2). Aborting.")
        return

    builder = DatasetBuilder(cfg)
    d_gen_dir = output_dir / "d_gen"
    builder.build_from_trajectory(steps, d_gen_dir, trajectory_id=trajectory_id)
    logger.info("Pipeline complete for trajectory %s", trajectory_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI-Assisted Top-Down Regression Pipeline — build ETFT corpus."
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Algorithm family label, e.g. 'image_classification_cnn'.",
    )
    parser.add_argument(
        "--seed-code",
        default=None,
        help="Path to the seed SOTA Python file. If omitted, a placeholder is used.",
    )
    parser.add_argument("--seed-fitness", type=float, default=1.0, help="Fitness of the seed SOTA.")
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Number of regression steps (overrides config.yaml).",
    )
    parser.add_argument("--output-dir", default="./data", help="Root data directory.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    depth = args.depth or cfg["corpus"]["regression_pipeline"]["max_regression_depth"]

    if args.seed_code:
        seed_code = Path(args.seed_code).read_text()
    else:
        logger.warning("No --seed-code provided. Using a trivial placeholder.")
        seed_code = (
            "# Placeholder SOTA algorithm\n"
            "def model(x):\n"
            "    return x\n"
        )

    run_regression_pipeline(
        cfg=cfg,
        algorithm_family=args.target,
        seed_code=seed_code,
        seed_fitness=args.seed_fitness,
        depth=depth,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
