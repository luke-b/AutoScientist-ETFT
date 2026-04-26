"""
synthesis/sota_plus_one/generate.py — CLI entrypoint for SOTA+1 generation.

Usage:
    python -m synthesis.sota_plus_one.generate --trajectory <id> \\
        [--bottleneck <component>] [--arl-result ./data/arl_results/arl_X.json]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from corpus.regression_pipeline.schemas import ResearchBrief
from etft.config import load_config
from synthesis.sota_plus_one.synthesizer import SOTAPlusOneSynthesizer
from synthesis.sota_plus_one.triage import TriageFilter

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------


def generate(
    cfg: dict,
    trajectory_id: str,
    sota_code: str,
    bottleneck: str,
    brief: ResearchBrief,
    empirical_summary: dict | None,
    output_dir: Path,
) -> list[dict]:
    """
    Generate, triage, and persist SOTA+1 candidates.

    Returns
    -------
    list[dict]
        JSON-serialisable list of candidate dicts (triage passed only).
    """
    synthesizer = SOTAPlusOneSynthesizer(cfg)
    triage = TriageFilter(cfg)

    candidates = synthesizer.generate(
        sota_code=sota_code,
        trajectory_id=trajectory_id,
        bottleneck=bottleneck,
        brief=brief,
        empirical_summary=empirical_summary,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for candidate in candidates:
        scored = triage.evaluate(candidate)
        results.append(scored.model_dump())

        if scored.triage_passed:
            out_path = output_dir / f"candidate_{scored.candidate_id}.json"
            with open(out_path, "w") as f:
                json.dump(scored.model_dump(), f, indent=2)
            logger.info("Candidate %s saved to %s", scored.candidate_id, out_path)
        else:
            logger.warning(
                "Candidate %s rejected (risk=%.3f) — routing to feedback.",
                scored.candidate_id, scored.risk_score,
            )
            _route_triage_failure(scored, cfg)

    passed = sum(1 for r in results if r["triage_passed"])
    logger.info("%d / %d candidates passed triage.", passed, len(results))
    return results


def _route_triage_failure(candidate, cfg: dict) -> None:
    try:
        from feedback.rl_loop.feedback_router import FeedbackRouter
        router = FeedbackRouter(cfg)
        router.route_triage_failure(candidate)
    except Exception as exc:
        logger.debug("Feedback routing skipped: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and triage SOTA+1 candidates.")
    parser.add_argument("--trajectory", required=True, help="Trajectory ID.")
    parser.add_argument(
        "--sota-code",
        default=None,
        help="Path to the current SOTA Python file.",
    )
    parser.add_argument(
        "--bottleneck",
        default="unknown",
        help="Bottleneck component from Pareto analysis.",
    )
    parser.add_argument(
        "--arl-result",
        default=None,
        help="Path to ARL JSON result file (for research brief + empirical data).",
    )
    parser.add_argument("--output-dir", default="./data/candidates")
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    sota_code = ""
    if args.sota_code:
        sota_code = Path(args.sota_code).read_text()

    brief = ResearchBrief(
        bottleneck=args.bottleneck,
        query=args.bottleneck,
        synthesis="No literature synthesis available.",
        hypotheses=[],
    )
    empirical_summary = None

    if args.arl_result:
        with open(args.arl_result) as f:
            arl = json.load(f)
        if "research_brief" in arl:
            brief = ResearchBrief.model_validate(arl["research_brief"])
        empirical_summary = arl.get("experiment_metrics")

    generate(
        cfg=cfg,
        trajectory_id=args.trajectory,
        sota_code=sota_code,
        bottleneck=args.bottleneck,
        brief=brief,
        empirical_summary=empirical_summary,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
