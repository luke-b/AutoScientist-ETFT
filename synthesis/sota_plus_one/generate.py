"""
synthesis/sota_plus_one/generate.py — CLI entrypoint for SOTA+1 generation.

Usage:
    python -m synthesis.sota_plus_one.generate --trajectory <id> \\
        [--bottleneck <component>] [--arl-result ./data/arl_results/arl_X.json] \\
        [--submit]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corpus.regression_pipeline.schemas import SOTAPlusOneCandidate

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
    rl_context: str = "",
    submit: bool = False,
) -> list[dict]:
    """
    Generate, triage, and persist SOTA+1 candidates.

    Parameters
    ----------
    cfg:
        Full runtime configuration dict.
    trajectory_id:
        Identifier of the source trajectory.
    sota_code:
        Source code of the current SOTA algorithm.
    bottleneck:
        Bottleneck component identified by Pareto analysis.
    brief:
        ResearchBrief from the ARL literature phase.
    empirical_summary:
        MetricsCollector.to_dict() output from the ARL empirical phase (or None).
    output_dir:
        Directory to write accepted candidate JSON files.
    rl_context:
        Optional in-context RL feedback string (accumulated negative reward
        signals from a prior ARL run).  Forwarded to the synthesizer so the
        LLM avoids patterns that were already rejected or failed physically.
    submit:
        When True, triage-passing candidates are submitted via the configured
        cluster adapter (defaults to ``LocalSubprocessAdapter``).  Physical
        evaluation failures are routed back through ``FeedbackRouter``.

    Returns
    -------
    list[dict]
        JSON-serialisable list of candidate dicts (all candidates, triage status included).
    """
    synthesizer = SOTAPlusOneSynthesizer(cfg)
    triage = TriageFilter(cfg)

    candidates = synthesizer.generate(
        sota_code=sota_code,
        trajectory_id=trajectory_id,
        bottleneck=bottleneck,
        brief=brief,
        empirical_summary=empirical_summary,
        rl_context=rl_context,
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

            if submit:
                _submit_candidate(scored, cfg, output_dir)
        else:
            logger.warning(
                "Candidate %s rejected (risk=%.3f) — routing to feedback.",
                scored.candidate_id, scored.risk_score,
            )
            _route_triage_failure(scored, cfg)

    passed = sum(1 for r in results if r["triage_passed"])
    logger.info("%d / %d candidates passed triage.", passed, len(results))
    return results


def _route_triage_failure(candidate: SOTAPlusOneCandidate, cfg: dict) -> None:
    try:
        from feedback.rl_loop.feedback_router import FeedbackRouter
        router = FeedbackRouter(cfg, load_history=True)
        router.route_triage_failure(candidate)
    except Exception as exc:
        logger.debug("Feedback routing skipped: %s", exc)


def _submit_candidate(candidate: SOTAPlusOneCandidate, cfg: dict, output_dir: Path) -> None:
    """Submit a triage-passing candidate via the cluster adapter."""
    from synthesis.sota_plus_one.cluster_adapter import get_adapter

    adapter = get_adapter(cfg)
    try:
        job_id = adapter.submit(candidate)
        logger.info("Candidate %s submitted — job_id=%s", candidate.candidate_id, job_id)
    except Exception as exc:
        failure_reason = str(exc)
        logger.warning(
            "Physical evaluation of candidate %s failed: %s",
            candidate.candidate_id, failure_reason,
        )
        try:
            from feedback.rl_loop.feedback_router import FeedbackRouter
            router = FeedbackRouter(cfg, data_root=output_dir.parent)
            router.route_physical_eval_failure(candidate, failure_reason=failure_reason)
        except Exception as fb_exc:
            logger.debug("Physical eval feedback routing skipped: %s", fb_exc)


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
    parser.add_argument(
        "--submit",
        action="store_true",
        default=False,
        help=(
            "Submit triage-passing candidates to the cluster adapter for physical evaluation. "
            "Physical evaluation failures are automatically routed to the feedback loop."
        ),
    )
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
    rl_context = ""

    if args.arl_result:
        with open(args.arl_result) as f:
            arl = json.load(f)
        if "research_brief" in arl:
            brief = ResearchBrief.model_validate(arl["research_brief"])
        empirical_summary = arl.get("experiment_metrics")
        rl_context = arl.get("rl_context_prefix", "")

    generate(
        cfg=cfg,
        trajectory_id=args.trajectory,
        sota_code=sota_code,
        bottleneck=args.bottleneck,
        brief=brief,
        empirical_summary=empirical_summary,
        output_dir=Path(args.output_dir),
        rl_context=rl_context,
        submit=args.submit,
    )


if __name__ == "__main__":
    main()
