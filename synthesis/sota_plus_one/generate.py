"""
synthesis/sota_plus_one/generate.py — CLI entrypoint for SOTA+x generation.

Implements standard SOTA+1 synthesis and recursive SOTA+x discovery as
introduced by the Recursive Stage-Gate Calibration paper (Benda, 2026).

In recursive mode (``--recursive`` / ``generations > 1``), each verified
SOTA+k candidate is integrated back into the trajectory and the calibration
stage-gate is re-evaluated before hypothesising SOTA+(k+1).

Usage:
    python -m synthesis.sota_plus_one.generate --trajectory <id> \\
        [--bottleneck <component>] [--arl-result ./data/arl_results/arl_X.json] \\
        [--submit] [--recursive --generations 3]
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


def generate_recursive(
    cfg: dict,
    trajectory_id: str,
    sota_code: str,
    bottleneck: str,
    brief: ResearchBrief,
    empirical_summary: dict | None,
    output_dir: Path,
    rl_context: str = "",
    submit: bool = False,
    generations: int = 3,
) -> dict[str, list[dict]]:
    """
    Recursive SOTA+x discovery loop (§3, Benda 2026).

    After each successful SOTA+k generation, the best triage-passing candidate
    is promoted to the new SOTA and the calibration stage-gate is re-evaluated
    before hypothesising SOTA+(k+1).  This extends the "fossil record" by one
    generation per iteration, enabling multi-generational innovation.

    Parameters
    ----------
    cfg:
        Runtime configuration dict.
    trajectory_id:
        Base trajectory identifier.  Each generation is suffixed ``_gN``.
    sota_code:
        Source code of the *current* SOTA (SOTA+0).
    bottleneck:
        Bottleneck component from Pareto analysis (shared across generations).
    brief:
        ResearchBrief from the ARL literature phase.
    empirical_summary:
        Empirical metrics dict (or None).
    output_dir:
        Root directory for candidate files; each generation writes to
        ``output_dir/gen_N/``.
    rl_context:
        In-context RL feedback from prior ARL failures.
    submit:
        Forward to physical cluster submission.
    generations:
        Number of successive SOTA+x hypotheses to attempt (default 3).

    Returns
    -------
    dict[str, list[dict]]
        Mapping ``"gen_N" -> list_of_candidate_dicts`` for each generation.
    """
    from calibration.engine import CalibrationEngine
    from corpus.regression_pipeline.schemas import TrajectoryStep

    cal_cfg = cfg.get("calibration", {})
    cal_engine = CalibrationEngine(cfg)
    cal_output_dir = Path(cal_cfg.get("output_dir", "./data/calibration"))
    min_steps = int(cal_cfg.get("min_replay_steps", 2))
    skip_short = bool(cal_cfg.get("skip_on_short_trajectory", True))

    all_results: dict[str, list[dict]] = {}
    current_sota = sota_code
    # Seed the trajectory with a single step representing the starting SOTA
    trajectory_steps: list[TrajectoryStep] = [
        TrajectoryStep(
            step_index=0,
            algorithm_id=f"{trajectory_id}_g0",
            algorithm_family=trajectory_id,
            code=sota_code,
            fitness_score=1.0,
        )
    ]

    for gen in range(1, generations + 1):
        gen_label = f"gen_{gen}"
        gen_trajectory_id = f"{trajectory_id}_g{gen}"
        gen_output_dir = output_dir / gen_label

        logger.info("=== Recursive SOTA+%d Discovery (trajectory=%s) ===", gen, gen_trajectory_id)

        # --- Calibration stage-gate ---
        # Skip calibration when the trajectory is still too short AND the config
        # allows it (skip_on_short_trajectory=true).  When skip_short=False,
        # calibration runs unconditionally — every generation must earn the gate.
        needs_calibration = (
            len(trajectory_steps) >= min_steps + 1  # enough steps to replay
            or not skip_short                        # config says never skip
        )
        if needs_calibration:
            logger.info("[recursive] Calibrating before SOTA+%d synthesis …", gen)
            cal_record = cal_engine.run(
                trajectory_id=gen_trajectory_id,
                steps=trajectory_steps,
                output_dir=cal_output_dir,
            )
            if not cal_record.gate_passed:
                logger.error(
                    "[recursive] Stage-gate CLOSED at generation %d — C=%.3f < %.3f. "
                    "Halting recursive loop. %s",
                    gen, cal_record.confidence_level, cal_record.threshold,
                    cal_record.gate_diagnostic,
                )
                break
            logger.info("[recursive] Stage-gate OPEN — proceeding to SOTA+%d synthesis.", gen)
        else:
            logger.info(
                "[recursive] Skipping calibration for gen %d "
                "(trajectory depth=%d < min_replay_steps+1=%d, skip_on_short_trajectory=true).",
                gen, len(trajectory_steps), min_steps + 1,
            )

        # --- Generate SOTA+k candidates ---
        gen_results = generate(
            cfg=cfg,
            trajectory_id=gen_trajectory_id,
            sota_code=current_sota,
            bottleneck=bottleneck,
            brief=brief,
            empirical_summary=empirical_summary,
            output_dir=gen_output_dir,
            rl_context=rl_context,
            submit=submit,
        )
        all_results[gen_label] = gen_results

        # --- Promote best triage-passing candidate to new SOTA ---
        passed = [r for r in gen_results if r.get("triage_passed")]
        if not passed:
            logger.warning(
                "[recursive] No triage-passing candidates in generation %d — "
                "halting recursive loop.",
                gen,
            )
            break

        # Pick the candidate with the lowest risk score as the new SOTA
        best = min(passed, key=lambda r: r.get("risk_score", 1.0))
        current_sota = best["code"]
        logger.info(
            "[recursive] Promoting candidate %s (risk=%.3f) as SOTA+%d for next generation.",
            best.get("candidate_id"), best.get("risk_score", 0.0), gen,
        )

        # Extend the fossil record with the new SOTA step.
        # fitness_score is a monotonically increasing placeholder — replace
        # with the physically measured ℱ(aₙ₊ₖ) once GPU evaluation is complete.
        trajectory_steps.append(
            TrajectoryStep(
                step_index=gen,
                algorithm_id=f"{trajectory_id}_g{gen}",
                algorithm_family=trajectory_id,
                code=current_sota,
                fitness_score=1.0 + gen * 0.1,
                metadata={"source": "recursive_synthesis", "generation": gen},
            )
        )

    return all_results


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
    parser = argparse.ArgumentParser(description="Generate and triage SOTA+x candidates.")
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
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help=(
            "Enable Recursive SOTA+x Discovery (§3, Benda 2026). "
            "Each verified SOTA+k candidate is promoted to the new SOTA and the "
            "calibration stage-gate is re-evaluated before hypothesising SOTA+(k+1). "
            "Use --generations to control the number of successive hypotheses."
        ),
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=3,
        help=(
            "Number of successive SOTA+x generations to attempt in recursive mode "
            "(default: 3). Ignored when --recursive is not set."
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

    if args.recursive:
        generate_recursive(
            cfg=cfg,
            trajectory_id=args.trajectory,
            sota_code=sota_code,
            bottleneck=args.bottleneck,
            brief=brief,
            empirical_summary=empirical_summary,
            output_dir=Path(args.output_dir),
            rl_context=rl_context,
            submit=args.submit,
            generations=args.generations,
        )
    else:
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
