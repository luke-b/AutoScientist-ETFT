"""
pipeline.py — Top-level orchestrator for the full AutoScientist-ETFT pipeline.

Chains the four major pipeline stages:
    corpus   → build evolutionary regression trajectories (𝒟_Gen + 𝒟_Rationale)
    train    → fine-tune an LLM on the accumulated corpus
    arl      → run the Agentic Research Loop for a bottleneck component
    generate → synthesise and triage SOTA+1 candidates

State tracking
--------------
A lightweight ``data/pipeline_state.json`` records what has been completed
so runs can be resumed without re-executing finished stages.

Usage:
    python pipeline.py --target image_classification_cnn --seed-code sota.py
    python pipeline.py --target ... --resume              # skip done stages
    python pipeline.py --target ... --stage arl           # single stage only
    etft-pipeline --target ...                            # installed CLI
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_ALL_STAGES = ("corpus", "analyse", "train", "calibrate", "arl", "generate")

# ---------------------------------------------------------------------------
# Pipeline state helpers
# ---------------------------------------------------------------------------


def _state_path(data_root: Path) -> Path:
    return data_root / "pipeline_state.json"


def _load_state(data_root: Path) -> dict[str, Any]:
    path = _state_path(data_root)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            logger.warning("Could not read pipeline state: %s — starting fresh.", exc)
    return {
        "completed_stages": [],
        "trajectory_ids": [],
        "arl_runs": [],
        "candidates": [],
        "events": [],
    }


def _save_state(state: dict[str, Any], data_root: Path) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    _state_path(data_root).write_text(json.dumps(state, indent=2))


def _log_event(state: dict[str, Any], stage: str, detail: dict[str, Any]) -> None:
    """Append a structured event to the state's event log (consistent with RunLogger format)."""
    state.setdefault("events", []).append({
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "stage": stage,
        **detail,
    })


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------


def _load_trajectory_steps(
    data_root: Path,
    target: str,
    held_out_frac: float = 0.0,
) -> list:
    """
    Load ``TrajectoryStep`` objects from the 𝒟_Gen JSONL files for *target*.

    Parameters
    ----------
    data_root:
        Root data directory containing ``d_gen/``.
    target:
        Algorithm family label used to filter trajectories.
    held_out_frac:
        Fraction of trajectory steps (from the end of each sorted trajectory)
        that should be tagged ``split="calibration"`` instead of ``split="train"``.
        When 0.0 (default) all steps remain ``"train"``-split, which is the
        backwards-compatible behaviour.

    Returns an empty list when no trajectory files are found (the calibration
    stage will then auto-pass the gate via ``skip_on_short_trajectory``).
    """
    import json as _json
    from corpus.regression_pipeline.schemas import TrajectoryStep

    d_gen_dir = data_root / "d_gen"
    if not d_gen_dir.exists():
        logger.debug("[calibrate] No d_gen directory found — returning empty steps list.")
        return []

    steps: list[TrajectoryStep] = []
    total_skipped = 0
    for path in sorted(d_gen_dir.glob("*.jsonl")):
        file_records = 0
        file_skipped = 0
        try:
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    record = _json.loads(line)
                    # Each line in a 𝒟_Gen JSONL is a TrajectoryPair; extract both steps
                    for step_key in ("step_before", "step_after"):
                        step_data = record.get(step_key)
                        if step_data and record.get("trajectory_id", "").endswith(target):
                            file_records += 1
                            try:
                                steps.append(TrajectoryStep.model_validate(step_data))
                            except Exception as exc:
                                file_skipped += 1
                                logger.warning(
                                    "[calibrate] Skipping malformed step in %s: %s",
                                    path, exc,
                                )
        except Exception as exc:
            logger.debug("[calibrate] Could not parse %s: %s", path, exc)
            continue

        total_skipped += file_skipped
        if file_records > 0 and file_skipped > file_records * 0.5:
            raise RuntimeError(
                f"[calibrate] More than 50% of records in {path} failed to parse "
                f"({file_skipped}/{file_records}). "
                "Aborting to prevent silent data loss — inspect the JSONL file."
            )

    if total_skipped:
        logger.warning("[calibrate] Total malformed steps skipped: %d", total_skipped)

    # De-duplicate by (algorithm_id, step_index) and sort
    seen: set[tuple] = set()
    unique_steps = []
    for s in sorted(steps, key=lambda x: x.step_index):
        key = (s.algorithm_id, s.step_index)
        if key not in seen:
            seen.add(key)
            unique_steps.append(s)

    # A1: tag the last k steps of each trajectory as "calibration" split
    if held_out_frac > 0.0 and unique_steps:
        k = max(1, int(round(len(unique_steps) * held_out_frac)))
        for step in unique_steps[:-k]:
            step.split = "train"
        for step in unique_steps[-k:]:
            step.split = "calibration"
        logger.info(
            "[calibrate] Held-out split: %d train / %d calibration steps (held_out_frac=%.2f).",
            len(unique_steps) - k, k, held_out_frac,
        )

    logger.debug("[calibrate] Loaded %d unique trajectory steps for target '%s'.", len(unique_steps), target)
    return unique_steps


def _run_corpus_stage(
    cfg: dict,
    target: str,
    seed_code: str,
    seed_fitness: float,
    depth: int,
    data_root: Path,
    state: dict[str, Any],
) -> str:
    """Run the regression pipeline and return the trajectory JSONL path."""
    from corpus.regression_pipeline.run import run_regression_pipeline

    logger.info("[corpus] Building evolutionary trajectory for '%s' …", target)

    output_dir = data_root
    run_regression_pipeline(
        cfg=cfg,
        algorithm_family=target,
        seed_code=seed_code,
        seed_fitness=seed_fitness,
        depth=depth,
        output_dir=output_dir,
    )

    # Record generated trajectory IDs from d_gen
    d_gen_dir = data_root / "d_gen"
    new_files = [str(p) for p in sorted(d_gen_dir.glob("*.jsonl"))]
    state.setdefault("trajectory_ids", []).extend(new_files)
    _log_event(state, "corpus", {"target": target, "d_gen_files": new_files})
    logger.info("[corpus] Done. %d 𝒟_Gen file(s) written.", len(new_files))
    return str(d_gen_dir)


def _run_analyse_stage(
    cfg: dict,
    data_root: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Run Pareto delta analysis on the accumulated 𝒟_Gen corpus.

    Persists the report to ``data_root/analysis_report.json`` and records the
    top bottleneck component in ``pipeline_state.json`` so the ARL stage can
    use it as ``--bottleneck`` without manual intervention.

    Returns the report dict.
    """
    from analysis.pareto_delta.run import run_analyse

    logger.info("[analyse] Running Pareto delta analysis …")

    report_path = data_root / "analysis_report.json"
    report = run_analyse(
        data_root=data_root,
        pareto_threshold=cfg.get("analysis", {}).get("pareto_delta", {}).get("pareto_threshold", 0.80),
        min_delta_lines=cfg.get("analysis", {}).get("pareto_delta", {}).get("min_delta_lines", 5),
        as_json=False,
        save_to=report_path,
    )

    # Extract top bottleneck for downstream ARL stage
    pareto_set = report.get("pareto_set", [])
    top_bottleneck = pareto_set[0]["component"] if pareto_set else None
    state["top_bottleneck"] = top_bottleneck
    _log_event(state, "analyse", {
        "report_path": str(report_path),
        "top_bottleneck": top_bottleneck,
        "total_fitness_gain": report.get("total_fitness_gain", 0),
    })
    logger.info("[analyse] Done. Top bottleneck: %s", top_bottleneck or "(none identified)")
    return report


def _run_train_stage(
    cfg: dict,
    data_root: Path,
    state: dict[str, Any],
) -> str:
    """Fine-tune the model on the accumulated corpus and return checkpoint dir."""
    from train import train as _train

    logger.info("[train] Fine-tuning model on corpus …")
    train_cfg = cfg.get("training", {})
    model_name: str = train_cfg.get("base_model", "meta-llama/Meta-Llama-3-8B")
    output_dir = Path(train_cfg.get("output_dir", "./checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)

    _train(cfg, model_name, data_root, output_dir)

    _log_event(state, "train", {"model": model_name, "output_dir": str(output_dir)})
    logger.info("[train] Done. Checkpoint at %s", output_dir)
    return str(output_dir)


def _run_calibrate_stage(
    cfg: dict,
    trajectory_id: str,
    steps: list,
    data_root: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the Evolutionary Replay calibration protocol and evaluate the stage-gate.

    When ``calibration.skip_on_short_trajectory`` is True (default) and the
    trajectory has fewer than ``min_replay_steps + 1`` steps the stage-gate is
    automatically opened so short PoC trajectories are not blocked.

    Raises ``RuntimeError`` when the gate is closed and the caller must not
    proceed to synthesis.
    """
    from calibration.engine import CalibrationEngine

    cal_cfg = cfg.get("calibration", {})
    output_dir = Path(cal_cfg.get("output_dir", str(data_root / "calibration")))
    min_steps = int(cal_cfg.get("min_replay_steps", 2))
    skip_short = bool(cal_cfg.get("skip_on_short_trajectory", True))

    logger.info("[calibrate] Running Evolutionary Replay for trajectory '%s' …", trajectory_id)

    if skip_short and len(steps) < min_steps + 1:
        logger.info(
            "[calibrate] Trajectory has only %d step(s) — gate auto-opened "
            "(skip_on_short_trajectory=true).",
            len(steps),
        )
        # B3: use None for confidence_level and "skipped" gate_status to avoid
        # polluting calibration statistics with a synthetic C=1.0
        record = {
            "trajectory_id": trajectory_id,
            "gate_passed": True,
            "gate_status": "skipped",
            "confidence_level": None,
            "n_steps_replayed": 0,
            "n_steps_attempted": 0,
            "n_agent_failures": 0,
            "skipped": True,
        }
        _log_event(state, "calibrate", record)
        state["calibration"] = record
        _save_state(state, data_root)
        return record

    engine = CalibrationEngine(cfg)
    cal_record = engine.run(
        trajectory_id=trajectory_id,
        steps=steps,
        output_dir=output_dir,
    )

    record = {
        "trajectory_id": trajectory_id,
        "gate_passed": cal_record.gate_passed,
        "gate_status": cal_record.gate_status,
        "confidence_level": cal_record.confidence_level,
        "n_steps_replayed": cal_record.n_steps_replayed,
        "n_steps_attempted": cal_record.n_steps_attempted,
        "n_agent_failures": cal_record.n_agent_failures,
        "gate_diagnostic": cal_record.gate_diagnostic,
    }
    state["calibration"] = record
    _log_event(state, "calibrate", record)

    if not cal_record.gate_passed:
        logger.error(
            "[calibrate] Stage-gate CLOSED — C=%s < threshold=%.3f. "
            "SOTA+x synthesis is blocked. %s",
            f"{cal_record.confidence_level:.3f}" if cal_record.confidence_level is not None else "N/A",
            cal_record.threshold,
            cal_record.gate_diagnostic,
        )
        raise RuntimeError(
            f"Calibration stage-gate closed: {cal_record.gate_diagnostic}"
        )

    logger.info(
        "[calibrate] Stage-gate OPEN — C=%s. Model is Ready for Action.",
        f"{cal_record.confidence_level:.3f}" if cal_record.confidence_level is not None else "N/A",
    )
    return record


def _run_arl_stage(
    cfg: dict,
    bottleneck: str,
    data_root: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Run the Agentic Research Loop and return the summary dict."""
    from agents.run_arl import run_arl

    arl_output_dir = data_root / "arl_results"
    logger.info("[arl] Running ARL for bottleneck '%s' …", bottleneck)
    summary = run_arl(cfg=cfg, bottleneck=bottleneck, output_dir=arl_output_dir, data_root=data_root)

    run_record = {
        "bottleneck": bottleneck,
        "success_rate": summary.get("experiment_metrics", {}).get("success_rate", 0),
        "arl_output": str(arl_output_dir / f"arl_{bottleneck}.json"),
    }
    state.setdefault("arl_runs", []).append(run_record)
    _log_event(state, "arl", run_record)
    logger.info(
        "[arl] Done. Success rate: %.0f%%",
        run_record["success_rate"] * 100,
    )
    return summary


def _run_generate_stage(
    cfg: dict,
    sota_code: str,
    bottleneck: str,
    arl_summary: dict[str, Any],
    data_root: Path,
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate and triage SOTA+1 candidates, returning all candidate records."""
    from corpus.regression_pipeline.schemas import ResearchBrief
    from synthesis.sota_plus_one.generate import generate

    output_dir = data_root / "candidates"
    brief = ResearchBrief.model_validate(arl_summary.get("research_brief", {
        "bottleneck": bottleneck,
        "query": bottleneck,
        "synthesis": "No literature synthesis available.",
        "hypotheses": [],
    }))

    logger.info("[generate] Synthesising SOTA+1 candidates for '%s' …", bottleneck)
    results = generate(
        cfg=cfg,
        trajectory_id=f"pipeline_{bottleneck}",
        sota_code=sota_code,
        bottleneck=bottleneck,
        brief=brief,
        empirical_summary=arl_summary.get("experiment_metrics"),
        output_dir=output_dir,
        rl_context=arl_summary.get("rl_context_prefix", ""),
    )

    passed = [r for r in results if r.get("triage_passed")]
    for cand in passed:
        state.setdefault("candidates", []).append({
            "candidate_id": cand.get("candidate_id"),
            "triage_passed": True,
            "risk_score": cand.get("risk_score"),
        })
    _log_event(state, "generate", {
        "total": len(results),
        "passed": len(passed),
        "bottleneck": bottleneck,
    })
    logger.info("[generate] Done. %d / %d candidates passed triage.", len(passed), len(results))
    return results


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_pipeline(
    cfg: dict,
    target: str,
    seed_code: str,
    seed_fitness: float = 1.0,
    depth: int | None = None,
    data_root: Path | None = None,
    stage: str = "all",
    resume: bool = False,
) -> dict[str, Any]:
    """
    Run the full ETFT pipeline (or a single stage) with optional resume support.

    Parameters
    ----------
    cfg:
        Runtime configuration dict.
    target:
        Algorithm family label / bottleneck name.
    seed_code:
        Source code of the current SOTA algorithm.
    seed_fitness:
        Fitness score of the seed (default 1.0).
    depth:
        Regression depth (overrides ``cfg.corpus.regression_pipeline.max_regression_depth``).
    data_root:
        Root directory for all data outputs (default ``./data``).
    stage:
        One of ``"corpus"``, ``"analyse"``, ``"train"``, ``"calibrate"``,
        ``"arl"``, ``"generate"``, or ``"all"``.
    resume:
        When *True*, skip stages that are already recorded as completed in
        ``pipeline_state.json``.

    Returns
    -------
    dict
        Pipeline result summary.
    """
    if data_root is None:
        data_root = Path(cfg.get("data", {}).get("root", "./data"))

    regression_depth = depth or cfg.get("corpus", {}).get(
        "regression_pipeline", {}
    ).get("max_regression_depth", 10)

    state = _load_state(data_root)
    completed = set(state.get("completed_stages", []))

    stages_to_run = _ALL_STAGES if stage == "all" else (stage,)
    for s in stages_to_run:
        if s not in _ALL_STAGES:
            raise ValueError(f"Unknown stage {s!r}. Must be one of {_ALL_STAGES}.")

    result: dict[str, Any] = {}
    arl_summary: dict[str, Any] = {}

    for current_stage in stages_to_run:
        if resume and current_stage in completed:
            logger.info("[pipeline] Skipping already-completed stage: %s", current_stage)
            continue

        logger.info("[pipeline] ===== Stage: %s =====", current_stage)

        if current_stage == "corpus":
            result["corpus"] = _run_corpus_stage(
                cfg, target, seed_code, seed_fitness, regression_depth, data_root, state
            )

        elif current_stage == "analyse":
            result["analyse"] = _run_analyse_stage(cfg, data_root, state)

        elif current_stage == "train":
            result["train"] = _run_train_stage(cfg, data_root, state)

        elif current_stage == "calibrate":
            # Load the trajectory steps built during the corpus stage.
            # Falls back gracefully to an empty list (auto-pass) when corpus
            # has not yet been built in this run.
            cal_cfg = cfg.get("calibration", {})
            held_out_frac = float(cal_cfg.get("held_out_frac", 0.0))
            trajectory_steps = _load_trajectory_steps(data_root, target, held_out_frac)
            result["calibrate"] = _run_calibrate_stage(
                cfg, f"pipeline_{target}", trajectory_steps, data_root, state
            )

        elif current_stage == "arl":
            arl_summary = _run_arl_stage(cfg, target, data_root, state)
            result["arl"] = arl_summary

        elif current_stage == "generate":
            if not arl_summary and stage == "all":
                # Try to load ARL summary from disk
                arl_file = data_root / "arl_results" / f"arl_{target}.json"
                if arl_file.exists():
                    arl_summary = json.loads(arl_file.read_text())
                else:
                    logger.warning(
                        "[generate] No ARL summary found — generating with empty brief."
                    )
                    arl_summary = {}
            result["generate"] = _run_generate_stage(
                cfg, seed_code, target, arl_summary, data_root, state
            )

        state.setdefault("completed_stages", [])
        if current_stage not in state["completed_stages"]:
            state["completed_stages"].append(current_stage)
        _save_state(state, data_root)

    result["state"] = state
    logger.info("[pipeline] Pipeline complete for target '%s'.", target)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AutoScientist-ETFT full pipeline orchestrator."
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Algorithm family / bottleneck label, e.g. 'image_classification_cnn'.",
    )
    parser.add_argument(
        "--seed-code",
        default=None,
        help="Path to the seed SOTA Python file. Uses a placeholder if omitted.",
    )
    parser.add_argument(
        "--seed-fitness",
        type=float,
        default=1.0,
        help="Fitness score of the seed SOTA (default: 1.0).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Number of regression steps (overrides config).",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Root data directory (default: cfg.data.root or ./data).",
    )
    parser.add_argument(
        "--stage",
        choices=list(_ALL_STAGES) + ["all"],
        default="all",
        help="Run a specific stage only (default: all).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Skip stages already recorded as completed in pipeline_state.json.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml).",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        default=False,
        help="Validate config.yaml against the schema and exit (0 = valid, 1 = invalid).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.validate_config:
        import sys

        from pydantic import ValidationError

        from etft.config import load_config

        try:
            load_config(args.config, validate=True)
            print(f"Config '{args.config}' is valid.")
            sys.exit(0)
        except ValidationError as exc:
            print(f"Config validation failed for '{args.config}':\n{exc}", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:
            print(f"Failed to load config '{args.config}': {exc}", file=sys.stderr)
            sys.exit(1)

    cfg = load_config(args.config)

    seed_code = ""
    if args.seed_code:
        seed_code = Path(args.seed_code).read_text()
    else:
        logger.warning("No --seed-code provided. Using a trivial placeholder.")
        seed_code = "# Placeholder SOTA algorithm\ndef model(x):\n    return x\n"

    data_root = Path(args.data_root) if args.data_root else None

    run_pipeline(
        cfg=cfg,
        target=args.target,
        seed_code=seed_code,
        seed_fitness=args.seed_fitness,
        depth=args.depth,
        data_root=data_root,
        stage=args.stage,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
