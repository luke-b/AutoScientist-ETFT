"""
calibration/analysis/threshold_search.py — Empirical calibration threshold search.

Sweeps C thresholds from 0.3 to 1.0 in 0.05 steps and computes precision /
recall / F1 statistics linking C-pass decisions to downstream synthesis
success (triage_passed=True in pipeline_state.json).

Usage
-----
    # Analyse all CalibrationRecord JSON files under data/calibration/:
    python -m calibration.analysis.threshold_search \\
        --calibration-dir data/calibration \\
        --pipeline-state  data/pipeline_state.json \\
        --output          data/calibration_analysis.json

    # Also produce a plot (requires matplotlib):
    python -m calibration.analysis.threshold_search \\
        --calibration-dir data/calibration \\
        --pipeline-state  data/pipeline_state.json \\
        --output          data/calibration_analysis.json \\
        --plot

Output
------
``calibration_analysis.json`` contains:

.. code-block:: json

    {
      "optimal_threshold": 0.65,
      "optimal_f1": 0.84,
      "sweep": [
        {"threshold": 0.30, "precision": ..., "recall": ..., "f1": ...,
         "n_pass": ..., "n_fail": ...},
        ...
      ]
    }

The ``optimal_threshold`` value can be used to override
``calibration.confidence_threshold`` in ``config.yaml`` or passed as
``--config`` at runtime.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_calibration_records(calibration_dir: Path) -> list[dict[str, Any]]:
    """Load all CalibrationRecord JSON files from *calibration_dir*."""
    records = []
    for path in sorted(calibration_dir.glob("calibration_*.json")):
        try:
            data = json.loads(path.read_text())
            # Skip skipped (short-trajectory) records — confidence is None
            if data.get("gate_status") == "skipped" or data.get("confidence_level") is None:
                logger.debug("Skipping %s (no measured confidence_level)", path.name)
                continue
            records.append(data)
        except Exception as exc:
            logger.warning("Could not parse %s: %s", path, exc)
    logger.info("Loaded %d calibration records from %s", len(records), calibration_dir)
    return records


def _load_pipeline_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        logger.warning("pipeline_state.json not found at %s — assuming no data.", state_path)
        return {}
    try:
        return json.loads(state_path.read_text())
    except Exception as exc:
        logger.warning("Could not parse pipeline state: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------


def _build_label_index(state: dict[str, Any]) -> dict[str, bool]:
    """
    Build a dict mapping ``trajectory_id`` → ``synthesis_success`` (bool).

    ``synthesis_success`` is True when at least one candidate with
    ``triage_passed=True`` exists in the state for that trajectory.
    """
    labels: dict[str, bool] = {}
    for cand in state.get("candidates", []):
        tid = cand.get("trajectory_id") or cand.get("candidate_id", "").split("_")[0]
        if cand.get("triage_passed"):
            labels[tid] = True
        elif tid not in labels:
            labels[tid] = False
    return labels


def sweep_thresholds(
    records: list[dict[str, Any]],
    labels: dict[str, bool],
    thresholds: list[float] | None = None,
) -> list[dict[str, Any]]:
    """
    Sweep C thresholds and compute precision/recall/F1 for each.

    A True Positive is: gate says PASS (C ≥ threshold) AND synthesis succeeded.
    A False Positive is: gate says PASS but synthesis failed.
    A False Negative is: gate says FAIL but synthesis would have succeeded.

    Parameters
    ----------
    records:
        List of CalibrationRecord dicts (only measured records, no skipped).
    labels:
        Dict mapping trajectory_id → synthesis_success (True/False).
    thresholds:
        List of threshold values to sweep.  Defaults to 0.30 … 1.00 step 0.05.

    Returns
    -------
    list[dict]
        One entry per threshold with keys: threshold, precision, recall, f1,
        tp, fp, fn, tn, n_pass, n_fail, n_labelled.
    """
    if thresholds is None:
        thresholds = [round(0.30 + i * 0.05, 2) for i in range(15)]  # 0.30 … 1.00

    sweep = []
    for thr in thresholds:
        tp = fp = fn = tn = unlabelled = 0
        for rec in records:
            tid = rec.get("trajectory_id", "")
            c = rec["confidence_level"]
            gate_pass = c >= thr

            if tid not in labels:
                unlabelled += 1
                continue

            success = labels[tid]
            if gate_pass and success:
                tp += 1
            elif gate_pass and not success:
                fp += 1
            elif not gate_pass and success:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        if not math.isnan(precision) and not math.isnan(recall) and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = float("nan")

        sweep.append({
            "threshold": thr,
            "precision": round(precision, 4) if not math.isnan(precision) else None,
            "recall": round(recall, 4) if not math.isnan(recall) else None,
            "f1": round(f1, 4) if not math.isnan(f1) else None,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "n_pass": tp + fp,
            "n_fail": fn + tn,
            "n_labelled": tp + fp + fn + tn,
            "n_unlabelled": unlabelled,
        })
    return sweep


def find_optimal_threshold(sweep: list[dict[str, Any]]) -> tuple[float, float]:
    """Return the (threshold, f1) that maximises F1."""
    best = max(
        (entry for entry in sweep if entry["f1"] is not None),
        key=lambda e: e["f1"],
        default=None,
    )
    if best is None:
        return float("nan"), float("nan")
    return best["threshold"], best["f1"]


# ---------------------------------------------------------------------------
# Plotting (optional)
# ---------------------------------------------------------------------------


def _plot_sweep(sweep: list[dict[str, Any]], output_path: Path) -> None:
    """Save a precision-recall-F1 vs threshold plot to *output_path*."""
    try:
        import matplotlib.pyplot as plt  # type: ignore[import]
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot (pip install matplotlib).")
        return

    thresholds = [e["threshold"] for e in sweep]
    precisions = [e["precision"] or 0 for e in sweep]
    recalls = [e["recall"] or 0 for e in sweep]
    f1s = [e["f1"] or 0 for e in sweep]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions, label="Precision", marker="o")
    ax.plot(thresholds, recalls, label="Recall", marker="s")
    ax.plot(thresholds, f1s, label="F1", marker="^", linewidth=2)
    ax.set_xlabel("Confidence Threshold C")
    ax.set_ylabel("Score")
    ax.set_title("Calibration Threshold Sweep")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = output_path.with_suffix(".png")
    fig.savefig(plot_path, dpi=150)
    logger.info("Plot saved to %s", plot_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_threshold_search(
    calibration_dir: Path,
    pipeline_state_path: Path,
    output_path: Path,
    plot: bool = False,
) -> dict[str, Any]:
    """
    Run the full threshold sweep and write results to *output_path*.

    Returns the result dict.
    """
    records = _load_calibration_records(calibration_dir)
    if not records:
        logger.warning(
            "No measured calibration records found in %s — "
            "run the full pipeline first to accumulate data.",
            calibration_dir,
        )
        result: dict[str, Any] = {
            "optimal_threshold": None,
            "optimal_f1": None,
            "sweep": [],
            "note": "No calibration data available.",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))
        return result

    state = _load_pipeline_state(pipeline_state_path)
    labels = _build_label_index(state)
    logger.info(
        "Labels: %d trajectories with known synthesis outcome (%d success, %d fail).",
        len(labels),
        sum(labels.values()),
        sum(1 for v in labels.values() if not v),
    )

    sweep = sweep_thresholds(records, labels)
    optimal_thr, optimal_f1 = find_optimal_threshold(sweep)

    result = {
        "optimal_threshold": optimal_thr,
        "optimal_f1": optimal_f1,
        "n_calibration_records": len(records),
        "n_labelled_trajectories": len(labels),
        "sweep": sweep,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    logger.info(
        "Threshold search complete — optimal threshold=%.2f (F1=%.4f). "
        "Results written to %s",
        optimal_thr, optimal_f1, output_path,
    )

    if plot:
        _plot_sweep(sweep, output_path)

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep calibration thresholds and find the empirically optimal value.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("data/calibration"),
        help="Directory containing CalibrationRecord JSON files (default: data/calibration).",
    )
    parser.add_argument(
        "--pipeline-state",
        type=Path,
        default=Path("data/pipeline_state.json"),
        help="Path to pipeline_state.json (default: data/pipeline_state.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/calibration_analysis.json"),
        help="Output JSON path for sweep results (default: data/calibration_analysis.json).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Save a precision-recall-F1 plot alongside the JSON output.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    result = run_threshold_search(
        calibration_dir=args.calibration_dir,
        pipeline_state_path=args.pipeline_state,
        output_path=args.output,
        plot=args.plot,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
