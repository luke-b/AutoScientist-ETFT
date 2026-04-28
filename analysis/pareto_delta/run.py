"""
analysis/pareto_delta/run.py — ``etft-analyse`` CLI command.

Loads TrajectoryStep records from 𝒟_Gen JSONL files (or a direct trajectory
JSONL), runs the full Pareto delta analysis pipeline, and outputs a bottleneck
report.

Usage
-----
    etft-analyse --data-root ./data
    etft-analyse --trajectory-file ./data/d_gen/my_traj.jsonl
    etft-analyse --data-root ./data --json
    etft-analyse --data-root ./data --plot  # requires [viz] extra
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from analysis.pareto_delta.bottleneck_reporter import generate_report, save_report
from corpus.regression_pipeline.schemas import TrajectoryStep

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_steps_from_file(path: Path) -> list[TrajectoryStep]:
    """
    Load TrajectoryStep records from a single JSONL file.

    Accepts two JSONL formats:
    - Trajectory step dicts: records with ``step_index``, ``code``, ``fitness_score``
    - Training example dicts: ``{"prompt": "...", "completion": "..."}`` — skipped
      (these are 𝒟_Gen prompt/completion records, not step records).
    """
    steps: list[TrajectoryStep] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed JSON at %s:%d: %s", path, lineno, exc)
            continue

        # Skip prompt/completion pairs (𝒟_Gen training examples)
        if "prompt" in record and "completion" in record and "step_index" not in record:
            continue

        try:
            steps.append(TrajectoryStep.model_validate(record))
        except Exception as exc:
            logger.debug("Skipping record at %s:%d: %s", path, lineno, exc)
    return steps


def _load_steps_from_data_root(data_root: Path) -> list[TrajectoryStep]:
    """
    Walk ``data_root/d_gen/`` and load all TrajectoryStep records.

    Falls back to ``data_root/trajectories/`` if ``d_gen`` is absent.
    """
    candidates = [data_root / "d_gen", data_root / "trajectories"]
    steps: list[TrajectoryStep] = []
    for directory in candidates:
        if not directory.exists():
            continue
        for jsonl_file in sorted(directory.glob("*.jsonl")):
            file_steps = _load_steps_from_file(jsonl_file)
            if file_steps:
                logger.info("Loaded %d steps from %s", len(file_steps), jsonl_file)
            steps.extend(file_steps)
    return steps


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_rich_report(report: dict) -> None:
    """Print a Rich-formatted bottleneck report."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        console.print(
            f"\n[bold cyan]Trajectory:[/bold cyan] {report.get('trajectory_id', '—')}  "
            f"[bold]Steps:[/bold] {report.get('total_steps', 0)}  "
            f"[bold]Total Δℱ:[/bold] {report.get('total_fitness_gain', 0):.4f}"
        )

        pareto_set = report.get("pareto_set", [])
        if not pareto_set:
            console.print("[yellow]No significant component-level bottlenecks found.[/yellow]")
        else:
            table = Table(title="Pareto Bottleneck Analysis", show_lines=True)
            table.add_column("Rank", justify="right", style="cyan")
            table.add_column("Component", style="magenta")
            table.add_column("Fitness Contribution", justify="right")
            table.add_column("Cumulative %", justify="right")
            table.add_column("Occurrences", justify="right")
            table.add_column("Lines Changed", justify="right")

            for entry in pareto_set:
                table.add_row(
                    str(entry.get("rank", "")),
                    entry.get("component", "—"),
                    f"{entry.get('total_fitness_contribution', 0):.4f}",
                    f"{entry.get('cumulative_fraction', 0):.0%}",
                    str(entry.get("occurrence_count", 0)),
                    str(entry.get("total_lines_changed", 0)),
                )
            console.print(table)

        console.print(f"\n[bold]Summary:[/bold] {report.get('bottleneck_summary', '—')}")

    except ImportError:
        _print_plain_report(report)


def _print_plain_report(report: dict) -> None:
    """Plain-text fallback report."""
    print(f"Trajectory: {report.get('trajectory_id', '—')}")
    print(f"Steps: {report.get('total_steps', 0)}  Total ΔF: {report.get('total_fitness_gain', 0):.4f}")
    pareto_set = report.get("pareto_set", [])
    if pareto_set:
        print(f"\n{'Rank':>4} {'Component':<30} {'Contribution':>14} {'Cumulative%':>12}")
        print("-" * 65)
        for entry in pareto_set:
            print(
                f"{entry.get('rank', ''):>4} {entry.get('component', '—'):<30} "
                f"{entry.get('total_fitness_contribution', 0):>14.4f} "
                f"{entry.get('cumulative_fraction', 0):>11.0%}"
            )
    else:
        print("No significant component-level bottlenecks found.")
    print(f"\nSummary: {report.get('bottleneck_summary', '—')}")


def _print_plot(report: dict, pareto_threshold: float) -> None:
    """Render a Pareto bar chart (requires [viz] extra)."""
    try:
        from analysis.pareto_delta.visualizer import plot_pareto_bar

        all_ranks = report.get("all_ranks", [])
        if not all_ranks:
            print("No ranked components to plot.")
            return
        names = [r.get("component", "?") for r in all_ranks]
        contributions = [r.get("total_fitness_contribution", 0.0) for r in all_ranks]
        fig = plot_pareto_bar(names, contributions, pareto_threshold=pareto_threshold)
        fig.show()
    except ImportError:
        print(
            "matplotlib is required for --plot. "
            "Install with: pip install 'autoscientist-etft[viz]'"
        )


# ---------------------------------------------------------------------------
# Core run function (importable for tests)
# ---------------------------------------------------------------------------


def run_analyse(
    data_root: Path | None = None,
    trajectory_file: Path | None = None,
    pareto_threshold: float = 0.80,
    min_delta_lines: int = 5,
    as_json: bool = False,
    plot: bool = False,
    save_to: Path | None = None,
) -> dict:
    """
    Run the full Pareto delta analysis and return the report dict.

    Parameters
    ----------
    data_root:
        Root data directory; walks ``d_gen/*.jsonl``.
    trajectory_file:
        Direct path to a JSONL trajectory file.
    pareto_threshold:
        Cumulative fraction threshold (default 0.80).
    min_delta_lines:
        Noise filter for step deltas.
    as_json:
        When True, print JSON report to stdout.
    plot:
        When True, render a matplotlib Pareto bar chart.
    save_to:
        Optional path to save the JSON report.

    Returns
    -------
    dict
        The bottleneck report dict.
    """
    steps: list[TrajectoryStep] = []

    if trajectory_file is not None:
        steps.extend(_load_steps_from_file(trajectory_file))

    if data_root is not None:
        steps.extend(_load_steps_from_data_root(data_root))

    if not steps:
        logger.warning("No TrajectoryStep records found.")
        return {"error": "No trajectory data found."}

    logger.info("Loaded %d total trajectory steps.", len(steps))
    report = generate_report(steps, pareto_threshold=pareto_threshold, min_delta_lines=min_delta_lines)

    if save_to:
        save_report(report, save_to)
        logger.info("Report saved to %s", save_to)

    if as_json:
        print(json.dumps(report, indent=2))
    elif not plot:
        _print_rich_report(report)

    if plot:
        _print_plot(report, pareto_threshold)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AutoScientist-ETFT Pareto Delta Analysis CLI. "
            "Computes bottleneck components from evolutionary trajectory data."
        )
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--data-root",
        default=None,
        help="Root data directory (walks d_gen/*.jsonl). Default: none.",
    )
    group.add_argument(
        "--trajectory-file",
        default=None,
        help="Direct path to a trajectory JSONL file.",
    )
    parser.add_argument(
        "--pareto-threshold",
        type=float,
        default=0.80,
        help="Cumulative contribution threshold for the Pareto set (default: 0.80).",
    )
    parser.add_argument(
        "--min-delta-lines",
        type=int,
        default=5,
        help="Minimum changed lines to include a step delta (default: 5).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        dest="as_json",
        help="Output the report as machine-readable JSON.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Render a matplotlib Pareto bar chart (requires [viz] extra).",
    )
    parser.add_argument(
        "--save",
        default=None,
        dest="save_to",
        help="Save the JSON report to this file path.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    if args.data_root is None and args.trajectory_file is None:
        print("Error: provide --data-root or --trajectory-file.", file=sys.stderr)
        sys.exit(1)

    report = run_analyse(
        data_root=Path(args.data_root) if args.data_root else None,
        trajectory_file=Path(args.trajectory_file) if args.trajectory_file else None,
        pareto_threshold=args.pareto_threshold,
        min_delta_lines=args.min_delta_lines,
        as_json=args.as_json,
        plot=args.plot,
        save_to=Path(args.save_to) if args.save_to else None,
    )

    if "error" in report:
        print(f"Error: {report['error']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
