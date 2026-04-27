"""
reporting.py — ``etft-report`` CLI command.

Reads all ``run_summary.json`` files under ``data/runs/`` and prints a
formatted summary table.  Supports both human-readable (Rich) output and
machine-readable JSON output for integration with external dashboards.

Usage:
    etft-report                          # Rich table, default data dir
    etft-report --data-root ./my_data    # custom data dir
    etft-report --json                   # JSON output
    python reporting.py --json | jq '.runs[].success_rate'
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def _load_run_summaries(data_root: Path) -> list[dict[str, Any]]:
    """
    Walk ``data_root/runs/`` and load every ``run_summary.json``.

    Missing or malformed files are silently skipped with a warning.

    Returns
    -------
    list[dict]
        Run summary dicts sorted by ``timestamp`` (oldest first).
    """
    runs_dir = data_root / "runs"
    summaries: list[dict[str, Any]] = []

    if not runs_dir.exists():
        return summaries

    for run_dir in sorted(runs_dir.iterdir()):
        summary_path = run_dir / "run_summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except Exception as exc:
            logger.warning("Skipping malformed summary %s: %s", summary_path, exc)
            continue

        # Count alerts from the sibling alerts.jsonl if present
        alerts_path = run_dir / "alerts.jsonl"
        if alerts_path.exists():
            alert_count = sum(1 for ln in alerts_path.read_text().splitlines() if ln.strip())
        else:
            alert_count = summary.get("alert_count", 0)
        summary["_alert_count"] = alert_count

        # Top skills from experiments.jsonl if present
        exp_path = run_dir / "experiments.jsonl"
        top_skills: list[str] = []
        if exp_path.exists():
            top_skills = _extract_top_skills(exp_path)
        summary["_top_skills"] = top_skills

        summaries.append(summary)

    summaries.sort(key=lambda s: s.get("timestamp", ""))
    return summaries


def _extract_top_skills(experiments_path: Path, top_n: int = 3) -> list[str]:
    """Return the top *top_n* skill names from the experiments.jsonl file."""
    skill_counts: dict[str, int] = {}
    try:
        for line in experiments_path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            for skill in record.get("skills_invoked", []):
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
    except Exception:
        pass
    sorted_skills = sorted(skill_counts, key=lambda k: skill_counts[k], reverse=True)
    return sorted_skills[:top_n]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _to_report_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert raw summaries to normalised row dicts for the report."""
    rows = []
    for s in summaries:
        metrics = s.get("metrics") or {}
        row = {
            "run_id": s.get("run_id", "—"),
            "bottleneck": s.get("bottleneck", "—"),
            "timestamp": (s.get("timestamp") or "")[:19],  # trim microseconds
            "total_experiments": s.get("total_experiments", 0),
            "success_rate": metrics.get("success_rate", 0.0),
            "top_skills": ", ".join(s.get("_top_skills") or []) or "—",
            "alert_count": s.get("_alert_count", 0),
        }
        rows.append(row)
    return rows


def _print_rich_table(rows: list[dict[str, Any]]) -> None:
    """Print a Rich-formatted table.  Falls back to plain text if Rich is absent."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="AutoScientist-ETFT Run Report", show_lines=True)

        table.add_column("Run ID", style="cyan", no_wrap=True)
        table.add_column("Bottleneck", style="magenta")
        table.add_column("Timestamp", style="dim")
        table.add_column("Experiments", justify="right")
        table.add_column("Success %", justify="right")
        table.add_column("Top Skills")
        table.add_column("Alerts", justify="right")

        for row in rows:
            success_pct = f"{row['success_rate']:.0%}"
            alert_str = str(row["alert_count"])
            alert_style = "red" if row["alert_count"] > 0 else "green"

            table.add_row(
                row["run_id"],
                row["bottleneck"],
                row["timestamp"],
                str(row["total_experiments"]),
                success_pct,
                row["top_skills"],
                f"[{alert_style}]{alert_str}[/{alert_style}]",
            )

        console.print(table)

        total_exp = sum(r["total_experiments"] for r in rows)
        avg_success = (
            sum(r["success_rate"] for r in rows) / len(rows) if rows else 0.0
        )
        total_alerts = sum(r["alert_count"] for r in rows)
        console.print(
            f"\n[bold]Total runs:[/bold] {len(rows)} | "
            f"[bold]Total experiments:[/bold] {total_exp} | "
            f"[bold]Avg success:[/bold] {avg_success:.0%} | "
            f"[bold]Total alerts:[/bold] {total_alerts}"
        )

    except ImportError:
        _print_plain_table(rows)


def _print_plain_table(rows: list[dict[str, Any]]) -> None:
    """Fallback plain-text table (no Rich dependency)."""
    header = (
        f"{'Run ID':<14} {'Bottleneck':<25} {'Timestamp':<20} "
        f"{'Exps':>6} {'Succ%':>7} {'Alerts':>7}  Top Skills"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        success_pct = f"{row['success_rate']:.0%}"
        print(
            f"{row['run_id']:<14} {row['bottleneck']:<25} {row['timestamp']:<20} "
            f"{row['total_experiments']:>6} {success_pct:>7} {row['alert_count']:>7}  "
            f"{row['top_skills']}"
        )


# ---------------------------------------------------------------------------
# Main report function
# ---------------------------------------------------------------------------


def generate_report(
    data_root: Path,
    as_json: bool = False,
) -> dict[str, Any] | None:
    """
    Load run summaries and either print a table or return a dict.

    Parameters
    ----------
    data_root:
        Root data directory (must contain a ``runs/`` sub-directory).
    as_json:
        When *True* return the report as a dict instead of printing.

    Returns
    -------
    dict | None
        When *as_json* is *True* returns ``{"runs": [...], "totals": {...}}``.
        Otherwise returns *None* after printing to stdout.
    """
    summaries = _load_run_summaries(data_root)
    rows = _to_report_rows(summaries)

    if not rows:
        if as_json:
            return {"runs": [], "totals": {"run_count": 0}}
        print("No run summaries found under", data_root / "runs")
        return None

    if as_json:
        totals: dict[str, Any] = {
            "run_count": len(rows),
            "total_experiments": sum(r["total_experiments"] for r in rows),
            "avg_success_rate": (
                sum(r["success_rate"] for r in rows) / len(rows) if rows else 0.0
            ),
            "total_alerts": sum(r["alert_count"] for r in rows),
        }
        return {"runs": rows, "totals": totals}

    _print_rich_table(rows)
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a summary report of all AutoScientist-ETFT ARL runs."
    )
    parser.add_argument(
        "--data-root",
        default="./data",
        help="Root data directory (default: ./data).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        dest="as_json",
        help="Output the report as machine-readable JSON.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    args = parse_args()
    data_root = Path(args.data_root)
    result = generate_report(data_root, as_json=args.as_json)
    if args.as_json and result is not None:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
