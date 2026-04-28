"""
etft/tests/test_run_logger.py — Unit tests for RunLogger observability enhancements.
"""

from __future__ import annotations

import json
from pathlib import Path

from corpus.regression_pipeline.schemas import ExperimentResult
from etft.run_logger import ALERT_CRITICAL, ALERT_WARNING, RunLogger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(success: bool = True) -> ExperimentResult:
    return ExperimentResult(
        experiment_id="exp_001",
        script="pass",
        success=success,
        metrics={"acc": 0.9} if success else {},
        error_message=None if success else "boom",
    )


def _make_logger(tmp_path: Path, alert_threshold: float = 0.2) -> RunLogger:
    return RunLogger(
        output_root=tmp_path / "runs",
        run_id="test-run",
        alert_threshold=alert_threshold,
    )


# ---------------------------------------------------------------------------
# log_alert
# ---------------------------------------------------------------------------


def test_log_alert_creates_alerts_file(tmp_path):
    """log_alert writes to alerts.jsonl."""
    rl = _make_logger(tmp_path)
    rl.log_alert(ALERT_WARNING, "Something bad happened", {"detail": 42})

    alerts_file = tmp_path / "runs" / "test-run" / "alerts.jsonl"
    assert alerts_file.exists()


def test_log_alert_content(tmp_path):
    """Alert record contains all required fields."""
    rl = _make_logger(tmp_path)
    rl.log_alert(ALERT_CRITICAL, "Critical failure", {"x": 1})

    alerts_file = tmp_path / "runs" / "test-run" / "alerts.jsonl"
    lines = [ln for ln in alerts_file.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["level"] == ALERT_CRITICAL
    assert record["message"] == "Critical failure"
    assert record["context"]["x"] == 1
    assert "timestamp" in record
    assert record["run_id"] == "test-run"


def test_log_alert_increments_alert_count(tmp_path):
    """alert_count property increments with each log_alert call."""
    rl = _make_logger(tmp_path)
    assert rl.alert_count == 0
    rl.log_alert(ALERT_WARNING, "msg1")
    rl.log_alert(ALERT_WARNING, "msg2")
    assert rl.alert_count == 2


def test_log_alert_multiple_appended(tmp_path):
    """Multiple alerts are appended as separate lines."""
    rl = _make_logger(tmp_path)
    for i in range(3):
        rl.log_alert(ALERT_WARNING, f"alert {i}")

    alerts_file = tmp_path / "runs" / "test-run" / "alerts.jsonl"
    lines = [ln for ln in alerts_file.read_text().splitlines() if ln.strip()]
    assert len(lines) == 3


# ---------------------------------------------------------------------------
# Automatic alert on low success rate
# ---------------------------------------------------------------------------


def test_auto_alert_triggers_below_threshold(tmp_path):
    """log_experiment emits an alert when success rate drops below threshold."""
    rl = _make_logger(tmp_path, alert_threshold=0.5)

    # 0 successes out of 3 → success_rate=0.0 < 0.5
    for _ in range(3):
        rl.log_experiment("h1", _make_result(success=False))

    assert rl.alert_count >= 1
    alerts_file = tmp_path / "runs" / "test-run" / "alerts.jsonl"
    assert alerts_file.exists()
    records = [json.loads(ln) for ln in alerts_file.read_text().splitlines() if ln.strip()]
    assert any("success rate" in r["message"].lower() for r in records)


def test_auto_alert_not_triggered_above_threshold(tmp_path):
    """No automatic alert when success rate stays above threshold."""
    rl = _make_logger(tmp_path, alert_threshold=0.2)

    # 3 successes → success_rate=1.0 > 0.2
    for _ in range(3):
        rl.log_experiment("h1", _make_result(success=True))

    assert rl.alert_count == 0


def test_auto_alert_not_triggered_before_warmup(tmp_path):
    """Automatic alerting waits until at least 3 experiments have been logged."""
    rl = _make_logger(tmp_path, alert_threshold=0.9)

    # Only 2 experiments (below warm-up threshold)
    for _ in range(2):
        rl.log_experiment("h1", _make_result(success=False))

    assert rl.alert_count == 0


def test_auto_alert_disabled_when_threshold_zero(tmp_path):
    """When alert_threshold=0.0 no automatic alerts are emitted."""
    rl = _make_logger(tmp_path, alert_threshold=0.0)
    for _ in range(5):
        rl.log_experiment("h", _make_result(success=False))
    assert rl.alert_count == 0


# ---------------------------------------------------------------------------
# alert_count in run_summary
# ---------------------------------------------------------------------------


def test_run_summary_includes_alert_count(tmp_path):
    """log_run_summary writes alert_count to run_summary.json."""
    rl = _make_logger(tmp_path)
    rl.log_alert(ALERT_WARNING, "test alert")
    rl.log_run_summary(bottleneck="test_bn", metrics={"success_rate": 0.5})

    summary_path = tmp_path / "runs" / "test-run" / "run_summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["alert_count"] == 1


# ---------------------------------------------------------------------------
# generate_report (reporting.py)
# ---------------------------------------------------------------------------


def _populate_runs(data_root: Path, n_runs: int = 2) -> None:
    """Create synthetic run_summary.json files for testing."""
    runs_dir = data_root / "runs"
    for i in range(n_runs):
        run_dir = runs_dir / f"run-{i:03d}"
        run_dir.mkdir(parents=True)
        summary = {
            "timestamp": f"2026-01-0{i+1}T00:00:00+00:00",
            "run_id": f"run-{i:03d}",
            "bottleneck": f"bottleneck_{i}",
            "total_experiments": 10 + i * 5,
            "alert_count": i,
            "metrics": {"success_rate": 0.5 + i * 0.1},
        }
        (run_dir / "run_summary.json").write_text(json.dumps(summary))


def test_generate_report_json_output(tmp_path):
    """generate_report returns expected structure with as_json=True."""
    from reporting import generate_report

    _populate_runs(tmp_path, n_runs=3)
    report = generate_report(tmp_path, as_json=True)

    assert report is not None
    assert "runs" in report
    assert "totals" in report
    assert report["totals"]["run_count"] == 3
    assert len(report["runs"]) == 3


def test_generate_report_empty_dir(tmp_path):
    """generate_report returns empty structure when no runs exist."""
    from reporting import generate_report

    result = generate_report(tmp_path, as_json=True)
    assert result["runs"] == []
    assert result["totals"]["run_count"] == 0


def test_generate_report_totals_correct(tmp_path):
    """Totals correctly aggregate experiment counts and alert counts."""
    from reporting import generate_report

    _populate_runs(tmp_path, n_runs=2)  # runs have 10 and 15 experiments, 0 and 1 alert
    report = generate_report(tmp_path, as_json=True)

    assert report["totals"]["total_experiments"] == 25  # 10 + 15
    assert report["totals"]["total_alerts"] == 1  # 0 + 1


def test_generate_report_runs_sorted_by_timestamp(tmp_path):
    """Runs in JSON report are sorted oldest-first by timestamp."""
    from reporting import generate_report

    _populate_runs(tmp_path, n_runs=3)
    report = generate_report(tmp_path, as_json=True)

    timestamps = [r["timestamp"] for r in report["runs"]]
    assert timestamps == sorted(timestamps)


def test_generate_report_row_fields(tmp_path):
    """Each run row contains the expected keys."""
    from reporting import generate_report

    _populate_runs(tmp_path, n_runs=1)
    report = generate_report(tmp_path, as_json=True)

    row = report["runs"][0]
    for key in ("run_id", "bottleneck", "timestamp", "total_experiments", "success_rate", "alert_count"):
        assert key in row, f"Missing key: {key}"


def test_generate_report_corrupt_summary_skipped(tmp_path):
    """Corrupt summary files are skipped gracefully."""
    from reporting import generate_report

    _populate_runs(tmp_path, n_runs=2)
    # Corrupt one summary
    corrupt_dir = tmp_path / "runs" / "corrupt-run"
    corrupt_dir.mkdir()
    (corrupt_dir / "run_summary.json").write_text("{bad json{{")

    report = generate_report(tmp_path, as_json=True)
    assert report["totals"]["run_count"] == 2  # corrupt one skipped


def test_generate_report_plain_text_no_error(tmp_path, capsys):
    """generate_report prints without error when Rich is absent."""
    import sys
    from unittest.mock import patch

    from reporting import generate_report

    _populate_runs(tmp_path, n_runs=1)
    # Simulate Rich not being installed
    with patch.dict(sys.modules, {"rich": None, "rich.console": None, "rich.table": None}):
        generate_report(tmp_path, as_json=False)

    captured = capsys.readouterr()
    # Should have printed something (the plain-text fallback)
    assert "run-000" in captured.out or len(captured.out) > 0
