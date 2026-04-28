"""
etft/run_logger.py — Structured JSON-lines run logger for ARL campaigns.

Writes three artefact types for every ARL run:
  runs/<run_id>/experiments.jsonl   — one record per micro-experiment
  runs/<run_id>/run_summary.json    — summary written at the end of the run
  runs/<run_id>/alerts.jsonl        — timestamped alert events (see log_alert)

Each record contains a UTC ISO-8601 timestamp so results can be correlated
across runs and tool calls without an external time-series database.

Alerting
--------
``log_alert(level, message, context)`` writes a structured alert record to
``alerts.jsonl``.  The ARL runner calls this automatically when the rolling
experiment success rate drops below the configured ``alert_threshold``.

Optional MLflow integration
---------------------------
If ``mlflow`` is importable the logger transparently logs metrics and params
to the active MLflow run (or creates a new one named ``arl/<run_id>``).
Install MLflow with: ``pip install mlflow``

Usage
-----
    from etft.run_logger import RunLogger

    log = RunLogger(output_root=Path("./data/runs"))
    log.log_experiment(hypothesis="...", result=result, rl_prefix_length=42)
    log.log_alert("WARNING", "Low success rate", {"success_rate": 0.1})
    log.log_run_summary(bottleneck="batch_norm", metrics=collector.to_dict())
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from corpus.regression_pipeline.schemas import ExperimentResult

logger = logging.getLogger(__name__)

# Alert levels (mirrors standard log levels as strings)
ALERT_INFO = "INFO"
ALERT_WARNING = "WARNING"
ALERT_CRITICAL = "CRITICAL"


class RunLogger:
    """
    Writes structured experiment logs for a single ARL run.

    Parameters
    ----------
    output_root:
        Root directory under which ``runs/<run_id>/`` will be created.
    run_id:
        Optional run identifier.  A UUID4 is generated automatically when
        omitted.
    alert_threshold:
        Rolling success-rate threshold below which :meth:`log_experiment` will
        automatically emit a ``WARNING`` alert to ``alerts.jsonl``.  Set to
        ``0.0`` to disable automatic alerting.
    """

    def __init__(
        self,
        output_root: Path,
        run_id: str | None = None,
        alert_threshold: float = 0.2,
    ) -> None:
        self.run_id: str = run_id or str(uuid.uuid4())[:12]
        self._run_dir = output_root / self.run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._experiments_path = self._run_dir / "experiments.jsonl"
        self._alerts_path = self._run_dir / "alerts.jsonl"
        self._experiment_count = 0
        self._success_count = 0
        self._alert_count = 0
        self._alert_threshold: float = alert_threshold
        self._last_alert_rate: float | None = None  # track last alerted rate to avoid spam

        logger.info("RunLogger initialised (run_id=%s, dir=%s)", self.run_id, self._run_dir)
        self._mlflow_start()

    # ------------------------------------------------------------------
    def log_experiment(
        self,
        hypothesis: str,
        result: ExperimentResult,
        rl_prefix_length: int = 0,
    ) -> None:
        """
        Append one experiment record to ``experiments.jsonl``.

        Also checks the rolling success rate and emits a ``WARNING`` alert
        when it drops below ``alert_threshold`` (after at least 3 experiments).

        Parameters
        ----------
        hypothesis:
            The natural-language hypothesis that was tested.
        result:
            ExperimentResult returned by ExperimentRunner.run().
        rl_prefix_length:
            Character length of the RL context prefix that was prepended to
            the design prompt for this experiment (0 = no prior failures).
        """
        self._experiment_count += 1
        if result.success:
            self._success_count += 1

        record = {
            "timestamp": _utcnow(),
            "run_id": self.run_id,
            "experiment_id": result.experiment_id,
            "hypothesis": hypothesis,
            "success": result.success,
            "metrics": result.metrics,
            "error": result.error_message,
            "rl_prefix_length": rl_prefix_length,
        }
        with open(self._experiments_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        self._mlflow_log_experiment(record)

        # Auto-alert on low success rate (after initial warm-up of 3 experiments).
        # Only emit one alert per 10-experiment window to avoid spam.
        if (
            self._alert_threshold > 0.0
            and self._experiment_count >= 3
        ):
            rolling_rate = self._success_count / self._experiment_count
            if (
                rolling_rate < self._alert_threshold
                and (self._last_alert_rate is None or self._experiment_count % 10 == 0)
            ):
                self._last_alert_rate = rolling_rate
                self.log_alert(
                    ALERT_WARNING,
                    f"Low experiment success rate: {rolling_rate:.0%}",
                    {
                        "success_rate": rolling_rate,
                        "total_experiments": self._experiment_count,
                        "alert_threshold": self._alert_threshold,
                    },
                )

    # ------------------------------------------------------------------
    def log_alert(
        self,
        level: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """
        Write a timestamped alert record to ``runs/<run_id>/alerts.jsonl``.

        Parameters
        ----------
        level:
            Severity level string, e.g. ``"WARNING"``, ``"CRITICAL"``.
        message:
            Human-readable alert message.
        context:
            Optional dict of additional context values to include in the
            alert record (e.g. ``{"success_rate": 0.1}``).
        """
        self._alert_count += 1
        alert = {
            "timestamp": _utcnow(),
            "run_id": self.run_id,
            "level": level,
            "message": message,
            "context": context or {},
        }
        with open(self._alerts_path, "a") as fh:
            fh.write(json.dumps(alert) + "\n")

        log_fn = logger.critical if level == ALERT_CRITICAL else logger.warning
        log_fn("[ALERT %s] %s | context=%s", level, message, context)

        self._mlflow_log_alert(alert)

    # ------------------------------------------------------------------
    def log_run_summary(self, bottleneck: str, metrics: dict) -> None:
        """
        Write the final run summary to ``run_summary.json``.

        Parameters
        ----------
        bottleneck:
            The bottleneck component targeted in this ARL campaign.
        metrics:
            MetricsCollector.to_dict() output.
        """
        summary = {
            "timestamp": _utcnow(),
            "run_id": self.run_id,
            "bottleneck": bottleneck,
            "total_experiments": self._experiment_count,
            "alert_count": self._alert_count,
            "metrics": metrics,
        }
        out_path = self._run_dir / "run_summary.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Run summary written to %s", out_path)

        self._mlflow_log_summary(summary)
        self._mlflow_end()

    # ------------------------------------------------------------------
    @property
    def alert_count(self) -> int:
        """Number of alerts emitted during this run."""
        return self._alert_count

    # ------------------------------------------------------------------
    # Optional MLflow helpers (all guarded by try/except ImportError)
    # ------------------------------------------------------------------

    def _mlflow_start(self) -> None:
        try:
            import mlflow  # noqa: PLC0415

            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)

            if mlflow.active_run() is None:
                mlflow.start_run(run_name=f"arl/{self.run_id}")
                self._mlflow_owned_run = True
            else:
                self._mlflow_owned_run = False
            mlflow.set_tag("etft.run_id", self.run_id)
        except ImportError:
            self._mlflow_owned_run = False

    def _mlflow_log_experiment(self, record: dict) -> None:
        try:
            import mlflow  # noqa: PLC0415

            step = self._experiment_count
            if record.get("success"):
                for k, v in (record.get("metrics") or {}).items():
                    mlflow.log_metric(f"exp/{k}", v, step=step)
            mlflow.log_metric("exp/rl_prefix_length", record.get("rl_prefix_length", 0), step=step)
        except ImportError:
            pass

    def _mlflow_log_alert(self, alert: dict) -> None:
        try:
            import mlflow  # noqa: PLC0415

            mlflow.log_metric("run/alert_count", self._alert_count)
            mlflow.set_tag(f"etft.alert.{self._alert_count}", alert.get("message", "")[:250])
        except ImportError:
            pass

    def _mlflow_log_summary(self, summary: dict) -> None:
        try:
            import mlflow  # noqa: PLC0415

            metrics = summary.get("metrics") or {}
            mlflow.log_metric("run/success_rate", metrics.get("success_rate", 0.0))
            mlflow.log_metric("run/total_experiments", summary.get("total_experiments", 0))
            mlflow.log_metric("run/alert_count", summary.get("alert_count", 0))
            mlflow.set_tag("etft.bottleneck", summary.get("bottleneck", ""))
        except ImportError:
            pass

    def _mlflow_end(self) -> None:
        try:
            import mlflow  # noqa: PLC0415

            if self._mlflow_owned_run and mlflow.active_run() is not None:
                mlflow.end_run()
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(tz=timezone.utc).isoformat()

