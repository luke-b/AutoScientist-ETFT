"""
etft/run_logger.py — Structured JSON-lines run logger for ARL campaigns.

Writes two artefact types for every ARL run:
  runs/<run_id>/experiments.jsonl   — one record per micro-experiment
  runs/<run_id>/run_summary.json    — summary written at the end of the run

Each record contains a UTC ISO-8601 timestamp so results can be correlated
across runs and tool calls without an external time-series database.

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
    log.log_run_summary(bottleneck="batch_norm", metrics=collector.to_dict())
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corpus.regression_pipeline.schemas import ExperimentResult

logger = logging.getLogger(__name__)


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
    """

    def __init__(self, output_root: Path, run_id: str | None = None) -> None:
        self.run_id: str = run_id or str(uuid.uuid4())[:12]
        self._run_dir = output_root / self.run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._experiments_path = self._run_dir / "experiments.jsonl"
        self._experiment_count = 0

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
            "metrics": metrics,
        }
        out_path = self._run_dir / "run_summary.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Run summary written to %s", out_path)

        self._mlflow_log_summary(summary)
        self._mlflow_end()

    # ------------------------------------------------------------------
    # Optional MLflow helpers (all guarded by try/except ImportError)
    # ------------------------------------------------------------------

    def _mlflow_start(self) -> None:
        try:
            import mlflow  # noqa: PLC0415

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

    def _mlflow_log_summary(self, summary: dict) -> None:
        try:
            import mlflow  # noqa: PLC0415

            metrics = summary.get("metrics") or {}
            mlflow.log_metric("run/success_rate", metrics.get("success_rate", 0.0))
            mlflow.log_metric("run/total_experiments", summary.get("total_experiments", 0))
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
