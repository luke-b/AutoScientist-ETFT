"""
calibration/engine.py — CalibrationEngine: Evolutionary Replay orchestrator.

The CalibrationEngine is the central coordinator of the Recursive Stage-Gate
Calibration protocol (§2, Benda 2026).  It:

  1. Runs an ``ReplaySession`` over the supplied trajectory, asking the LLM
     to reconstruct each successor algorithm from its predecessor.
  2. Computes the Confidence Level C = (1/n) Σ Sim(a_pred_i, a_true_i).
  3. Evaluates the ``StageGate``: only when C ≥ threshold is the model
     declared "Ready for Action" for SOTA+x synthesis.
  4. Persists a ``CalibrationRecord`` so runs can be audited and replayed.

The engine is invoked by ``pipeline.py`` as the ``calibrate`` stage, which
must complete successfully (gate open) before the ``generate`` stage runs.

Usage
-----
    from calibration.engine import CalibrationEngine
    from corpus.regression_pipeline.schemas import TrajectoryStep

    engine = CalibrationEngine(cfg)
    record = engine.run(trajectory_id="t1", steps=trajectory_steps)
    if not record.gate_passed:
        raise RuntimeError(record.gate_diagnostic)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from corpus.regression_pipeline.schemas import CalibrationRecord, TrajectoryStep
from etft.agent import AgentClient

logger = logging.getLogger(__name__)


class CalibrationEngine:
    """
    Orchestrates the Evolutionary Replay calibration protocol.

    Parameters
    ----------
    cfg:
        Runtime configuration dict from ``config.yaml``.
    """

    def __init__(self, cfg: dict | None = None) -> None:
        self._cfg = cfg or {}
        cal_cfg = self._cfg.get("calibration", {})

        self._threshold: float = float(cal_cfg.get("confidence_threshold", 0.65))
        self._min_steps: int = int(cal_cfg.get("min_replay_steps", 2))
        self._similarity_metric: str = cal_cfg.get("similarity_metric", "composite")

        # Build sub-components
        from calibration.similarity import SIMILARITY_FUNCTIONS, code_similarity
        self._sim_fn = SIMILARITY_FUNCTIONS.get(self._similarity_metric, code_similarity)

        from calibration.stage_gate import StageGate
        self._gate = StageGate(
            threshold=self._threshold,
            min_steps=self._min_steps,
        )

        self._agent = AgentClient(cfg)

    # ------------------------------------------------------------------
    def run(
        self,
        trajectory_id: str,
        steps: list[TrajectoryStep],
        output_dir: Path | None = None,
    ) -> CalibrationRecord:
        """
        Execute the full Evolutionary Replay protocol for *trajectory_id*.

        Parameters
        ----------
        trajectory_id:
            Identifier of the trajectory being calibrated.
        steps:
            Ordered list of ``TrajectoryStep`` objects (will be sorted by
            step_index internally).
        output_dir:
            When provided, persists the ``CalibrationRecord`` as JSON to
            ``output_dir/calibration_<trajectory_id>.json``.

        Returns
        -------
        CalibrationRecord
            Contains per-step reconstruction results, C, and gate outcome.
        """
        logger.info(
            "CalibrationEngine starting Evolutionary Replay for trajectory '%s' "
            "(%d steps, threshold=%.2f, metric='%s') …",
            trajectory_id, len(steps), self._threshold, self._similarity_metric,
        )

        from calibration.replay import ReplaySession

        session = ReplaySession(agent=self._agent, similarity_fn=self._sim_fn)
        reconstruction_results = session.run(steps)

        confidence = session.confidence_level
        gate_result = self._gate.evaluate(
            confidence_level=confidence,
            n_steps=len(reconstruction_results),
        )

        record = CalibrationRecord(
            trajectory_id=trajectory_id,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            n_steps_replayed=len(reconstruction_results),
            confidence_level=confidence,
            threshold=self._threshold,
            gate_passed=gate_result.ready,
            gate_diagnostic=gate_result.diagnostic,
            similarity_metric=self._similarity_metric,
            reconstruction_results=reconstruction_results,
        )

        logger.info(
            "CalibrationEngine done — C=%.3f, gate=%s",
            confidence,
            "OPEN" if gate_result.ready else "CLOSED",
        )

        if output_dir is not None:
            self._persist(record, output_dir)

        return record

    # ------------------------------------------------------------------
    @staticmethod
    def _persist(record: CalibrationRecord, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"calibration_{record.trajectory_id}.json"
        with open(out_path, "w") as fh:
            json.dump(record.model_dump(), fh, indent=2)
        logger.info("CalibrationRecord persisted to %s", out_path)
