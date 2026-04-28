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
from functools import partial
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
        self._max_code_chars: int = int(cal_cfg.get("max_code_chars", 8000))
        self._blind_mode: bool = bool(cal_cfg.get("blind_mode", False))
        self._max_step_gap = cal_cfg.get("max_step_gap", None)
        if self._max_step_gap is not None:
            self._max_step_gap = int(self._max_step_gap)

        # C1: configurable similarity weights with validation
        raw_weights = cal_cfg.get("similarity_weights", [0.4, 0.3, 0.3])
        weights = tuple(float(w) for w in raw_weights)
        if len(weights) != 3:
            raise ValueError(
                f"calibration.similarity_weights must have exactly 3 elements; "
                f"got {len(weights)}: {weights}"
            )
        weight_sum = sum(weights)
        if abs(weight_sum - 1.0) > 0.001:
            raise ValueError(
                f"calibration.similarity_weights must sum to 1.0 ± 0.001; "
                f"got {weight_sum:.6f} (weights={weights})"
            )
        self._weights: tuple[float, float, float] = weights  # type: ignore[assignment]

        # Build sub-components
        from calibration.similarity import SIMILARITY_FUNCTIONS, code_similarity

        base_fn = SIMILARITY_FUNCTIONS.get(self._similarity_metric, code_similarity)
        # Inject configured weights into composite metric via partial application
        if self._similarity_metric == "composite":
            self._sim_fn = partial(code_similarity, weights=self._weights)
        else:
            self._sim_fn = base_fn

        # A4: per-step floor and recency weighting
        min_step_sim = float(cal_cfg.get("min_step_similarity", 0.35))
        recency_weighting = bool(cal_cfg.get("recency_weighting", False))
        max_agent_failure_rate = float(cal_cfg.get("max_agent_failure_rate", 0.5))

        from calibration.stage_gate import StageGate
        self._gate = StageGate(
            threshold=self._threshold,
            min_steps=self._min_steps,
            min_step_similarity=min_step_sim,
            recency_weighting=recency_weighting,
            max_agent_failure_rate=max_agent_failure_rate,
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
            "(%d steps, threshold=%.2f, metric='%s', blind=%s) …",
            trajectory_id, len(steps), self._threshold,
            self._similarity_metric, self._blind_mode,
        )

        from calibration.replay import ReplaySession

        session = ReplaySession(
            agent=self._agent,
            similarity_fn=self._sim_fn,
            max_code_chars=self._max_code_chars,
            blind_mode=self._blind_mode,
            max_step_gap=self._max_step_gap,
        )
        reconstruction_results = session.run(steps)

        confidence = session.confidence_level
        gate_result = self._gate.evaluate(
            confidence_level=confidence,
            n_steps=len(reconstruction_results),
            reconstruction_results=reconstruction_results,
            n_steps_attempted=session.n_steps_attempted,
            n_agent_failures=session.n_agent_failures,
        )

        # Determine gate_status
        if session.n_agent_failures >= max(1, session.n_steps_attempted) * self._gate.max_agent_failure_rate:
            gate_status = "agent_failure"
        else:
            gate_status = "evaluated"

        record = CalibrationRecord(
            trajectory_id=trajectory_id,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            n_steps_replayed=len(reconstruction_results),
            n_steps_attempted=session.n_steps_attempted,
            n_agent_failures=session.n_agent_failures,
            confidence_level=confidence,
            threshold=self._threshold,
            gate_passed=gate_result.ready,
            gate_status=gate_status,
            gate_diagnostic=gate_result.diagnostic,
            similarity_metric=self._similarity_metric,
            reconstruction_results=reconstruction_results,
            metadata={
                "similarity_weights": list(self._weights),
                "blind_mode": self._blind_mode,
                "max_code_chars": self._max_code_chars,
            },
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
