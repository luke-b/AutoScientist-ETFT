"""
calibration/ — Evolutionary Replay Calibration Engine.

Implements the Recursive Stage-Gate Calibration protocol described in
"Evolutionary Trajectory Fine-Tuning: Reverse-Engineering Algorithmic
Progression and the Recursive Stage-Gate Calibration for SOTA+x Discovery"
(Benda, April 2026).

The Calibration Engine enforces a hard algorithmic stage-gate: before the
synthesiser is permitted to propose a speculative SOTA+1 hypothesis, the
model must demonstrate it has internalised the historical trajectory by
accurately reconstructing each step from its predecessor.  Only when the
mean reconstruction similarity C exceeds the configured threshold is the
gate opened for SOTA+x synthesis.

Public API
----------
    from calibration.engine import CalibrationEngine
    from calibration.stage_gate import StageGate
    from calibration.similarity import code_similarity
    from calibration.replay import ReplaySession
"""

from calibration.engine import CalibrationEngine
from calibration.similarity import code_similarity
from calibration.stage_gate import StageGate

__all__ = ["CalibrationEngine", "StageGate", "code_similarity"]
