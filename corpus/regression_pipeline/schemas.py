"""
schemas.py — Pydantic models and dataclasses for the ETFT corpus.

These types are shared across the regression pipeline, performance estimator,
agents, and synthesis modules.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ValidationStatus(str, Enum):
    PASS = "pass"
    FAIL_SYNTAX = "fail_syntax"
    FAIL_OOM = "fail_oom"
    FAIL_RUNTIME = "fail_runtime"
    FAIL_TIMEOUT = "fail_timeout"


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


# ---------------------------------------------------------------------------
# Trajectory & Regression Pipeline
# ---------------------------------------------------------------------------


class TrajectoryStep(BaseModel):
    """A single algorithm in an evolutionary trajectory."""

    step_index: int = Field(..., description="Position in the trajectory (0 = simplest).")
    algorithm_id: str = Field(..., description="Unique identifier, e.g. 'resnet_v1'.")
    algorithm_family: str = Field(..., description="e.g. 'image_classification_cnn'.")
    code: str = Field(..., description="Full source code of the algorithm.")
    fitness_score: float = Field(
        ..., description="Objective fitness ℱ(a); higher is better."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrajectoryPair(BaseModel):
    """Adjacent pair (aᵢ₋₁, aᵢ) extracted from a trajectory — core of 𝒟_Gen."""

    trajectory_id: str
    step_before: TrajectoryStep
    step_after: TrajectoryStep

    @property
    def fitness_delta(self) -> float:
        return self.step_after.fitness_score - self.step_before.fitness_score

    def to_training_example(self) -> dict[str, str]:
        """Convert to a prompt/completion dict for LLM fine-tuning."""
        prompt = (
            f"# Algorithm family: {self.step_before.algorithm_family}\n"
            f"# Task: Improve the following algorithm.\n\n"
            f"{self.step_before.code}"
        )
        return {"prompt": prompt, "completion": self.step_after.code}


class RationaleRecord(BaseModel):
    """Structural-rationale metadata for a single aᵢ₋₁ → aᵢ transition — core of 𝒟_Rationale."""

    trajectory_id: str
    step_index_before: int
    step_index_after: int
    delta_summary: str = Field(..., description="Natural-language description of what changed.")
    changed_components: list[str] = Field(
        default_factory=list,
        description="List of architectural sub-components modified.",
    )
    performance_impact: float = Field(
        ..., description="Estimated fractional performance gain from this transition."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_training_example(self) -> dict[str, str]:
        prompt = (
            f"# Trajectory: {self.trajectory_id}\n"
            f"# Transition: step {self.step_index_before} → {self.step_index_after}\n"
            f"# Task: Explain the architectural changes and their impact.\n"
        )
        return {"prompt": prompt, "completion": self.delta_summary}


# ---------------------------------------------------------------------------
# Validation / CI-CD
# ---------------------------------------------------------------------------


class ValidationResult(BaseModel):
    """Outcome of the CI/CD validator for a generated code artifact."""

    status: ValidationStatus
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == ValidationStatus.PASS


# ---------------------------------------------------------------------------
# Performance Estimator (𝒟_Perf)
# ---------------------------------------------------------------------------


class PerfSample(BaseModel):
    """One labelled example for the Probabilistic Heuristic Filter training."""

    algorithm_id: str
    features: dict[str, float] = Field(
        ..., description="Extracted structural features (see feature_extractor.py)."
    )
    label: int = Field(..., description="1 = failed (OOM/diverge), 0 = passed.")
    failure_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


class ResearchBrief(BaseModel):
    """Output of the literature synthesis agent."""

    bottleneck: str
    query: str
    papers: list[dict[str, Any]] = Field(default_factory=list)
    synthesis: str = Field(..., description="LLM-generated summary of retrieved literature.")
    hypotheses: list[str] = Field(default_factory=list)


class ExperimentResult(BaseModel):
    """Outcome of a single micro-experiment."""

    experiment_id: str
    script: str
    stdout: str = ""
    stderr: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)
    success: bool = True
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


class SOTAPlusOneCandidate(BaseModel):
    """A proposed next-generation algorithm candidate."""

    candidate_id: str
    trajectory_id: str
    code: str
    rationale: str
    risk_score: float = Field(
        0.0,
        description="P(failure) from the triage filter; lower is safer.",
    )
    triage_passed: bool = True
    provenance: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Calibration Engine (Recursive Stage-Gate)
# ---------------------------------------------------------------------------


class ReconstructionResult(BaseModel):
    """
    Outcome of a single step in the Evolutionary Replay protocol.

    The Calibration Engine asks the model to reconstruct algorithm a_i from
    a_{i-1} and the causal meta-data M_i.  This record captures the predicted
    code and its similarity to the ground-truth successor so the Confidence
    Level C can be computed.  (§2, Benda 2026)
    """

    trajectory_id: str
    step_index_before: int
    step_index_after: int
    predicted_code: str = Field(..., description="Code reconstructed by the model.")
    true_code: str = Field(..., description="Ground-truth successor code from the trajectory.")
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Sim(a_pred_i, a_true_i) ∈ [0, 1]; higher is better.",
    )
    agent_success: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class CalibrationRecord(BaseModel):
    """
    Full record of one Evolutionary Replay calibration run for a trajectory.

    Persisted by ``CalibrationEngine`` to ``data/calibration/`` so the
    history of calibration attempts is auditable.

    The ``gate_passed`` flag indicates whether the Confidence Level C met
    the configured threshold — this is the hard stage-gate that must be
    satisfied before SOTA+x synthesis is permitted.  (§2, Benda 2026)
    """

    trajectory_id: str
    timestamp: str = Field(..., description="UTC ISO-8601 timestamp of the calibration run.")
    n_steps_replayed: int
    confidence_level: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="C = (1/n) Σ Sim(a_pred_i, a_true_i) [Eq. 1, Benda 2026].",
    )
    threshold: float = Field(..., description="Minimum C required for gate_passed=True.")
    gate_passed: bool = Field(
        ..., description="True when C ≥ threshold — SOTA+x synthesis is unlocked."
    )
    gate_diagnostic: str = Field(
        "", description="Human-readable explanation of the gate decision."
    )
    similarity_metric: str = Field("composite", description="Similarity function used.")
    reconstruction_results: list[ReconstructionResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Feedback / RL
# ---------------------------------------------------------------------------


class FailureRecord(BaseModel):
    """A high-value failure event recycled as negative feedback."""

    source: str = Field(..., description="'triage' | 'physical_eval' | 'micro_experiment'")
    candidate_id: str | None = None
    experiment_id: str | None = None
    failure_reason: str
    features: dict[str, float] = Field(default_factory=dict)
    reward_signal: float = Field(
        -1.0,
        description="Negative reward injected into the in-context RL loop.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
