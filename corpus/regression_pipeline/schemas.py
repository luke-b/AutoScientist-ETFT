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


class TransitionType(str, Enum):
    INCREMENTAL = "incremental"
    PARADIGM_SHIFT = "paradigm_shift"
    MAJOR_LEAP = "major_leap"


class EvolutionaryRationaleRecord(BaseModel):
    """
    Rich evolutionary rationale for a single aᵢ₋₁ → aᵢ transition.

    Extends RationaleRecord with the historical context, forcing function
    (predecessor bottleneck), conceptual breakthrough, and enabling conditions
    that made the transition inevitable or possible. Used for 𝒟_Rationale
    extended training and the Open-Loop ARL context window.
    """

    trajectory_id: str
    step_before_id: str = Field(..., description="algorithm_id of aᵢ₋₁.")
    step_after_id: str = Field(..., description="algorithm_id of aᵢ.")
    step_index_before: int
    step_index_after: int
    transition_type: TransitionType = Field(
        ..., description="Characterises the magnitude of the evolutionary jump."
    )
    historical_context: str = Field(
        ...,
        description=(
            "The time period, competitive landscape, and external pressures "
            "(hardware availability, benchmark results, paper releases) that "
            "surrounded the transition."
        ),
    )
    predecessor_bottleneck: str = Field(
        ...,
        description=(
            "The concrete limitation in aᵢ₋₁ that acted as the forcing function "
            "for the transition — what problem could not be solved within the "
            "existing paradigm."
        ),
    )
    breakthrough_insight: str = Field(
        ...,
        description=(
            "The key conceptual or mathematical insight introduced in aᵢ that "
            "resolved the predecessor bottleneck."
        ),
    )
    enabling_factors: list[str] = Field(
        default_factory=list,
        description=(
            "Conditions that made the transition feasible: hardware advances, "
            "dataset availability, theoretical results, software tooling, etc."
        ),
    )
    delta_summary: str = Field(
        ..., description="Concise natural-language description of the architectural delta."
    )
    changed_components: list[str] = Field(
        default_factory=list,
        description="Top-level architectural sub-components modified.",
    )
    performance_impact: float = Field(
        ..., description="Estimated fractional fitness gain ∈ [0, 1] from this transition."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_training_example(self) -> dict[str, str]:
        prompt = (
            f"# Trajectory: {self.trajectory_id}\n"
            f"# Transition: {self.step_before_id} (step {self.step_index_before})"
            f" → {self.step_after_id} (step {self.step_index_after})\n"
            f"# Transition type: {self.transition_type.value}\n"
            f"# Task: Explain the evolutionary rationale for this architectural transition.\n"
        )
        completion = (
            f"Historical context: {self.historical_context}\n\n"
            f"Predecessor bottleneck: {self.predecessor_bottleneck}\n\n"
            f"Breakthrough insight: {self.breakthrough_insight}\n\n"
            f"Enabling factors: {'; '.join(self.enabling_factors)}\n\n"
            f"Architectural delta: {self.delta_summary}"
        )
        return {"prompt": prompt, "completion": completion}


class ParetoComponentEntry(BaseModel):
    """A single component in a Pareto transition record."""

    component: str = Field(..., description="Name of the architectural sub-component.")
    fraction_of_code_changed: float = Field(
        ..., description="Fraction of total changed lines attributable to this component ∈ [0, 1]."
    )
    fraction_of_impact: float = Field(
        ..., description="Estimated fraction of fitness delta attributable to this component ∈ [0, 1]."
    )
    description: str = Field(
        ..., description="What specifically changed in this component and why it mattered."
    )


class ParetoTransitionRecord(BaseModel):
    """
    Retrospective 80/20 Pareto analysis for a single aᵢ₋₁ → aᵢ transition.

    Identifies the vital-few components (≈20% of the changed code surface area)
    that are responsible for ≈80% of the fitness improvement, versus the
    useful-many incremental refinements. Used for 𝒟_Pareto and to train the
    Pareto lens projected forward onto aₙ → aₙ₊₁ hypothesis generation.
    """

    trajectory_id: str
    step_before_id: str
    step_after_id: str
    step_index_before: int
    step_index_after: int
    vital_few: list[ParetoComponentEntry] = Field(
        ...,
        description=(
            "The ~20% of changed components responsible for ~80% of the "
            "fitness gain (the Pareto set)."
        ),
    )
    useful_many: list[ParetoComponentEntry] = Field(
        default_factory=list,
        description=(
            "The remaining ~80% of changed components responsible for the "
            "residual ~20% of fitness gain."
        ),
    )
    vital_few_code_fraction: float = Field(
        ...,
        description="Observed fraction of code change surface area in vital_few ∈ [0, 1].",
    )
    vital_few_impact_fraction: float = Field(
        ...,
        description="Observed fraction of fitness delta attributable to vital_few ∈ [0, 1].",
    )
    pareto_narrative: str = Field(
        ...,
        description=(
            "Human-readable summary identifying which 20% of the predecessor's "
            "code evolved into 80% of the successor's innovative impact, and why."
        ),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_training_example(self) -> dict[str, str]:
        vital_few_names = [e.component for e in self.vital_few]
        prompt = (
            f"# Trajectory: {self.trajectory_id}\n"
            f"# Transition: {self.step_before_id} → {self.step_after_id}\n"
            f"# Task: Identify the 20% of architectural changes responsible for 80% of the "
            f"performance improvement in this evolutionary transition.\n"
        )
        completion = (
            f"Vital few components ({self.vital_few_code_fraction:.0%} of code change, "
            f"{self.vital_few_impact_fraction:.0%} of impact):\n"
            + "\n".join(
                f"  - {e.component} (impact {e.fraction_of_impact:.0%}): {e.description}"
                for e in self.vital_few
            )
            + f"\n\nPareto narrative: {self.pareto_narrative}"
        )
        return {"prompt": prompt, "completion": completion}


class MetricDelta(BaseModel):
    """Quantitative before/after measurement for a single performance metric."""

    metric_name: str
    unit: str
    value_before: float
    value_after: float
    delta_absolute: float
    delta_percent: float
    higher_is_better: bool = True
    notes: str = ""


class QuantitativePerformanceRecord(BaseModel):
    """
    Quantitative performance hike between aᵢ₋₁ and aᵢ.

    Captures the empirical numeric evidence for the fitness improvement, covering
    accuracy, efficiency, memory, latency, and any domain-specific metrics.
    Used as ground-truth labels for 𝒟_Perf and for calibrating the Probabilistic
    Heuristic Filter's regression targets.
    """

    trajectory_id: str
    step_before_id: str
    step_after_id: str
    step_index_before: int
    step_index_after: int
    primary_metric: str = Field(
        ..., description="The headline metric that best characterises the fitness gain."
    )
    metrics: list[MetricDelta] = Field(
        ..., description="All measured metric deltas for this transition."
    )
    normalized_fitness_delta: float = Field(
        ...,
        description=(
            "ℱ(aᵢ) − ℱ(aᵢ₋₁) normalised to [0, 1] relative to the total "
            "trajectory fitness range."
        ),
    )
    benchmark_suite: str = Field(
        ..., description="The evaluation benchmark or dataset used for measurements."
    )
    source_references: list[str] = Field(
        default_factory=list,
        description="Paper titles, arXiv IDs, or report names for the numeric claims.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_training_example(self) -> dict[str, str]:
        primary = next((m for m in self.metrics if m.metric_name == self.primary_metric), None)
        headline = (
            f"{self.primary_metric}: {primary.value_before}{primary.unit} → "
            f"{primary.value_after}{primary.unit} "
            f"(Δ {primary.delta_percent:+.1f}%)"
            if primary
            else ""
        )
        prompt = (
            f"# Trajectory: {self.trajectory_id}\n"
            f"# Transition: {self.step_before_id} → {self.step_after_id}\n"
            f"# Task: State the quantitative performance improvement from this "
            f"architectural transition.\n"
        )
        rows = "\n".join(
            f"  {m.metric_name}: {m.value_before}{m.unit} → {m.value_after}{m.unit} "
            f"(Δ {m.delta_absolute:+.3g} / {m.delta_percent:+.1f}%)"
            for m in self.metrics
        )
        completion = (
            f"Headline: {headline}\n\n"
            f"All metrics:\n{rows}\n\n"
            f"Normalised fitness delta: {self.normalized_fitness_delta:.3f}\n"
            f"Benchmark: {self.benchmark_suite}"
        )
        return {"prompt": prompt, "completion": completion}


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
