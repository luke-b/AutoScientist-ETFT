"""
etft/config_schema.py — Pydantic v2 models for the full config.yaml structure.

These models provide:
  - Type-safe access to all configuration values
  - Clear validation errors pointing to the offending key and expected type
  - Default values matching config.yaml defaults

All models use ``model_config = ConfigDict(extra="allow")`` so unknown keys
(from future config additions) are silently accepted rather than rejected.
This keeps backward compatibility as config.yaml grows.

Usage
-----
    from etft.config_schema import ETFTConfig
    from pydantic import ValidationError

    try:
        cfg = ETFTConfig.model_validate(raw_dict)
    except ValidationError as exc:
        print(exc)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _AllowExtra(BaseModel):
    """Base model that allows unknown keys (forward-compat)."""

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------


class SandboxConfig(_AllowExtra):
    backend: str = Field("subprocess", description="'docker' | 'subprocess'")
    image: str = "python:3.11-slim"
    mem_limit: str = "512m"
    cpu_count: int = 1
    network_disabled: bool = True
    timeout_seconds: int = 60


# ---------------------------------------------------------------------------
# Agent (ReAct loop)
# ---------------------------------------------------------------------------


class AgentConfig(_AllowExtra):
    max_steps: int = 10
    system_prompt: str = (
        "You are an expert ML researcher and software engineer with access to ETFT tools."
    )


# ---------------------------------------------------------------------------
# Agent proxy container
# ---------------------------------------------------------------------------


class ContainerConfig(_AllowExtra):
    mode: str = "local"
    image: str = "autoscientist-etft-agent:latest"
    name: str = "etft-agent-proxy"
    port: int = 8080
    mem_limit: str = "2g"
    cpu_count: int = 2
    proxy_llm_backend: str = "openai"
    proxy_llm_model: str = "gpt-4o"


class AgentProxyConfig(_AllowExtra):
    url: str = "http://localhost:8080"
    token: str = ""
    timeout_seconds: int = 120
    max_retries: int = 3
    retry_wait_seconds: int = 5
    container: ContainerConfig = Field(default_factory=ContainerConfig)


# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------


class DataConfig(_AllowExtra):
    root: str = "./data"
    d_gen: str = "./data/d_gen"
    d_rationale: str = "./data/d_rationale"
    d_perf: str = "./data/d_perf"
    trajectories: str = "./data/trajectories"


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


class RegressionPipelineConfig(_AllowExtra):
    max_regression_depth: int = 10
    validation_timeout_seconds: int = 60
    max_code_size_bytes: int = 524288


class PerformanceEstimatorConfig(_AllowExtra):
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 200


class CorpusConfig(_AllowExtra):
    regression_pipeline: RegressionPipelineConfig = Field(
        default_factory=RegressionPipelineConfig
    )
    performance_estimator: PerformanceEstimatorConfig = Field(
        default_factory=PerformanceEstimatorConfig
    )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


class ParetoDeltaConfig(_AllowExtra):
    pareto_threshold: float = 0.80
    min_delta_lines: int = 5


class AnalysisConfig(_AllowExtra):
    pareto_delta: ParetoDeltaConfig = Field(default_factory=ParetoDeltaConfig)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


class LiteratureAgentConfig(_AllowExtra):
    max_papers: int = 10
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    arxiv_max_results: int = 20
    request_delay_seconds: float = 0.4


class EmpiricalAgentConfig(_AllowExtra):
    experiment_timeout_seconds: int = 120
    max_script_size_bytes: int = 65536
    allowed_packages: list[str] = Field(
        default_factory=lambda: ["numpy", "scipy", "torch", "sklearn"]
    )
    blocked_builtins: list[str] = Field(
        default_factory=lambda: ["exec", "eval", "__import__", "compile"]
    )


class AgentsConfig(_AllowExtra):
    literature: LiteratureAgentConfig = Field(default_factory=LiteratureAgentConfig)
    empirical: EmpiricalAgentConfig = Field(default_factory=EmpiricalAgentConfig)


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------


class VectorStoreConfig(_AllowExtra):
    backend: str = "memory"
    persist_dir: str = "./data/vector_store"
    collection: str = "literature"
    n_results: int = 20


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


class TriageConfig(_AllowExtra):
    risk_threshold: float = 0.7


class SOTAPlusOneConfig(_AllowExtra):
    max_candidates: int = 3


class SynthesisConfig(_AllowExtra):
    triage: TriageConfig = Field(default_factory=TriageConfig)
    sota_plus_one: SOTAPlusOneConfig = Field(default_factory=SOTAPlusOneConfig)


# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------


class SlurmConfig(_AllowExtra):
    partition: str = "gpu"
    n_gpus: int = 1
    mem_gb: int = 32
    time_limit: str = "01:00:00"
    python_path: str = "python3"
    poll_interval_seconds: int = 30
    timeout_seconds: int = 3600


class K8sConfig(_AllowExtra):
    namespace: str = "default"
    image: str = "python:3.11-slim"
    gpu_count: int = 1
    memory_limit: str = "8Gi"
    cpu_limit: str = "4"
    poll_interval_seconds: int = 10
    timeout_seconds: int = 1800
    service_account: str = ""


class ClusterConfig(_AllowExtra):
    adapter: str = Field("local", description="'local' | 'slurm' | 'kubernetes'")
    slurm: SlurmConfig = Field(default_factory=SlurmConfig)
    k8s: K8sConfig = Field(default_factory=K8sConfig)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TrainingConfig(_AllowExtra):
    base_model: str = "meta-llama/Meta-Llama-3-8B"
    output_dir: str = "./checkpoints"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2.0e-4
    fp16: bool = True
    val_split: float = 0.1
    dataset_path: str = "./data"
    openai_finetune_model: str = "gpt-3.5-turbo"


# ---------------------------------------------------------------------------
# Calibration (Recursive Stage-Gate)
# ---------------------------------------------------------------------------


class CalibrationConfig(_AllowExtra):
    """
    Configuration for the Evolutionary Replay Calibration Engine.

    The calibration stage runs before SOTA+x synthesis.  The model must
    reconstruct each step of the trajectory from its predecessor; only when
    the mean reconstruction similarity C meets ``confidence_threshold`` is
    the stage-gate opened and synthesis permitted.
    """

    confidence_threshold: float = Field(
        0.65,
        ge=0.0,
        le=1.0,
        description="Minimum C = (1/n) Σ Sim(pred, true) to open the stage-gate.",
    )
    min_replay_steps: int = Field(
        2,
        ge=1,
        description="Minimum number of trajectory pairs that must be replayed.",
    )
    similarity_metric: str = Field(
        "composite",
        description="'composite' | 'token_jaccard' | 'line_lcs' | 'ast_edit'",
    )
    skip_on_short_trajectory: bool = Field(
        True,
        description=(
            "When True, skip calibration (and open the gate automatically) for "
            "trajectories with fewer than min_replay_steps + 1 steps.  "
            "Set to False to enforce strict calibration regardless of trajectory length."
        ),
    )
    output_dir: str = Field(
        "./data/calibration",
        description="Directory where CalibrationRecord JSON files are persisted.",
    )


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class OrthogonalWidthConfig(_AllowExtra):
    lateral_variants_per_jump: int = 4
    performance_tolerance: float = 0.05
    adapter_output_dir: str = "./checkpoints/width_adapters"


class OrthogonalDepthConfig(_AllowExtra):
    adapter_output_dir: str = "./checkpoints/depth_adapter"


class OrthogonalRecursiveConfig(_AllowExtra):
    enabled: bool = True
    max_sota_plus_x: int = 2


class ObjectiveCalibrationConfig(_AllowExtra):
    diversity_threshold: float = Field(0.3, ge=0.0, le=1.0)
    performance_tolerance: float = Field(0.05, ge=0.0)
    similarity_metric: str = "composite"
    similarity_weights: list[float] = Field(default_factory=lambda: [0.4, 0.3, 0.3])


class OrthogonalCalibrationConfig(_AllowExtra):
    width: OrthogonalWidthConfig = Field(default_factory=OrthogonalWidthConfig)
    depth: OrthogonalDepthConfig = Field(default_factory=OrthogonalDepthConfig)
    recursive: OrthogonalRecursiveConfig = Field(default_factory=OrthogonalRecursiveConfig)
    objective_calibration: ObjectiveCalibrationConfig = Field(
        default_factory=ObjectiveCalibrationConfig
    )


class LoRARoutingConfig(_AllowExtra):
    enabled: bool = False
    foundation_model: str | None = None
    device: str = "auto"
    max_new_tokens: int = 2048
    adapter_index_path: str = "./checkpoints/adapter_index.json"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class ETFTConfig(_AllowExtra):
    """Root configuration model for the full config.yaml."""

    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    agent_proxy: AgentProxyConfig = Field(default_factory=AgentProxyConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    orthogonal_calibration: OrthogonalCalibrationConfig = Field(
        default_factory=OrthogonalCalibrationConfig
    )
    lora_routing: LoRARoutingConfig = Field(default_factory=LoRARoutingConfig)
