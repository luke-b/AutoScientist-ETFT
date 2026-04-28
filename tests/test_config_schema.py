"""
tests for etft/config_schema.py — ETFTConfig Pydantic validation
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Test 1: empty dict produces a fully-defaulted config
# ---------------------------------------------------------------------------


def test_etft_config_empty_dict():
    """ETFTConfig.model_validate({}) succeeds and returns defaults."""
    from etft.config_schema import ETFTConfig

    cfg = ETFTConfig.model_validate({})
    assert cfg.sandbox.backend == "subprocess"
    assert cfg.agent.max_steps == 10
    assert cfg.cluster.adapter == "local"
    assert cfg.synthesis.triage.risk_threshold == 0.7
    assert cfg.training.lora_r == 16


# ---------------------------------------------------------------------------
# Test 2: real config.yaml parses cleanly
# ---------------------------------------------------------------------------


def test_etft_config_from_real_yaml():
    """The repository's config.yaml passes ETFTConfig validation."""
    from etft.config_schema import ETFTConfig

    repo_root = Path(__file__).parent.parent
    raw = yaml.safe_load((repo_root / "config.yaml").read_text())
    cfg = ETFTConfig.model_validate(raw)
    assert cfg.sandbox.backend == "docker"
    assert cfg.agents.literature.max_papers == 10
    assert cfg.analysis.pareto_delta.pareto_threshold == 0.80
    assert cfg.cluster.adapter == "local"


# ---------------------------------------------------------------------------
# Test 3: load_config() validates by default
# ---------------------------------------------------------------------------


def test_load_config_validates():
    """load_config() returns a dict with correct nested values."""
    from etft.config import load_config

    cfg = load_config()
    assert isinstance(cfg, dict)
    assert cfg["sandbox"]["backend"] == "docker"
    assert cfg["cluster"]["adapter"] == "local"


# ---------------------------------------------------------------------------
# Test 4: load_config(validate=False) skips validation and returns raw dict
# ---------------------------------------------------------------------------


def test_load_config_no_validate():
    """load_config(validate=False) skips Pydantic parsing."""
    from etft.config import load_config

    cfg = load_config(validate=False)
    assert isinstance(cfg, dict)
    # Raw dict should still have the sandbox key
    assert "sandbox" in cfg


# ---------------------------------------------------------------------------
# Test 5: wrong type for max_steps raises ValidationError
# ---------------------------------------------------------------------------


def test_etft_config_wrong_type_max_steps():
    """agent.max_steps must be an int; a string raises ValidationError."""
    from etft.config_schema import ETFTConfig

    with pytest.raises(ValidationError) as exc_info:
        ETFTConfig.model_validate({"agent": {"max_steps": "not-an-int"}})
    # The error should mention the offending field
    errors = exc_info.value.errors()
    field_paths = [".".join(str(loc) for loc in e["loc"]) for e in errors]
    assert any("max_steps" in path for path in field_paths)


# ---------------------------------------------------------------------------
# Test 6: unknown extra keys are silently allowed
# ---------------------------------------------------------------------------


def test_etft_config_allows_extra_keys():
    """Unknown keys in config are silently accepted (forward-compat)."""
    from etft.config_schema import ETFTConfig

    cfg = ETFTConfig.model_validate({"sandbox": {"future_key": "value"}})
    assert cfg.sandbox.backend == "subprocess"  # default preserved


# ---------------------------------------------------------------------------
# Test 7: nested models default correctly when section is absent
# ---------------------------------------------------------------------------


def test_etft_config_missing_sections():
    """When a section is absent from the dict, defaults apply."""
    from etft.config_schema import ETFTConfig

    # Supply only training section; everything else should use defaults
    cfg = ETFTConfig.model_validate({"training": {"lora_r": 32}})
    assert cfg.training.lora_r == 32
    assert cfg.analysis.pareto_delta.pareto_threshold == 0.80  # default


# ---------------------------------------------------------------------------
# Test 8: ClusterConfig adapter field accepts valid values
# ---------------------------------------------------------------------------


def test_etft_config_cluster_adapters():
    """Cluster adapter can be 'local', 'slurm', or 'kubernetes'."""
    from etft.config_schema import ETFTConfig

    for adapter in ("local", "slurm", "kubernetes"):
        cfg = ETFTConfig.model_validate({"cluster": {"adapter": adapter}})
        assert cfg.cluster.adapter == adapter


# ---------------------------------------------------------------------------
# Test 9: model_dump() returns plain dict (backward compat)
# ---------------------------------------------------------------------------


def test_etft_config_model_dump():
    """model_dump() returns a plain dict compatible with cfg.get() access."""
    from etft.config_schema import ETFTConfig

    raw = yaml.safe_load(
        (Path(__file__).parent.parent / "config.yaml").read_text()
    )
    cfg_dict = ETFTConfig.model_validate(raw).model_dump()
    # Standard cfg.get() access pattern used across codebase
    assert cfg_dict.get("sandbox", {}).get("backend") == "docker"
    assert cfg_dict.get("cluster", {}).get("adapter") == "local"
