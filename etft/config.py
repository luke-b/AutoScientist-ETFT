"""
etft/config.py — Configuration loader shared across all modules.
"""

from __future__ import annotations

from pathlib import Path

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load config.yaml and return as a dict."""
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    with open(config_path) as f:
        return yaml.safe_load(f)
