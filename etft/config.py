"""
etft/config.py — Configuration loader shared across all modules.

``load_config`` loads a YAML file and validates it against
:class:`~etft.config_schema.ETFTConfig`.  It returns a plain ``dict``
(via ``model.model_dump()``) so all existing callers that use ``cfg.get(...)``
continue to work unchanged.  A ``pydantic.ValidationError`` is raised when
the file contains structurally invalid values.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(path: str | Path | None = None, validate: bool = True) -> dict:
    """
    Load *path* (default: config.yaml) and return a validated config dict.

    Parameters
    ----------
    path:
        Path to the YAML config file.  Defaults to the ``config.yaml`` in
        the repository root.
    validate:
        When *True* (default) the raw YAML is validated against
        :class:`~etft.config_schema.ETFTConfig` and a
        ``pydantic.ValidationError`` is raised on schema violations.
        Pass *False* to skip validation and return the raw dict.

    Returns
    -------
    dict
        Validated configuration dict.  Nested values are plain Python dicts
        and lists, fully compatible with the ``cfg.get(...)`` access pattern
        used throughout the codebase.

    Raises
    ------
    pydantic.ValidationError
        When *validate* is True and the config file contains structurally
        invalid values (wrong type, out-of-range numbers, etc.).
    FileNotFoundError
        When the config file does not exist.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    with open(config_path) as f:
        raw: dict = yaml.safe_load(f) or {}

    if not validate:
        return raw

    from etft.config_schema import ETFTConfig  # local import to keep startup fast

    model = ETFTConfig.model_validate(raw)
    validated = model.model_dump()
    logger.debug("Config loaded and validated from %s", config_path)
    return validated
