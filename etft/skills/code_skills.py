"""
etft/skills/code_skills.py — Skills that wrap code validation, feature
extraction, script execution, and syntax checking.

All skills are pure Python — no LLM calls.
"""

from __future__ import annotations

import ast
from typing import Any

from etft.skills.base import Skill


class ValidateCodeSkill(Skill):
    """Validate a Python code string using CICDValidator."""

    name = "validate_code"
    description = (
        "Validate a Python code snippet. Checks syntax, size limits, and "
        "runs the code in a sandboxed subprocess. Returns passed=true/false "
        "and any error details."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python source code to validate."},
        },
        "required": ["code"],
    }

    def __init__(self, cfg: dict | None = None) -> None:
        self._cfg = cfg

    def execute(self, **kwargs: Any) -> dict:
        from corpus.regression_pipeline.cicd_validator import CICDValidator

        code: str = kwargs["code"]
        validator = CICDValidator(self._cfg)
        result = validator.validate(code)
        return {
            "passed": result.passed,
            "status": result.status.value,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
            "duration_seconds": result.duration_seconds,
        }


class ExtractCodeFeaturesSkill(Skill):
    """Extract structural features from Python code using AST analysis."""

    name = "extract_code_features"
    description = (
        "Extract numeric structural features from a Python code snippet "
        "(function count, class count, nesting depth, etc.)."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python source code."},
        },
        "required": ["code"],
    }

    def execute(self, **kwargs: Any) -> dict:
        from corpus.performance_estimator.feature_extractor import extract_features

        code: str = kwargs["code"]
        return extract_features(code)


class ExecuteScriptSkill(Skill):
    """Execute a Python script in a sandboxed subprocess and return metrics."""

    name = "execute_script"
    description = (
        "Execute a Python experiment script in a sandboxed subprocess. "
        "Returns success status, stdout/stderr, and parsed METRIC lines."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "script": {"type": "string", "description": "Python script to execute."},
        },
        "required": ["script"],
    }

    def __init__(self, cfg: dict | None = None) -> None:
        self._cfg = cfg

    def execute(self, **kwargs: Any) -> dict:
        from agents.empirical.runner import ExperimentRunner

        script: str = kwargs["script"]
        runner = ExperimentRunner(self._cfg)
        result = runner.run(script)
        return {
            "experiment_id": result.experiment_id,
            "success": result.success,
            "metrics": result.metrics,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
            "error_message": result.error_message or "",
        }


class ParseMetricsSkill(Skill):
    """Parse METRIC: name=value lines from script stdout."""

    name = "parse_metrics"
    description = (
        "Parse 'METRIC: <name>=<value>' lines from experiment stdout. "
        "Returns a dict of metric name → float value."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "stdout": {"type": "string", "description": "stdout from a script execution."},
        },
        "required": ["stdout"],
    }

    def execute(self, **kwargs: Any) -> dict:
        from agents.empirical.runner import _parse_metrics

        stdout: str = kwargs["stdout"]
        return _parse_metrics(stdout)


class SyntaxCheckSkill(Skill):
    """Quick AST syntax check for a Python code string."""

    name = "check_syntax"
    description = (
        "Perform a fast AST syntax check on a Python code snippet. "
        "Returns valid=true/false and the error message if invalid."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python source code to check."},
        },
        "required": ["code"],
    }

    def execute(self, **kwargs: Any) -> dict:
        code: str = kwargs["code"]
        try:
            ast.parse(code)
            return {"valid": True, "error": None}
        except SyntaxError as exc:
            return {"valid": False, "error": str(exc)}
