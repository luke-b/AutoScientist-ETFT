"""
cicd_validator.py — Validates generated Python code for the ETFT corpus.

Checks performed (in order):
  1. Syntax parse via ast.parse
  2. File size guard (configurable max_code_size_bytes)
  3. Sandboxed execution (Docker if available, subprocess fallback)
  4. OOM / timeout detection
"""

from __future__ import annotations

import ast
import logging

from corpus.regression_pipeline.schemas import ValidationResult, ValidationStatus
from etft.sandbox import DockerSandbox

logger = logging.getLogger(__name__)

# OOM heuristics — keywords that strongly suggest an out-of-memory failure
_OOM_PATTERNS = ("MemoryError", "CUDA out of memory", "Killed", "OutOfMemoryError")


class CICDValidator:
    """Validates generated Python code artifacts."""

    def __init__(self, cfg: dict | None = None) -> None:
        pipeline_cfg = (cfg or {}).get("corpus", {}).get("regression_pipeline", {})
        self.timeout: int = int(pipeline_cfg.get("validation_timeout_seconds", 60))
        self.max_code_size: int = int(pipeline_cfg.get("max_code_size_bytes", 524288))
        self._sandbox = DockerSandbox(cfg)

    # ------------------------------------------------------------------
    def validate(self, code: str) -> ValidationResult:
        """Run all checks and return a ValidationResult."""
        # 1. Size guard
        if len(code.encode()) > self.max_code_size:
            return ValidationResult(
                status=ValidationStatus.FAIL_SYNTAX,
                stderr=f"Code exceeds max size ({self.max_code_size} bytes).",
            )

        # 2. Syntax check
        syntax_result = self._check_syntax(code)
        if not syntax_result.passed:
            return syntax_result

        # 3. Sandbox execution
        return self._run_subprocess(code)

    # ------------------------------------------------------------------
    @staticmethod
    def _check_syntax(code: str) -> ValidationResult:
        try:
            ast.parse(code)
            return ValidationResult(status=ValidationStatus.PASS)
        except SyntaxError as exc:
            return ValidationResult(
                status=ValidationStatus.FAIL_SYNTAX,
                stderr=f"SyntaxError: {exc}",
            )

    # ------------------------------------------------------------------
    def _run_subprocess(self, code: str) -> ValidationResult:
        result = self._sandbox.run_script(code, timeout=self.timeout)

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        if result.returncode == 124:
            return ValidationResult(
                status=ValidationStatus.FAIL_TIMEOUT,
                stderr=stderr or f"Execution timed out after {self.timeout}s.",
                duration_seconds=float(self.timeout),
            )

        if any(pat in (stderr + stdout) for pat in _OOM_PATTERNS):
            return ValidationResult(
                status=ValidationStatus.FAIL_OOM,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=result.duration_seconds,
            )

        if result.returncode != 0:
            return ValidationResult(
                status=ValidationStatus.FAIL_RUNTIME,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=result.duration_seconds,
            )

        return ValidationResult(
            status=ValidationStatus.PASS,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=result.duration_seconds,
        )
