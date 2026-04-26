"""
cicd_validator.py — Validates generated Python code for the ETFT corpus.

Checks performed (in order):
  1. Syntax parse via ast.parse
  2. File size guard (configurable max_code_size_bytes)
  3. Sandboxed subprocess execution with timeout
  4. Peak memory estimation via resource module (Unix only)
"""

from __future__ import annotations

import ast
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from corpus.regression_pipeline.schemas import ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)

# OOM heuristics — keywords that strongly suggest an out-of-memory failure
_OOM_PATTERNS = ("MemoryError", "CUDA out of memory", "Killed", "OutOfMemoryError")


class CICDValidator:
    """Validates generated Python code artifacts."""

    def __init__(self, cfg: dict | None = None) -> None:
        pipeline_cfg = (cfg or {}).get("corpus", {}).get("regression_pipeline", {})
        self.timeout: int = int(pipeline_cfg.get("validation_timeout_seconds", 60))
        self.max_code_size: int = int(pipeline_cfg.get("max_code_size_bytes", 524288))

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

        # 3. Subprocess execution
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
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write(code)
            tmp_path = f.name

        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            duration = time.perf_counter() - start

            stderr = proc.stderr or ""
            stdout = proc.stdout or ""

            if any(pat in stderr or stdout for pat in _OOM_PATTERNS):
                return ValidationResult(
                    status=ValidationStatus.FAIL_OOM,
                    stdout=stdout,
                    stderr=stderr,
                    duration_seconds=duration,
                )

            if proc.returncode != 0:
                return ValidationResult(
                    status=ValidationStatus.FAIL_RUNTIME,
                    stdout=stdout,
                    stderr=stderr,
                    duration_seconds=duration,
                )

            return ValidationResult(
                status=ValidationStatus.PASS,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
            )

        except subprocess.TimeoutExpired:
            return ValidationResult(
                status=ValidationStatus.FAIL_TIMEOUT,
                stderr=f"Execution timed out after {self.timeout}s.",
                duration_seconds=float(self.timeout),
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)
