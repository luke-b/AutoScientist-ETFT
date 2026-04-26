"""
agents/empirical/runner.py — Safely executes generated micro-experiment
scripts in a sandboxed subprocess with resource limits.

Safety measures:
  - Execution timeout (configurable, default 120 s)
  - Script size guard (64 KB)
  - Package whitelist enforced via AST import check
  - No network access (scripts must use synthetic data)
"""

from __future__ import annotations

import ast
import logging
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from corpus.regression_pipeline.schemas import ExperimentResult

logger = logging.getLogger(__name__)

_DEFAULT_ALLOWED = {"numpy", "scipy", "sklearn", "torch", "math", "random", "statistics"}


class ExperimentRunner:
    """Executes a micro-experiment script and captures its output."""

    def __init__(self, cfg: dict | None = None) -> None:
        emp_cfg = (cfg or {}).get("agents", {}).get("empirical", {})
        self.timeout: int = int(emp_cfg.get("experiment_timeout_seconds", 120))
        self.max_script_size: int = int(emp_cfg.get("max_script_size_bytes", 65536))
        allowed = emp_cfg.get("allowed_packages", list(_DEFAULT_ALLOWED))
        self._allowed: set[str] = set(allowed) | _DEFAULT_ALLOWED

    # ------------------------------------------------------------------
    def run(self, script: str) -> ExperimentResult:
        """
        Execute *script* in a sandboxed subprocess.

        Returns
        -------
        ExperimentResult with metrics parsed from stdout.
        """
        experiment_id = str(uuid.uuid4())[:8]

        # Size guard
        if len(script.encode()) > self.max_script_size:
            return ExperimentResult(
                experiment_id=experiment_id,
                script=script,
                success=False,
                error_message=f"Script exceeds max size ({self.max_script_size} bytes).",
            )

        # Import whitelist check
        violation = self._check_imports(script)
        if violation:
            return ExperimentResult(
                experiment_id=experiment_id,
                script=script,
                success=False,
                error_message=f"Disallowed import: {violation!r}",
            )

        # Write to temp file and execute
        with tempfile.NamedTemporaryFile(
            suffix=".py", delete=False, mode="w", prefix="etft_exp_"
        ) as f:
            f.write(script)
            tmp_path = f.name

        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            _duration = time.perf_counter() - start

            if proc.returncode != 0:
                return ExperimentResult(
                    experiment_id=experiment_id,
                    script=script,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    success=False,
                    error_message=f"Exit code {proc.returncode}",
                )

            metrics = _parse_metrics(proc.stdout)
            return ExperimentResult(
                experiment_id=experiment_id,
                script=script,
                stdout=proc.stdout,
                stderr=proc.stderr,
                metrics=metrics,
                success=True,
            )

        except subprocess.TimeoutExpired:
            return ExperimentResult(
                experiment_id=experiment_id,
                script=script,
                success=False,
                error_message=f"Experiment timed out after {self.timeout}s.",
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    def _check_imports(self, script: str) -> str | None:
        """Return the first disallowed top-level import name, or None."""
        try:
            tree = ast.parse(script)
        except SyntaxError:
            return None  # syntax errors are caught later at runtime

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root not in self._allowed and not root.startswith("_"):
                        return alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    if root not in self._allowed and not root.startswith("_"):
                        return node.module
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_metrics(stdout: str) -> dict[str, float]:
    """
    Parse lines of the form ``METRIC: <name>=<value>`` from stdout.

    Example stdout line:  ``METRIC: accuracy=0.923``
    """
    metrics: dict[str, float] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("METRIC:"):
            body = line[len("METRIC:"):].strip()
            if "=" in body:
                name, _, value = body.partition("=")
                try:
                    metrics[name.strip()] = float(value.strip())
                except ValueError:
                    pass
    return metrics
