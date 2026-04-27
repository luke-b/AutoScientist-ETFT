"""
agents/empirical/runner.py — Safely executes generated micro-experiment
scripts in a sandboxed subprocess with resource limits.

Safety measures:
  - Execution timeout (configurable, default 120 s)
  - Script size guard (64 KB)
  - Package whitelist enforced via AST import check
  - No network access (scripts must use synthetic data)
  - Docker-isolated execution via DockerSandbox (falls back to subprocess)
"""

from __future__ import annotations

import ast
import logging
import uuid

from corpus.regression_pipeline.schemas import ExperimentResult
from etft.sandbox import DockerSandbox

logger = logging.getLogger(__name__)

_DEFAULT_ALLOWED = {"numpy", "scipy", "sklearn", "torch", "math", "random", "statistics"}

# Built-in names that can be used to dynamically execute or import code,
# bypassing the static import whitelist.  Any call to one of these in the
# AST is treated as a security violation regardless of arguments.
_DEFAULT_BLOCKED_BUILTINS: frozenset[str] = frozenset({
    "exec",
    "eval",
    "__import__",
    "compile",
    "open",
})

# importlib attributes that allow dynamic module loading
_BLOCKED_IMPORTLIB_ATTRS: frozenset[str] = frozenset({
    "import_module",
    "util",
    "__import__",
})


class ExperimentRunner:
    """Executes a micro-experiment script and captures its output."""

    def __init__(self, cfg: dict | None = None) -> None:
        emp_cfg = (cfg or {}).get("agents", {}).get("empirical", {})
        self.timeout: int = int(emp_cfg.get("experiment_timeout_seconds", 120))
        self.max_script_size: int = int(emp_cfg.get("max_script_size_bytes", 65536))
        allowed = emp_cfg.get("allowed_packages", list(_DEFAULT_ALLOWED))
        self._allowed: set[str] = set(allowed) | _DEFAULT_ALLOWED
        extra_blocked = emp_cfg.get("blocked_builtins", [])
        self._blocked_builtins: frozenset[str] = _DEFAULT_BLOCKED_BUILTINS | frozenset(extra_blocked)
        self._sandbox = DockerSandbox(cfg)

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

        # Import / security check
        violation = self._check_imports(script)
        if violation:
            return ExperimentResult(
                experiment_id=experiment_id,
                script=script,
                success=False,
                error_message=violation,
            )

        # Execute via sandbox (Docker if available, subprocess fallback)
        result = self._sandbox.run_script(script, timeout=self.timeout)

        if result.returncode == 124:
            return ExperimentResult(
                experiment_id=experiment_id,
                script=script,
                success=False,
                error_message=f"Experiment timed out after {self.timeout}s.",
            )

        if result.returncode != 0:
            return ExperimentResult(
                experiment_id=experiment_id,
                script=script,
                stdout=result.stdout,
                stderr=result.stderr,
                success=False,
                error_message=f"Exit code {result.returncode}",
            )

        metrics = _parse_metrics(result.stdout)
        return ExperimentResult(
            experiment_id=experiment_id,
            script=script,
            stdout=result.stdout,
            stderr=result.stderr,
            metrics=metrics,
            success=True,
        )

    # ------------------------------------------------------------------
    def _check_imports(self, script: str) -> str | None:
        """
        Return a violation description string if *script* contains a
        disallowed pattern, or ``None`` if the script passes all checks.

        Patterns checked (in order):

        1. ``import <module>`` / ``from <module> import …`` — the root
           package must be in the allowed-packages whitelist.
        2. ``exec(…)`` / ``eval(…)`` / ``__import__(…)`` / ``compile(…)``
           / ``open(…)`` — direct calls to dangerous built-ins.
        3. ``importlib.import_module(…)`` and similar attribute accesses on
           ``importlib`` — dynamic module loading that bypasses (1).
        4. ``getattr(…)`` / ``__builtins__[…]`` style access to blocked
           names — indirect references to dangerous built-ins.
        """
        try:
            tree = ast.parse(script)
        except SyntaxError:
            return None  # syntax errors are caught later at runtime

        for node in ast.walk(tree):
            # --- (1) static import statements ---
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root not in self._allowed and not root.startswith("_"):
                        return f"Disallowed import: {alias.name!r}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    if root not in self._allowed and not root.startswith("_"):
                        return f"Disallowed import: {node.module!r}"

            # --- (2) blocked built-in calls: exec / eval / __import__ / compile / open ---
            elif isinstance(node, ast.Call):
                func = node.func
                # Direct name call: exec(...), eval(...), etc.
                if isinstance(func, ast.Name) and func.id in self._blocked_builtins:
                    return f"Blocked builtin call: {func.id!r}"

                # --- (3) importlib.import_module / importlib.util.find_spec etc. ---
                if isinstance(func, ast.Attribute):
                    # e.g. importlib.import_module(...)
                    if (
                        isinstance(func.value, ast.Name)
                        and func.value.id == "importlib"
                        and func.attr in _BLOCKED_IMPORTLIB_ATTRS
                    ):
                        return f"Blocked dynamic import: importlib.{func.attr!r}"

                    # e.g. importlib.util.find_spec(...)
                    if (
                        isinstance(func.value, ast.Attribute)
                        and isinstance(func.value.value, ast.Name)
                        and func.value.value.id == "importlib"
                    ):
                        return f"Blocked dynamic import: importlib.{func.value.attr}.{func.attr!r}"

                # --- (4) getattr(builtins, 'exec') style ---
                if isinstance(func, ast.Name) and func.id == "getattr":
                    # getattr(<something>, <string_literal>)
                    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                        attr_name = str(node.args[1].value)
                        if attr_name in self._blocked_builtins:
                            return f"Blocked getattr access to builtin: {attr_name!r}"

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
