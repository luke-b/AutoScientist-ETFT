"""
etft/sandbox.py — Isolated execution sandbox for generated code.

The sandbox executes untrusted Python scripts in a Docker container with:
  - ``--network none``   (no network access)
  - ``--memory``         (configurable; default 512 MB)
  - ``--cpus``           (configurable; default 1)
  - ``--read-only``      (root filesystem is read-only)
  - a bind-mounted, ephemeral temp directory for the script

If Docker is unavailable (ImportError or Docker daemon not running), the
sandbox falls back to a plain subprocess execution and logs a warning.

Architecture
------------
  ExperimentRunner / CICDValidator
      │
      ▼
  DockerSandbox.run_script(script, timeout)
      ├── [docker available] → docker run --rm --network none ...
      └── [docker unavailable] → subprocess.run (fallback)

Usage
-----
    from etft.sandbox import DockerSandbox
    sandbox = DockerSandbox(cfg)
    stdout, stderr, returncode = sandbox.run_script(code, timeout=60)
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

# Sentinel value — truthy on first call, lazy-initialised afterwards
_docker_available: bool | None = None


def _check_docker() -> bool:
    """Return True if the Docker CLI is available and the daemon is reachable."""
    global _docker_available
    if _docker_available is not None:
        return _docker_available
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        _docker_available = result.returncode == 0
    except Exception:
        _docker_available = False
    if not _docker_available:
        logger.warning(
            "Docker unavailable — DockerSandbox will fall back to subprocess execution."
        )
    return _docker_available


class SandboxResult(NamedTuple):
    stdout: str
    stderr: str
    returncode: int
    duration_seconds: float


class DockerSandbox:
    """
    Executes a Python script in an isolated Docker container.

    Parameters
    ----------
    cfg:
        Full config dict.  Reads the ``sandbox`` section:
          backend          : "docker" | "subprocess"
          image            : Docker image (default "python:3.11-slim")
          mem_limit        : Memory limit string (default "512m")
          cpu_count        : CPU quota (default 1)
          network_disabled : Disable container networking (default True)
          timeout_seconds  : Execution timeout (default 60)
    """

    def __init__(self, cfg: dict | None = None) -> None:
        sandbox_cfg = (cfg or {}).get("sandbox", {})
        self._backend: str = sandbox_cfg.get("backend", "subprocess")  # subprocess is the safe default
        self._image: str = sandbox_cfg.get("image", "python:3.11-slim")
        self._mem_limit: str = str(sandbox_cfg.get("mem_limit", "512m"))
        self._cpu_count: int = int(sandbox_cfg.get("cpu_count", 1))
        self._network_disabled: bool = bool(sandbox_cfg.get("network_disabled", True))
        self._timeout: int = int(sandbox_cfg.get("timeout_seconds", 60))

    # ------------------------------------------------------------------
    def run_script(self, script: str, timeout: int | None = None) -> SandboxResult:
        """
        Execute *script* and return a :class:`SandboxResult`.

        Falls back to subprocess execution if ``backend`` is ``"subprocess"``
        or if Docker is unavailable at runtime.
        """
        effective_timeout = timeout if timeout is not None else self._timeout

        if self._backend == "subprocess" or not _check_docker():
            return self._run_subprocess(script, effective_timeout)

        return self._run_docker(script, effective_timeout)

    # ------------------------------------------------------------------
    def _run_docker(self, script: str, timeout: int) -> SandboxResult:
        with tempfile.TemporaryDirectory(prefix="etft_sandbox_") as tmp_dir:
            script_path = Path(tmp_dir) / "run.py"
            script_path.write_text(script)

            cmd = [
                "docker", "run", "--rm",
                "--memory", self._mem_limit,
                "--cpus", str(self._cpu_count),
                "--read-only",
                # Temp dirs required by Python itself (e.g. tempfile module)
                "--tmpfs", "/tmp",
                "-v", f"{tmp_dir}:/script:ro",
            ]
            if self._network_disabled:
                cmd += ["--network", "none"]

            cmd += [self._image, "python", "/script/run.py"]

            start = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                duration = time.perf_counter() - start
                return SandboxResult(
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    returncode=proc.returncode,
                    duration_seconds=duration,
                )
            except subprocess.TimeoutExpired:
                duration = time.perf_counter() - start
                return SandboxResult(
                    stdout="",
                    stderr=f"Docker execution timed out after {timeout}s.",
                    returncode=124,
                    duration_seconds=duration,
                )
            except Exception as exc:
                logger.error("Docker sandbox error: %s — falling back to subprocess.", exc)
                return self._run_subprocess(script, timeout)

    # ------------------------------------------------------------------
    @staticmethod
    def _run_subprocess(script: str, timeout: int) -> SandboxResult:
        with tempfile.NamedTemporaryFile(
            suffix=".py", delete=False, mode="w", prefix="etft_sub_"
        ) as f:
            f.write(script)
            tmp_path = f.name

        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            duration = time.perf_counter() - start
            return SandboxResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                returncode=proc.returncode,
                duration_seconds=duration,
            )
        except subprocess.TimeoutExpired:
            duration = time.perf_counter() - start
            return SandboxResult(
                stdout="",
                stderr=f"Subprocess execution timed out after {timeout}s.",
                returncode=124,
                duration_seconds=duration,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)
