"""
etft/tests/test_sandbox.py — Unit tests for DockerSandbox security features.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from etft.sandbox import DockerSandbox

# ---------------------------------------------------------------------------
# Subprocess mode security warning
# ---------------------------------------------------------------------------


def test_subprocess_mode_logs_warning(caplog):
    """DockerSandbox must log a WARNING when instantiated with backend='subprocess'."""
    with caplog.at_level(logging.WARNING, logger="etft.sandbox"):
        DockerSandbox(cfg={"sandbox": {"backend": "subprocess"}})

    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "subprocess" in msg.lower() or "SUBPROCESS" in msg
        for msg in warning_messages
    ), "Expected a WARNING about subprocess mode security risk"


def test_docker_mode_no_subprocess_warning(caplog):
    """DockerSandbox must NOT log the subprocess security warning when backend='docker'."""
    with caplog.at_level(logging.WARNING, logger="etft.sandbox"):
        DockerSandbox(cfg={"sandbox": {"backend": "docker"}})

    subprocess_warnings = [
        r.message for r in caplog.records
        if r.levelno == logging.WARNING and "SUBPROCESS" in r.message
    ]
    assert not subprocess_warnings


# ---------------------------------------------------------------------------
# Seccomp profile injection into Docker command
# ---------------------------------------------------------------------------


def test_seccomp_arg_injected_when_profile_exists():
    """
    When the bundled docker/seccomp-etft.json exists, _run_docker must include
    '--security-opt seccomp=<path>' in the Docker command.
    """
    import etft.sandbox as _sandbox_mod

    # The real seccomp profile lives at <repo>/docker/seccomp-etft.json.
    # Verify it exists (it is committed to the repo).
    real_seccomp = Path(_sandbox_mod.__file__).parent.parent / "docker" / "seccomp-etft.json"
    if not real_seccomp.exists():
        pytest.skip("docker/seccomp-etft.json not present in this environment")

    sandbox = DockerSandbox(cfg={"sandbox": {"backend": "docker"}})

    captured: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        captured.append(list(cmd))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with (
        patch("subprocess.run", side_effect=_fake_run),
        patch.object(_sandbox_mod, "_check_docker", return_value=True),
    ):
        try:
            sandbox._run_docker("x=1\n", timeout=5)
        except Exception:
            pass

    assert captured, "subprocess.run was not called — Docker command was not built"
    cmd_str = " ".join(str(t) for t in captured[0])
    assert "--security-opt" in cmd_str and "seccomp" in cmd_str, (
        f"Expected '--security-opt seccomp=...' in Docker cmd, got: {cmd_str}"
    )


def test_seccomp_arg_absent_when_profile_missing():
    """When no seccomp profile file exists at the resolved path, --security-opt is omitted."""
    import etft.sandbox as _sandbox_mod

    sandbox = DockerSandbox(cfg={"sandbox": {"backend": "docker"}})

    captured: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        captured.append(list(cmd))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    # Redirect __file__ so the seccomp path resolves to a non-existent file
    with (
        patch("subprocess.run", side_effect=_fake_run),
        patch.object(_sandbox_mod, "_check_docker", return_value=True),
        patch.object(_sandbox_mod, "__file__", "/nonexistent_dir/etft/sandbox.py"),
    ):
        try:
            sandbox._run_docker("x=1\n", timeout=5)
        except Exception:
            pass

    if captured:
        cmd_str = " ".join(str(t) for t in captured[0])
        assert "--security-opt" not in cmd_str, (
            "Should not inject --security-opt when seccomp file is absent"
        )
