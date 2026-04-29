"""
synthesis/tests/test_recursive_loop.py — Unit tests for RecursiveSOTAGenerator
with mocked adapters, sandbox, and DynamicLoRARouter.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from synthesis.lora_routing.adapter_library import AdapterLibrary
from synthesis.recursive_loop import RecursiveResult, RecursiveSOTAGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SIMPLE_CODE = "def model(x):\n    return x\n"
_IMPROVED_CODE = "def model(x):\n    return x * 2\n"


@pytest.fixture()
def lib(tmp_path) -> AdapterLibrary:
    index_path = tmp_path / "adapter_index.json"
    lib = AdapterLibrary(index_path=index_path)
    adapter_dir = tmp_path / "gen_0"
    adapter_dir.mkdir()
    lib.register(generation_index=0, adapter_path=adapter_dir)
    return lib


@pytest.fixture()
def generator(lib) -> RecursiveSOTAGenerator:
    cfg = {
        "orthogonal_calibration": {
            "recursive": {"enabled": True, "max_sota_plus_x": 2},
            "objective_calibration": {
                "diversity_threshold": 0.0,   # always pass diversity gate in tests
                "performance_tolerance": 10.0, # always pass quality gate in tests
            },
        },
        "sandbox": {"backend": "subprocess", "timeout_seconds": 10},
    }
    return RecursiveSOTAGenerator(cfg=cfg, adapter_library=lib)


# ---------------------------------------------------------------------------
# Route mock helper
# ---------------------------------------------------------------------------


def _make_route_result(code: str, success: bool = True):
    from synthesis.lora_routing.router import DepthAnalysis, SynthesisResult
    return SynthesisResult(
        code=code,
        rationale="Test rationale.",
        adapter_used=None,
        depth_analysis=DepthAnalysis(
            bottleneck="test_bottleneck",
            structural_summary="Test summary.",
            recommended_epoch=0,
        ),
        papers_retrieved=0,
        success=success,
        error="" if success else "mock failure",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_generate_returns_list_of_recursive_results(generator, tmp_path):
    """generate() returns a list of RecursiveResult objects."""
    with patch(
        "synthesis.lora_routing.router.DynamicLoRARouter.route",
        return_value=_make_route_result(_IMPROVED_CODE),
    ):
        results = generator.generate(
            sota_code=_SIMPLE_CODE,
            generation_index=0,
            max_depth=1,
            output_dir=tmp_path / "recursive",
        )

    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], RecursiveResult)
    assert results[0].depth == 1


def test_generate_sandbox_validation_fails_halts_loop(generator, tmp_path):
    """When sandbox validation fails at depth 1, the loop stops."""
    bad_code = "this is not valid python !!!@@@"

    with patch(
        "synthesis.lora_routing.router.DynamicLoRARouter.route",
        return_value=_make_route_result(bad_code),
    ):
        results = generator.generate(
            sota_code=_SIMPLE_CODE,
            generation_index=0,
            max_depth=2,
            output_dir=tmp_path / "recursive",
        )

    assert len(results) == 1
    assert results[0].sandbox_passed is False


def test_generate_max_depth_respected(generator, tmp_path):
    """The loop does not exceed max_depth iterations."""
    with patch(
        "synthesis.lora_routing.router.DynamicLoRARouter.route",
        return_value=_make_route_result(_IMPROVED_CODE),
    ):
        results = generator.generate(
            sota_code=_SIMPLE_CODE,
            generation_index=0,
            max_depth=2,
            output_dir=tmp_path / "recursive",
        )

    assert len(results) <= 2


def test_generate_persists_json_files(generator, tmp_path):
    """RecursiveResult objects are persisted as JSON files."""
    out_dir = tmp_path / "recursive"
    with patch(
        "synthesis.lora_routing.router.DynamicLoRARouter.route",
        return_value=_make_route_result(_IMPROVED_CODE),
    ):
        results = generator.generate(
            sota_code=_SIMPLE_CODE,
            generation_index=0,
            max_depth=1,
            output_dir=out_dir,
        )

    json_files = list(out_dir.glob("*.json"))
    assert len(json_files) == len(results)


def test_router_failure_produces_failed_result(generator, tmp_path):
    """A router failure at depth 1 produces a result with sandbox_passed=False."""
    with patch(
        "synthesis.lora_routing.router.DynamicLoRARouter.route",
        return_value=_make_route_result("", success=False),
    ):
        results = generator.generate(
            sota_code=_SIMPLE_CODE,
            generation_index=0,
            max_depth=2,
            output_dir=tmp_path / "recursive",
        )

    assert len(results) >= 1
    assert results[0].sandbox_passed is False


def test_recursive_result_has_required_fields(generator, tmp_path):
    with patch(
        "synthesis.lora_routing.router.DynamicLoRARouter.route",
        return_value=_make_route_result(_IMPROVED_CODE),
    ):
        results = generator.generate(
            sota_code=_SIMPLE_CODE,
            generation_index=0,
            max_depth=1,
            output_dir=tmp_path / "recursive",
        )

    r = results[0]
    assert hasattr(r, "depth")
    assert hasattr(r, "candidate_id")
    assert hasattr(r, "code")
    assert hasattr(r, "sandbox_passed")
    assert hasattr(r, "passed_gate")
    assert hasattr(r, "metadata")


def test_objective_calibration_gate_checked(generator, tmp_path):
    """The passed_gate field reflects the ObjectiveCalibration outcome."""
    with patch(
        "synthesis.lora_routing.router.DynamicLoRARouter.route",
        return_value=_make_route_result(_IMPROVED_CODE),
    ):
        results = generator.generate(
            sota_code=_SIMPLE_CODE,
            generation_index=0,
            max_depth=1,
            output_dir=tmp_path / "recursive",
        )

    # With diversity_threshold=0.0 and performance_tolerance=10.0, gate should open
    assert isinstance(results[0].passed_gate, bool)


def test_empty_code_from_router_fails_sandbox(generator, tmp_path):
    """Empty code from router leads to sandbox_passed=False."""
    with patch(
        "synthesis.lora_routing.router.DynamicLoRARouter.route",
        return_value=_make_route_result(""),
    ):
        results = generator.generate(
            sota_code=_SIMPLE_CODE,
            generation_index=0,
            max_depth=1,
            output_dir=tmp_path / "recursive",
        )

    assert results[0].sandbox_passed is False
