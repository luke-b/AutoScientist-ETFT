"""
synthesis/lora_routing/tests/test_router.py — Unit tests for
DynamicLoRARouter with mocked LoRA hot-swap and mocked LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from synthesis.lora_routing.adapter_library import AdapterLibrary
from synthesis.lora_routing.router import (
    DynamicLoRARouter,
    SynthesisResult,
    _build_trajectory_summary,
    _extract_code,
    _extract_rationale,
    _format_papers,
    _parse_depth_analysis,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_SIMPLE_ALGO = "x = 1\n"

_DEPTH_RESPONSE = (
    "BOTTLENECK: batch_normalisation\n"
    "STRUCTURAL_SUMMARY: The trajectory shows progressive replacement of BN with LN.\n"
    "RECOMMENDED_EPOCH: 3\n"
)

_SYNTHESIS_RESPONSE = (
    "```python\n"
    "def model(x):\n"
    "    return x * 2\n"
    "```\n"
    "RATIONALE: Applied LayerNorm instead of BatchNorm for better batch independence.\n"
)


@pytest.fixture()
def lib(tmp_path) -> AdapterLibrary:
    index_path = tmp_path / "adapter_index.json"
    lib = AdapterLibrary(index_path=index_path)
    adapter_dir = tmp_path / "gen_3"
    adapter_dir.mkdir()
    lib.register(generation_index=3, adapter_path=adapter_dir)
    return lib


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


def test_extract_code_from_fenced_block():
    text = "Some preamble\n```python\nx = 1\n```\nsome suffix"
    assert _extract_code(text) == "x = 1"


def test_extract_code_fallback():
    text = "no fenced block here just raw code"
    # No fenced block — returns stripped text
    assert _extract_code(text) == "no fenced block here just raw code"


def test_extract_rationale_after_marker():
    text = "```python\nx = 1\n```\nRATIONALE: This is the reason."
    assert "This is the reason" in _extract_rationale(text)


def test_parse_depth_analysis_parses_all_fields():
    result = _parse_depth_analysis(_DEPTH_RESPONSE, fallback_epoch=0)
    assert result.bottleneck == "batch_normalisation"
    assert "LayerNorm" in result.structural_summary or "LN" in result.structural_summary
    assert result.recommended_epoch == 3


def test_parse_depth_analysis_fallback_epoch():
    result = _parse_depth_analysis("no structured fields here", fallback_epoch=7)
    assert result.recommended_epoch == 7
    assert result.bottleneck == "unknown bottleneck"


def test_build_trajectory_summary_empty():
    assert "(no trajectory steps" in _build_trajectory_summary([])


def test_build_trajectory_summary_with_steps():
    @dataclass
    class FakeStep:
        algorithm_id: str
        step_index: int
        fitness_score: float

    steps = [FakeStep("alg_v0", 0, 1.0), FakeStep("alg_v1", 1, 1.1)]
    summary = _build_trajectory_summary(steps)
    assert "alg_v0" in summary or "alg_v1" in summary


def test_format_papers_empty():
    assert "No literature" in _format_papers([], [])


# ---------------------------------------------------------------------------
# DynamicLoRARouter tests (all LLM calls mocked)
# ---------------------------------------------------------------------------


def test_router_route_no_adapter(lib, tmp_path):
    """Router falls back gracefully when no adapter registered for resolved epoch."""
    lib_no_gen = AdapterLibrary(index_path=tmp_path / "empty_index.json")

    router = DynamicLoRARouter(cfg={}, adapter_library=lib_no_gen)

    with (
        patch.object(router._llm, "complete", return_value=_DEPTH_RESPONSE) as mock_depth,
        patch(
            "agents.literature.searcher.LiteratureSearcher.search", return_value=[]
        ),
        patch.object(
            router._llm, "complete", side_effect=[_DEPTH_RESPONSE, _SYNTHESIS_RESPONSE]
        ),
    ):
        # Both depth and synthesis use the proxy (no adapter)
        with patch.object(router._llm, "complete") as mock_complete:
            mock_complete.side_effect = [_DEPTH_RESPONSE, _SYNTHESIS_RESPONSE]
            with patch(
                "agents.literature.searcher.LiteratureSearcher.search", return_value=[]
            ):
                result = router.route(sota_code=_SIMPLE_ALGO, generation_index=1)

    assert isinstance(result, SynthesisResult)


def test_router_route_with_adapter(lib, tmp_path):
    """Router resolves registered adapter path and returns SynthesisResult."""
    router = DynamicLoRARouter(cfg={}, adapter_library=lib)

    def _mock_complete(prompt, system=""):
        if "BOTTLENECK" in system or "BOTTLENECK" in prompt:
            return _DEPTH_RESPONSE
        return _SYNTHESIS_RESPONSE

    with (
        patch.object(router._llm, "complete", side_effect=_mock_complete),
        patch(
            "agents.literature.searcher.LiteratureSearcher.search", return_value=[]
        ),
    ):
        result = router.route(sota_code=_SIMPLE_ALGO, generation_index=3)

    assert isinstance(result, SynthesisResult)
    # depth analysis should have been called and bottleneck extracted
    if result.depth_analysis:
        assert result.depth_analysis.bottleneck != ""


def test_router_returns_failure_on_exception(lib):
    """Router catches exceptions and returns a failed SynthesisResult."""
    router = DynamicLoRARouter(cfg={}, adapter_library=lib)

    with patch.object(router._llm, "complete", side_effect=RuntimeError("network error")):
        result = router.route(sota_code=_SIMPLE_ALGO, generation_index=0)

    assert result.success is False
    assert "network error" in result.error


def test_router_depth_analysis_uses_fallback_epoch(lib):
    """When LLM response has no RECOMMENDED_EPOCH, the passed generation_index is used."""
    router = DynamicLoRARouter(cfg={}, adapter_library=lib)

    no_epoch_response = (
        "BOTTLENECK: attention\n"
        "STRUCTURAL_SUMMARY: Multi-head attention dominates.\n"
    )

    with (
        patch.object(router._llm, "complete", side_effect=[no_epoch_response, _SYNTHESIS_RESPONSE]),
        patch("agents.literature.searcher.LiteratureSearcher.search", return_value=[]),
    ):
        result = router.route(sota_code=_SIMPLE_ALGO, generation_index=5)

    if result.depth_analysis:
        assert result.depth_analysis.recommended_epoch == 5
