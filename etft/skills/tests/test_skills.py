"""
etft/skills/tests/test_skills.py — Unit tests for all native skills.
"""

from __future__ import annotations

import pytest

from etft.skills.base import AgentResult, LLMResponse, Skill, SkillRegistry, ToolCall


# ---------------------------------------------------------------------------
# Base types
# ---------------------------------------------------------------------------


def test_tool_call_dataclass():
    tc = ToolCall(id="call_1", name="check_syntax", args={"code": "x = 1"})
    assert tc.id == "call_1"
    assert tc.args["code"] == "x = 1"


def test_llm_response_defaults():
    resp = LLMResponse(content="hello")
    assert resp.tool_calls == []
    assert resp.finish_reason == "stop"


def test_agent_result_success():
    result = AgentResult(
        output="done",
        steps=[{"step": 1}],
        skills_invoked=["check_syntax"],
        success=True,
    )
    assert result.success
    assert result.error is None
    assert result.metadata == {}


def test_agent_result_failure():
    result = AgentResult(
        output="",
        steps=[],
        skills_invoked=[],
        success=False,
        error="something went wrong",
    )
    assert not result.success
    assert result.error == "something went wrong"


# ---------------------------------------------------------------------------
# SkillRegistry
# ---------------------------------------------------------------------------


class _DummySkill(Skill):
    name = "dummy_skill"
    description = "A dummy skill for testing."
    parameters_schema = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
    }

    def execute(self, **kwargs):
        return {"doubled": kwargs["x"] * 2}


def test_registry_register_and_get():
    registry = SkillRegistry()
    skill = _DummySkill()
    registry.register(skill)
    assert registry.get("dummy_skill") is skill
    assert registry.get("nonexistent") is None


def test_registry_get_tool_definitions():
    registry = SkillRegistry()
    registry.register(_DummySkill())
    defs = registry.get_tool_definitions()
    assert len(defs) == 1
    assert defs[0]["type"] == "function"
    assert defs[0]["function"]["name"] == "dummy_skill"


def test_registry_execute():
    registry = SkillRegistry()
    registry.register(_DummySkill())
    result = registry.execute("dummy_skill", x=5)
    assert result == {"doubled": 10}


def test_registry_execute_unknown():
    registry = SkillRegistry()
    result = registry.execute("missing_skill")
    assert "error" in result


def test_registry_execute_propagates_error():
    class _BrokenSkill(Skill):
        name = "broken"
        description = "Broken"
        parameters_schema = {"type": "object", "properties": {}}

        def execute(self, **kwargs):
            raise ValueError("intentional error")

    registry = SkillRegistry()
    registry.register(_BrokenSkill())
    result = registry.execute("broken")
    assert "error" in result
    assert "intentional error" in result["error"]


def test_skill_to_tool_definition():
    skill = _DummySkill()
    defn = skill.to_tool_definition()
    assert defn["function"]["name"] == "dummy_skill"
    assert defn["function"]["description"] == "A dummy skill for testing."


# ---------------------------------------------------------------------------
# SyntaxCheckSkill
# ---------------------------------------------------------------------------


def test_syntax_check_valid():
    from etft.skills.code_skills import SyntaxCheckSkill

    skill = SyntaxCheckSkill()
    result = skill.execute(code="def foo(): return 42")
    assert result["valid"] is True
    assert result["error"] is None


def test_syntax_check_invalid():
    from etft.skills.code_skills import SyntaxCheckSkill

    skill = SyntaxCheckSkill()
    result = skill.execute(code="def foo(\n")
    assert result["valid"] is False
    assert result["error"] is not None


# ---------------------------------------------------------------------------
# ExtractCodeFeaturesSkill
# ---------------------------------------------------------------------------


def test_extract_code_features():
    from etft.skills.code_skills import ExtractCodeFeaturesSkill

    skill = ExtractCodeFeaturesSkill()
    code = "def foo():\n    for i in range(10):\n        pass\n"
    result = skill.execute(code=code)
    assert "num_functions" in result
    assert result["num_functions"] == 1.0
    assert result["num_loops"] == 1.0


# ---------------------------------------------------------------------------
# ParseMetricsSkill
# ---------------------------------------------------------------------------


def test_parse_metrics_skill():
    from etft.skills.code_skills import ParseMetricsSkill

    skill = ParseMetricsSkill()
    stdout = "METRIC: loss=0.1\nsome other line\nMETRIC: acc=0.9"
    result = skill.execute(stdout=stdout)
    assert abs(result["loss"] - 0.1) < 1e-9
    assert abs(result["acc"] - 0.9) < 1e-9


# ---------------------------------------------------------------------------
# ComputeTrajectoryDeltasSkill
# ---------------------------------------------------------------------------


def test_compute_trajectory_deltas_skill():
    from etft.skills.analysis_skills import ComputeTrajectoryDeltasSkill

    skill = ComputeTrajectoryDeltasSkill()
    steps = [
        {
            "step_index": 0,
            "algorithm_family": "test",
            "code": "x = 1\ny = 2\nz = 3\na = 4\nb = 5\n",
            "fitness_score": 0.5,
        },
        {
            "step_index": 1,
            "algorithm_family": "test",
            "code": "x = 1\ny = 2\nz = 3\na = 4\nb = 5\nw = 6\nv = 7\n",
            "fitness_score": 0.8,
        },
    ]
    result = skill.execute(steps=steps, min_delta_lines=1)
    assert "deltas" in result


# ---------------------------------------------------------------------------
# TriageCandidateSkill
# ---------------------------------------------------------------------------


def test_triage_candidate_skill_no_model(tmp_path):
    from etft.skills.filter_skills import TriageCandidateSkill

    skill = TriageCandidateSkill()
    candidate = {
        "candidate_id": "c1",
        "trajectory_id": "t1",
        "code": "x = 1\n",
        "rationale": "test",
    }
    result = skill.execute(
        candidate=candidate,
        model_path=str(tmp_path / "nonexistent.joblib"),
    )
    assert result["triage_passed"] is True
    assert result["risk_score"] == pytest.approx(0.0)
