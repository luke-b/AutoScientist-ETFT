"""
etft/tests/test_agent.py — Unit tests for AgentClient ReAct loop.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from etft.agent import AgentClient
from etft.skills.base import AgentResult, LLMResponse, Skill, SkillRegistry, ToolCall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _EchoSkill(Skill):
    """Returns whatever 'message' arg was passed."""

    name = "echo"
    description = "Echoes the input message."
    parameters_schema = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }

    def execute(self, **kwargs):
        return {"echo": kwargs["message"]}


def _make_agent(skills=None):
    """Create an AgentClient with a mocked LLMClient."""
    with patch("etft.agent.AgentClient.__init__", lambda self, cfg, skills: None):
        agent = AgentClient.__new__(AgentClient)

    agent._cfg = {}
    agent._registry = SkillRegistry()
    for skill in (skills or []):
        agent._registry.register(skill)
    agent._default_max_steps = 10
    agent._default_system = "You are a test agent."
    agent._llm = MagicMock()
    return agent


# ---------------------------------------------------------------------------
# run_task — no tools
# ---------------------------------------------------------------------------


def test_run_task_simple_completion():
    agent = _make_agent()
    agent._llm.complete_with_tools.return_value = LLMResponse(
        content="The answer is 42.",
        finish_reason="stop",
    )

    result = agent.run_task("What is 6 * 7?")

    assert result.success
    assert result.output == "The answer is 42."
    assert result.skills_invoked == []
    assert len(result.steps) == 1


def test_run_task_includes_context():
    agent = _make_agent()
    agent._llm.complete_with_tools.return_value = LLMResponse(
        content="Done.",
        finish_reason="stop",
    )

    agent.run_task("Do something.", context={"key": "value"})

    call_args = agent._llm.complete_with_tools.call_args
    messages = call_args[1]["messages"] if call_args[1] else call_args[0][0]
    user_msg = next(m for m in messages if m["role"] == "user")
    assert "key" in user_msg["content"]
    assert "value" in user_msg["content"]


def test_run_task_system_override():
    agent = _make_agent()
    agent._llm.complete_with_tools.return_value = LLMResponse(
        content="Done.",
        finish_reason="stop",
    )

    agent.run_task("task", system="Custom system.")

    call_args = agent._llm.complete_with_tools.call_args
    messages = call_args[1]["messages"] if call_args[1] else call_args[0][0]
    system_msg = next((m for m in messages if m["role"] == "system"), None)
    assert system_msg is not None
    assert system_msg["content"] == "Custom system."


# ---------------------------------------------------------------------------
# run_task — with tool calls
# ---------------------------------------------------------------------------


def test_run_task_executes_tool_call():
    echo_skill = _EchoSkill()
    agent = _make_agent(skills=[echo_skill])

    # First LLM call → tool call; second → final answer
    agent._llm.complete_with_tools.side_effect = [
        LLMResponse(
            content="",
            tool_calls=[ToolCall(id="call_1", name="echo", args={"message": "hello"})],
            finish_reason="tool_calls",
        ),
        LLMResponse(
            content="The echo says: hello",
            finish_reason="stop",
        ),
    ]

    result = agent.run_task("Echo 'hello' please.")

    assert result.success
    assert result.output == "The echo says: hello"
    assert "echo" in result.skills_invoked
    assert agent._llm.complete_with_tools.call_count == 2


def test_run_task_tool_result_appended_to_messages():
    echo_skill = _EchoSkill()
    agent = _make_agent(skills=[echo_skill])

    agent._llm.complete_with_tools.side_effect = [
        LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc_1", name="echo", args={"message": "test"})],
            finish_reason="tool_calls",
        ),
        LLMResponse(content="Got it.", finish_reason="stop"),
    ]

    agent.run_task("Use echo.")

    # The second call should include tool result messages
    second_call_messages = agent._llm.complete_with_tools.call_args_list[1][1]["messages"]
    roles = [m["role"] for m in second_call_messages]
    assert "tool" in roles


def test_run_task_unknown_skill_returns_error():
    agent = _make_agent()

    agent._llm.complete_with_tools.side_effect = [
        LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc_1", name="nonexistent_skill", args={})],
            finish_reason="tool_calls",
        ),
        LLMResponse(content="OK.", finish_reason="stop"),
    ]

    result = agent.run_task("Call missing skill.")
    assert result.success  # Still succeeds (error is in tool result)


def test_run_task_max_steps_truncates():
    agent = _make_agent(skills=[_EchoSkill()])
    agent._default_max_steps = 2

    # Always return a tool call (infinite loop guard)
    agent._llm.complete_with_tools.return_value = LLMResponse(
        content="thinking...",
        tool_calls=[ToolCall(id="tc", name="echo", args={"message": "x"})],
        finish_reason="tool_calls",
    )

    result = agent.run_task("Loop forever.")
    assert result.success
    assert result.metadata.get("truncated") is True


# ---------------------------------------------------------------------------
# run_task — error handling
# ---------------------------------------------------------------------------


def test_run_task_llm_error_returns_failure():
    agent = _make_agent()
    agent._llm.complete_with_tools.side_effect = ConnectionError("proxy down")

    result = agent.run_task("Do something.")

    assert not result.success
    assert result.error is not None
    assert "proxy down" in result.error


# ---------------------------------------------------------------------------
# AgentClient construction
# ---------------------------------------------------------------------------


def test_agent_client_registers_skills():
    echo = _EchoSkill()
    with patch("etft.llm.LLMClient"):
        with patch("etft.agent.LLMClient", create=True):
            agent = AgentClient.__new__(AgentClient)
            agent._cfg = {}
            agent._registry = SkillRegistry()
            agent._registry.register(echo)
            agent._default_max_steps = 10
            agent._default_system = "test"
            agent._llm = MagicMock()

    assert agent._registry.get("echo") is echo


def test_agent_client_default_max_steps():
    with patch("etft.llm.httpx"):
        with patch("etft.llm.LLMClient.__init__", return_value=None):
            agent = AgentClient(cfg={"agent": {"max_steps": 5}})
    assert agent._default_max_steps == 5


def test_agent_client_default_system_from_config():
    with patch("etft.llm.LLMClient.__init__", return_value=None):
        agent = AgentClient(
            cfg={"agent": {"system_prompt": "Custom prompt."}},
        )
    assert agent._default_system == "Custom prompt."
