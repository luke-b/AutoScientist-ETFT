"""
etft/agent.py — AgentClient with ReAct (Reason + Act) loop.

This is the ONLY place in ETFT code that imports LLMClient.  All domain
modules interact with the LLM exclusively through AgentClient.run_task().

Architecture
------------
  Domain module
      │  AgentClient.run_task(task, context)
      ▼
  AgentClient ──── LLMClient.complete_with_tools(messages, tools)
      │                  │
      │                  ▼  HTTP
      │           Coding-Agent Proxy (docker/agent/server.py)
      │                  │
      │                  ▼  provider SDK
      │           OpenAI / Anthropic / local model
      │
      ├── tool_calls returned? ──► SkillRegistry.execute(name, **args)
      │       ↑ loop until finish_reason == "stop" or max_steps exceeded
      └── AgentResult(output, steps, skills_invoked, success)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from etft.skills.base import AgentResult, Skill, SkillRegistry

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM = (
    "You are an expert ML researcher and software engineer with access to ETFT tools. "
    "Use available tools to gather information, validate code, and complete the task. "
    "When you have enough information, provide a final answer."
)


class AgentClient:
    """
    ReAct agent that orchestrates LLM reasoning with local skill execution.

    Parameters
    ----------
    cfg:
        Config dict (from config.yaml).  Forwarded to LLMClient and Skills.
    skills:
        Optional list of Skill instances to register.  If None, the agent
        has no tools and behaves as a plain chat client.
    """

    def __init__(
        self,
        cfg: dict | None = None,
        skills: list[Skill] | None = None,
    ) -> None:
        # LLMClient is ONLY imported here in ETFT code
        from etft.llm import LLMClient

        self._cfg = cfg or {}
        self._llm = LLMClient(cfg)

        self._registry = SkillRegistry()
        for skill in skills or []:
            self._registry.register(skill)

        agent_cfg = self._cfg.get("agent", {})
        self._default_max_steps: int = int(agent_cfg.get("max_steps", 10))
        self._default_system: str = agent_cfg.get("system_prompt", _DEFAULT_SYSTEM)

    # ------------------------------------------------------------------
    def run_task(
        self,
        task: str,
        context: dict | None = None,
        system: str = "",
        max_steps: int | None = None,
    ) -> AgentResult:
        """
        Execute a task with the ReAct loop.

        Parameters
        ----------
        task:
            Semantically rich description of what the agent must accomplish.
        context:
            Optional dict of named context values included in the first message.
        system:
            System prompt override.  Defaults to the agent-level system prompt.
        max_steps:
            Maximum reasoning + tool-execution iterations.

        Returns
        -------
        AgentResult
        """
        effective_system = system or self._default_system
        effective_max_steps = max_steps if max_steps is not None else self._default_max_steps

        # Build initial messages
        messages: list[dict] = []
        if effective_system:
            messages.append({"role": "system", "content": effective_system})

        user_content = task
        if context:
            ctx_lines = ["=== CONTEXT ==="]
            for key, val in context.items():
                if isinstance(val, (dict, list)):
                    ctx_lines.append(f"{key}:\n{json.dumps(val, indent=2)}")
                else:
                    ctx_lines.append(f"{key}: {val}")
            ctx_lines.append("=== END CONTEXT ===\n")
            user_content = "\n".join(ctx_lines) + "\n" + task
        messages.append({"role": "user", "content": user_content})

        tools = self._registry.get_tool_definitions()
        steps: list[dict] = []
        skills_invoked: list[str] = []

        try:
            for step_idx in range(effective_max_steps):
                logger.debug("Agent step %d / %d", step_idx + 1, effective_max_steps)

                llm_response = self._llm.complete_with_tools(
                    messages=messages,
                    tools=tools if tools else None,
                )

                step_record: dict[str, Any] = {
                    "step": step_idx + 1,
                    "finish_reason": llm_response.finish_reason,
                    "content": llm_response.content,
                    "tool_calls": [],
                }

                # Append assistant message
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": llm_response.content,
                }
                if llm_response.tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.args),
                            },
                        }
                        for tc in llm_response.tool_calls
                    ]
                messages.append(assistant_msg)

                if not llm_response.tool_calls:
                    # No tool calls → final answer
                    steps.append(step_record)
                    return AgentResult(
                        output=llm_response.content,
                        steps=steps,
                        skills_invoked=skills_invoked,
                        success=True,
                    )

                # Execute tool calls and append results
                for tc in llm_response.tool_calls:
                    logger.info("Executing skill: %s", tc.name)
                    skills_invoked.append(tc.name)

                    result = self._registry.execute(tc.name, **tc.args)

                    tool_result_msg: dict[str, Any] = {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    }
                    messages.append(tool_result_msg)

                    step_record["tool_calls"].append(
                        {"name": tc.name, "args": tc.args, "result": result}
                    )

                steps.append(step_record)

            # Exhausted max_steps — return last assistant content
            last_content = ""
            for msg in reversed(messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    last_content = msg["content"]
                    break

            logger.warning(
                "Agent reached max_steps=%d without a final answer.", effective_max_steps
            )
            return AgentResult(
                output=last_content,
                steps=steps,
                skills_invoked=skills_invoked,
                success=True,
                metadata={"truncated": True},
            )

        except Exception as exc:
            logger.exception("AgentClient.run_task failed: %s", exc)
            return AgentResult(
                output="",
                steps=steps,
                skills_invoked=skills_invoked,
                success=False,
                error=str(exc),
            )
