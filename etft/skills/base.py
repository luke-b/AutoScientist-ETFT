"""
etft/skills/base.py — Foundation types for the Skills + AgentClient architecture.

All domain skills inherit from Skill.  AgentClient uses SkillRegistry to
dispatch tool calls returned by the LLM.  LLMResponse and AgentResult carry
the structured results of each reasoning step and the full task, respectively.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Skill base class
# ---------------------------------------------------------------------------


class Skill(ABC):
    """
    Abstract base for all ETFT skills.

    Subclasses MUST define class-level attributes:
        name: str                  — unique tool name (snake_case)
        description: str           — human-readable description for the LLM
        parameters_schema: dict    — JSON Schema object for the parameters

    Skills NEVER call LLMClient.  They are pure Python computation.
    """

    name: str
    description: str
    parameters_schema: dict

    @abstractmethod
    def execute(self, **kwargs: Any) -> dict:
        """
        Execute the skill with the given keyword arguments.

        Returns a JSON-serialisable dict that is appended to the agent
        conversation as a tool result message.
        """

    def to_tool_definition(self) -> dict:
        """Return the OpenAI tool-definition dict for this skill."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }


# ---------------------------------------------------------------------------
# Skill registry
# ---------------------------------------------------------------------------


class SkillRegistry:
    """Central registry that maps skill names to Skill instances."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """Register *skill* under its ``name``."""
        self._skills[skill.name] = skill
        logger.debug("Registered skill: %s", skill.name)

    def get(self, name: str) -> Skill | None:
        """Return the skill with *name*, or None if not found."""
        return self._skills.get(name)

    def get_tool_definitions(self) -> list[dict]:
        """Return a list of OpenAI tool-definition dicts for all skills."""
        return [skill.to_tool_definition() for skill in self._skills.values()]

    def execute(self, name: str, **kwargs: Any) -> dict:
        """
        Execute the named skill with *kwargs*.

        Returns a result dict; on error returns ``{"error": <message>}``.
        """
        skill = self.get(name)
        if skill is None:
            return {"error": f"Unknown skill: {name!r}"}
        try:
            return skill.execute(**kwargs)
        except Exception as exc:
            logger.exception("Skill %r raised: %s", name, exc)
            return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Data-transfer types
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """A single tool-call request returned by the LLM."""

    id: str
    name: str
    args: dict


@dataclass
class LLMResponse:
    """Parsed response from the LLM proxy."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"


@dataclass
class AgentResult:
    """Final result produced by AgentClient.run_task."""

    output: str
    steps: list[dict]            # Full reasoning + tool-call history
    skills_invoked: list[str]    # Names of skills called during the run
    success: bool
    error: str | None = None
    metadata: dict = field(default_factory=dict)
