"""
regression_agent.py — LLM agent that de-optimises a SOTA algorithm into its predecessor.

The agent is given the current algorithm and asked to produce a simpler (but
still functional) version, implementing one step of the top-down regression
that builds the evolutionary trajectory.
"""

from __future__ import annotations

import logging
import re

from etft.agent import AgentClient
from etft.skills.code_skills import SyntaxCheckSkill, ValidateCodeSkill

logger = logging.getLogger(__name__)

DEFAULT_SKILLS = [ValidateCodeSkill(), SyntaxCheckSkill()]

_SYSTEM_PROMPT = """You are an expert software engineer and machine learning researcher.
Your task is to reverse-engineer a given algorithm into a simpler, earlier version of itself.
The simplified version must:
  1. Solve the same problem.
  2. Be syntactically valid Python.
  3. Be measurably simpler — fewer parameters, fewer layers, or a less sophisticated optimisation strategy.
  4. NOT be a trivially empty or no-op implementation.
Respond with ONLY the Python source code, enclosed in ```python ... ``` fences.
"""

_USER_TEMPLATE = """Algorithm family: {family}
Current fitness score: {fitness:.4f}

=== CURRENT ALGORITHM ===
{code}

=== TASK ===
Produce a valid Python predecessor of this algorithm that is simpler/less optimised.
Return ONLY the Python source code.
"""


def _extract_code(text: str) -> str:
    """Extract the first ```python ... ``` block from the LLM output."""
    pattern = r"```python\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: strip fences if present
    return text.strip().removeprefix("```python").removesuffix("```").strip()


class RegressionAgent:
    """Wraps an LLM agent to de-optimise one algorithm step."""

    def __init__(self, cfg: dict | None = None) -> None:
        self._agent = AgentClient(cfg, skills=DEFAULT_SKILLS)

    def regress(self, code: str, algorithm_family: str, fitness: float) -> str:
        """
        Produce a predecessor algorithm for *code*.

        Parameters
        ----------
        code:
            Source code of the current algorithm.
        algorithm_family:
            Human-readable family label (e.g. 'image_classification_cnn').
        fitness:
            Fitness score of the current algorithm.

        Returns
        -------
        str
            Source code of the proposed predecessor algorithm.
        """
        task = (
            f"Produce a valid Python predecessor of the given algorithm that is simpler and "
            f"less optimised. The algorithm family is '{algorithm_family}' with fitness "
            f"{fitness:.4f}. The predecessor must solve the same problem, be syntactically "
            f"valid Python, and be measurably simpler. Respond with ONLY the Python source "
            f"code, enclosed in ```python ... ``` fences."
        )
        logger.info(
            "Calling regression agent for family=%s fitness=%.4f", algorithm_family, fitness
        )
        result = self._agent.run_task(
            task=task,
            context={"current_code": code, "system_instructions": _SYSTEM_PROMPT},
            system=_SYSTEM_PROMPT,
        )
        predecessor_code = _extract_code(result.output)
        logger.debug("Predecessor code length: %d chars", len(predecessor_code))
        return predecessor_code
