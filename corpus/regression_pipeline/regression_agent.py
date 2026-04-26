"""
regression_agent.py — LLM agent that de-optimises a SOTA algorithm into its predecessor.

The agent is given the current algorithm and asked to produce a simpler (but
still functional) version, implementing one step of the top-down regression
that builds the evolutionary trajectory.
"""

from __future__ import annotations

import logging
import re

from etft.llm import LLMClient

logger = logging.getLogger(__name__)

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
    """Wraps an LLM to de-optimise one algorithm step."""

    def __init__(self, cfg: dict | None = None) -> None:
        self._llm = LLMClient(cfg)

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
        prompt = _USER_TEMPLATE.format(family=algorithm_family, fitness=fitness, code=code)
        logger.info("Calling regression agent for family=%s fitness=%.4f", algorithm_family, fitness)
        raw = self._llm.complete(prompt, system=_SYSTEM_PROMPT)
        predecessor_code = _extract_code(raw)
        logger.debug("Predecessor code length: %d chars", len(predecessor_code))
        return predecessor_code
