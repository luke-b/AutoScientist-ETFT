"""
agents/empirical/experiment_designer.py — LLM agent that designs a
micro-experiment from a ResearchBrief.

The coding-agent proxy receives the brief's hypotheses and generates a
self-contained Python script that can be executed to validate one hypothesis.
"""

from __future__ import annotations

import logging
import re

from corpus.regression_pipeline.schemas import ResearchBrief
from etft.agent import AgentClient
from etft.skills.code_skills import ParseMetricsSkill, ValidateCodeSkill
from etft.skills.experiment_skills import RunExperimentSkill

logger = logging.getLogger(__name__)

DEFAULT_SKILLS = [ValidateCodeSkill(), RunExperimentSkill(), ParseMetricsSkill()]

_SYSTEM = (
    "You are an expert ML engineer. "
    "Write a self-contained Python micro-experiment script that validates a specific hypothesis. "
    "The script must:\n"
    "  1. Run to completion in under 2 minutes on a CPU.\n"
    "  2. Print exactly one line: 'METRIC: <name>=<value>' for each measured metric.\n"
    "  3. Use only packages from this whitelist: numpy, scipy, sklearn, torch (CPU only).\n"
    "  4. NOT download large datasets — use synthetic data only.\n"
    "Respond with ONLY the Python source code, in ```python ... ``` fences."
)

_TEMPLATE = """\
Bottleneck: {bottleneck}
Synthesis: {synthesis}

Hypotheses:
{hypotheses}

Target hypothesis to test: {target_hypothesis}

Write a micro-experiment that validates this hypothesis using synthetic data.
"""


def _extract_code(text: str) -> str:
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip().removeprefix("```python").removesuffix("```").strip()


class ExperimentDesigner:
    """Generates runnable micro-experiment scripts from a ResearchBrief."""

    def __init__(self, cfg: dict | None = None) -> None:
        self._agent = AgentClient(cfg, skills=DEFAULT_SKILLS)
        emp_cfg = (cfg or {}).get("agents", {}).get("empirical", {})
        self.max_script_size: int = int(emp_cfg.get("max_script_size_bytes", 65536))

    # ------------------------------------------------------------------
    def design(
        self,
        brief: ResearchBrief,
        hypothesis_index: int = 0,
        rl_context: str = "",
    ) -> str:
        """
        Generate a micro-experiment script for the *hypothesis_index*-th
        hypothesis in *brief*.

        Parameters
        ----------
        brief:
            ResearchBrief produced by the literature synthesis agent.
        hypothesis_index:
            Index of the target hypothesis within *brief.hypotheses*.
        rl_context:
            Optional in-context RL feedback string (formatted negative reward
            signals from prior failed experiments in this campaign).  When
            provided it is prepended to the agent context so the LLM avoids
            repeating previously observed failure patterns.

        Returns
        -------
        str
            Python source code of the experiment script.
        """
        hypotheses = brief.hypotheses
        if not hypotheses:
            raise ValueError("ResearchBrief contains no hypotheses.")
        idx = min(hypothesis_index, len(hypotheses) - 1)
        target = hypotheses[idx]

        hypothesis_list = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(hypotheses))
        task = (
            f"Write a self-contained Python micro-experiment script that validates this "
            f"hypothesis: '{target}'. The script must run in under 2 minutes on CPU, print "
            f"'METRIC: <name>=<value>' for each metric, use only whitelisted packages "
            f"(numpy, scipy, sklearn, torch CPU only), and use synthetic data only. "
            f"Respond with ONLY the Python source code in ```python ... ``` fences."
        )

        context: dict = {
            "bottleneck": brief.bottleneck,
            "synthesis": brief.synthesis,
            "all_hypotheses": hypothesis_list,
        }
        if rl_context:
            context["rl_feedback"] = rl_context

        logger.info("Designing experiment for hypothesis: %r", target)
        result = self._agent.run_task(
            task=task,
            context=context,
            system=_SYSTEM,
        )
        script = _extract_code(result.output)

        if len(script.encode()) > self.max_script_size:
            logger.warning("Generated script exceeds max size; truncating.")
            script = script[: self.max_script_size]

        return script
