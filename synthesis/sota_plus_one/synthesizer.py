"""
synthesis/sota_plus_one/synthesizer.py — LLM agent that combines three evidence
streams to propose SOTA+1 algorithm candidates:

  1. Historical evolutionary trajectories (𝒟_Gen / 𝒟_Rationale)
  2. Live literature synthesis (ResearchBrief from ARL Phase 1)
  3. Empirical metrics (MetricsSummary from ARL Phase 2)
"""

from __future__ import annotations

import logging
import re
import uuid

from corpus.regression_pipeline.schemas import ResearchBrief, SOTAPlusOneCandidate
from etft.agent import AgentClient
from etft.skills.code_skills import ValidateCodeSkill
from etft.skills.filter_skills import TriageCandidateSkill

logger = logging.getLogger(__name__)

DEFAULT_SKILLS = [ValidateCodeSkill(), TriageCandidateSkill()]

_SYSTEM = (
    "You are an elite ML researcher and software engineer. "
    "Your task is to design SOTA+1: a next-generation algorithm that improves upon the "
    "current state-of-the-art by applying the insights from recent literature and "
    "empirical evidence. Respond with:\n"
    "1. A Python implementation of the proposed algorithm (```python ... ``` block).\n"
    "2. A concise rationale explaining what changed and why it should improve performance."
)

_TEMPLATE = """\
=== CURRENT SOTA (trajectory tail) ===
{sota_code}

=== BOTTLENECK (from 80/20 Δ analysis) ===
{bottleneck}

=== LITERATURE SYNTHESIS ===
{synthesis}

=== EMPIRICAL EVIDENCE ===
{empirical_summary}

=== TASK ===
Propose SOTA+1: an improved version of the current SOTA that addresses the bottleneck,
incorporates the literature findings, and is consistent with the empirical evidence.

Return:
IMPLEMENTATION:
```python
<code here>
```

RATIONALE:
<explanation here>
"""


def _extract_code_and_rationale(text: str) -> tuple[str, str]:
    code_match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    code = code_match.group(1).strip() if code_match else ""

    rationale = ""
    if "RATIONALE:" in text:
        rationale = text.split("RATIONALE:", 1)[1].strip()
    elif not code:
        rationale = text.strip()

    return code, rationale


class SOTAPlusOneSynthesizer:
    """Generates SOTA+1 candidates by fusing trajectory context with ARL outputs."""

    def __init__(self, cfg: dict | None = None) -> None:
        self._agent = AgentClient(cfg, skills=DEFAULT_SKILLS)
        synth_cfg = (cfg or {}).get("synthesis", {}).get("sota_plus_one", {})
        self.max_candidates: int = int(synth_cfg.get("max_candidates", 3))

    # ------------------------------------------------------------------
    def generate(
        self,
        sota_code: str,
        trajectory_id: str,
        bottleneck: str,
        brief: ResearchBrief,
        empirical_summary: dict | None = None,
        rl_context: str = "",
    ) -> list[SOTAPlusOneCandidate]:
        """
        Generate up to ``max_candidates`` SOTA+1 proposals.

        Parameters
        ----------
        sota_code:
            Source code of the current SOTA (top of the trajectory).
        trajectory_id:
            Identifier of the source trajectory.
        bottleneck:
            The bottleneck component from Pareto analysis.
        brief:
            ResearchBrief from the literature agent.
        empirical_summary:
            MetricsCollector.to_dict() output (or None).
        rl_context:
            Optional in-context RL feedback string containing negative reward
            signals from prior triage rejections or physical evaluation failures
            in the same campaign.  When provided it is prepended to the agent
            context so the synthesizer avoids repeating known failure patterns.
        """
        emp_text = _format_empirical(empirical_summary)

        candidates: list[SOTAPlusOneCandidate] = []
        for i in range(self.max_candidates):
            logger.info(
                "Generating SOTA+1 candidate %d / %d …", i + 1, self.max_candidates
            )
            task = (
                "Propose SOTA+1: an improved Python algorithm that addresses the bottleneck, "
                "incorporates the literature findings, and is consistent with the empirical "
                "evidence. Return:\nIMPLEMENTATION:\n```python\n<code here>\n```\n\n"
                "RATIONALE:\n<explanation here>"
            )
            context: dict = {
                "sota_code": sota_code[:3000],
                "bottleneck": bottleneck,
                "literature_synthesis": brief.synthesis,
                "empirical_summary": emp_text,
            }
            if rl_context:
                context["rl_feedback"] = rl_context

            result = self._agent.run_task(
                task=task,
                context=context,
                system=_SYSTEM,
            )
            code, rationale = _extract_code_and_rationale(result.output)

            if not code:
                logger.warning("Candidate %d produced no code block — skipping.", i + 1)
                continue

            candidates.append(
                SOTAPlusOneCandidate(
                    candidate_id=str(uuid.uuid4())[:8],
                    trajectory_id=trajectory_id,
                    code=code,
                    rationale=rationale,
                    provenance={
                        "bottleneck": bottleneck,
                        "hypotheses": brief.hypotheses,
                        "generation_index": i,
                    },
                )
            )

        logger.info("Generated %d valid SOTA+1 candidates.", len(candidates))
        return candidates


def _format_empirical(summary: dict | None) -> str:
    if not summary:
        return "No empirical data available."
    lines = [
        f"Experiments run: {summary.get('total_experiments', 0)}",
        f"Success rate: {summary.get('success_rate', 0):.0%}",
    ]
    for metric, stats in summary.get("metrics", {}).items():
        lines.append(f"  {metric}: mean={stats.get('mean', 0):.4f} ± {stats.get('stdev', 0):.4f}")
    return "\n".join(lines)
