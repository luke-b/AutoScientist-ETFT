"""
etft/agent_factory.py — Factory for creating a fully-equipped research agent.

``create_research_agent(cfg)`` instantiates an :class:`~etft.agent.AgentClient`
pre-loaded with every domain skill available in the ETFT framework.  Callers
only need this one function to get an agent capable of running the full
research loop from the CLI or from application code.

Usage
-----
    from etft.agent_factory import create_research_agent
    agent = create_research_agent(cfg)
    result = agent.run_task("Improve the batch normalisation component.")
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def create_research_agent(
    cfg: dict | None = None,
    data_root: str | Path | None = None,
):
    """
    Create an :class:`~etft.agent.AgentClient` with all ETFT domain skills.

    Parameters
    ----------
    cfg:
        Runtime configuration dict (from ``config.yaml``).  When *None* an
        empty dict is used and module defaults apply.
    data_root:
        Optional root data directory forwarded to data-root-sensitive skills
        (FeedbackRouter, FeedbackSkills).  Defaults to
        ``cfg['data']['root']`` or ``./data``.

    Returns
    -------
    AgentClient
        A fully configured agent with all 16 domain skills registered.
    """
    from etft.agent import AgentClient
    from etft.skills.analysis_skills import (
        ComputeTrajectoryDeltasSkill,
        GenerateBottleneckReportSkill,
        RankParetoSkill,
    )
    from etft.skills.code_skills import (
        ExecuteScriptSkill,
        ExtractCodeFeaturesSkill,
        ParseMetricsSkill,
        SyntaxCheckSkill,
        ValidateCodeSkill,
    )
    from etft.skills.experiment_skills import (
        CollectExperimentMetricsSkill,
        RunExperimentSkill,
    )
    from etft.skills.feedback_skills import (
        BuildRLContextSkill,
        RouteExperimentFailureSkill,
        RouteTriageFailureSkill,
    )
    from etft.skills.filter_skills import (
        TrainPerformanceFilterSkill,
        TriageCandidateSkill,
    )
    from etft.skills.literature_skills import (
        FetchAndChunkPapersSkill,
        SearchArxivSkill,
    )
    from etft.skills.synthesis_skill import EvolutionaryResearchSynthesisSkill

    effective_cfg = cfg or {}

    # Resolve data_root: explicit arg → config key → default
    if data_root is not None:
        dr = str(data_root)
    else:
        dr = effective_cfg.get("data", {}).get("root", "./data")

    skills = [
        # Literature
        SearchArxivSkill(cfg=effective_cfg),
        FetchAndChunkPapersSkill(cfg=effective_cfg),
        # Code / validation
        ValidateCodeSkill(cfg=effective_cfg),
        ExtractCodeFeaturesSkill(),
        ExecuteScriptSkill(cfg=effective_cfg),
        ParseMetricsSkill(),
        SyntaxCheckSkill(),
        # Experiments
        RunExperimentSkill(cfg=effective_cfg),
        CollectExperimentMetricsSkill(),
        # Analysis
        ComputeTrajectoryDeltasSkill(),
        RankParetoSkill(),
        GenerateBottleneckReportSkill(),
        # Feedback / RL
        RouteExperimentFailureSkill(cfg=effective_cfg, data_root=dr),
        RouteTriageFailureSkill(cfg=effective_cfg, data_root=dr),
        BuildRLContextSkill(),
        # Filter / triage
        TrainPerformanceFilterSkill(cfg=effective_cfg),
        TriageCandidateSkill(cfg=effective_cfg),
        # Top-level synthesis skill
        EvolutionaryResearchSynthesisSkill(cfg=effective_cfg, data_root=dr),
    ]

    logger.debug(
        "create_research_agent: registering %d skills: %s",
        len(skills),
        [s.name for s in skills],
    )
    return AgentClient(cfg=effective_cfg, skills=skills)
