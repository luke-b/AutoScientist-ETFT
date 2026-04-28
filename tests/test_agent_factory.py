"""
tests for etft/agent_factory.py — create_research_agent()
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXPECTED_SKILL_NAMES = {
    "search_arxiv",
    "fetch_and_chunk_papers",
    "validate_code",
    "extract_code_features",
    "execute_script",
    "parse_metrics",
    "check_syntax",
    "run_experiment",
    "collect_metrics",
    "compute_trajectory_deltas",
    "rank_pareto",
    "generate_bottleneck_report",
    "route_experiment_failure",
    "route_triage_failure",
    "build_rl_context",
    "train_performance_filter",
    "triage_candidate",
    "run_evolutionary_synthesis",
}


# ---------------------------------------------------------------------------
# Test 1: factory returns an AgentClient with all skills registered
# ---------------------------------------------------------------------------


def test_create_research_agent_all_skills():
    """create_research_agent registers all 18 expected skills."""
    from etft.agent import AgentClient
    from etft.agent_factory import create_research_agent

    agent = create_research_agent(cfg={})
    assert isinstance(agent, AgentClient)

    registered = {td["function"]["name"] for td in agent._registry.get_tool_definitions()}
    assert registered == _EXPECTED_SKILL_NAMES


# ---------------------------------------------------------------------------
# Test 2: factory works with None cfg
# ---------------------------------------------------------------------------


def test_create_research_agent_none_cfg():
    """create_research_agent works with cfg=None."""
    from etft.agent_factory import create_research_agent

    agent = create_research_agent(cfg=None)
    registered = {td["function"]["name"] for td in agent._registry.get_tool_definitions()}
    assert "run_evolutionary_synthesis" in registered


# ---------------------------------------------------------------------------
# Test 3: data_root forwarded to feedback skills
# ---------------------------------------------------------------------------


def test_create_research_agent_data_root(tmp_path):
    """data_root parameter is accepted without error."""
    from etft.agent_factory import create_research_agent

    agent = create_research_agent(cfg={}, data_root=str(tmp_path))
    registered = {td["function"]["name"] for td in agent._registry.get_tool_definitions()}
    assert "route_experiment_failure" in registered
    assert "route_triage_failure" in registered


# ---------------------------------------------------------------------------
# Test 4: factory uses cfg data.root when data_root arg is omitted
# ---------------------------------------------------------------------------


def test_create_research_agent_cfg_data_root(tmp_path):
    """When data_root arg is None, cfg['data']['root'] is used."""
    cfg = {"data": {"root": str(tmp_path)}}
    from etft.agent_factory import create_research_agent

    agent = create_research_agent(cfg=cfg)
    registered = {td["function"]["name"] for td in agent._registry.get_tool_definitions()}
    assert len(registered) == len(_EXPECTED_SKILL_NAMES)


# ---------------------------------------------------------------------------
# Test 5: factory is importable from etft package root
# ---------------------------------------------------------------------------


def test_create_research_agent_importable_from_etft():
    """create_research_agent is exported from the etft package."""
    import etft

    assert hasattr(etft, "create_research_agent")
    assert callable(etft.create_research_agent)
