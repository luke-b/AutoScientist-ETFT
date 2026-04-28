"""
tests for etft/skills/synthesis_skill.py — EvolutionaryResearchSynthesisSkill
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(candidate_id="c1", triage_passed=True, risk_score=0.1):
    from corpus.regression_pipeline.schemas import SOTAPlusOneCandidate

    return SOTAPlusOneCandidate(
        candidate_id=candidate_id,
        trajectory_id="traj1",
        code="def model(x): return x",
        rationale="test rationale",
        risk_score=risk_score,
        triage_passed=triage_passed,
    )


def _make_brief():
    from corpus.regression_pipeline.schemas import ResearchBrief

    return ResearchBrief(
        bottleneck="batch_norm",
        query="batch norm deep learning",
        synthesis="Batch norm improves training.",
        hypotheses=["Use layer norm instead", "Try group norm"],
    )


def _make_experiment_result(success=True, experiment_id="e1"):
    from corpus.regression_pipeline.schemas import ExperimentResult

    return ExperimentResult(
        experiment_id=experiment_id,
        script="print('METRIC: accuracy=0.9')",
        success=success,
        metrics={"accuracy": 0.9} if success else {},
        error_message=None if success else "timeout",
    )


# ---------------------------------------------------------------------------
# Test 1: skill has correct metadata
# ---------------------------------------------------------------------------


def test_synthesis_skill_metadata():
    from etft.skills.synthesis_skill import EvolutionaryResearchSynthesisSkill

    skill = EvolutionaryResearchSynthesisSkill()
    assert skill.name == "run_evolutionary_synthesis"
    assert "sota_code" in skill.parameters_schema["properties"]
    assert "bottleneck" in skill.parameters_schema["properties"]
    assert "trajectory_steps" in skill.parameters_schema["properties"]
    assert "max_hypotheses" in skill.parameters_schema["properties"]
    assert skill.parameters_schema["required"] == ["sota_code", "bottleneck"]


# ---------------------------------------------------------------------------
# Test 2: to_tool_definition shape
# ---------------------------------------------------------------------------


def test_synthesis_skill_tool_definition():
    from etft.skills.synthesis_skill import EvolutionaryResearchSynthesisSkill

    skill = EvolutionaryResearchSynthesisSkill()
    td = skill.to_tool_definition()
    assert td["type"] == "function"
    assert td["function"]["name"] == "run_evolutionary_synthesis"
    assert "parameters" in td["function"]


# ---------------------------------------------------------------------------
# Test 3: full execute path with mocked internals
# ---------------------------------------------------------------------------


def test_synthesis_skill_execute_success(tmp_path):
    """execute() returns the correct structured dict when all phases succeed."""
    from etft.skills.synthesis_skill import EvolutionaryResearchSynthesisSkill

    brief = _make_brief()
    candidate = _make_candidate()
    exp_result = _make_experiment_result(success=True)
    collector_dict = {"total_experiments": 2, "success_rate": 1.0, "metrics": {}}

    with (
        patch("agents.literature.searcher.LiteratureSearcher.search", return_value=[]),
        patch("agents.literature.retriever.LiteratureRetriever.retrieve_and_chunk", return_value=[]),
        patch("agents.literature.synthesizer.LiteratureSynthesizer.synthesize", return_value=brief),
        patch("agents.literature.vector_store.build_vector_store", return_value=MagicMock()),
        patch("agents.empirical.experiment_designer.ExperimentDesigner.design", return_value="print('METRIC: x=1')"),
        patch("agents.empirical.runner.ExperimentRunner.run", return_value=exp_result),
        patch("agents.empirical.metrics_collector.MetricsCollector.to_dict", return_value=collector_dict),
        patch("feedback.rl_loop.feedback_router.FeedbackRouter.build_rl_context_prefix", return_value=""),
        patch("feedback.rl_loop.feedback_router.FeedbackRouter.route_experiment_failure"),
        patch("synthesis.sota_plus_one.synthesizer.SOTAPlusOneSynthesizer.generate", return_value=[candidate]),
        patch("synthesis.sota_plus_one.triage.TriageFilter.evaluate", return_value=candidate),
    ):
        skill = EvolutionaryResearchSynthesisSkill(cfg={}, data_root=str(tmp_path))
        result = skill.execute(sota_code="def model(x): return x", bottleneck="batch_norm")

    assert "candidates" in result
    assert "brief" in result
    assert "experiment_metrics" in result
    assert "pareto_bottlenecks" in result
    assert "rl_context_length" in result
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["candidate_id"] == "c1"
    assert result["brief"]["bottleneck"] == "batch_norm"


# ---------------------------------------------------------------------------
# Test 4: execute handles experiment design failure gracefully
# ---------------------------------------------------------------------------


def test_synthesis_skill_execute_design_failure(tmp_path):
    """Experiment design failures are caught; pipeline continues."""
    from etft.skills.synthesis_skill import EvolutionaryResearchSynthesisSkill

    brief = _make_brief()
    candidate = _make_candidate()
    collector_dict = {"total_experiments": 0, "success_rate": 0.0, "metrics": {}}

    with (
        patch("agents.literature.searcher.LiteratureSearcher.search", return_value=[]),
        patch("agents.literature.retriever.LiteratureRetriever.retrieve_and_chunk", return_value=[]),
        patch("agents.literature.synthesizer.LiteratureSynthesizer.synthesize", return_value=brief),
        patch("agents.literature.vector_store.build_vector_store", return_value=MagicMock()),
        patch(
            "agents.empirical.experiment_designer.ExperimentDesigner.design",
            side_effect=ValueError("design failed"),
        ),
        patch("agents.empirical.metrics_collector.MetricsCollector.to_dict", return_value=collector_dict),
        patch("feedback.rl_loop.feedback_router.FeedbackRouter.build_rl_context_prefix", return_value=""),
        patch("synthesis.sota_plus_one.synthesizer.SOTAPlusOneSynthesizer.generate", return_value=[candidate]),
        patch("synthesis.sota_plus_one.triage.TriageFilter.evaluate", return_value=candidate),
    ):
        skill = EvolutionaryResearchSynthesisSkill(cfg={}, data_root=str(tmp_path))
        result = skill.execute(sota_code="def model(x): return x", bottleneck="batch_norm")

    # Pipeline still completes despite design failure
    assert "candidates" in result
    assert len(result["candidates"]) == 1


# ---------------------------------------------------------------------------
# Test 5: Pareto analysis is run when trajectory_steps are supplied
# ---------------------------------------------------------------------------


def test_synthesis_skill_pareto_analysis(tmp_path):
    """When trajectory_steps are provided, Pareto analysis is called."""
    from etft.skills.synthesis_skill import EvolutionaryResearchSynthesisSkill

    brief = _make_brief()
    candidate = _make_candidate()
    collector_dict = {"total_experiments": 0, "success_rate": 0.0, "metrics": {}}

    steps = [
        {
            "step_index": 0,
            "algorithm_family": "cnn",
            "code": "def f(): pass",
            "fitness_score": 0.5,
        },
        {
            "step_index": 1,
            "algorithm_family": "cnn",
            "code": "def f():\n    pass\n    pass\n    pass\n    pass\n    pass",
            "fitness_score": 0.8,
        },
    ]

    pareto_result = ["batch_normalisation"]

    with (
        patch("agents.literature.searcher.LiteratureSearcher.search", return_value=[]),
        patch("agents.literature.retriever.LiteratureRetriever.retrieve_and_chunk", return_value=[]),
        patch("agents.literature.synthesizer.LiteratureSynthesizer.synthesize", return_value=brief),
        patch("agents.literature.vector_store.build_vector_store", return_value=MagicMock()),
        patch("agents.empirical.experiment_designer.ExperimentDesigner.design", side_effect=ValueError("skip")),
        patch("agents.empirical.metrics_collector.MetricsCollector.to_dict", return_value=collector_dict),
        patch("feedback.rl_loop.feedback_router.FeedbackRouter.build_rl_context_prefix", return_value=""),
        patch("synthesis.sota_plus_one.synthesizer.SOTAPlusOneSynthesizer.generate", return_value=[candidate]),
        patch("synthesis.sota_plus_one.triage.TriageFilter.evaluate", return_value=candidate),
        patch.object(
            EvolutionaryResearchSynthesisSkill,
            "_run_pareto_analysis",
            return_value=pareto_result,
        ) as mock_pareto,
    ):
        skill = EvolutionaryResearchSynthesisSkill(cfg={}, data_root=str(tmp_path))
        result = skill.execute(
            sota_code="def model(x): return x",
            bottleneck="batch_norm",
            trajectory_steps=steps,
        )

    mock_pareto.assert_called_once()
    assert result["pareto_bottlenecks"] == pareto_result


# ---------------------------------------------------------------------------
# Test 6: RL context length is captured
# ---------------------------------------------------------------------------


def test_synthesis_skill_rl_context_length(tmp_path):
    """rl_context_length reflects the length of the accumulated RL prefix."""
    from etft.skills.synthesis_skill import EvolutionaryResearchSynthesisSkill

    brief = _make_brief()
    candidate = _make_candidate()
    collector_dict = {"total_experiments": 0, "success_rate": 0.0, "metrics": {}}
    fake_prefix = "=== IN-CONTEXT RL FEEDBACK ===\nsome failure\n=== END ==="

    with (
        patch("agents.literature.searcher.LiteratureSearcher.search", return_value=[]),
        patch("agents.literature.retriever.LiteratureRetriever.retrieve_and_chunk", return_value=[]),
        patch("agents.literature.synthesizer.LiteratureSynthesizer.synthesize", return_value=brief),
        patch("agents.literature.vector_store.build_vector_store", return_value=MagicMock()),
        patch("agents.empirical.experiment_designer.ExperimentDesigner.design", side_effect=ValueError("skip")),
        patch("agents.empirical.metrics_collector.MetricsCollector.to_dict", return_value=collector_dict),
        patch(
            "feedback.rl_loop.feedback_router.FeedbackRouter.build_rl_context_prefix",
            return_value=fake_prefix,
        ),
        patch("synthesis.sota_plus_one.synthesizer.SOTAPlusOneSynthesizer.generate", return_value=[candidate]),
        patch("synthesis.sota_plus_one.triage.TriageFilter.evaluate", return_value=candidate),
    ):
        skill = EvolutionaryResearchSynthesisSkill(cfg={}, data_root=str(tmp_path))
        result = skill.execute(sota_code="def model(x): return x", bottleneck="batch_norm")

    assert result["rl_context_length"] == len(fake_prefix)


# ---------------------------------------------------------------------------
# Test 7: max_hypotheses limits experiments
# ---------------------------------------------------------------------------


def test_synthesis_skill_max_hypotheses(tmp_path):
    """max_hypotheses=1 only tries to design one experiment."""
    from etft.skills.synthesis_skill import EvolutionaryResearchSynthesisSkill

    brief = _make_brief()  # has 2 hypotheses
    candidate = _make_candidate()
    collector_dict = {"total_experiments": 0, "success_rate": 0.0, "metrics": {}}

    design_calls = []

    def fake_design(brief_arg, hypothesis_index=0, rl_context=""):
        design_calls.append(hypothesis_index)
        return "print('METRIC: x=1')"

    exp_result = _make_experiment_result(success=True)

    with (
        patch("agents.literature.searcher.LiteratureSearcher.search", return_value=[]),
        patch("agents.literature.retriever.LiteratureRetriever.retrieve_and_chunk", return_value=[]),
        patch("agents.literature.synthesizer.LiteratureSynthesizer.synthesize", return_value=brief),
        patch("agents.literature.vector_store.build_vector_store", return_value=MagicMock()),
        patch("agents.empirical.experiment_designer.ExperimentDesigner.design", side_effect=fake_design),
        patch("agents.empirical.runner.ExperimentRunner.run", return_value=exp_result),
        patch("agents.empirical.metrics_collector.MetricsCollector.to_dict", return_value=collector_dict),
        patch("feedback.rl_loop.feedback_router.FeedbackRouter.build_rl_context_prefix", return_value=""),
        patch("synthesis.sota_plus_one.synthesizer.SOTAPlusOneSynthesizer.generate", return_value=[candidate]),
        patch("synthesis.sota_plus_one.triage.TriageFilter.evaluate", return_value=candidate),
    ):
        skill = EvolutionaryResearchSynthesisSkill(cfg={}, data_root=str(tmp_path))
        skill.execute(sota_code="def model(x): return x", bottleneck="batch_norm", max_hypotheses=1)

    assert len(design_calls) == 1
