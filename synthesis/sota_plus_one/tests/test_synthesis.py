"""
tests/test_synthesis.py — Unit tests for synthesis/sota_plus_one.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from corpus.regression_pipeline.schemas import ResearchBrief, SOTAPlusOneCandidate
from synthesis.sota_plus_one.synthesizer import SOTAPlusOneSynthesizer, _extract_code_and_rationale
from synthesis.sota_plus_one.triage import TriageFilter

# ---------------------------------------------------------------------------
# _extract_code_and_rationale
# ---------------------------------------------------------------------------


def test_extract_with_both_sections():
    text = (
        "IMPLEMENTATION:\n```python\ndef foo(): pass\n```\n\n"
        "RATIONALE:\nAdded dropout for regularisation."
    )
    code, rationale = _extract_code_and_rationale(text)
    assert "def foo" in code
    assert "dropout" in rationale


def test_extract_no_code_block():
    text = "No code here."
    code, rationale = _extract_code_and_rationale(text)
    assert code == ""
    assert rationale == text


# ---------------------------------------------------------------------------
# SOTAPlusOneSynthesizer
# ---------------------------------------------------------------------------


def _make_brief() -> ResearchBrief:
    return ResearchBrief(
        bottleneck="dropout",
        query="dropout regularisation",
        synthesis="Dropout reduces overfitting.",
        hypotheses=["Add Dropout(0.3) after linear layers."],
    )


def test_synthesizer_generates_candidates():
    from etft.skills.base import AgentResult

    synth = SOTAPlusOneSynthesizer.__new__(SOTAPlusOneSynthesizer)
    synth.max_candidates = 2
    synth._agent = MagicMock()
    synth._agent.run_task.return_value = AgentResult(
        output="IMPLEMENTATION:\n```python\ndef model(): pass\n```\n\nRATIONALE:\nAdded dropout.",
        steps=[],
        skills_invoked=[],
        success=True,
    )

    candidates = synth.generate(
        sota_code="def old_model(): pass",
        trajectory_id="t1",
        bottleneck="dropout",
        brief=_make_brief(),
    )

    assert len(candidates) == 2
    assert all(c.triage_passed for c in candidates)  # default risk = 0.0
    assert synth._agent.run_task.call_count == 2


def test_synthesizer_skips_empty_code():
    from etft.skills.base import AgentResult

    synth = SOTAPlusOneSynthesizer.__new__(SOTAPlusOneSynthesizer)
    synth.max_candidates = 1
    synth._agent = MagicMock()
    synth._agent.run_task.return_value = AgentResult(
        output="No code here.",
        steps=[],
        skills_invoked=[],
        success=True,
    )

    candidates = synth.generate("def x(): pass", "t1", "x", _make_brief())
    assert candidates == []


# ---------------------------------------------------------------------------
# TriageFilter
# ---------------------------------------------------------------------------


def test_triage_pass_when_no_model(tmp_path):
    filt = TriageFilter(cfg=None, model_path=tmp_path / "nonexistent.joblib")
    candidate = SOTAPlusOneCandidate(
        candidate_id="c1",
        trajectory_id="t1",
        code="x = 1\n",
        rationale="test",
    )
    result = filt.evaluate(candidate)
    assert result.triage_passed
    assert result.risk_score == pytest.approx(0.0)


def test_triage_reject_above_threshold(tmp_path):
    filt = TriageFilter(
        cfg={"synthesis": {"triage": {"risk_threshold": 0.5}}},
        model_path=tmp_path / "nonexistent.joblib",
    )
    # Patch the internal filter to return high risk
    filt._filter = MagicMock()
    filt._filter.predict_failure_probability.return_value = 0.9

    candidate = SOTAPlusOneCandidate(
        candidate_id="c2", trajectory_id="t1", code="x = 1\n", rationale="test"
    )
    result = filt.evaluate(candidate)
    assert not result.triage_passed
    assert result.risk_score == pytest.approx(0.9)


def test_triage_pass_below_threshold(tmp_path):
    filt = TriageFilter(
        cfg={"synthesis": {"triage": {"risk_threshold": 0.7}}},
        model_path=tmp_path / "nonexistent.joblib",
    )
    filt._filter = MagicMock()
    filt._filter.predict_failure_probability.return_value = 0.3

    candidate = SOTAPlusOneCandidate(
        candidate_id="c3", trajectory_id="t1", code="x = 1\n", rationale="test"
    )
    result = filt.evaluate(candidate)
    assert result.triage_passed


# ---------------------------------------------------------------------------
# SOTAPlusOneSynthesizer — rl_context injection
# ---------------------------------------------------------------------------


def test_synthesizer_rl_context_forwarded():
    """rl_context is passed as 'rl_feedback' in the agent context dict."""
    from etft.skills.base import AgentResult

    synth = SOTAPlusOneSynthesizer.__new__(SOTAPlusOneSynthesizer)
    synth.max_candidates = 1
    synth._agent = MagicMock()
    synth._agent.run_task.return_value = AgentResult(
        output="IMPLEMENTATION:\n```python\ndef model(): pass\n```\n\nRATIONALE:\nTest.",
        steps=[],
        skills_invoked=[],
        success=True,
    )

    rl_signal = "=== IN-CONTEXT RL FEEDBACK ===\nsome rejection\n=== END FEEDBACK ===\n"
    brief = ResearchBrief(
        bottleneck="dropout",
        query="dropout",
        synthesis="Dropout reduces overfitting.",
        hypotheses=["Add Dropout(0.3)."],
    )
    synth.generate("def old(): pass", "t1", "dropout", brief, rl_context=rl_signal)

    call_kwargs = synth._agent.run_task.call_args.kwargs
    assert "rl_feedback" in call_kwargs.get("context", {}), (
        "rl_context must be forwarded as 'rl_feedback' in the context dict"
    )
    assert rl_signal in call_kwargs["context"]["rl_feedback"]


def test_synthesizer_no_rl_context_not_injected():
    """Empty rl_context must not add 'rl_feedback' key to the context dict."""
    from etft.skills.base import AgentResult

    synth = SOTAPlusOneSynthesizer.__new__(SOTAPlusOneSynthesizer)
    synth.max_candidates = 1
    synth._agent = MagicMock()
    synth._agent.run_task.return_value = AgentResult(
        output="IMPLEMENTATION:\n```python\ndef m(): pass\n```\n\nRATIONALE:\nTest.",
        steps=[],
        skills_invoked=[],
        success=True,
    )
    brief = ResearchBrief(bottleneck="x", query="x", synthesis="s", hypotheses=["h"])
    synth.generate("def old(): pass", "t1", "x", brief)

    call_kwargs = synth._agent.run_task.call_args.kwargs
    assert "rl_feedback" not in call_kwargs.get("context", {})


# ---------------------------------------------------------------------------
# ClusterSubmissionAdapter
# ---------------------------------------------------------------------------


_SIMPLE_SCRIPT = "print('METRIC: accuracy=0.8')\n"
_FAILING_SCRIPT = "raise RuntimeError('GPU OOM')\n"


def _make_candidate(code: str = _SIMPLE_SCRIPT) -> SOTAPlusOneCandidate:
    return SOTAPlusOneCandidate(
        candidate_id="cand01",
        trajectory_id="traj01",
        code=code,
        rationale="test candidate",
    )


def test_local_adapter_success_simple():
    """LocalSubprocessAdapter returns a job_id string when the script succeeds."""
    from synthesis.sota_plus_one.cluster_adapter import LocalSubprocessAdapter

    adapter = LocalSubprocessAdapter()
    job_id = adapter.submit(_make_candidate(_SIMPLE_SCRIPT))
    assert isinstance(job_id, str)
    assert "cand01" in job_id


def test_local_adapter_failure_raises_simple():
    """LocalSubprocessAdapter raises RuntimeError when the script fails."""
    from synthesis.sota_plus_one.cluster_adapter import LocalSubprocessAdapter

    adapter = LocalSubprocessAdapter()
    with pytest.raises(RuntimeError):
        adapter.submit(_make_candidate(_FAILING_SCRIPT))


def test_slurm_adapter_raises_on_missing_sbatch():
    """SlurmAdapter raises RuntimeError (not NotImplementedError) when sbatch is absent."""
    from unittest.mock import patch

    from synthesis.sota_plus_one.cluster_adapter import SlurmAdapter

    adapter = SlurmAdapter()
    with patch("subprocess.run", side_effect=FileNotFoundError("sbatch not found")):
        with pytest.raises(RuntimeError, match="sbatch not found"):
            adapter.submit(_make_candidate())


def test_kubernetes_adapter_raises_on_missing_package():
    """KubernetesAdapter raises ImportError when the kubernetes package is absent."""
    import sys
    from unittest.mock import patch

    from synthesis.sota_plus_one.cluster_adapter import KubernetesAdapter

    adapter = KubernetesAdapter()
    with patch.dict(sys.modules, {"kubernetes": None}):
        with pytest.raises((ImportError, TypeError)):
            adapter._k8s_clients()


# ---------------------------------------------------------------------------
# FeedbackRouter.route_physical_eval_failure
# ---------------------------------------------------------------------------


def test_physical_eval_failure_routed(tmp_path: Path) -> None:
    """route_physical_eval_failure persists a FailureRecord with reward=-2.0."""
    from feedback.rl_loop.feedback_router import FeedbackRouter

    router = FeedbackRouter(data_root=tmp_path)
    candidate = _make_candidate()
    record = router.route_physical_eval_failure(
        candidate,
        failure_reason="OOM on H100",
        also_rationale=True,
    )

    assert record.source == "physical_eval"
    assert record.reward_signal == pytest.approx(-2.0)
    assert record.candidate_id == "cand01"

    d_perf_file = tmp_path / "d_perf" / "feedback_failures.jsonl"
    assert d_perf_file.exists()
    import json
    line = json.loads(d_perf_file.read_text().strip().splitlines()[0])
    assert line["label"] == 1
    assert "OOM" in line["failure_reason"]


def test_physical_eval_failure_in_rl_context(tmp_path: Path) -> None:
    """route_physical_eval_failure contributes to the RL context prefix."""
    from feedback.rl_loop.feedback_router import FeedbackRouter

    router = FeedbackRouter(data_root=tmp_path)
    router.route_physical_eval_failure(_make_candidate(), failure_reason="divergent loss")

    prefix = router.build_rl_context_prefix()
    assert "physical_eval" in prefix
    assert "divergent loss" in prefix
    assert "reward=-2.00" in prefix


# ---------------------------------------------------------------------------
# ClusterSubmissionAdapter tests
# ---------------------------------------------------------------------------


def _make_cluster_candidate(code: str = "print('METRIC: acc=1.0')\n") -> SOTAPlusOneCandidate:
    return SOTAPlusOneCandidate(
        candidate_id="cand_test",
        trajectory_id="traj_test",
        code=code,
        rationale="Test candidate.",
        triage_passed=True,
    )


def test_local_adapter_success(tmp_path):
    """LocalSubprocessAdapter returns a job_id string on success."""
    from unittest.mock import MagicMock

    from synthesis.sota_plus_one.cluster_adapter import LocalSubprocessAdapter

    adapter = LocalSubprocessAdapter.__new__(LocalSubprocessAdapter)
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "METRIC: acc=1.0"
    mock_result.stderr = ""
    adapter._sandbox = MagicMock()
    adapter._sandbox.run_script.return_value = mock_result
    adapter._timeout = 60

    job_id = adapter.submit(_make_cluster_candidate())
    assert job_id.startswith("local-")


def test_local_adapter_failure_raises(tmp_path):
    """LocalSubprocessAdapter raises RuntimeError on non-zero exit code."""
    from synthesis.sota_plus_one.cluster_adapter import LocalSubprocessAdapter

    adapter = LocalSubprocessAdapter.__new__(LocalSubprocessAdapter)
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "RuntimeError: boom"
    adapter._sandbox = MagicMock()
    adapter._sandbox.run_script.return_value = mock_result
    adapter._timeout = 60

    import pytest
    with pytest.raises(RuntimeError, match="exit code 1"):
        adapter.submit(_make_cluster_candidate())


def test_local_adapter_timeout_raises():
    """LocalSubprocessAdapter raises RuntimeError on timeout (returncode=124)."""
    from synthesis.sota_plus_one.cluster_adapter import LocalSubprocessAdapter

    adapter = LocalSubprocessAdapter.__new__(LocalSubprocessAdapter)
    mock_result = MagicMock()
    mock_result.returncode = 124
    mock_result.stderr = ""
    adapter._sandbox = MagicMock()
    adapter._sandbox.run_script.return_value = mock_result
    adapter._timeout = 60

    import pytest
    with pytest.raises(RuntimeError, match="timed out"):
        adapter.submit(_make_cluster_candidate())


def test_slurm_adapter_no_sbatch_raises():
    """SlurmAdapter raises RuntimeError when sbatch is not on PATH."""
    from unittest.mock import patch

    from synthesis.sota_plus_one.cluster_adapter import SlurmAdapter

    adapter = SlurmAdapter()

    with patch(
        "subprocess.run",
        side_effect=FileNotFoundError("sbatch not found"),
    ):
        import pytest
        with pytest.raises(RuntimeError, match="sbatch not found"):
            adapter.submit(_make_cluster_candidate())


def test_slurm_adapter_sbatch_failure_raises():
    """SlurmAdapter raises RuntimeError when sbatch returns non-zero."""
    import subprocess
    from unittest.mock import patch

    from synthesis.sota_plus_one.cluster_adapter import SlurmAdapter

    adapter = SlurmAdapter()

    with patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "sbatch", stderr="Invalid partition"),
    ):
        import pytest
        with pytest.raises(RuntimeError, match="sbatch submission failed"):
            adapter.submit(_make_cluster_candidate())


def test_kubernetes_adapter_missing_package():
    """KubernetesAdapter._k8s_clients() raises ImportError when kubernetes not installed."""
    import sys
    from unittest.mock import patch

    from synthesis.sota_plus_one.cluster_adapter import KubernetesAdapter

    adapter = KubernetesAdapter()

    with patch.dict(sys.modules, {"kubernetes": None}):
        import pytest
        with pytest.raises((ImportError, TypeError)):
            adapter._k8s_clients()


def test_kubernetes_adapter_build_manifest_structure():
    """_build_job_manifest returns a valid batch/v1 Job manifest dict."""
    from synthesis.sota_plus_one.cluster_adapter import KubernetesAdapter

    adapter = KubernetesAdapter(cfg={
        "cluster": {
            "k8s": {
                "namespace": "test-ns",
                "image": "python:3.11",
                "gpu_count": 2,
                "memory_limit": "16Gi",
                "cpu_limit": "8",
            }
        }
    })

    manifest = adapter._build_job_manifest("test-job", "test-script-cm")

    assert manifest["kind"] == "Job"
    assert manifest["apiVersion"] == "batch/v1"
    assert manifest["metadata"]["name"] == "test-job"
    assert manifest["metadata"]["namespace"] == "test-ns"
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == "python:3.11"
    limits = container["resources"]["limits"]
    assert limits["nvidia.com/gpu"] == "2"
    assert limits["memory"] == "16Gi"
    assert "candidate.py" in container["command"][-1]


def test_kubernetes_adapter_zero_gpus_no_gpu_limit():
    """When gpu_count=0 the manifest must not include an nvidia.com/gpu limit."""
    from synthesis.sota_plus_one.cluster_adapter import KubernetesAdapter

    adapter = KubernetesAdapter(cfg={"cluster": {"k8s": {"gpu_count": 0}}})
    manifest = adapter._build_job_manifest("j", "cm")
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    assert "nvidia.com/gpu" not in container["resources"]["limits"]
