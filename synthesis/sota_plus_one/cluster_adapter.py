"""
synthesis/sota_plus_one/cluster_adapter.py — Pluggable cluster submission
adapters for physical GPU evaluation of SOTA+1 candidates.

Architecture
------------
  generate.py (--submit flag)
      │
      ▼
  ClusterSubmissionAdapter.submit(candidate) → job_id: str
      ├── LocalSubprocessAdapter   (default — runs via DockerSandbox)
      ├── SlurmAdapter             (stub — not yet implemented)
      └── KubernetesAdapter        (stub — not yet implemented)

On failure the caller catches the exception and routes it to
``FeedbackRouter.route_physical_eval_failure()`` so the negative signal
hardens the Probabilistic Heuristic Filter automatically.

Usage
-----
    from synthesis.sota_plus_one.cluster_adapter import LocalSubprocessAdapter

    adapter = LocalSubprocessAdapter(cfg)
    job_id = adapter.submit(candidate)   # raises on failure
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod

from corpus.regression_pipeline.schemas import SOTAPlusOneCandidate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ClusterSubmissionAdapter(ABC):
    """
    Abstract interface for submitting a SOTA+1 candidate to a compute cluster.

    Subclasses implement :meth:`submit` and return a job identifier string on
    success.  They MUST raise an exception on failure so the caller can route
    the failure to :class:`~feedback.rl_loop.feedback_router.FeedbackRouter`.
    """

    @abstractmethod
    def submit(self, candidate: SOTAPlusOneCandidate) -> str:
        """
        Submit *candidate* for physical evaluation.

        Parameters
        ----------
        candidate:
            A triage-passing SOTA+1 candidate.

        Returns
        -------
        str
            A unique job identifier string (cluster-specific format).

        Raises
        ------
        RuntimeError
            If the submission or execution fails.  The exception message is
            used as ``failure_reason`` in the resulting FailureRecord.
        """


# ---------------------------------------------------------------------------
# LocalSubprocessAdapter
# ---------------------------------------------------------------------------


class LocalSubprocessAdapter(ClusterSubmissionAdapter):
    """
    Evaluates a SOTA+1 candidate locally using the existing DockerSandbox.

    This is the default adapter for local development and CI.  It runs the
    candidate's code in the same sandboxed environment used for
    micro-experiments and raises ``RuntimeError`` if execution fails, allowing
    the feedback loop to capture the failure automatically.

    Parameters
    ----------
    cfg:
        Full runtime configuration dict (forwarded to DockerSandbox).
    """

    def __init__(self, cfg: dict | None = None) -> None:
        from etft.sandbox import DockerSandbox

        self._sandbox = DockerSandbox(cfg)
        emp_cfg = (cfg or {}).get("agents", {}).get("empirical", {})
        self._timeout: int = int(emp_cfg.get("experiment_timeout_seconds", 120))

    def submit(self, candidate: SOTAPlusOneCandidate) -> str:
        job_id = f"local-{candidate.candidate_id}-{str(uuid.uuid4())[:6]}"
        logger.info(
            "LocalSubprocessAdapter: submitting candidate %s (job_id=%s)",
            candidate.candidate_id, job_id,
        )

        result = self._sandbox.run_script(candidate.code, timeout=self._timeout)

        if result.returncode == 124:
            raise RuntimeError(
                f"Physical evaluation timed out after {self._timeout}s "
                f"(job_id={job_id})."
            )
        if result.returncode != 0:
            stderr_tail = (result.stderr or "")[-500:]
            raise RuntimeError(
                f"Physical evaluation failed with exit code {result.returncode} "
                f"(job_id={job_id}): {stderr_tail}"
            )

        logger.info(
            "LocalSubprocessAdapter: candidate %s completed successfully (job_id=%s).",
            candidate.candidate_id, job_id,
        )
        return job_id


# ---------------------------------------------------------------------------
# Stub adapters (not yet implemented)
# ---------------------------------------------------------------------------


class SlurmAdapter(ClusterSubmissionAdapter):
    """
    Submits SOTA+1 candidates to a SLURM HPC cluster.

    .. note::
        Not yet implemented.  Override :meth:`submit` with real ``sbatch``
        invocation logic when a SLURM cluster is available.
    """

    def submit(self, candidate: SOTAPlusOneCandidate) -> str:
        raise NotImplementedError(
            "SlurmAdapter.submit() is not yet implemented. "
            "Implement sbatch invocation and job-ID extraction here."
        )


class KubernetesAdapter(ClusterSubmissionAdapter):
    """
    Submits SOTA+1 candidates as Kubernetes Jobs.

    .. note::
        Not yet implemented.  Override :meth:`submit` with real
        ``kubernetes`` client calls (e.g. using the ``kubernetes`` Python
        package) when a cluster is available.
    """

    def submit(self, candidate: SOTAPlusOneCandidate) -> str:
        raise NotImplementedError(
            "KubernetesAdapter.submit() is not yet implemented. "
            "Implement k8s Job creation and status polling here."
        )
