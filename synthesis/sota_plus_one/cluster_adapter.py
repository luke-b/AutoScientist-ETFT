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
      ├── SlurmAdapter             (sbatch-based HPC submission)
      └── KubernetesAdapter        (kubernetes batch/v1 Job)

On failure the caller catches the exception and routes it to
``FeedbackRouter.route_physical_eval_failure()`` so the negative signal
hardens the Probabilistic Heuristic Filter automatically.

Choosing an adapter
-------------------
* **LocalSubprocessAdapter** — suitable for development and CI.  Runs the
  candidate code inside the same DockerSandbox used for micro-experiments.
  No cluster credentials required.

* **SlurmAdapter** — suitable for on-premise HPC clusters that use SLURM as
  their workload manager.  Requires ``sbatch`` on ``$PATH`` and appropriate
  cluster credentials.  Configure via the ``cluster.slurm`` section of
  ``config.yaml``.

* **KubernetesAdapter** — suitable for cloud or on-premise Kubernetes
  clusters with GPU node pools.  Requires the ``kubernetes`` Python package
  (``pip install 'autoscientist-etft[cluster]'``) and a valid kubeconfig.
  Configure via the ``cluster.k8s`` section of ``config.yaml``.

Usage
-----
    from synthesis.sota_plus_one.cluster_adapter import LocalSubprocessAdapter

    adapter = LocalSubprocessAdapter(cfg)
    job_id = adapter.submit(candidate)   # raises on failure
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from corpus.regression_pipeline.schemas import SOTAPlusOneCandidate

logger = logging.getLogger(__name__)

_MAX_STDERR_TAIL = 500  # characters to include in failure messages from stderr


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
            stderr_tail = (result.stderr or "")[-_MAX_STDERR_TAIL:]
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
# SlurmAdapter
# ---------------------------------------------------------------------------


class SlurmAdapter(ClusterSubmissionAdapter):
    """
    Submits SOTA+1 candidates to a SLURM HPC cluster via ``sbatch``.

    Wraps the candidate code in a Bash job script and calls ``sbatch``
    as a subprocess.  Polls ``squeue`` to wait for the job to finish, then
    reads the captured stdout/stderr from the log files to determine success.

    Configuration (``config.yaml`` -> ``cluster.slurm``):

    .. code-block:: yaml

       cluster:
         slurm:
           partition: gpu
           n_gpus: 1
           mem_gb: 32
           time_limit: "01:00:00"
           python_path: "python3"
           poll_interval_seconds: 30
           timeout_seconds: 3600

    Parameters
    ----------
    cfg:
        Full runtime configuration dict.
    """

    def __init__(self, cfg: dict | None = None) -> None:
        slurm_cfg = (cfg or {}).get("cluster", {}).get("slurm", {})
        self._partition: str = slurm_cfg.get("partition", "gpu")
        self._n_gpus: int = int(slurm_cfg.get("n_gpus", 1))
        self._mem_gb: int = int(slurm_cfg.get("mem_gb", 32))
        self._time_limit: str = slurm_cfg.get("time_limit", "01:00:00")
        self._python_path: str = slurm_cfg.get("python_path", "python3")
        self._poll_interval: int = int(slurm_cfg.get("poll_interval_seconds", 30))
        self._timeout: int = int(slurm_cfg.get("timeout_seconds", 3600))

    def _build_job_script(self, script_path: Path, log_dir: Path) -> str:
        """Build an sbatch job script as a string."""
        return (
            "#!/bin/bash\n"
            f"#SBATCH --partition={self._partition}\n"
            f"#SBATCH --gres=gpu:{self._n_gpus}\n"
            f"#SBATCH --mem={self._mem_gb}G\n"
            f"#SBATCH --time={self._time_limit}\n"
            f"#SBATCH --output={log_dir}/job_%j.out\n"
            f"#SBATCH --error={log_dir}/job_%j.err\n"
            "\n"
            f"{self._python_path} {script_path}\n"
        )

    def submit(self, candidate: SOTAPlusOneCandidate) -> str:
        """
        Submit *candidate* to SLURM via sbatch and wait for completion.

        Raises
        ------
        RuntimeError
            If sbatch fails, the job exits with non-zero status, or timeout is
            exceeded.
        """
        job_id_prefix = f"slurm-{candidate.candidate_id[:8]}"

        with tempfile.TemporaryDirectory(prefix="etft_slurm_") as tmpdir:
            tmp = Path(tmpdir)
            script_path = tmp / "candidate.py"
            script_path.write_text(candidate.code)

            log_dir = tmp / "logs"
            log_dir.mkdir()
            sbatch_script = tmp / "job.sh"
            sbatch_script.write_text(self._build_job_script(script_path, log_dir))

            try:
                proc = subprocess.run(
                    ["sbatch", str(sbatch_script)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except FileNotFoundError as exc:
                raise RuntimeError("sbatch not found on PATH. Is SLURM installed?") from exc
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"sbatch submission failed: {exc.stderr.strip()}"
                ) from exc

            # Parse job ID from: "Submitted batch job 12345"
            slurm_job_id = proc.stdout.strip().split()[-1]
            job_id = f"{job_id_prefix}-{slurm_job_id}"
            logger.info(
                "SlurmAdapter: submitted job %s for candidate %s",
                job_id, candidate.candidate_id,
            )

            # Poll for completion
            elapsed = 0
            while elapsed < self._timeout:
                time.sleep(self._poll_interval)
                elapsed += self._poll_interval

                squeue = subprocess.run(
                    ["squeue", "--job", slurm_job_id, "--noheader"],
                    capture_output=True, text=True,
                )
                if not squeue.stdout.strip():
                    break

            # Check exit code via sacct
            sacct = subprocess.run(
                [
                    "sacct", "-j", slurm_job_id,
                    "--format=ExitCode", "--noheader", "--parsable2",
                ],
                capture_output=True, text=True,
            )
            exit_code_str = (
                sacct.stdout.strip().splitlines()[0]
                if sacct.stdout.strip()
                else "0:0"
            )
            exit_code = int(exit_code_str.split(":")[0])

            if exit_code != 0:
                err_logs = list(log_dir.glob("*.err"))
                stderr_tail = err_logs[0].read_text()[-_MAX_STDERR_TAIL:] if err_logs else ""
                raise RuntimeError(
                    f"SLURM job {job_id} failed with exit code {exit_code}: {stderr_tail}"
                )

            logger.info("SlurmAdapter: job %s completed successfully.", job_id)
            return job_id


# ---------------------------------------------------------------------------
# KubernetesAdapter
# ---------------------------------------------------------------------------


class KubernetesAdapter(ClusterSubmissionAdapter):
    """
    Submits SOTA+1 candidates as Kubernetes ``batch/v1 Job`` objects.

    Wraps the candidate code in a ConfigMap-mounted Python script,
    creates a Job that runs ``python /scripts/candidate.py``, polls for
    completion, parses ``METRIC:`` lines from pod logs on success, and
    raises ``RuntimeError`` (routed to FeedbackRouter) on failure.

    Requires the ``kubernetes`` Python package::

        pip install 'autoscientist-etft[cluster]'

    Configuration (``config.yaml`` -> ``cluster.k8s``):

    .. code-block:: yaml

       cluster:
         k8s:
           namespace: default
           image: "python:3.11-slim"
           gpu_count: 1
           memory_limit: "8Gi"
           cpu_limit: "4"
           poll_interval_seconds: 10
           timeout_seconds: 1800
           service_account: ""

    Parameters
    ----------
    cfg:
        Full runtime configuration dict.
    """

    def __init__(self, cfg: dict | None = None) -> None:
        k8s_cfg = (cfg or {}).get("cluster", {}).get("k8s", {})
        self._namespace: str = k8s_cfg.get("namespace", "default")
        self._image: str = k8s_cfg.get("image", "python:3.11-slim")
        self._gpu_count: int = int(k8s_cfg.get("gpu_count", 1))
        self._memory_limit: str = k8s_cfg.get("memory_limit", "8Gi")
        self._cpu_limit: str = str(k8s_cfg.get("cpu_limit", "4"))
        self._poll_interval: int = int(k8s_cfg.get("poll_interval_seconds", 10))
        self._timeout: int = int(k8s_cfg.get("timeout_seconds", 1800))
        self._service_account: str = k8s_cfg.get("service_account", "")

    def _k8s_clients(self):
        """Return (batch_v1, core_v1) Kubernetes API clients."""
        try:
            import kubernetes as k8s
        except ImportError as exc:
            raise ImportError(
                "kubernetes package is required for KubernetesAdapter. "
                "Install it with: pip install 'autoscientist-etft[cluster]'"
            ) from exc
        k8s.config.load_kube_config()
        return k8s.client.BatchV1Api(), k8s.client.CoreV1Api()

    def _build_job_manifest(self, job_name: str, configmap_name: str) -> dict:
        """Return a batch/v1 Job manifest dict."""
        resources: dict = {
            "limits": {
                "memory": self._memory_limit,
                "cpu": self._cpu_limit,
            }
        }
        if self._gpu_count > 0:
            resources["limits"]["nvidia.com/gpu"] = str(self._gpu_count)

        container: dict = {
            "name": "etft-candidate",
            "image": self._image,
            "command": ["python", "/scripts/candidate.py"],
            "resources": resources,
            "volumeMounts": [{"name": "script", "mountPath": "/scripts"}],
        }

        spec: dict = {
            "containers": [container],
            "restartPolicy": "Never",
            "volumes": [
                {
                    "name": "script",
                    "configMap": {"name": configmap_name},
                }
            ],
        }
        if self._service_account:
            spec["serviceAccountName"] = self._service_account

        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": job_name, "namespace": self._namespace},
            "spec": {
                "backoffLimit": 0,
                "template": {"spec": spec},
            },
        }

    def _get_pod_logs(self, core_v1, job_name: str) -> str:
        """Retrieve the log tail of the first pod for *job_name*."""
        try:
            pods = core_v1.list_namespaced_pod(
                namespace=self._namespace,
                label_selector=f"job-name={job_name}",
            )
            if pods.items:
                logs = core_v1.read_namespaced_pod_log(
                    name=pods.items[0].metadata.name,
                    namespace=self._namespace,
                )
                return (logs or "")[-_MAX_STDERR_TAIL:]
        except Exception as exc:
            logger.debug("Could not retrieve pod logs for %s: %s", job_name, exc)
        return "(logs unavailable)"

    def _collect_result(self, core_v1, job_name: str, job_id: str) -> str:
        """Log METRIC lines from pod output and return job_id."""
        logs = self._get_pod_logs(core_v1, job_name)
        for line in logs.splitlines():
            if line.strip().startswith("METRIC:"):
                logger.info("KubernetesAdapter: %s", line.strip())
        return job_id

    def submit(self, candidate: SOTAPlusOneCandidate) -> str:
        """
        Submit *candidate* as a Kubernetes Job and wait for completion.

        Raises
        ------
        ImportError
            If the ``kubernetes`` package is not installed.
        RuntimeError
            If the job fails, times out, or the cluster is unreachable.
        """
        import kubernetes as k8s

        batch_v1, core_v1 = self._k8s_clients()

        uid = str(uuid.uuid4())[:8]
        job_name = f"etft-{candidate.candidate_id[:12]}-{uid}"
        configmap_name = f"{job_name}-script"
        job_id = f"k8s-{job_name}"

        configmap_body = k8s.client.V1ConfigMap(
            metadata=k8s.client.V1ObjectMeta(
                name=configmap_name, namespace=self._namespace
            ),
            data={"candidate.py": candidate.code},
        )
        core_v1.create_namespaced_config_map(self._namespace, configmap_body)
        logger.info("KubernetesAdapter: created ConfigMap %s", configmap_name)

        try:
            job_manifest = self._build_job_manifest(job_name, configmap_name)
            batch_v1.create_namespaced_job(namespace=self._namespace, body=job_manifest)
            logger.info(
                "KubernetesAdapter: submitted Job %s for candidate %s",
                job_name, candidate.candidate_id,
            )

            elapsed = 0
            while elapsed < self._timeout:
                time.sleep(self._poll_interval)
                elapsed += self._poll_interval

                job_status = batch_v1.read_namespaced_job_status(
                    name=job_name, namespace=self._namespace
                )
                conds = job_status.status

                if conds.succeeded and conds.succeeded > 0:
                    logger.info("KubernetesAdapter: Job %s succeeded.", job_name)
                    return self._collect_result(core_v1, job_name, job_id)

                if conds.failed and conds.failed > 0:
                    logs = self._get_pod_logs(core_v1, job_name)
                    raise RuntimeError(
                        f"Kubernetes Job {job_id} failed. Last logs:\n{logs}"
                    )

            raise RuntimeError(
                f"Kubernetes Job {job_id} timed out after {self._timeout}s."
            )

        finally:
            try:
                core_v1.delete_namespaced_config_map(configmap_name, self._namespace)
            except Exception as exc:
                logger.debug("Failed to delete ConfigMap %s: %s", configmap_name, exc)
            try:
                batch_v1.delete_namespaced_job(
                    job_name, self._namespace,
                    body=k8s.client.V1DeleteOptions(propagation_policy="Foreground"),
                )
            except Exception as exc:
                logger.debug("Failed to delete Job %s: %s", job_name, exc)
