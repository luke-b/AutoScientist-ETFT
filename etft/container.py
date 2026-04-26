"""
etft/container.py — Local Docker container lifecycle management for the
coding-agent proxy.

When ``agent_proxy.container.mode`` is ``"local"``, ETFT can start and stop
the proxy container automatically.  In ``"cloud"`` mode this module is a
no-op; the proxy URL is simply read from config / env.

Usage
-----
    from etft.container import ContainerManager
    mgr = ContainerManager(cfg)
    mgr.ensure_running()   # starts container if not already up
    ...
    mgr.stop()
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)

_HEALTH_PATH = "/health"
_POLL_INTERVAL = 2      # seconds between readiness polls
_MAX_WAIT = 60          # seconds before giving up


class ContainerManager:
    """
    Manages the lifecycle of the local coding-agent proxy Docker container.

    If ``mode`` is ``"cloud"`` this class does nothing — the proxy URL is
    assumed to be externally managed.
    """

    def __init__(self, cfg: dict | None = None) -> None:
        proxy_cfg = (cfg or {}).get("agent_proxy", {})
        container_cfg = proxy_cfg.get("container", {})

        self.mode: str = container_cfg.get("mode", "local")
        self.image: str = container_cfg.get("image", "autoscientist-etft-agent:latest")
        self.name: str = container_cfg.get("name", "etft-agent-proxy")
        self.port: int = int(container_cfg.get("port", 8080))
        self.mem_limit: str = container_cfg.get("mem_limit", "2g")
        self.cpu_count: int = int(container_cfg.get("cpu_count", 2))

        # Proxy's internal LLM settings passed as container env vars
        self._proxy_env: dict[str, str] = {
            "PROXY_LLM_BACKEND": os.getenv(
                "PROXY_LLM_BACKEND", container_cfg.get("proxy_llm_backend", "openai")
            ),
            "PROXY_LLM_MODEL": os.getenv(
                "PROXY_LLM_MODEL", container_cfg.get("proxy_llm_model", "gpt-4o")
            ),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
            "LOCAL_LLM_BASE_URL": os.getenv("LOCAL_LLM_BASE_URL", ""),
            "LOCAL_LLM_MODEL": os.getenv("LOCAL_LLM_MODEL", ""),
            "AGENT_PROXY_TOKEN": os.getenv("AGENT_PROXY_TOKEN", ""),
        }

        self._proxy_url: str = os.getenv(
            "AGENT_PROXY_URL", proxy_cfg.get("url", f"http://localhost:{self.port}")
        )
        self._container = None  # docker.models.containers.Container

    # ------------------------------------------------------------------
    def ensure_running(self) -> str:
        """
        Ensure the proxy container is running.

        Returns
        -------
        str
            The proxy base URL.
        """
        if self.mode != "local":
            logger.info("Container mode is %r — skipping local Docker management.", self.mode)
            return self._proxy_url

        import docker  # optional dependency

        client = docker.from_env()

        # Check if already running
        try:
            existing = client.containers.get(self.name)
            if existing.status == "running":
                logger.info("Container %r is already running.", self.name)
                self._container = existing
                return self._proxy_url
            logger.info("Container %r exists but is not running — restarting.", self.name)
            existing.remove(force=True)
        except docker.errors.NotFound:
            pass

        logger.info("Starting container %r from image %r …", self.name, self.image)
        self._container = client.containers.run(
            self.image,
            name=self.name,
            detach=True,
            ports={f"{self.port}/tcp": self.port},
            environment=self._proxy_env,
            mem_limit=self.mem_limit,
            nano_cpus=int(self.cpu_count * 1_000_000_000),  # Docker expects nanoseconds-equivalent CPU units
            remove=False,
        )

        self._wait_for_ready()
        return self._proxy_url

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Stop (and remove) the local proxy container."""
        if self.mode != "local":
            return
        if self._container is not None:
            logger.info("Stopping container %r …", self.name)
            self._container.stop()
            self._container.remove()
            self._container = None
        else:
            # Try by name
            try:
                import docker

                c = docker.from_env().containers.get(self.name)
                c.stop()
                c.remove()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _wait_for_ready(self) -> None:
        import httpx

        health_url = f"{self._proxy_url}{_HEALTH_PATH}"
        deadline = time.monotonic() + _MAX_WAIT
        while time.monotonic() < deadline:
            try:
                r = httpx.get(health_url, timeout=5)
                if r.status_code == 200:
                    logger.info("Proxy container is ready at %s", self._proxy_url)
                    return
            except Exception:
                pass
            time.sleep(_POLL_INTERVAL)
        raise TimeoutError(
            f"Proxy container did not become ready within {_MAX_WAIT}s. "
            f"Check container logs: docker logs {self.name}"
        )
