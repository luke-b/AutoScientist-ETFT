"""
etft/llm.py — Coding-agent proxy client.

All LLM traffic in AutoScientist-ETFT flows through a coding-agent proxy
container (local Docker or cloud endpoint) rather than calling provider
SDKs directly.  The proxy exposes an OpenAI-compatible
POST /v1/chat/completions endpoint and handles provider credentials,
model selection, rate-limiting, and high-level coding-agent capabilities
internally.

Architecture
------------
  ETFT module
      │
      ▼  HTTP  (X-Agent-Token header)
  ┌──────────────────────────────┐
  │  Coding-Agent Proxy          │  ← docker/agent/server.py
  │  (local container or cloud)  │
  └──────────────────────────────┘
      │
      ▼  provider SDK (inside container)
  OpenAI Codex / Anthropic / local model

Usage
-----
    from etft.llm import LLMClient
    client = LLMClient(cfg)          # cfg from config.yaml
    answer = client.complete(prompt, system="…")
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_DEFAULT_PROXY_URL = "http://localhost:8080"
_COMPLETIONS_PATH = "/v1/chat/completions"


class LLMClient:
    """
    Sends completion requests to the coding-agent proxy container.

    The proxy is responsible for all provider credentials and model routing;
    this client only needs the proxy URL and a shared token.
    """

    def __init__(self, cfg: dict | None = None) -> None:
        proxy_cfg = (cfg or {}).get("agent_proxy", {})
        self._url: str = os.getenv(
            "AGENT_PROXY_URL", proxy_cfg.get("url", _DEFAULT_PROXY_URL)
        ).rstrip("/")
        self._token: str = os.getenv(
            "AGENT_PROXY_TOKEN", proxy_cfg.get("token", "")
        )
        self._timeout: float = float(proxy_cfg.get("timeout_seconds", 120))
        self._headers: dict[str, str] = {
            "Content-Type": "application/json",
            **({"X-Agent-Token": self._token} if self._token else {}),
        }

    # ------------------------------------------------------------------
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def complete(self, prompt: str, system: str = "") -> str:
        """
        Send *prompt* to the proxy and return the assistant's reply text.

        Parameters
        ----------
        prompt:
            User message / task description.
        system:
            Optional system instruction forwarded to the proxy.

        Returns
        -------
        str
            The coding agent's response text.

        Raises
        ------
        httpx.HTTPError
            On network failures after all retries are exhausted.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {"messages": messages}

        logger.debug("Proxy request → %s%s", self._url, _COMPLETIONS_PATH)

        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._url}{_COMPLETIONS_PATH}",
                json=payload,
                headers=self._headers,
            )
            response.raise_for_status()

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise ValueError(f"Unexpected proxy response format: {data}") from exc

    # ------------------------------------------------------------------
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """
        Send *messages* to the proxy, optionally with tool definitions.

        Returns
        -------
        LLMResponse
            Parsed response with content, tool_calls, and finish_reason.
        """
        from etft.skills.base import LLMResponse, ToolCall

        payload: dict[str, Any] = {"messages": messages}
        if tools:
            payload["tools"] = tools

        logger.debug(
            "Proxy request (with_tools=%s) → %s%s",
            bool(tools),
            self._url,
            _COMPLETIONS_PATH,
        )

        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._url}{_COMPLETIONS_PATH}",
                json=payload,
                headers=self._headers,
            )
            response.raise_for_status()

        data = response.json()
        try:
            choice = data["choices"][0]
            message = choice["message"]
            finish_reason: str = choice.get("finish_reason", "stop")
            content: str = message.get("content") or ""

            tool_calls: list[ToolCall] = []
            for tc in message.get("tool_calls") or []:
                func = tc.get("function", {})
                raw_args = func.get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", ""),
                        name=func.get("name", ""),
                        args=args,
                    )
                )

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            )
        except (KeyError, IndexError) as exc:
            raise ValueError(f"Unexpected proxy response format: {data}") from exc

