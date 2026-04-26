"""
etft/llm.py — Provider-agnostic LLM adapter.

Supports OpenAI, Anthropic, and local (OpenAI-compatible) backends.
Configure via config.yaml or environment variables.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin, provider-agnostic wrapper around LLM completion APIs."""

    def __init__(self, cfg: dict | None = None) -> None:
        self._cfg = cfg or {}
        llm_cfg = self._cfg.get("llm", {})
        self.backend: str = os.getenv("LLM_BACKEND", llm_cfg.get("backend", "openai"))
        self.temperature: float = float(llm_cfg.get("temperature", 0.2))
        self.max_tokens: int = int(llm_cfg.get("max_tokens", 4096))
        self._client: Any = None
        self._model: str = ""
        self._init_client(llm_cfg)

    # ------------------------------------------------------------------
    def _init_client(self, llm_cfg: dict) -> None:
        if self.backend == "openai":
            import openai

            self._client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self._model = os.getenv("OPENAI_MODEL", llm_cfg.get("openai_model", "gpt-4o"))

        elif self.backend == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self._model = os.getenv(
                "ANTHROPIC_MODEL",
                llm_cfg.get("anthropic_model", "claude-3-5-sonnet-20241022"),
            )

        elif self.backend == "local":
            import openai

            base_url = os.getenv(
                "LOCAL_LLM_BASE_URL", llm_cfg.get("local_base_url", "http://localhost:11434/v1")
            )
            self._client = openai.OpenAI(api_key="local", base_url=base_url)
            self._model = os.getenv("LOCAL_LLM_MODEL", llm_cfg.get("local_model", "llama3"))

        else:
            raise ValueError(f"Unknown LLM backend: {self.backend!r}")

    # ------------------------------------------------------------------
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def complete(self, prompt: str, system: str = "") -> str:
        """Return the model's text completion for *prompt*."""
        logger.debug("LLM complete | backend=%s model=%s", self.backend, self._model)

        if self.backend == "anthropic":
            msg = self._client.messages.create(
                model=self._model,
                max_tokens=self.max_tokens,
                system=system or "You are a helpful AI research assistant.",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            return msg.content[0].text

        # OpenAI-compatible (openai + local)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""
