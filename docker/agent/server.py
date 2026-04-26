"""
docker/agent/server.py — FastAPI proxy server that runs inside the
coding-agent container.

The server exposes an OpenAI-compatible POST /v1/chat/completions endpoint.
ETFT modules call this endpoint; the server fulfills requests using whichever
LLM provider is configured in the container's environment (PROXY_LLM_BACKEND).

Endpoints
---------
GET  /health                   Readiness probe.
POST /v1/chat/completions      OpenAI-compatible completion (passthrough).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

logger = logging.getLogger("etft.proxy")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AutoScientist-ETFT Coding-Agent Proxy",
    description="Lightweight proxy that routes ETFT completion requests to a coding-agent LLM backend.",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

_TOKEN_HEADER = APIKeyHeader(name="X-Agent-Token", auto_error=False)
_EXPECTED_TOKEN: str = os.getenv("AGENT_PROXY_TOKEN", "")


def _verify_token(token: str | None = Security(_TOKEN_HEADER)) -> None:
    if _EXPECTED_TOKEN and token != _EXPECTED_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Agent-Token header.",
        )


# ---------------------------------------------------------------------------
# LLM backend (initialised once at startup)
# ---------------------------------------------------------------------------


class _Backend:
    def __init__(self) -> None:
        self.backend = os.getenv("PROXY_LLM_BACKEND", "openai")
        self.model = os.getenv("PROXY_LLM_MODEL", "gpt-4o")
        self._client: Any = None
        self._init()

    def _init(self) -> None:
        if self.backend == "openai":
            import openai

            self._client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        elif self.backend == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        elif self.backend == "local":
            import openai

            self._client = openai.OpenAI(
                api_key="local",
                base_url=os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1"),
            )
            self.model = os.getenv("LOCAL_LLM_MODEL", "llama3")

        else:
            raise RuntimeError(f"Unknown PROXY_LLM_BACKEND: {self.backend!r}")

        logger.info("Proxy LLM backend: %s / %s", self.backend, self.model)

    def complete(self, messages: list[dict[str, str]]) -> str:
        if self.backend == "anthropic":
            system = next((m["content"] for m in messages if m["role"] == "system"), "")
            user_msgs = [m for m in messages if m["role"] != "system"]
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system or "You are an expert coding agent.",
                messages=user_msgs,
            )
            return resp.content[0].text

        # OpenAI-compatible
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=4096,
        )
        return resp.choices[0].message.content or ""


_backend = _Backend()

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class CompletionRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float = 0.2
    max_tokens: int = 4096


class CompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class CompletionResponse(BaseModel):
    object: str = "chat.completion"
    choices: list[CompletionChoice]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", status_code=200)
def health() -> dict[str, str]:
    return {"status": "ok", "backend": _backend.backend, "model": _backend.model}


@app.post(
    "/v1/chat/completions",
    response_model=CompletionResponse,
    dependencies=[Depends(_verify_token)],
)
def chat_completions(req: CompletionRequest) -> CompletionResponse:
    messages = [m.model_dump() for m in req.messages]
    logger.info("Completion request | messages=%d", len(messages))
    try:
        content = _backend.complete(messages)
    except Exception as exc:
        logger.exception("LLM backend error: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return CompletionResponse(
        choices=[CompletionChoice(message=ChatMessage(role="assistant", content=content))]
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")), log_level="info")
