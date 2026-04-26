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

import json
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

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY environment variable is required for the 'openai' backend."
                )
            self._client = openai.OpenAI(api_key=api_key)

        elif self.backend == "anthropic":
            import anthropic

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY environment variable is required for the 'anthropic' backend."
                )
            self._client = anthropic.Anthropic(api_key=api_key)

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

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        """
        Call the provider and return a response dict in OpenAI message format:
        ``{"content": str, "tool_calls": list | None, "finish_reason": str}``
        """
        if self.backend == "anthropic":
            return self._complete_anthropic(messages, tools)

        # OpenAI-compatible (openai + local)
        return self._complete_openai(messages, tools)

    def _complete_openai(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 4096,
        }
        if tools:
            kwargs["tools"] = tools

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message

        tool_calls = None
        if getattr(msg, "tool_calls", None):
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]

        return {
            "content": msg.content or "",
            "tool_calls": tool_calls,
            "finish_reason": choice.finish_reason or "stop",
        }

    def _complete_anthropic(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs = [m for m in messages if m["role"] != "system"]

        kwargs: dict = {
            "model": self.model,
            "max_tokens": 4096,
            "system": system or "You are an expert coding agent.",
            "messages": user_msgs,
        }

        if tools:
            # Convert OpenAI tool format to Anthropic tool format
            anthropic_tools = []
            for t in tools:
                func = t.get("function", {})
                anthropic_tools.append(
                    {
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                    }
                )
            kwargs["tools"] = anthropic_tools

        resp = self._client.messages.create(**kwargs)

        content_text = ""
        tool_calls = None

        for block in resp.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    }
                )

        finish_reason = "tool_calls" if tool_calls else "stop"
        return {
            "content": content_text,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
        }


_backend = _Backend()

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class CompletionRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float = 0.2
    max_tokens: int = 4096
    tools: list[dict] | None = None


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
    messages = [m.model_dump(exclude_none=True) for m in req.messages]
    logger.info("Completion request | messages=%d tools=%s", len(messages), bool(req.tools))
    try:
        result = _backend.complete(messages, tools=req.tools or None)
    except Exception as exc:
        logger.exception("LLM backend error: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    response_message = ChatMessage(
        role="assistant",
        content=result["content"] or None,
        tool_calls=result.get("tool_calls"),
    )
    return CompletionResponse(
        choices=[
            CompletionChoice(
                message=response_message,
                finish_reason=result.get("finish_reason", "stop"),
            )
        ]
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")), log_level="info")
