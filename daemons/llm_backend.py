"""LLM Backend abstraction — pluggable interface for agent LLM calls.

Supported backends:
  - "vllm": Direct vLLM OpenAI-compatible API (default, known working)
  - "openclaw": OpenClaw Gateway with per-agent routing
"""

from __future__ import annotations

import abc
import asyncio
import logging
import os

import httpx

logger = logging.getLogger(__name__)

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_TIMEOUT = 120

OPENCLAW_GATEWAY_URL = os.environ.get("OPENCLAW_GATEWAY_URL", "http://localhost:18789")
OPENCLAW_GATEWAY_TOKEN = os.environ.get("OPENCLAW_GATEWAY_TOKEN")


class LLMBackend(abc.ABC):
    """Abstract base for LLM backends."""

    @abc.abstractmethod
    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Send a chat completion request. Returns the response text."""
        ...

    @abc.abstractmethod
    async def health_check(self) -> dict:
        """Check backend health."""
        ...

    async def close(self):
        """Cleanup resources."""
        pass


class VLLMBackend(LLMBackend):
    """Direct vLLM OpenAI-compatible API backend."""

    def __init__(self, base_url: str | None = None, timeout: int = VLLM_TIMEOUT):
        self.base_url = base_url or VLLM_BASE_URL
        self.timeout = timeout
        self._http: httpx.AsyncClient | None = None

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=self.timeout)
        return self._http

    async def chat_completion(self, model, messages, max_tokens=512, temperature=0.7) -> str:
        http = await self._get_http()
        try:
            async with asyncio.timeout(self.timeout):
                response = await http.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning(f"vLLM request timed out after {self.timeout}s for model {model}")
            raise
        except httpx.ConnectError:
            logger.error(f"Cannot connect to vLLM at {self.base_url}")
            raise

    async def health_check(self) -> dict:
        try:
            http = await self._get_http()
            r = await http.get(f"{self.base_url}/models")
            r.raise_for_status()
            return {"status": "ok", "backend": "vllm"}
        except Exception as e:
            return {"status": "error", "backend": "vllm", "error": str(e)}

    async def close(self):
        if self._http and not self._http.is_closed:
            await self._http.aclose()


class OpenClawBackend(LLMBackend):
    """OpenClaw Gateway backend with per-agent routing.

    Calls the Gateway's OpenAI-compatible endpoint with model names
    prefixed as "openclaw:{agent_id}" for per-agent routing.
    """

    def __init__(
        self,
        gateway_url: str | None = None,
        gateway_token: str | None = None,
        timeout: int = 120,
    ):
        self.gateway_url = (gateway_url or OPENCLAW_GATEWAY_URL).rstrip("/")
        self.gateway_token = gateway_token or OPENCLAW_GATEWAY_TOKEN
        self.timeout = timeout
        self._http: httpx.AsyncClient | None = None

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=self.timeout)
        return self._http

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.gateway_token:
            headers["Authorization"] = f"Bearer {self.gateway_token}"
        return headers

    async def chat_completion(self, model, messages, max_tokens=512, temperature=0.7) -> str:
        http = await self._get_http()

        # Route via openclaw:{agent_name} model prefix
        model_name = model
        if not model.startswith("openclaw:"):
            # Extract agent name from model if it looks like "agent-N"
            # Otherwise just use the raw model name
            model_name = f"openclaw:{model}"

        response = await http.post(
            f"{self.gateway_url}/v1/chat/completions",
            headers=self._headers(),
            json={
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def health_check(self) -> dict:
        try:
            http = await self._get_http()
            headers = self._headers()
            # Try /health first, then /v1/models
            for endpoint in ["/health", "/v1/models"]:
                try:
                    r = await http.get(f"{self.gateway_url}{endpoint}", headers=headers, timeout=5)
                    if r.status_code < 500:
                        return {"status": "ok", "backend": "openclaw"}
                except Exception:
                    continue
            return {"status": "error", "backend": "openclaw", "error": "all endpoints failed"}
        except Exception as e:
            return {"status": "error", "backend": "openclaw", "error": str(e)}

    async def close(self):
        if self._http and not self._http.is_closed:
            await self._http.aclose()


def create_backend(config: dict) -> LLMBackend:
    """Factory: create backend from experiment config.

    Config format (in experiment YAML):
        backend:
          type: vllm          # or "openclaw"
          url: http://localhost:8000/v1
          timeout: 120
    """
    backend_type = config.get("type", "vllm")

    if backend_type == "vllm":
        return VLLMBackend(
            base_url=config.get("url"),
            timeout=config.get("timeout", VLLM_TIMEOUT),
        )
    elif backend_type == "openclaw":
        return OpenClawBackend(
            gateway_url=config.get("url"),
            gateway_token=config.get("token"),
            timeout=config.get("timeout", 120),
        )
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
