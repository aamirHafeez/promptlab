"""Local Ollama provider (no API key, runs against a local server)."""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from promptlab.providers.base import LLMProvider, ProviderResponse


class OllamaProvider(LLMProvider):
    name = "ollama"
    default_timeout: float = 120.0

    def __init__(self, model: str, **options: Any) -> None:
        super().__init__(model, **options)
        self.base_url = options.get("base_url", os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    async def complete(self, prompt: str, *, system: Optional[str] = None, **kwargs: Any) -> ProviderResponse:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        options: dict[str, Any] = {}
        if (temperature := kwargs.get("temperature", self.options.get("temperature"))) is not None:
            options["temperature"] = temperature
        if options:
            payload["options"] = options

        start = time.perf_counter()
        resp = await self._client.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        latency_ms = (time.perf_counter() - start) * 1000

        text = data.get("message", {}).get("content", "")
        prompt_tokens = int(data.get("prompt_eval_count", 0))
        completion_tokens = int(data.get("eval_count", 0))
        return ProviderResponse(
            text=text,
            model=self.model,
            tokens_used=prompt_tokens + completion_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cost=0.0,  # local models are free
            raw_response=data,
        )
