"""OpenAI and OpenAI-compatible provider (Azure, vLLM, LiteLLM, etc.)."""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from promptlab.providers.base import LLMProvider, ProviderResponse

# Approximate USD per 1M tokens (input, output). EDIT THESE — prices change often.
# Source of truth is each provider's pricing page, not this file.
_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "o3": (2.00, 8.00),
    "o4-mini": (1.10, 4.40),
}


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str, **options: Any) -> None:
        super().__init__(model, **options)
        self.api_key = options.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = options.get("base_url", "https://api.openai.com/v1").rstrip("/")

    async def complete(self, prompt: str, *, system: Optional[str] = None, **kwargs: Any) -> ProviderResponse:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (or pass api_key in the provider config).")

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        for key in ("temperature", "max_tokens", "top_p"):
            value = kwargs.get(key, self.options.get(key))
            if value is not None:
                payload[key] = value

        start = time.perf_counter()
        resp = await self._client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        latency_ms = (time.perf_counter() - start) * 1000

        choices = data.get("choices")
        if not choices:
            raise RuntimeError(f"OpenAI returned no choices: {str(data)[:200]}")
        text = choices[0].get("message", {}).get("content") or ""
        usage = data.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        return ProviderResponse(
            text=text,
            model=data.get("model", self.model),
            tokens_used=int(usage.get("total_tokens", prompt_tokens + completion_tokens)),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cost=self.estimate_cost(prompt_tokens, completion_tokens),
            raw_response=data,
        )

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        in_price, out_price = _PRICES.get(self.model, (0.0, 0.0))
        return (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000
