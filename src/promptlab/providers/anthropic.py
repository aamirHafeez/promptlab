"""Anthropic Claude provider (Messages API).

Endpoint:  https://api.anthropic.com/v1/messages
Headers:   x-api-key, anthropic-version, content-type
Docs:      https://docs.claude.com/en/api/overview
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from promptlab.providers.base import LLMProvider, ProviderResponse

ANTHROPIC_VERSION = "2023-06-01"

# Approximate USD per 1M tokens (input, output). EDIT THESE — prices change often.
_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (15.00, 75.00),
    "claude-opus-4-7": (15.00, 75.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str, **options: Any) -> None:
        super().__init__(model, **options)
        self.api_key = options.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = options.get("base_url", "https://api.anthropic.com").rstrip("/")

    async def complete(self, prompt: str, *, system: Optional[str] = None, **kwargs: Any) -> ProviderResponse:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (or pass api_key in the provider config).")

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.options.get("max_tokens", 4096)),
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        temperature = kwargs.get("temperature", self.options.get("temperature"))
        if temperature is not None:
            payload["temperature"] = temperature

        start = time.perf_counter()
        resp = await self._client.post(
            f"{self.base_url}/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        latency_ms = (time.perf_counter() - start) * 1000

        # Concatenate all text blocks in the content array.
        text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
        usage = data.get("usage", {})
        prompt_tokens = int(usage.get("input_tokens", 0))
        completion_tokens = int(usage.get("output_tokens", 0))
        return ProviderResponse(
            text=text,
            model=data.get("model", self.model),
            tokens_used=prompt_tokens + completion_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cost=self.estimate_cost(prompt_tokens, completion_tokens),
            raw_response=data,
        )

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        in_price, out_price = _PRICES.get(self.model, (0.0, 0.0))
        return (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000
