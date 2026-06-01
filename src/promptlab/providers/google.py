"""Google Gemini provider (Generative Language API)."""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from promptlab.providers.base import LLMProvider, ProviderResponse

# Approximate USD per 1M tokens (input, output). EDIT THESE — prices change often.
_PRICES: dict[str, tuple[float, float]] = {
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
}


class GoogleProvider(LLMProvider):
    name = "google"

    def __init__(self, model: str, **options: Any) -> None:
        super().__init__(model, **options)
        self.api_key = options.get("api_key") or os.environ.get("GOOGLE_API_KEY", "")
        self.base_url = options.get("base_url", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")

    async def complete(self, prompt: str, *, system: Optional[str] = None, **kwargs: Any) -> ProviderResponse:
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set (or pass api_key in the provider config).")

        payload: dict[str, Any] = {"contents": [{"parts": [{"text": prompt}]}]}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        gen_config: dict[str, Any] = {}
        for key, api_key in (("temperature", "temperature"), ("max_tokens", "maxOutputTokens")):
            value = kwargs.get(key, self.options.get(key))
            if value is not None:
                gen_config[api_key] = value
        if gen_config:
            payload["generationConfig"] = gen_config

        start = time.perf_counter()
        resp = await self._client.post(
            f"{self.base_url}/models/{self.model}:generateContent",
            headers={"x-goog-api-key": self.api_key},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        latency_ms = (time.perf_counter() - start) * 1000

        candidates = data.get("candidates", [])
        text = ""
        if candidates:
            finish_reason = candidates[0].get("finishReason", "")
            if finish_reason and finish_reason != "STOP":
                text = f"[blocked: {finish_reason}]"
            else:
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(part.get("text", "") for part in parts)
        elif data.get("promptFeedback", {}).get("blockReason"):
            text = f"[blocked: {data['promptFeedback']['blockReason']}]"
        usage = data.get("usageMetadata", {})
        prompt_tokens = int(usage.get("promptTokenCount", 0))
        completion_tokens = int(usage.get("candidatesTokenCount", 0))
        return ProviderResponse(
            text=text,
            model=self.model,
            tokens_used=int(usage.get("totalTokenCount", prompt_tokens + completion_tokens)),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cost=self.estimate_cost(prompt_tokens, completion_tokens),
            raw_response=data,
        )

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        in_price, out_price = _PRICES.get(self.model, (0.0, 0.0))
        return (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000
