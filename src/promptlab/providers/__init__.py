"""Provider registry. Resolves "provider/model" ids to provider instances."""

from __future__ import annotations

from typing import Any

from promptlab.providers.anthropic import AnthropicProvider
from promptlab.providers.base import LLMProvider, ProviderResponse
from promptlab.providers.google import GoogleProvider
from promptlab.providers.ollama import OllamaProvider
from promptlab.providers.openai import OpenAIProvider

_REGISTRY: dict[str, type[LLMProvider]] = {
    OpenAIProvider.name: OpenAIProvider,
    AnthropicProvider.name: AnthropicProvider,
    GoogleProvider.name: GoogleProvider,
    OllamaProvider.name: OllamaProvider,
}


def available_providers() -> list[str]:
    return sorted(_REGISTRY)


def parse_id(provider_id: str) -> tuple[str, str]:
    """Split "openai/gpt-4o" into ("openai", "gpt-4o")."""
    if "/" not in provider_id:
        raise ValueError(f"Invalid provider id {provider_id!r}. Use the form 'provider/model', e.g. 'openai/gpt-4o'.")
    provider, model = provider_id.split("/", 1)
    return provider, model


def get_provider(provider_id: str, **options: Any) -> LLMProvider:
    """Build a provider instance from a "provider/model" id."""
    provider, model = parse_id(provider_id)
    if provider not in _REGISTRY:
        raise ValueError(f"Unknown provider {provider!r}. Available: {', '.join(available_providers())}.")
    return _REGISTRY[provider](model=model, **options)


__all__ = [
    "LLMProvider",
    "ProviderResponse",
    "available_providers",
    "get_provider",
    "parse_id",
]
