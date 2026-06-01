"""Shared fixtures."""

from __future__ import annotations

import pytest

from promptlab.config import Assertion, Config, ProviderConfig, TestCase
from promptlab.providers.base import ProviderResponse


@pytest.fixture
def response() -> ProviderResponse:
    return ProviderResponse(
        text="Hello, Paris is the capital of France.",
        model="test/model",
        tokens_used=20,
        prompt_tokens=8,
        completion_tokens=12,
        latency_ms=120.0,
        cost=0.0001,
    )


@pytest.fixture
def config() -> Config:
    return Config(
        description="fixture suite",
        providers=[ProviderConfig(id="openai/gpt-4o"), ProviderConfig(id="ollama/llama3")],
        prompts=["Answer: {{query}}", "Reply: {{query}}"],
        test_cases=[
            TestCase(
                vars={"query": "capital of France"},
                assertions=[Assertion(type="contains", value="Paris")],
            )
        ],
    )
