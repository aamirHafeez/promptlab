"""Tests for config loading and validation."""

from __future__ import annotations

import pytest

from promptlab.config import ConfigError, parse_config


def test_minimal_valid_config():
    cfg = parse_config({"providers": ["openai/gpt-4o"], "prompts": ["Hello {{name}}"]})
    assert cfg.providers[0].id == "openai/gpt-4o"
    assert cfg.prompts == ["Hello {{name}}"]
    assert cfg.test_cases == []


def test_provider_dict_options():
    cfg = parse_config(
        {
            "providers": [{"id": "openai/gpt-4o", "temperature": 0.5, "base_url": "http://x"}],
            "prompts": ["hi"],
        }
    )
    assert cfg.providers[0].options == {"temperature": 0.5, "base_url": "http://x"}


def test_single_prompt_string_is_wrapped():
    cfg = parse_config({"providers": ["ollama/llama3"], "prompts": "just one"})
    assert cfg.prompts == ["just one"]


def test_assertions_parsed():
    cfg = parse_config(
        {
            "providers": ["openai/gpt-4o"],
            "prompts": ["hi"],
            "test_cases": [{"vars": {"q": "x"}, "assert": [{"type": "contains", "value": "y"}]}],
        }
    )
    assert cfg.test_cases[0].assertions[0].type == "contains"
    assert cfg.test_cases[0].assertions[0].value == "y"


def test_missing_providers_raises():
    with pytest.raises(ConfigError):
        parse_config({"prompts": ["hi"]})


def test_missing_prompts_raises():
    with pytest.raises(ConfigError):
        parse_config({"providers": ["openai/gpt-4o"]})


def test_assertion_without_type_raises():
    with pytest.raises(ConfigError):
        parse_config(
            {
                "providers": ["openai/gpt-4o"],
                "prompts": ["hi"],
                "test_cases": [{"assert": [{"value": "y"}]}],
            }
        )
