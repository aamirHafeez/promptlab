"""Load and validate promptlab.yml into typed dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml


class ConfigError(ValueError):
    """Raised when a config file is malformed."""


@dataclass
class ProviderConfig:
    id: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class Assertion:
    type: str
    value: Any = None
    threshold: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestCase:
    vars: dict[str, Any] = field(default_factory=dict)
    assertions: list[Assertion] = field(default_factory=list)
    name: Optional[str] = None


@dataclass
class Config:
    description: str = ""
    providers: list[ProviderConfig] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    test_cases: list[TestCase] = field(default_factory=list)


def _parse_provider(raw: Union[str, dict[str, Any]]) -> ProviderConfig:
    if isinstance(raw, str):
        return ProviderConfig(id=raw)
    if isinstance(raw, dict):
        if "id" not in raw:
            raise ConfigError("Each provider needs an 'id' (e.g. 'openai/gpt-4o').")
        options = {k: v for k, v in raw.items() if k != "id"}
        return ProviderConfig(id=raw["id"], options=options)
    raise ConfigError(f"Invalid provider entry: {raw!r}")


def _parse_assertion(raw: dict[str, Any]) -> Assertion:
    if "type" not in raw:
        raise ConfigError(f"Each assertion needs a 'type'. Got: {raw!r}")
    known = {"type", "value", "threshold"}
    extra = {k: v for k, v in raw.items() if k not in known}
    return Assertion(
        type=raw["type"],
        value=raw.get("value"),
        threshold=raw.get("threshold"),
        extra=extra,
    )


def _parse_test_case(raw: dict[str, Any]) -> TestCase:
    assertions = [_parse_assertion(a) for a in raw.get("assert", [])]
    return TestCase(vars=raw.get("vars", {}), assertions=assertions, name=raw.get("name"))


def parse_config(data: dict[str, Any]) -> Config:
    """Turn a parsed-YAML dict into a validated :class:`Config`."""
    if not isinstance(data, dict):
        raise ConfigError("Top-level config must be a mapping.")
    if not data.get("providers"):
        raise ConfigError("Config must list at least one provider under 'providers'.")
    if not data.get("prompts"):
        raise ConfigError("Config must list at least one prompt under 'prompts'.")

    prompts = data["prompts"]
    if isinstance(prompts, str):
        prompts = [prompts]

    return Config(
        description=data.get("description", ""),
        providers=[_parse_provider(p) for p in data["providers"]],
        prompts=list(prompts),
        test_cases=[_parse_test_case(t) for t in data.get("test_cases", [])],
    )


def load_config(path: Union[str, Path]) -> Config:
    """Read and validate a YAML config file."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - passthrough
        raise ConfigError(f"Could not parse YAML: {exc}") from exc
    return parse_config(data or {})
