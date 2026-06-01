"""Built-in evaluators. Each maps an Assertion to a pass/fail result.

An evaluator signature is:
    evaluator(response, assertion, context) -> EvalResult

`response` is a ProviderResponse, `assertion` is a config.Assertion, and
`context` carries optional services (e.g. a judge provider for llm_judge).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from promptlab.config import Assertion
from promptlab.providers.base import ProviderResponse


@dataclass
class EvalResult:
    passed: bool
    type: str
    message: str = ""


Evaluator = Callable[[ProviderResponse, Assertion, "EvalContext"], EvalResult]


@dataclass
class EvalContext:
    """Optional services available to evaluators."""

    judge: Any = None  # an LLMProvider used by llm_judge


_REGISTRY: dict[str, Evaluator] = {}


def register(name: str) -> Callable[[Evaluator], Evaluator]:
    def deco(fn: Evaluator) -> Evaluator:
        _REGISTRY[name] = fn
        return fn

    return deco


def available_evaluators() -> list[str]:
    return sorted(_REGISTRY)


def evaluate(response: ProviderResponse, assertion: Assertion, context: Optional[EvalContext] = None) -> EvalResult:
    evaluator = _REGISTRY.get(assertion.type)
    if evaluator is None:
        return EvalResult(False, assertion.type, f"Unknown assertion type: {assertion.type!r}")
    return evaluator(response, assertion, context or EvalContext())


# --- string evaluators -------------------------------------------------------


@register("contains")
def _contains(response: ProviderResponse, a: Assertion, _: EvalContext) -> EvalResult:
    ok = str(a.value) in response.text
    return EvalResult(ok, "contains", "" if ok else f"output does not contain {a.value!r}")


@register("not_contains")
def _not_contains(response: ProviderResponse, a: Assertion, _: EvalContext) -> EvalResult:
    ok = str(a.value) not in response.text
    return EvalResult(ok, "not_contains", "" if ok else f"output unexpectedly contains {a.value!r}")


@register("matches_regex")
def _matches_regex(response: ProviderResponse, a: Assertion, _: EvalContext) -> EvalResult:
    try:
        ok = re.search(str(a.value), response.text) is not None
    except re.error as exc:
        return EvalResult(False, "matches_regex", f"invalid regex: {exc}")
    return EvalResult(ok, "matches_regex", "" if ok else f"output does not match /{a.value}/")


@register("starts_with")
def _starts_with(response: ProviderResponse, a: Assertion, _: EvalContext) -> EvalResult:
    ok = response.text.lstrip().startswith(str(a.value))
    return EvalResult(ok, "starts_with", "" if ok else f"output does not start with {a.value!r}")


# --- budget evaluators -------------------------------------------------------


@register("max_tokens")
def _max_tokens(response: ProviderResponse, a: Assertion, _: EvalContext) -> EvalResult:
    if a.value is None:
        return EvalResult(False, "max_tokens", "assertion requires a numeric 'value'")
    try:
        limit = int(a.value)
    except (TypeError, ValueError):
        return EvalResult(False, "max_tokens", f"cannot convert {a.value!r} to int")
    ok = response.completion_tokens <= limit
    return EvalResult(ok, "max_tokens", "" if ok else f"{response.completion_tokens} > {limit} tokens")


@register("max_latency")
def _max_latency(response: ProviderResponse, a: Assertion, _: EvalContext) -> EvalResult:
    if a.value is None:
        return EvalResult(False, "max_latency", "assertion requires a numeric 'value'")
    try:
        limit = float(a.value)
    except (TypeError, ValueError):
        return EvalResult(False, "max_latency", f"cannot convert {a.value!r} to float")
    ok = response.latency_ms <= limit
    return EvalResult(ok, "max_latency", "" if ok else f"{response.latency_ms:.0f}ms > {limit:.0f}ms")


@register("max_cost")
def _max_cost(response: ProviderResponse, a: Assertion, _: EvalContext) -> EvalResult:
    if a.value is None:
        return EvalResult(False, "max_cost", "assertion requires a numeric 'value'")
    try:
        limit = float(a.value)
    except (TypeError, ValueError):
        return EvalResult(False, "max_cost", f"cannot convert {a.value!r} to float")
    ok = response.cost <= limit
    return EvalResult(ok, "max_cost", "" if ok else f"${response.cost:.4f} > ${limit:.4f}")


# --- structure evaluators ----------------------------------------------------


@register("is_json")
def _is_json(response: ProviderResponse, a: Assertion, _: EvalContext) -> EvalResult:
    expect_valid = True if a.value is None else bool(a.value)
    try:
        json.loads(response.text)
        valid = True
    except (json.JSONDecodeError, TypeError):
        valid = False
    ok = valid == expect_valid
    return EvalResult(ok, "is_json", "" if ok else f"expected valid JSON={expect_valid}, got {valid}")


@register("json_schema")
def _json_schema(response: ProviderResponse, a: Assertion, _: EvalContext) -> EvalResult:
    try:
        import jsonschema
    except ImportError:  # pragma: no cover
        return EvalResult(False, "json_schema", "jsonschema not installed")
    try:
        instance = json.loads(response.text)
    except (json.JSONDecodeError, TypeError):
        return EvalResult(False, "json_schema", "output is not valid JSON")
    try:
        jsonschema.validate(instance, a.value)
        return EvalResult(True, "json_schema")
    except jsonschema.ValidationError as exc:
        return EvalResult(False, "json_schema", exc.message)


# --- model-graded evaluators -------------------------------------------------


@register("llm_judge")
def _llm_judge(response: ProviderResponse, a: Assertion, ctx: EvalContext) -> EvalResult:
    """Ask a judge model to score the output 1-10 against a rubric.

    Requires a judge provider in the context (set up by the runner). The
    threshold defaults to 7.
    """
    if ctx.judge is None:
        return EvalResult(False, "llm_judge", "no judge provider configured (set 'judge' in config)")

    import asyncio

    rubric = str(a.value)
    threshold = a.threshold if a.threshold is not None else 7
    prompt = (
        "You are grading an AI output. Score from 1 to 10 how well it meets the rubric.\n"
        f"Rubric: {rubric}\n\n"
        f"Output:\n{response.text}\n\n"
        "Respond with ONLY the integer score."
    )
    try:
        judged = asyncio.run(ctx.judge.complete(prompt))
    except RuntimeError:
        # A loop is already running (rare in the sync eval phase); use a fresh one.
        loop = asyncio.new_event_loop()
        try:
            judged = loop.run_until_complete(ctx.judge.complete(prompt))
        finally:
            loop.close()
    matches = re.findall(r"\d+", judged.text)
    if not matches:
        return EvalResult(False, "llm_judge", f"could not parse score from {judged.text!r}")
    score = min(max(int(matches[-1]), 1), 10)  # last integer, clamped 1-10
    ok = score >= threshold
    return EvalResult(ok, "llm_judge", f"score {score}/10 (threshold {threshold})")


@register("similarity")
def _similarity(response: ProviderResponse, a: Assertion, _: EvalContext) -> EvalResult:
    """Cosine similarity to an expected output.

    Not implemented yet: this needs an embedding model, which adds an API
    dependency and cost. Tracked separately so the tool stays honest about
    what works today.
    """
    return EvalResult(
        False,
        "similarity",
        "similarity is not implemented yet (requires an embedding model). See ROADMAP.md.",
    )
