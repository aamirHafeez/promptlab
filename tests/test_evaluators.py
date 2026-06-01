"""Tests for the built-in evaluators."""

from __future__ import annotations

from promptlab.config import Assertion
from promptlab.evaluators import evaluate
from promptlab.providers.base import ProviderResponse


def _resp(text: str, **kw) -> ProviderResponse:
    return ProviderResponse(text=text, model="t/m", **kw)


def test_contains_pass(response):
    assert evaluate(response, Assertion(type="contains", value="Paris")).passed


def test_contains_fail(response):
    assert not evaluate(response, Assertion(type="contains", value="London")).passed


def test_not_contains(response):
    assert evaluate(response, Assertion(type="not_contains", value="London")).passed
    assert not evaluate(response, Assertion(type="not_contains", value="Paris")).passed


def test_matches_regex(response):
    assert evaluate(response, Assertion(type="matches_regex", value=r"capital of \w+")).passed


def test_starts_with():
    assert evaluate(_resp("  Yes, indeed"), Assertion(type="starts_with", value="Yes")).passed


def test_max_tokens():
    r = _resp("hi", completion_tokens=5)
    assert evaluate(r, Assertion(type="max_tokens", value=10)).passed
    assert not evaluate(r, Assertion(type="max_tokens", value=3)).passed


def test_max_latency():
    r = _resp("hi", latency_ms=200)
    assert evaluate(r, Assertion(type="max_latency", value=500)).passed
    assert not evaluate(r, Assertion(type="max_latency", value=100)).passed


def test_max_cost():
    r = _resp("hi", cost=0.001)
    assert evaluate(r, Assertion(type="max_cost", value=0.01)).passed
    assert not evaluate(r, Assertion(type="max_cost", value=0.0001)).passed


def test_is_json():
    assert evaluate(_resp('{"a": 1}'), Assertion(type="is_json", value=True)).passed
    assert evaluate(_resp("not json"), Assertion(type="is_json", value=False)).passed
    assert not evaluate(_resp("not json"), Assertion(type="is_json", value=True)).passed


def test_json_schema():
    schema = {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}
    assert evaluate(_resp('{"name": "x"}'), Assertion(type="json_schema", value=schema)).passed
    assert not evaluate(_resp('{"age": 1}'), Assertion(type="json_schema", value=schema)).passed


def test_unknown_assertion():
    assert not evaluate(_resp("x"), Assertion(type="nope")).passed


def test_similarity_not_implemented():
    result = evaluate(_resp("x"), Assertion(type="similarity", value="y"))
    assert not result.passed
    assert "not implemented" in result.message.lower()


# --- B1: guards against bad input ---


def test_max_tokens_missing_value():
    r = _resp("hi", completion_tokens=5)
    result = evaluate(r, Assertion(type="max_tokens"))
    assert not result.passed
    assert "numeric" in result.message.lower() or "value" in result.message.lower()


def test_max_latency_missing_value():
    r = _resp("hi", latency_ms=200)
    result = evaluate(r, Assertion(type="max_latency"))
    assert not result.passed


def test_max_cost_missing_value():
    r = _resp("hi", cost=0.001)
    result = evaluate(r, Assertion(type="max_cost"))
    assert not result.passed


def test_max_tokens_non_numeric_value():
    r = _resp("hi", completion_tokens=5)
    result = evaluate(r, Assertion(type="max_tokens", value="abc"))
    assert not result.passed
    assert "cannot convert" in result.message.lower()


def test_matches_regex_invalid_pattern():
    result = evaluate(_resp("hello"), Assertion(type="matches_regex", value="[invalid"))
    assert not result.passed
    assert "invalid regex" in result.message.lower()
