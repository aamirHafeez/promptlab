"""Tests for the runner using dry-run mode (no network)."""

from __future__ import annotations

from promptlab.config import Config, ProviderConfig
from promptlab.runner import Runner
from promptlab.utils import render, template_vars


def test_dry_run_matrix_size(config):
    report = Runner(config, dry_run=True).run()
    # 2 providers x 2 prompts x 1 case = 4 cells
    assert report.total == 4
    assert all(c.response is not None for c in report.cells)
    assert all(c.error is None for c in report.cells)
    assert all(c.skipped for c in report.cells)
    assert report.failed == 0


def test_dry_run_no_cost(config):
    report = Runner(config, dry_run=True).run()
    assert report.total_cost == 0.0


def test_render_substitutes_vars():
    assert render("Hi {{name}}", {"name": "Sam"}) == "Hi Sam"


def test_render_leaves_unknown_vars():
    assert render("Hi {{name}}", {}) == "Hi {{name}}"


def test_template_vars_extraction():
    assert template_vars("{{a}} and {{ b }}") == ["a", "b"]


def test_rendered_prompt_uses_case_vars(config):
    report = Runner(config, dry_run=True).run()
    assert any("capital of France" in c.rendered_prompt for c in report.cells)


def test_duplicate_provider_id_different_options():
    """B4: two providers sharing the same id but different options must both be exercised."""
    cfg = Config(
        description="dup-id test",
        providers=[
            ProviderConfig(id="openai/gpt-4o", options={"temperature": 0.0}),
            ProviderConfig(id="openai/gpt-4o", options={"temperature": 1.0}),
        ],
        prompts=["Hello"],
        test_cases=[],
    )
    report = Runner(cfg, dry_run=True).run()
    # 2 providers × 1 prompt × 1 (empty) case = 2 cells
    assert report.total == 2
    # Verify cells reference different provider config indices
    indices = {c.provider_cfg_index for c in report.cells}
    assert indices == {0, 1}


# --- F2: judge provider is closed after evaluation ---


def test_judge_provider_closed_after_run(config, monkeypatch):
    """F2: the judge provider's aclose() must be called even in dry-run."""
    from unittest.mock import AsyncMock, MagicMock

    from promptlab.providers.base import LLMProvider

    mock_judge = MagicMock(spec=LLMProvider)
    mock_judge.aclose = AsyncMock()

    # Track which provider ID was requested so we can return distinct mocks.
    real_get = __import__("promptlab.providers", fromlist=["get_provider"]).get_provider

    def _patched_get_provider(pid, **kw):
        if pid == "fake-judge/model":
            return mock_judge
        return real_get(pid, **kw)

    monkeypatch.setattr("promptlab.runner.get_provider", _patched_get_provider)

    runner = Runner(config, dry_run=True, judge_id="fake-judge/model")
    runner.run()
    mock_judge.aclose.assert_awaited_once()


# --- F3: retry on transient network errors, fail fast on 401 ---


def test_retry_on_connect_error(monkeypatch):
    """F3: ConnectError should be retried; cell should succeed after transient failures."""
    from unittest.mock import AsyncMock, MagicMock

    import httpx

    from promptlab.providers.base import LLMProvider, ProviderResponse

    call_count = 0

    async def flaky_complete(prompt, **kw):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise httpx.ConnectError("connection refused")
        return ProviderResponse(text="ok", model="stub/model")

    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.complete = AsyncMock(side_effect=flaky_complete)
    mock_provider.aclose = AsyncMock()

    monkeypatch.setattr("promptlab.runner.get_provider", lambda pid, **kw: mock_provider)
    # Zero out backoff so test is fast.
    monkeypatch.setattr("promptlab.runner._RETRY_BASE_SECONDS", 0.0)

    cfg = Config(
        description="retry test",
        providers=[ProviderConfig(id="stub/model")],
        prompts=["hello"],
        test_cases=[],
    )
    report = Runner(cfg, dry_run=False).run()
    assert report.total == 1
    cell = report.cells[0]
    assert cell.response is not None
    assert cell.response.text == "ok"
    assert cell.error is None
    assert call_count == 3  # 2 failures + 1 success


def test_no_retry_on_401(monkeypatch):
    """F3: a 401 HTTPStatusError must fail immediately — no retries, no backoff."""
    from unittest.mock import AsyncMock, MagicMock

    import httpx

    from promptlab.providers.base import LLMProvider

    call_count = 0

    async def auth_fail(prompt, **kw):
        nonlocal call_count
        call_count += 1
        response = httpx.Response(401, request=httpx.Request("POST", "https://api.example.com"))
        raise httpx.HTTPStatusError("Unauthorized", request=response.request, response=response)

    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.complete = AsyncMock(side_effect=auth_fail)
    mock_provider.aclose = AsyncMock()

    monkeypatch.setattr("promptlab.runner.get_provider", lambda pid, **kw: mock_provider)

    cfg = Config(
        description="auth-fail test",
        providers=[ProviderConfig(id="stub/model")],
        prompts=["hello"],
        test_cases=[],
    )
    report = Runner(cfg, dry_run=False).run()
    assert report.total == 1
    cell = report.cells[0]
    assert cell.error is not None
    assert "401" in cell.error or "Unauthorized" in cell.error
    assert call_count == 1  # no retries
