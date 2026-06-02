"""The test execution engine.

Builds a (provider x prompt x test_case) matrix, runs completions
concurrently, then evaluates assertions. Completion is async; evaluation
is synchronous and happens after all completions finish.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import httpx

from promptlab.config import Config, TestCase
from promptlab.evaluators import EvalContext, EvalResult, evaluate
from promptlab.providers import get_provider
from promptlab.providers.base import LLMProvider, ProviderResponse
from promptlab.utils import render

# Retry settings for transient HTTP errors (429, 5xx).
_MAX_RETRIES = 2
_RETRY_BASE_SECONDS = 1.0
_RETRYABLE_CODES = frozenset({429, 500, 502, 503, 504})


@dataclass
class CellResult:
    provider_id: str
    prompt: str
    prompt_index: int
    case_index: int
    case_name: Optional[str]
    rendered_prompt: str
    provider_cfg_index: int = 0
    response: Optional[ProviderResponse] = None
    error: Optional[str] = None
    skipped: bool = False
    evaluations: list[EvalResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        if self.error is not None:
            return False
        if self.skipped:  # dry-run: previewed, not judged
            return True
        return all(e.passed for e in self.evaluations)


@dataclass
class RunReport:
    description: str
    cells: list[CellResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.cells)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cells if c.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def total_cost(self) -> float:
        return sum(c.response.cost for c in self.cells if c.response)


class Runner:
    def __init__(
        self, config: Config, *, dry_run: bool = False, concurrency: int = 5, judge_id: Optional[str] = None
    ) -> None:
        self.config = config
        self.dry_run = dry_run
        self.concurrency = max(1, concurrency)
        self.judge_id = judge_id

    def _build_cells(self) -> list[CellResult]:
        cases = self.config.test_cases or [TestCase()]
        cells: list[CellResult] = []
        for cfg_idx, provider_cfg in enumerate(self.config.providers):
            for p_idx, prompt in enumerate(self.config.prompts):
                for c_idx, case in enumerate(cases):
                    cells.append(
                        CellResult(
                            provider_id=provider_cfg.id,
                            prompt=prompt,
                            prompt_index=p_idx,
                            case_index=c_idx,
                            case_name=case.name,
                            rendered_prompt=render(prompt, case.vars),
                            provider_cfg_index=cfg_idx,
                        )
                    )
        return cells

    def _build_provider_cache(self) -> dict[int, LLMProvider]:
        """Create exactly one provider instance per unique config index."""
        cache: dict[int, LLMProvider] = {}
        for idx, cfg in enumerate(self.config.providers):
            cache[idx] = get_provider(cfg.id, **cfg.options)
        return cache

    async def _complete_cell(
        self,
        cell: CellResult,
        provider: LLMProvider,
        sem: asyncio.Semaphore,
        progress_cb: Optional[Callable[[], Any]] = None,
    ) -> None:
        if self.dry_run:
            cell.skipped = True
            cell.response = ProviderResponse(text="[dry-run: no request sent]", model=cell.provider_id, latency_ms=0.0)
            if progress_cb:
                progress_cb()
            return
        async with sem:
            last_exc: Optional[Exception] = None
            for attempt in range(_MAX_RETRIES + 1):
                try:
                    cell.response = await provider.complete(cell.rendered_prompt)
                    last_exc = None
                    break
                except Exception as exc:  # noqa: BLE001 - surface any provider error per-cell
                    last_exc = exc
                    should_retry = False
                    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
                        # Transient network errors — always worth retrying.
                        should_retry = True
                    elif isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in _RETRYABLE_CODES:
                        # HTTP 429 (rate limit) or 5xx (server error).
                        should_retry = True
                    if should_retry and attempt < _MAX_RETRIES:
                        await asyncio.sleep(_RETRY_BASE_SECONDS * (2**attempt))
                        continue
                    break
            if last_exc is not None:
                cell.error = f"{type(last_exc).__name__}: {last_exc}"
        if progress_cb:
            progress_cb()

    async def _run_async(
        self,
        cells: list[CellResult],
        provider_cache: dict[int, LLMProvider],
        progress_cb: Optional[Callable[[], Any]] = None,
    ) -> None:
        sem = asyncio.Semaphore(self.concurrency)
        try:
            await asyncio.gather(
                *(self._complete_cell(c, provider_cache[c.provider_cfg_index], sem, progress_cb) for c in cells)
            )
        finally:
            # Close all provider HTTP clients.
            for provider in provider_cache.values():
                await provider.aclose()

    def run(self, progress_cb: Optional[Callable[[], Any]] = None) -> RunReport:
        cells = self._build_cells()
        provider_cache = self._build_provider_cache()
        asyncio.run(self._run_async(cells, provider_cache, progress_cb))

        # Evaluation phase (synchronous; no event loop running here).
        cases = self.config.test_cases or [TestCase()]
        judge = get_provider(self.judge_id) if self.judge_id else None
        ctx = EvalContext(judge=judge)
        try:
            for cell in cells:
                if cell.error is not None or cell.response is None or cell.skipped:
                    continue
                for assertion in cases[cell.case_index].assertions:
                    try:
                        cell.evaluations.append(evaluate(cell.response, assertion, ctx))
                    except Exception as exc:  # noqa: BLE001 – never let a broken evaluator abort the run
                        cell.evaluations.append(
                            EvalResult(False, assertion.type, f"evaluator error: {type(exc).__name__}: {exc}")
                        )
        finally:
            if judge is not None:
                asyncio.run(judge.aclose())

        return RunReport(description=self.config.description, cells=cells)
