"""Abstract provider interface shared by every backend."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx


@dataclass
class ProviderResponse:
    """A single completion result, normalized across providers."""

    text: str
    model: str
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0
    raw_response: Optional[dict[str, Any]] = field(default=None, repr=False)


class LLMProvider(abc.ABC):
    """Base class for all providers.

    A provider knows how to turn a prompt into a :class:`ProviderResponse`
    for one specific model. Provider instances are cheap; create one per
    (provider, model) pair.
    """

    #: Short registry name, e.g. "openai". Set on each subclass.
    name: str = "base"

    #: Default request timeout in seconds. Override per-subclass.
    default_timeout: float = 60.0

    def __init__(self, model: str, **options: Any) -> None:
        self.model = model
        self.options = options
        self._http: Optional[httpx.AsyncClient] = None

    @property
    def _client(self) -> httpx.AsyncClient:
        """Lazily create a shared ``httpx.AsyncClient`` for connection reuse."""
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=self.options.get("timeout", self.default_timeout))
        return self._http

    async def aclose(self) -> None:
        """Close the underlying HTTP client. Safe to call multiple times."""
        if self._http is not None and not self._http.is_closed:
            await self._http.aclose()
            self._http = None

    @abc.abstractmethod
    async def complete(self, prompt: str, *, system: Optional[str] = None, **kwargs: Any) -> ProviderResponse:
        """Run a single completion and return a normalized response."""
        raise NotImplementedError

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate USD cost. Override per-provider with a real price table.

        Prices drift constantly, so the default returns 0.0 rather than
        pretending to be accurate.
        """
        return 0.0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(model={self.model!r})"
