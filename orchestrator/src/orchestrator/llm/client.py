# orchestrator/src/orchestrator/llm/client.py
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel

if TYPE_CHECKING:
    from orchestrator.llm.backend import LLMBackend

logger = structlog.get_logger(__name__)


class LLMCallResult(BaseModel, frozen=True):
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int


class LLMClient:
    def __init__(
        self,
        backend: LLMBackend,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        max_concurrent: int = 0,
    ) -> None:
        self._backend = backend
        self.model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None

    @asynccontextmanager
    async def throttle(self) -> AsyncIterator[None]:
        """Acquire concurrency slot. No-op when max_concurrent is 0."""
        if self._semaphore is not None:
            async with self._semaphore:
                yield
        else:
            yield

    async def call(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMCallResult:
        effective_model = model or self.model

        result = await self._backend.complete(
            messages,
            model=effective_model,
            temperature=temperature if temperature is not None else self._temperature,
            max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
        )

        logger.info(
            "llm_call_complete",
            latency_ms=result.latency_ms,
            model=effective_model,
            tokens_in=result.input_tokens,
            tokens_out=result.output_tokens,
        )

        return result
