from __future__ import annotations

import time
from abc import ABC, abstractmethod

from litellm import acompletion

from orchestrator.llm.client import LLMCallResult


class LLMBackend(ABC):
    """Abstract interface for LLM call execution."""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMCallResult: ...


class LiteLLMBackend(LLMBackend):
    """API-based backend via LiteLLM."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMCallResult:
        start = time.monotonic()

        response = await acompletion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self._api_key,
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        content = response.choices[0].message.content or ""
        usage = response.usage

        return LLMCallResult(
            content=content,
            model=model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=elapsed_ms,
        )
