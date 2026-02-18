from __future__ import annotations

import time

import structlog
from litellm import acompletion
from pydantic import BaseModel

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
        model: str,
        api_key: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def call(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMCallResult:
        start = time.monotonic()

        response = await acompletion(
            model=self.model,
            messages=messages,
            temperature=temperature or self._temperature,
            max_tokens=max_tokens or self._max_tokens,
            api_key=self._api_key,
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        content = response.choices[0].message.content

        result = LLMCallResult(
            content=content,
            model=self.model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            latency_ms=elapsed_ms,
        )

        logger.info(
            "llm_call_complete",
            model=self.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms,
        )

        return result
