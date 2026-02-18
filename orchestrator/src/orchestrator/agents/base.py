from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import structlog
from pydantic import BaseModel

from orchestrator.llm.client import LLMCallResult, LLMClient
from orchestrator.llm.schema_validator import ValidationSuccess, validate_llm_output

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class AgentResult(BaseModel, Generic[T]):
    output: T
    degraded: bool = False
    llm_calls: list[LLMCallResult] = []

    model_config = {"arbitrary_types_allowed": True}


class BaseAgent(ABC, Generic[T]):
    output_model: type[T]

    def __init__(self, client: LLMClient, max_retries: int = 1) -> None:
        self._client = client
        self._max_retries = max_retries

    async def analyze(self, *, model_override: str | None = None, **kwargs) -> AgentResult[T]:
        messages = self._build_messages(**kwargs)
        llm_calls: list[LLMCallResult] = []

        for attempt in range(1 + self._max_retries):
            call_result = await self._client.call(messages, model=model_override)
            llm_calls.append(call_result)

            validation = validate_llm_output(call_result.content, self.output_model)

            if isinstance(validation, ValidationSuccess):
                logger.info(
                    "agent_success",
                    agent=self.__class__.__name__,
                    attempt=attempt + 1,
                )
                return AgentResult(
                    output=validation.value,
                    degraded=False,
                    llm_calls=llm_calls,
                )

            # Retry with error feedback
            logger.warning(
                "agent_validation_failed",
                agent=self.__class__.__name__,
                attempt=attempt + 1,
                error=validation.error_message,
            )

            if attempt < self._max_retries:
                messages = self._build_retry_messages(messages, validation.error_message)

        # All retries exhausted — degrade
        logger.warning(
            "agent_degraded",
            agent=self.__class__.__name__,
            total_attempts=1 + self._max_retries,
        )
        return AgentResult(
            output=self._get_default_output(),
            degraded=True,
            llm_calls=llm_calls,
        )

    @abstractmethod
    def _build_messages(self, **kwargs) -> list[dict]:
        ...

    @abstractmethod
    def _get_default_output(self) -> T:
        ...

    def _build_retry_messages(self, original_messages: list[dict], error: str) -> list[dict]:
        return original_messages + [
            {"role": "assistant", "content": "(invalid output)"},
            {
                "role": "user",
                "content": (
                    f"Your previous response failed validation: {error}\n"
                    "Please respond with ONLY a valid JSON object matching the schema."
                ),
            },
        ]
