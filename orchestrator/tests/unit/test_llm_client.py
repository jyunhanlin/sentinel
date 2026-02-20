# orchestrator/tests/unit/test_llm_client.py
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from orchestrator.llm.backend import LLMBackend
from orchestrator.llm.client import LLMCallResult, LLMClient


def _make_backend(
    content: str = "response",
    input_tokens: int = 50,
    output_tokens: int = 25,
) -> LLMBackend:
    """Create a mock LLMBackend that returns a fixed LLMCallResult."""
    backend = AsyncMock(spec=LLMBackend)
    backend.complete.return_value = LLMCallResult(
        content=content,
        model="test-model",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=10,
    )
    return backend


class TestLLMClient:
    def test_create_client(self):
        backend = AsyncMock(spec=LLMBackend)
        client = LLMClient(backend=backend, model="anthropic/claude-sonnet-4-6")
        assert client.model == "anthropic/claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_call_returns_result(self):
        backend = _make_backend(
            content='{"sentiment_score": 72}',
            input_tokens=100,
            output_tokens=50,
        )
        client = LLMClient(backend=backend, model="anthropic/claude-sonnet-4-6")

        result = await client.call(messages=[{"role": "user", "content": "analyze"}])

        assert isinstance(result, LLMCallResult)
        assert result.content == '{"sentiment_score": 72}'
        assert result.input_tokens == 100
        assert result.output_tokens == 50

    @pytest.mark.asyncio
    async def test_call_delegates_params_to_backend(self):
        backend = _make_backend()
        client = LLMClient(
            backend=backend,
            model="anthropic/claude-sonnet-4-6",
            temperature=0.5,
            max_tokens=500,
        )

        await client.call(messages=[{"role": "user", "content": "test"}])

        backend.complete.assert_called_once_with(
            [{"role": "user", "content": "test"}],
            model="anthropic/claude-sonnet-4-6",
            temperature=0.5,
            max_tokens=500,
        )

    @pytest.mark.asyncio
    async def test_call_with_zero_temperature(self):
        backend = _make_backend()
        client = LLMClient(
            backend=backend,
            model="anthropic/claude-sonnet-4-6",
            temperature=0.5,
        )

        await client.call(
            messages=[{"role": "user", "content": "test"}],
            temperature=0.0,
        )

        call_kwargs = backend.complete.call_args
        assert call_kwargs.kwargs["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_call_with_model_override(self):
        backend = _make_backend()
        client = LLMClient(backend=backend, model="anthropic/claude-sonnet-4-6")

        await client.call(
            messages=[{"role": "user", "content": "test"}],
            model="anthropic/claude-opus-4-6",
        )

        call_kwargs = backend.complete.call_args
        assert call_kwargs.kwargs["model"] == "anthropic/claude-opus-4-6"


class TestLLMCallResult:
    def test_result_is_frozen(self):
        result = LLMCallResult(
            content="test",
            model="test-model",
            input_tokens=10,
            output_tokens=5,
            latency_ms=100,
        )
        with pytest.raises(Exception):
            result.content = "modified"  # type: ignore
