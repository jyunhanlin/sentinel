from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.llm.backend import (
    LiteLLMBackend,
    LLMBackend,
    _extract_system_prompt,
    _flatten_messages,
)
from orchestrator.llm.client import LLMCallResult


class TestLiteLLMBackend:
    def test_is_llm_backend(self):
        backend = LiteLLMBackend(api_key="test-key")
        assert isinstance(backend, LLMBackend)

    @pytest.mark.asyncio
    async def test_complete_returns_llm_call_result(self):
        backend = LiteLLMBackend(api_key="test-key")

        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = '{"score": 42}'
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50

        with patch("orchestrator.llm.backend.acompletion", return_value=mock_response):
            result = await backend.complete(
                messages=[{"role": "user", "content": "test"}],
                model="anthropic/claude-sonnet-4-6",
                temperature=0.2,
                max_tokens=2000,
            )

        assert isinstance(result, LLMCallResult)
        assert result.content == '{"score": 42}'
        assert result.model == "anthropic/claude-sonnet-4-6"
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_complete_passes_all_params_to_litellm(self):
        backend = LiteLLMBackend(api_key="my-key")

        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = "ok"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        with patch("orchestrator.llm.backend.acompletion", return_value=mock_response) as mock_call:
            await backend.complete(
                messages=[{"role": "user", "content": "hello"}],
                model="anthropic/claude-opus-4-6",
                temperature=0.7,
                max_tokens=500,
            )
            mock_call.assert_called_once_with(
                model="anthropic/claude-opus-4-6",
                messages=[{"role": "user", "content": "hello"}],
                temperature=0.7,
                max_tokens=500,
                api_key="my-key",
            )

    @pytest.mark.asyncio
    async def test_complete_handles_none_usage(self):
        backend = LiteLLMBackend(api_key="test-key")

        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = "ok"
        mock_response.usage = None

        with patch("orchestrator.llm.backend.acompletion", return_value=mock_response):
            result = await backend.complete(
                messages=[{"role": "user", "content": "test"}],
                model="test-model",
                temperature=0.2,
                max_tokens=2000,
            )

        assert result.input_tokens == 0
        assert result.output_tokens == 0


class TestMessageConversion:
    def test_extract_system_prompt_from_messages(self):
        messages = [
            {"role": "system", "content": "You are an analyst."},
            {"role": "user", "content": "Analyze BTC."},
        ]
        system, remaining = _extract_system_prompt(messages)
        assert system == "You are an analyst."
        assert remaining == [{"role": "user", "content": "Analyze BTC."}]

    def test_extract_system_prompt_no_system_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        system, remaining = _extract_system_prompt(messages)
        assert system is None
        assert remaining == [{"role": "user", "content": "Hello"}]

    def test_flatten_single_user_message(self):
        messages = [{"role": "user", "content": "Analyze BTC."}]
        result = _flatten_messages(messages)
        assert result == "Analyze BTC."

    def test_flatten_multi_turn_retry(self):
        messages = [
            {"role": "user", "content": "Analyze BTC."},
            {"role": "assistant", "content": "(invalid output)"},
            {"role": "user", "content": "Your previous response failed validation."},
        ]
        result = _flatten_messages(messages)
        assert "Analyze BTC." in result
        assert "(invalid output)" in result
        assert "Your previous response failed validation." in result
