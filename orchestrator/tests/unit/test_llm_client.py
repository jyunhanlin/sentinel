import pytest
from unittest.mock import AsyncMock, patch

from orchestrator.llm.client import LLMClient, LLMCallResult


class TestLLMClient:
    def test_create_client(self):
        client = LLMClient(model="anthropic/claude-sonnet-4-20250514", api_key="test-key")
        assert client.model == "anthropic/claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_call_returns_result(self):
        client = LLMClient(model="anthropic/claude-sonnet-4-20250514", api_key="test-key")

        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = '{"sentiment_score": 72}'
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50

        with patch("orchestrator.llm.client.acompletion", return_value=mock_response):
            result = await client.call(
                messages=[{"role": "user", "content": "analyze"}],
            )

        assert isinstance(result, LLMCallResult)
        assert result.content == '{"sentiment_score": 72}'
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_call_with_custom_params(self):
        client = LLMClient(
            model="anthropic/claude-sonnet-4-20250514",
            api_key="test-key",
            temperature=0.5,
            max_tokens=500,
        )

        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = "response"
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 25

        with patch("orchestrator.llm.client.acompletion", return_value=mock_response) as mock_call:
            await client.call(messages=[{"role": "user", "content": "test"}])
            mock_call.assert_called_once()
            call_kwargs = mock_call.call_args[1]
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["max_tokens"] == 500


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
