import pytest
from unittest.mock import AsyncMock

from orchestrator.agents.base import BaseAgent, AgentResult
from orchestrator.llm.client import LLMCallResult, LLMClient
from orchestrator.llm.schema_validator import ValidationFailure, ValidationSuccess
from orchestrator.models import SentimentReport, KeyEvent


class FakeAgent(BaseAgent[SentimentReport]):
    output_model = SentimentReport

    def _build_messages(self, **kwargs) -> list[dict]:
        return [{"role": "user", "content": "analyze sentiment"}]

    def _get_default_output(self) -> SentimentReport:
        return SentimentReport(
            sentiment_score=50,
            key_events=[],
            sources=[],
            confidence=0.3,
        )

    def _build_retry_messages(self, original_messages: list[dict], error: str) -> list[dict]:
        return original_messages + [
            {"role": "assistant", "content": "invalid"},
            {"role": "user", "content": f"Fix: {error}. Respond with valid JSON only."},
        ]


class TestBaseAgent:
    @pytest.mark.asyncio
    async def test_successful_call(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content='{"sentiment_score": 72, "key_events": [], "sources": ["news"], "confidence": 0.8}',
            model="test",
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        agent = FakeAgent(client=mock_client, max_retries=1)
        result = await agent.analyze()

        assert result.output.sentiment_score == 72
        assert result.degraded is False
        assert len(result.llm_calls) == 1

    @pytest.mark.asyncio
    async def test_retry_then_succeed(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.side_effect = [
            LLMCallResult(
                content='{"sentiment_score": "bad"}',  # invalid
                model="test",
                input_tokens=100,
                output_tokens=50,
                latency_ms=500,
            ),
            LLMCallResult(
                content='{"sentiment_score": 72, "key_events": [], "sources": [], "confidence": 0.8}',
                model="test",
                input_tokens=150,
                output_tokens=60,
                latency_ms=600,
            ),
        ]

        agent = FakeAgent(client=mock_client, max_retries=1)
        result = await agent.analyze()

        assert result.output.sentiment_score == 72
        assert result.degraded is False
        assert len(result.llm_calls) == 2

    @pytest.mark.asyncio
    async def test_degrade_after_all_retries_fail(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content="not json at all",
            model="test",
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        agent = FakeAgent(client=mock_client, max_retries=1)
        result = await agent.analyze()

        assert result.output.sentiment_score == 50  # default
        assert result.output.confidence == 0.3  # default
        assert result.degraded is True
        assert len(result.llm_calls) == 2  # original + 1 retry

    @pytest.mark.asyncio
    async def test_model_override_passed_to_client(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content='{"sentiment_score": 72, "key_events": [], "sources": [], "confidence": 0.8}',
            model="anthropic/claude-opus-4-6",
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        agent = FakeAgent(client=mock_client, max_retries=0)
        await agent.analyze(model_override="anthropic/claude-opus-4-6")

        call_kwargs = mock_client.call.call_args[1]
        assert call_kwargs["model"] == "anthropic/claude-opus-4-6"
