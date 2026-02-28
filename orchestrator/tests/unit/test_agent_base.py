from unittest.mock import AsyncMock

import pytest

from orchestrator.agents.base import BaseAgent
from orchestrator.llm.client import LLMCallResult, LLMClient
from orchestrator.models import SentimentReport


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


class SkillFakeAgent(BaseAgent[SentimentReport]):
    """Agent that uses skill-based prompt instead of messages."""

    output_model = SentimentReport
    _skill_name = "sentiment"

    def _build_prompt(self, **kwargs) -> str:
        return f"Use the {self._skill_name} skill.\n\nData: test data"

    def _get_default_output(self) -> SentimentReport:
        return SentimentReport(
            sentiment_score=50,
            key_events=[],
            sources=[],
            confidence=0.1,
        )


class TestSkillBasedAgent:
    @pytest.mark.asyncio
    async def test_skill_prompt_sent_as_user_message(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=(
                "Analysis: looks bullish\n\n"
                '```json\n{"sentiment_score": 72, "key_events": [],'
                ' "sources": ["data"], "confidence": 0.8}\n```'
            ),
            model="test",
            input_tokens=200,
            output_tokens=100,
            latency_ms=500,
        )

        agent = SkillFakeAgent(client=mock_client)
        result = await agent.analyze()

        assert result.output.sentiment_score == 72
        assert result.degraded is False

        # Verify prompt is sent as single user message (no system message)
        call_args = mock_client.call.call_args
        messages = call_args[0][0]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "sentiment" in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_skill_response_with_analysis_text_before_json(self):
        """Skill responses include thinking text before the JSON block."""
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=(
                "## Analysis\n\n"
                "The funding rate is slightly positive at 0.03%, "
                "indicating mild bullish sentiment.\n"
                "Volume is stable. Price action shows higher lows.\n\n"
                "```json\n"
                '{"sentiment_score": 62, "key_events": '
                '[{"event": "stable funding rate", "impact": "positive", "source": "market"}], '
                '"sources": ["market_data"], "confidence": 0.6}\n'
                "```\n\n"
                "This concludes the analysis."
            ),
            model="test",
            input_tokens=300,
            output_tokens=200,
            latency_ms=800,
        )

        agent = SkillFakeAgent(client=mock_client)
        result = await agent.analyze()

        assert result.output.sentiment_score == 62
        assert result.degraded is False


class TestBaseAgent:
    @pytest.mark.asyncio
    async def test_successful_call(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=(
                '{"sentiment_score": 72, "key_events": [],'
                ' "sources": ["news"], "confidence": 0.8}'
            ),
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
                content=(
                    '{"sentiment_score": 72, "key_events": [],'
                    ' "sources": [], "confidence": 0.8}'
                ),
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
            content=(
                '{"sentiment_score": 72, "key_events": [],'
                ' "sources": [], "confidence": 0.8}'
            ),
            model="anthropic/claude-opus-4-6",
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        agent = FakeAgent(client=mock_client, max_retries=0)
        await agent.analyze(model_override="anthropic/claude-opus-4-6")

        call_kwargs = mock_client.call.call_args[1]
        assert call_kwargs["model"] == "anthropic/claude-opus-4-6"
