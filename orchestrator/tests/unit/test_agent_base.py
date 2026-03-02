from unittest.mock import AsyncMock

import pytest

from orchestrator.agents.base import BaseAgent
from orchestrator.llm.client import LLMCallResult, LLMClient
from orchestrator.models import Momentum, TechnicalAnalysis, Trend, VolatilityRegime


def _default_technical() -> TechnicalAnalysis:
    return TechnicalAnalysis(
        label="short_term", trend=Trend.RANGE, trend_strength=15.0,
        volatility_regime=VolatilityRegime.LOW, volatility_pct=1.0,
        momentum=Momentum.NEUTRAL, rsi=50.0, key_levels=[], risk_flags=[],
    )


_VALID_JSON = (
    '{"label": "short_term", "trend": "up", "trend_strength": 28.0,'
    ' "volatility_regime": "medium", "volatility_pct": 2.5,'
    ' "momentum": "bullish", "rsi": 62.0, "key_levels": [], "risk_flags": []}'
)


class FakeAgent(BaseAgent[TechnicalAnalysis]):
    output_model = TechnicalAnalysis

    def _build_messages(self, **kwargs) -> list[dict]:
        return [{"role": "user", "content": "analyze market"}]

    def _get_default_output(self) -> TechnicalAnalysis:
        return _default_technical()

    def _build_retry_messages(self, original_messages: list[dict], error: str) -> list[dict]:
        return original_messages + [
            {"role": "assistant", "content": "invalid"},
            {"role": "user", "content": f"Fix: {error}. Respond with valid JSON only."},
        ]


class SkillFakeAgent(BaseAgent[TechnicalAnalysis]):
    """Agent that uses skill-based prompt instead of messages."""

    output_model = TechnicalAnalysis
    _skill_name = "technical"

    def _build_prompt(self, **kwargs) -> str:
        return f"Use the {self._skill_name} skill.\n\nData: test data"

    def _get_default_output(self) -> TechnicalAnalysis:
        return _default_technical()


class TestSkillBasedAgent:
    @pytest.mark.asyncio
    async def test_skill_prompt_sent_as_user_message(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=f"Analysis: looks bullish\n\n```json\n{_VALID_JSON}\n```",
            model="test",
            input_tokens=200,
            output_tokens=100,
            latency_ms=500,
        )

        agent = SkillFakeAgent(client=mock_client)
        result = await agent.analyze()

        assert result.output.trend == Trend.UP
        assert result.degraded is False

        # Verify prompt is sent as single user message (no system message)
        call_args = mock_client.call.call_args
        messages = call_args[0][0]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "technical" in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_skill_response_with_analysis_text_before_json(self):
        """Skill responses include thinking text before the JSON block."""
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=(
                "## Analysis\n\n"
                "The market shows higher lows with expanding volume.\n\n"
                f"```json\n{_VALID_JSON}\n```\n\n"
                "This concludes the analysis."
            ),
            model="test",
            input_tokens=300,
            output_tokens=200,
            latency_ms=800,
        )

        agent = SkillFakeAgent(client=mock_client)
        result = await agent.analyze()

        assert result.output.trend == Trend.UP
        assert result.degraded is False


class TestBaseAgent:
    @pytest.mark.asyncio
    async def test_successful_call(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=_VALID_JSON,
            model="test",
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        agent = FakeAgent(client=mock_client, max_retries=1)
        result = await agent.analyze()

        assert result.output.trend == Trend.UP
        assert result.degraded is False
        assert len(result.llm_calls) == 1

    @pytest.mark.asyncio
    async def test_retry_then_succeed(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.side_effect = [
            LLMCallResult(
                content='{"trend": "bad"}',  # invalid
                model="test",
                input_tokens=100,
                output_tokens=50,
                latency_ms=500,
            ),
            LLMCallResult(
                content=_VALID_JSON,
                model="test",
                input_tokens=150,
                output_tokens=60,
                latency_ms=600,
            ),
        ]

        agent = FakeAgent(client=mock_client, max_retries=1)
        result = await agent.analyze()

        assert result.output.trend == Trend.UP
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

        assert result.output.trend == Trend.RANGE  # default
        assert result.degraded is True
        assert len(result.llm_calls) == 2  # original + 1 retry

    @pytest.mark.asyncio
    async def test_model_override_passed_to_client(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=_VALID_JSON,
            model="anthropic/claude-opus-4-6",
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        agent = FakeAgent(client=mock_client, max_retries=0)
        await agent.analyze(model_override="anthropic/claude-opus-4-6")

        call_kwargs = mock_client.call.call_args[1]
        assert call_kwargs["model"] == "anthropic/claude-opus-4-6"
