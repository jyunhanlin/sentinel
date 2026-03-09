from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.agents.base import AgentResult
from orchestrator.llm.client import LLMCallResult
from orchestrator.models import (
    CritiqueResult,
    DimensionVerdict,
    EntryOrder,
    Side,
    TakeProfit,
    TradeProposal,
)
from orchestrator.pipeline.refinement import RefinementLoop, RefinementResult


def _make_proposal(*, confidence=0.75) -> TradeProposal:
    return TradeProposal(
        symbol="BTC/USDT:USDT", side=Side.LONG,
        entry=EntryOrder(type="market"),
        position_size_risk_pct=1.5, stop_loss=93000.0,
        take_profit=[TakeProfit(price=97000.0, close_pct=100)],
        time_horizon="4h", confidence=confidence,
        invalid_if=[], rationale="Bullish momentum",
    )


def _make_pass_critique() -> CritiqueResult:
    return CritiqueResult(
        verdicts=[
            DimensionVerdict(dimension="consistency", passed=True, reason="ok"),
            DimensionVerdict(dimension="risk_reward", passed=True, reason="ok"),
            DimensionVerdict(dimension="input_respect", passed=True, reason="ok"),
            DimensionVerdict(dimension="parameter_sanity", passed=True, reason="ok"),
        ],
        overall_passed=True, suggestions=[], summary="All good",
    )


def _make_fail_critique() -> CritiqueResult:
    return CritiqueResult(
        verdicts=[
            DimensionVerdict(dimension="consistency", passed=False, reason="Side contradicts"),
        ],
        overall_passed=False,
        suggestions=["Reconsider side"],
        summary="Consistency failed",
    )


def _make_llm_call() -> LLMCallResult:
    return LLMCallResult(
        content="{}", model="test",
        input_tokens=100, output_tokens=50, latency_ms=500,
    )


class TestRefinementLoop:
    @pytest.mark.asyncio
    async def test_first_critique_passes(self):
        """Proposer → Critic passes → done in 1 round."""
        proposer = AsyncMock()
        proposer.analyze.return_value = AgentResult(
            output=_make_proposal(), llm_calls=[_make_llm_call()],
        )
        critic = AsyncMock()
        critic.analyze.return_value = AgentResult(
            output=_make_pass_critique(), llm_calls=[_make_llm_call()],
        )

        loop = RefinementLoop(proposer=proposer, critic=critic, max_rounds=2)
        result = await loop.run(snapshot=MagicMock())

        assert isinstance(result, RefinementResult)
        assert result.rounds == 1
        assert result.exhausted is False
        assert result.proposal.side == Side.LONG
        assert result.critique is not None
        assert result.critique.overall_passed is True
        assert proposer.analyze.call_count == 1
        assert critic.analyze.call_count == 1

    @pytest.mark.asyncio
    async def test_critique_fails_then_passes(self):
        """Proposer → Critic fails → Proposer revises → Critic passes."""
        proposal_v1 = _make_proposal(confidence=0.6)
        proposal_v2 = _make_proposal(confidence=0.8)

        proposer = AsyncMock()
        proposer.analyze.side_effect = [
            AgentResult(output=proposal_v1, llm_calls=[_make_llm_call()]),
            AgentResult(output=proposal_v2, llm_calls=[_make_llm_call()]),
        ]
        critic = AsyncMock()
        critic.analyze.side_effect = [
            AgentResult(output=_make_fail_critique(), llm_calls=[_make_llm_call()]),
            AgentResult(output=_make_pass_critique(), llm_calls=[_make_llm_call()]),
        ]

        loop = RefinementLoop(proposer=proposer, critic=critic, max_rounds=2)
        result = await loop.run(snapshot=MagicMock())

        assert result.rounds == 2
        assert result.exhausted is False
        assert result.proposal == proposal_v2
        assert proposer.analyze.call_count == 2
        assert critic.analyze.call_count == 2

    @pytest.mark.asyncio
    async def test_max_rounds_exhausted(self):
        """All rounds fail → returns last proposal with exhausted=True."""
        proposer = AsyncMock()
        proposer.analyze.return_value = AgentResult(
            output=_make_proposal(), llm_calls=[_make_llm_call()],
        )
        critic = AsyncMock()
        critic.analyze.return_value = AgentResult(
            output=_make_fail_critique(), llm_calls=[_make_llm_call()],
        )

        loop = RefinementLoop(proposer=proposer, critic=critic, max_rounds=2)
        result = await loop.run(snapshot=MagicMock())

        assert result.rounds == 2
        assert result.exhausted is True
        assert proposer.analyze.call_count == 2  # initial + 1 revision
        assert critic.analyze.call_count == 2

    @pytest.mark.asyncio
    async def test_proposer_degraded_skips_critique(self):
        """If proposer degrades, skip critic entirely."""
        proposer = AsyncMock()
        proposer.analyze.return_value = AgentResult(
            output=_make_proposal(), degraded=True, llm_calls=[_make_llm_call()],
        )
        critic = AsyncMock()

        loop = RefinementLoop(proposer=proposer, critic=critic, max_rounds=2)
        result = await loop.run(snapshot=MagicMock())

        assert result.proposer_degraded is True
        assert result.rounds == 0
        assert result.critique is None
        critic.analyze.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_llm_calls_collected(self):
        """All LLM calls from both agents across rounds are collected."""
        proposer = AsyncMock()
        proposer.analyze.return_value = AgentResult(
            output=_make_proposal(), llm_calls=[_make_llm_call()],
        )
        critic = AsyncMock()
        critic.analyze.return_value = AgentResult(
            output=_make_pass_critique(), llm_calls=[_make_llm_call()],
        )

        loop = RefinementLoop(proposer=proposer, critic=critic, max_rounds=2)
        result = await loop.run(snapshot=MagicMock())

        assert len(result.all_llm_calls) == 2  # 1 proposer + 1 critic

    @pytest.mark.asyncio
    async def test_revision_prompt_includes_critique_feedback(self):
        """When revising, proposer receives critique suggestions."""
        proposal_v1 = _make_proposal()

        proposer = AsyncMock()
        proposer.analyze.side_effect = [
            AgentResult(output=proposal_v1, llm_calls=[_make_llm_call()]),
            AgentResult(output=_make_proposal(), llm_calls=[_make_llm_call()]),
        ]
        proposer._build_prompt = MagicMock(return_value="prompt")

        critic = AsyncMock()
        fail_critique = _make_fail_critique()
        critic.analyze.side_effect = [
            AgentResult(output=fail_critique, llm_calls=[_make_llm_call()]),
            AgentResult(output=_make_pass_critique(), llm_calls=[_make_llm_call()]),
        ]

        loop = RefinementLoop(proposer=proposer, critic=critic, max_rounds=2)
        await loop.run(snapshot=MagicMock())

        # Second proposer call should have critique_feedback kwarg
        second_call_kwargs = proposer.analyze.call_args_list[1][1]
        assert "critique_feedback" in second_call_kwargs
        assert "Reconsider side" in second_call_kwargs["critique_feedback"]

    @pytest.mark.asyncio
    async def test_critic_degraded_returns_early(self):
        """If critic degrades, return proposal with critic_degraded flag."""
        proposer = AsyncMock()
        proposer.analyze.return_value = AgentResult(
            output=_make_proposal(), llm_calls=[_make_llm_call()],
        )
        critic = AsyncMock()
        critic.analyze.return_value = AgentResult(
            output=_make_pass_critique(), degraded=True,
            llm_calls=[_make_llm_call()],
        )

        loop = RefinementLoop(proposer=proposer, critic=critic, max_rounds=2)
        result = await loop.run(snapshot=MagicMock())

        assert result.critic_degraded is True
        assert result.rounds == 1
        assert result.proposal.side == Side.LONG
        # Should not attempt revision since critic is degraded
        assert proposer.analyze.call_count == 1
