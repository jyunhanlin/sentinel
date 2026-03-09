# Proposer Refinement Loop Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a CriticAgent and RefinementLoop that iteratively improves trade proposals via Proposer ↔ Critic feedback cycles, enabled per-run.

**Architecture:** New `CriticAgent` evaluates proposals on 4 dimensions. `RefinementLoop` coordinates Proposer → Critic → (revise if needed) iterations. The loop is injected into `PipelineRunner` and activated via a `refinement` flag on `execute()`. Only premium/manual runs enable it.

**Tech Stack:** Python 3.12+, Pydantic (frozen models), pytest + pytest-asyncio, structlog

---

### Task 1: Add CritiqueResult Model

**Files:**
- Modify: `orchestrator/src/orchestrator/models.py`
- Test: `orchestrator/tests/unit/test_models.py`

**Step 1: Write the failing test**

Add to `orchestrator/tests/unit/test_models.py`:

```python
from orchestrator.models import CritiqueResult, DimensionVerdict


class TestCritiqueResult:
    def test_dimension_verdict_frozen(self):
        v = DimensionVerdict(dimension="consistency", passed=True, reason="ok")
        assert v.dimension == "consistency"
        assert v.passed is True
        with pytest.raises(Exception):
            v.dimension = "other"

    def test_critique_result_overall_passed(self):
        verdicts = [
            DimensionVerdict(dimension="consistency", passed=True, reason="ok"),
            DimensionVerdict(dimension="risk_reward", passed=True, reason="ok"),
            DimensionVerdict(dimension="input_respect", passed=True, reason="ok"),
            DimensionVerdict(dimension="parameter_sanity", passed=True, reason="ok"),
        ]
        cr = CritiqueResult(
            verdicts=verdicts,
            overall_passed=True,
            suggestions=[],
            summary="All checks passed",
        )
        assert cr.overall_passed is True
        assert len(cr.verdicts) == 4

    def test_critique_result_with_failures(self):
        verdicts = [
            DimensionVerdict(dimension="consistency", passed=False, reason="Side contradicts analysis"),
            DimensionVerdict(dimension="risk_reward", passed=True, reason="ok"),
        ]
        cr = CritiqueResult(
            verdicts=verdicts,
            overall_passed=False,
            suggestions=["Reconsider side given bearish momentum"],
            summary="Consistency check failed",
        )
        assert cr.overall_passed is False
        assert len(cr.suggestions) == 1
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_models.py::TestCritiqueResult -v`
Expected: FAIL — `ImportError: cannot import name 'CritiqueResult'`

**Step 3: Write minimal implementation**

Add to `orchestrator/src/orchestrator/models.py`:

```python
# --- Critique ---


class DimensionVerdict(BaseModel, frozen=True):
    dimension: str  # "consistency" | "risk_reward" | "input_respect" | "parameter_sanity"
    passed: bool
    reason: str


class CritiqueResult(BaseModel, frozen=True):
    verdicts: list[DimensionVerdict]
    overall_passed: bool
    suggestions: list[str]
    summary: str
```

**Step 4: Run test to verify it passes**

Run: `cd orchestrator && uv run pytest tests/unit/test_models.py::TestCritiqueResult -v`
Expected: PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/models.py orchestrator/tests/unit/test_models.py
git commit -m "feat: add CritiqueResult and DimensionVerdict models"
```

---

### Task 2: Add CriticAgent

**Files:**
- Create: `orchestrator/src/orchestrator/agents/critic.py`
- Test: `orchestrator/tests/unit/test_agent_critic.py`

**Step 1: Write the failing test**

Create `orchestrator/tests/unit/test_agent_critic.py`:

```python
from unittest.mock import AsyncMock

import pytest

from orchestrator.agents.critic import CriticAgent
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.llm.client import LLMCallResult, LLMClient
from orchestrator.models import (
    CatalystReport,
    CorrelationAnalysis,
    CritiqueResult,
    EntryOrder,
    KeyLevel,
    Momentum,
    PositioningAnalysis,
    Side,
    TakeProfit,
    TechnicalAnalysis,
    TradeProposal,
    Trend,
    VolatilityRegime,
)


def _make_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        symbol="BTC/USDT:USDT", timeframe="4h",
        current_price=95200.0, volume_24h=1_000_000.0,
        funding_rate=0.0001,
        ohlcv=[[1700000000000, 94000.0, 95500.0, 93500.0, 95200.0, 1000.0]],
    )


def _make_proposal() -> TradeProposal:
    return TradeProposal(
        symbol="BTC/USDT:USDT", side=Side.LONG,
        entry=EntryOrder(type="market"),
        position_size_risk_pct=1.5, stop_loss=93000.0,
        take_profit=[TakeProfit(price=97000.0, close_pct=100)],
        time_horizon="4h", confidence=0.75,
        invalid_if=[], rationale="Bullish momentum",
    )


def _make_technical(label="short_term") -> TechnicalAnalysis:
    return TechnicalAnalysis(
        label=label, trend=Trend.UP, trend_strength=28.0,
        volatility_regime=VolatilityRegime.MEDIUM, volatility_pct=2.5,
        momentum=Momentum.BULLISH, rsi=62.0,
        key_levels=[KeyLevel(type="support", price=93000.0)],
        risk_flags=[],
    )


def _make_positioning() -> PositioningAnalysis:
    return PositioningAnalysis(
        funding_trend="stable", funding_extreme=False, oi_change_pct=2.0,
        retail_bias="neutral", smart_money_bias="long", squeeze_risk="none",
        liquidity_assessment="normal", risk_flags=[], confidence=0.7,
    )


def _make_catalyst() -> CatalystReport:
    return CatalystReport(
        upcoming_events=[], active_events=[],
        risk_level="low", recommendation="proceed", confidence=0.8,
    )


def _make_correlation() -> CorrelationAnalysis:
    return CorrelationAnalysis(
        dxy_trend="stable", dxy_impact="neutral",
        sp500_regime="risk_on", btc_dominance_trend="stable",
        cross_market_alignment="favorable", risk_flags=[], confidence=0.7,
    )


def _critic_kwargs():
    return {
        "proposal": _make_proposal(),
        "snapshot": _make_snapshot(),
        "technical_short": _make_technical("short_term"),
        "technical_long": _make_technical("long_term"),
        "positioning": _make_positioning(),
        "catalyst": _make_catalyst(),
        "correlation": _make_correlation(),
    }


PASS_RESPONSE = (
    '```json\n{"verdicts": ['
    '{"dimension": "consistency", "passed": true, "reason": "Side matches bullish analysis"},'
    '{"dimension": "risk_reward", "passed": true, "reason": "R:R of 1.8 is acceptable"},'
    '{"dimension": "input_respect", "passed": true, "reason": "No warnings ignored"},'
    '{"dimension": "parameter_sanity", "passed": true, "reason": "SL/TP within normal range"}'
    '], "overall_passed": true, "suggestions": [], '
    '"summary": "Proposal is consistent and well-structured"}\n```'
)

FAIL_RESPONSE = (
    '```json\n{"verdicts": ['
    '{"dimension": "consistency", "passed": false, "reason": "Side is long but momentum is bearish"},'
    '{"dimension": "risk_reward", "passed": true, "reason": "ok"},'
    '{"dimension": "input_respect", "passed": true, "reason": "ok"},'
    '{"dimension": "parameter_sanity", "passed": true, "reason": "ok"}'
    '], "overall_passed": false, '
    '"suggestions": ["Reconsider side given bearish momentum"], '
    '"summary": "Consistency check failed"}\n```'
)


class TestCriticAgent:
    @pytest.mark.asyncio
    async def test_prompt_contains_proposal_and_analysis(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=PASS_RESPONSE, model="test",
            input_tokens=500, output_tokens=200, latency_ms=1500,
        )

        agent = CriticAgent(client=mock_client)
        await agent.analyze(**_critic_kwargs())

        messages = mock_client.call.call_args[0][0]
        prompt = messages[0]["content"]

        assert "critic" in prompt.lower()
        assert "skill" in prompt.lower()
        assert "BTC/USDT:USDT" in prompt
        assert "long" in prompt.lower()       # proposal side
        assert "93000" in prompt               # stop loss
        assert "97000" in prompt               # take profit
        assert "bullish" in prompt.lower()     # momentum from technical

    @pytest.mark.asyncio
    async def test_pass_critique(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=PASS_RESPONSE, model="test",
            input_tokens=500, output_tokens=200, latency_ms=1500,
        )

        agent = CriticAgent(client=mock_client)
        result = await agent.analyze(**_critic_kwargs())

        assert isinstance(result.output, CritiqueResult)
        assert result.output.overall_passed is True
        assert result.degraded is False

    @pytest.mark.asyncio
    async def test_fail_critique(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=FAIL_RESPONSE, model="test",
            input_tokens=500, output_tokens=200, latency_ms=1500,
        )

        agent = CriticAgent(client=mock_client)
        result = await agent.analyze(**_critic_kwargs())

        assert isinstance(result.output, CritiqueResult)
        assert result.output.overall_passed is False
        assert len(result.output.suggestions) == 1

    @pytest.mark.asyncio
    async def test_degrade_returns_default_pass(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content="broken", model="test",
            input_tokens=100, output_tokens=50, latency_ms=500,
        )

        agent = CriticAgent(client=mock_client, max_retries=0)
        result = await agent.analyze(**_critic_kwargs())

        assert result.degraded is True
        assert result.output.overall_passed is True  # default: don't block
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_agent_critic.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'orchestrator.agents.critic'`

**Step 3: Write minimal implementation**

Create `orchestrator/src/orchestrator/agents/critic.py`:

```python
from __future__ import annotations

from orchestrator.agents.base import BaseAgent
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.models import (
    CatalystReport,
    CorrelationAnalysis,
    CritiqueResult,
    DimensionVerdict,
    PositioningAnalysis,
    TechnicalAnalysis,
    TradeProposal,
)


class CriticAgent(BaseAgent[CritiqueResult]):
    output_model = CritiqueResult
    _skill_name = "critic"

    def _build_prompt(self, **kwargs) -> str:
        proposal: TradeProposal = kwargs["proposal"]
        snapshot: MarketSnapshot = kwargs["snapshot"]
        tech_short: TechnicalAnalysis = kwargs["technical_short"]
        tech_long: TechnicalAnalysis = kwargs["technical_long"]
        positioning: PositioningAnalysis = kwargs["positioning"]
        catalyst: CatalystReport = kwargs["catalyst"]
        correlation: CorrelationAnalysis = kwargs["correlation"]

        def _format_risk_flags(flags: list[str]) -> str:
            return ", ".join(flags) or "none"

        tp_str = ", ".join(
            f"{tp.price} ({tp.close_pct}%)" for tp in proposal.take_profit
        ) or "none"

        data = (
            f"=== Proposal Under Review ===\n"
            f"Symbol: {proposal.symbol}\n"
            f"Side: {proposal.side}\n"
            f"Entry: {proposal.entry.type}"
            f"{f' @ {proposal.entry.price}' if proposal.entry.price else ''}\n"
            f"Stop Loss: {proposal.stop_loss}\n"
            f"Take Profit: {tp_str}\n"
            f"Position Size Risk: {proposal.position_size_risk_pct}%\n"
            f"Leverage: {proposal.suggested_leverage}x\n"
            f"Confidence: {proposal.confidence}\n"
            f"Time Horizon: {proposal.time_horizon}\n"
            f"Rationale: {proposal.rationale}\n\n"
            f"=== Market Context ===\n"
            f"Current Price: {snapshot.current_price}\n"
            f"24h Volume: {snapshot.volume_24h:,.0f}\n"
            f"Funding Rate: {snapshot.funding_rate:.6f}\n\n"
            f"=== Short-Term Technical ({tech_short.label}) ===\n"
            f"Trend: {tech_short.trend} (ADX: {tech_short.trend_strength:.1f})\n"
            f"Momentum: {tech_short.momentum} (RSI: {tech_short.rsi:.1f})\n"
            f"Volatility: {tech_short.volatility_regime} ({tech_short.volatility_pct:.1f}%)\n"
            f"Risk Flags: {_format_risk_flags(tech_short.risk_flags)}\n\n"
            f"=== Long-Term Technical ({tech_long.label}) ===\n"
            f"Trend: {tech_long.trend} (ADX: {tech_long.trend_strength:.1f})\n"
            f"Momentum: {tech_long.momentum} (RSI: {tech_long.rsi:.1f})\n"
            f"Volatility: {tech_long.volatility_regime} ({tech_long.volatility_pct:.1f}%)\n"
            f"Risk Flags: {_format_risk_flags(tech_long.risk_flags)}\n\n"
            f"=== Positioning ===\n"
            f"Funding Trend: {positioning.funding_trend} (extreme: {positioning.funding_extreme})\n"
            f"OI Change: {positioning.oi_change_pct:+.1f}%\n"
            f"Squeeze Risk: {positioning.squeeze_risk}\n"
            f"Risk Flags: {_format_risk_flags(positioning.risk_flags)}\n"
            f"Confidence: {positioning.confidence:.2f}\n\n"
            f"=== Catalyst ===\n"
            f"Risk Level: {catalyst.risk_level}\n"
            f"Recommendation: {catalyst.recommendation}\n"
            f"Confidence: {catalyst.confidence:.2f}\n\n"
            f"=== Cross-Market Correlation ===\n"
            f"DXY: {correlation.dxy_trend} ({correlation.dxy_impact})\n"
            f"S&P 500: {correlation.sp500_regime}\n"
            f"Alignment: {correlation.cross_market_alignment}\n"
            f"Risk Flags: {_format_risk_flags(correlation.risk_flags)}\n"
            f"Confidence: {correlation.confidence:.2f}"
        )

        return (
            f"Use the {self._skill_name} skill.\n\n"
            f"{data}"
        )

    def _get_default_output(self) -> CritiqueResult:
        return CritiqueResult(
            verdicts=[
                DimensionVerdict(dimension="consistency", passed=True, reason="degraded — skipped"),
                DimensionVerdict(dimension="risk_reward", passed=True, reason="degraded — skipped"),
                DimensionVerdict(dimension="input_respect", passed=True, reason="degraded — skipped"),
                DimensionVerdict(dimension="parameter_sanity", passed=True, reason="degraded — skipped"),
            ],
            overall_passed=True,
            suggestions=[],
            summary="Critic degraded — proposal accepted without review",
        )
```

**Step 4: Run test to verify it passes**

Run: `cd orchestrator && uv run pytest tests/unit/test_agent_critic.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/agents/critic.py orchestrator/tests/unit/test_agent_critic.py
git commit -m "feat: add CriticAgent with 4-dimension proposal evaluation"
```

---

### Task 3: Add Critic SKILL.md

**Files:**
- Create: `.claude/skills/critic/SKILL.md`

**Step 1: Write the skill file**

Create `.claude/skills/critic/SKILL.md` with:
- Context: Why this critic exists (improve proposal quality)
- Input Description: Table of all inputs (proposal + 5 analyses + snapshot)
- Methodology: Step-by-step evaluation across 4 dimensions
- Decision Criteria: Specific rules for pass/fail per dimension
- Output: JSON schema matching `CritiqueResult`

Key evaluation rules:
- **Consistency:** Side must align with majority of analysis signals. Rationale must match parameters.
- **Risk/Reward:** R:R ratio >= 1.5 for directional trades. Position size risk <= 2%. Size should scale with confidence.
- **Input Respect:** If catalyst.recommendation == "wait", side should be flat. If >= 2 analyses have risk_flags, extra scrutiny.
- **Parameter Sanity:** SL distance proportional to volatility. Leverage <= 20x for high volatility. TP distances reasonable.

**Step 2: Commit**

```bash
git add .claude/skills/critic/SKILL.md
git commit -m "feat: add critic skill with 4-dimension evaluation methodology"
```

---

### Task 4: Add RefinementLoop

**Files:**
- Create: `orchestrator/src/orchestrator/pipeline/refinement.py`
- Test: `orchestrator/tests/unit/test_refinement_loop.py`

**Step 1: Write the failing tests**

Create `orchestrator/tests/unit/test_refinement_loop.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_refinement_loop.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `orchestrator/src/orchestrator/pipeline/refinement.py`:

```python
from __future__ import annotations

import structlog
from pydantic import BaseModel

from orchestrator.agents.base import AgentResult, BaseAgent
from orchestrator.llm.client import LLMCallResult
from orchestrator.models import CritiqueResult, TradeProposal

logger = structlog.get_logger(__name__)


class RefinementResult(BaseModel, frozen=True):
    proposal: TradeProposal
    critique: CritiqueResult | None = None
    rounds: int = 0
    exhausted: bool = False
    all_llm_calls: list[LLMCallResult] = []
    proposer_degraded: bool = False

    model_config = {"arbitrary_types_allowed": True}


class RefinementLoop:
    def __init__(
        self,
        *,
        proposer: BaseAgent[TradeProposal],
        critic: BaseAgent[CritiqueResult],
        max_rounds: int = 2,
    ) -> None:
        self._proposer = proposer
        self._critic = critic
        self._max_rounds = max_rounds

    async def run(
        self,
        *,
        model_override: str | None = None,
        **proposer_kwargs,
    ) -> RefinementResult:
        all_llm_calls: list[LLMCallResult] = []

        # Initial proposer call
        proposer_result = await self._proposer.analyze(
            model_override=model_override, **proposer_kwargs,
        )
        all_llm_calls.extend(proposer_result.llm_calls)

        if proposer_result.degraded:
            logger.warning("refinement_proposer_degraded")
            return RefinementResult(
                proposal=proposer_result.output,
                proposer_degraded=True,
                all_llm_calls=all_llm_calls,
            )

        proposal = proposer_result.output
        critique: CritiqueResult | None = None

        for round_num in range(1, self._max_rounds + 1):
            logger.info("refinement_round", round=round_num)

            # Run critic
            critic_result = await self._critic.analyze(
                proposal=proposal,
                model_override=model_override,
                **proposer_kwargs,
            )
            all_llm_calls.extend(critic_result.llm_calls)
            critique = critic_result.output

            if critique.overall_passed:
                logger.info("refinement_passed", round=round_num)
                return RefinementResult(
                    proposal=proposal,
                    critique=critique,
                    rounds=round_num,
                    all_llm_calls=all_llm_calls,
                )

            # Critique failed — revise if more rounds remain
            if round_num < self._max_rounds:
                logger.info(
                    "refinement_revising",
                    round=round_num,
                    suggestions=critique.suggestions,
                )
                feedback = self._format_feedback(critique)
                proposer_result = await self._proposer.analyze(
                    model_override=model_override,
                    critique_feedback=feedback,
                    **proposer_kwargs,
                )
                all_llm_calls.extend(proposer_result.llm_calls)
                proposal = proposer_result.output

        # Exhausted all rounds
        logger.warning("refinement_exhausted", max_rounds=self._max_rounds)
        return RefinementResult(
            proposal=proposal,
            critique=critique,
            rounds=self._max_rounds,
            exhausted=True,
            all_llm_calls=all_llm_calls,
        )

    @staticmethod
    def _format_feedback(critique: CritiqueResult) -> str:
        failed = [v for v in critique.verdicts if not v.passed]
        lines = [f"Previous proposal failed critique ({critique.summary}):"]
        for v in failed:
            lines.append(f"- [{v.dimension}] {v.reason}")
        if critique.suggestions:
            lines.append("\nSuggested improvements:")
            for s in critique.suggestions:
                lines.append(f"- {s}")
        return "\n".join(lines)
```

**Step 4: Run test to verify it passes**

Run: `cd orchestrator && uv run pytest tests/unit/test_refinement_loop.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/pipeline/refinement.py orchestrator/tests/unit/test_refinement_loop.py
git commit -m "feat: add RefinementLoop for Proposer ↔ Critic iteration"
```

---

### Task 5: Update ProposerAgent to Accept Critique Feedback

**Files:**
- Modify: `orchestrator/src/orchestrator/agents/proposer.py`
- Modify: `orchestrator/tests/unit/test_agent_proposer.py`

**Step 1: Write the failing test**

Add to `orchestrator/tests/unit/test_agent_proposer.py`:

```python
    @pytest.mark.asyncio
    async def test_prompt_includes_critique_feedback(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=FLAT_RESPONSE, model="test",
            input_tokens=300, output_tokens=150, latency_ms=1000,
        )

        agent = ProposerAgent(client=mock_client)
        kwargs = _analysis_kwargs()
        kwargs["critique_feedback"] = (
            "Previous proposal failed critique:\n"
            "- [consistency] Side contradicts bearish analysis\n"
            "Suggested improvements:\n"
            "- Reconsider side"
        )
        await agent.analyze(**kwargs)

        messages = mock_client.call.call_args[0][0]
        prompt = messages[0]["content"]

        assert "Previous proposal failed critique" in prompt
        assert "Reconsider side" in prompt
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_agent_proposer.py::TestProposerAgent::test_prompt_includes_critique_feedback -v`
Expected: FAIL — critique feedback not in prompt

**Step 3: Update ProposerAgent._build_prompt()**

In `orchestrator/src/orchestrator/agents/proposer.py`, update `_build_prompt()` to
accept and append `critique_feedback`:

```python
    def _build_prompt(self, **kwargs) -> str:
        # ... existing code ...
        critique_feedback: str | None = kwargs.get("critique_feedback")

        prompt = (
            f"Use the {self._skill_name} skill.\n\n"
            f"{data}"
        )

        if critique_feedback:
            prompt += f"\n\n=== Critique Feedback (MUST ADDRESS) ===\n{critique_feedback}"

        return prompt
```

**Step 4: Run test to verify it passes**

Run: `cd orchestrator && uv run pytest tests/unit/test_agent_proposer.py -v`
Expected: PASS (all tests including new one)

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/agents/proposer.py orchestrator/tests/unit/test_agent_proposer.py
git commit -m "feat: ProposerAgent accepts critique_feedback for refinement loop"
```

---

### Task 6: Integrate RefinementLoop into PipelineRunner

**Files:**
- Modify: `orchestrator/src/orchestrator/pipeline/runner.py`
- Modify: `orchestrator/tests/unit/test_runner.py`

**Step 1: Write the failing tests**

Add to `orchestrator/tests/unit/test_runner.py`:

```python
from orchestrator.pipeline.refinement import RefinementLoop, RefinementResult
from orchestrator.models import CritiqueResult, DimensionVerdict


def _make_pass_critique() -> CritiqueResult:
    return CritiqueResult(
        verdicts=[
            DimensionVerdict(dimension="consistency", passed=True, reason="ok"),
        ],
        overall_passed=True, suggestions=[], summary="ok",
    )


class TestPipelineRunnerRefinement:
    @pytest.mark.asyncio
    async def test_refinement_enabled_uses_loop(self):
        """When refinement=True and loop is injected, uses RefinementLoop."""
        proposal = _make_proposal()
        refinement_loop = AsyncMock(spec=RefinementLoop)
        refinement_loop.run.return_value = RefinementResult(
            proposal=proposal,
            critique=_make_pass_critique(),
            rounds=1,
            all_llm_calls=[make_llm_call()],
        )

        runner, mocks = _make_runner(refinement_loop=refinement_loop)
        result = await runner.execute("BTC/USDT:USDT", refinement=True)

        assert result.status == "completed"
        assert result.refinement_rounds == 1
        assert result.refinement_exhausted is False
        refinement_loop.run.assert_called_once()
        # Proposer should NOT be called directly
        mocks["proposer_agent"].analyze.assert_not_called()

    @pytest.mark.asyncio
    async def test_refinement_disabled_uses_proposer_directly(self):
        """When refinement=False, falls back to direct proposer call."""
        refinement_loop = AsyncMock(spec=RefinementLoop)

        runner, mocks = _make_runner(refinement_loop=refinement_loop)
        result = await runner.execute("BTC/USDT:USDT", refinement=False)

        assert result.status == "completed"
        assert result.refinement_rounds == 0
        refinement_loop.run.assert_not_called()
        mocks["proposer_agent"].analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_refinement_loop_injected(self):
        """When no loop injected, refinement=True is ignored."""
        runner, mocks = _make_runner()
        result = await runner.execute("BTC/USDT:USDT", refinement=True)

        assert result.status == "completed"
        assert result.refinement_rounds == 0
        mocks["proposer_agent"].analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_refinement_exhausted_flag(self):
        """Exhausted refinement sets flag on PipelineResult."""
        fail_critique = CritiqueResult(
            verdicts=[DimensionVerdict(dimension="consistency", passed=False, reason="bad")],
            overall_passed=False, suggestions=["fix it"], summary="failed",
        )
        refinement_loop = AsyncMock(spec=RefinementLoop)
        refinement_loop.run.return_value = RefinementResult(
            proposal=_make_proposal(),
            critique=fail_critique,
            rounds=2,
            exhausted=True,
            all_llm_calls=[make_llm_call(), make_llm_call()],
        )

        runner, _ = _make_runner(refinement_loop=refinement_loop)
        result = await runner.execute("BTC/USDT:USDT", refinement=True)

        assert result.refinement_rounds == 2
        assert result.refinement_exhausted is True
        assert result.critique is not None
        assert result.critique.overall_passed is False
```

**Step 2: Run tests to verify they fail**

Run: `cd orchestrator && uv run pytest tests/unit/test_runner.py::TestPipelineRunnerRefinement -v`
Expected: FAIL

**Step 3: Update PipelineRunner and PipelineResult**

In `orchestrator/src/orchestrator/pipeline/runner.py`:

1. Add new fields to `PipelineResult`:
```python
    refinement_rounds: int = 0
    refinement_exhausted: bool = False
    critique: CritiqueResult | None = None
```

2. Add `refinement_loop` to `__init__`:
```python
    from orchestrator.pipeline.refinement import RefinementLoop
    # Add parameter:
    refinement_loop: RefinementLoop | None = None,
```

3. Add `refinement` parameter to `execute()`:
```python
    async def execute(
        self, symbol: str, *, timeframe: str = "1h",
        model_override: str | None = None,
        refinement: bool = False,
    ) -> PipelineResult:
```

4. Replace Step 3 logic:
```python
    # Step 3: Run Proposer (with optional refinement)
    if refinement and self._refinement_loop is not None:
        refinement_result = await self._refinement_loop.run(
            snapshot=snapshot,
            technical_short=tech_short_result.output,
            technical_long=tech_long_result.output,
            positioning=positioning_result.output,
            catalyst=catalyst_result.output,
            correlation=correlation_result.output,
            model_override=model_override,
        )
        # Save all LLM calls from refinement
        for call in refinement_result.all_llm_calls:
            self._llm_call_repo.save_call(
                run_id=run_id, agent_type="proposer_refinement",
                prompt="", response=call.content, model=call.model,
                latency_ms=call.latency_ms,
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
            )
        proposer_output = refinement_result.proposal
        proposer_degraded = refinement_result.proposer_degraded
        proposer_result_for_save = None  # already saved above
    else:
        proposer_result = await self._proposer_agent.analyze(
            snapshot=snapshot,
            technical_short=tech_short_result.output,
            technical_long=tech_long_result.output,
            positioning=positioning_result.output,
            catalyst=catalyst_result.output,
            correlation=correlation_result.output,
            model_override=model_override,
        )
        self._save_llm_calls(run_id, "proposer", proposer_result)
        proposer_output = proposer_result.output
        proposer_degraded = proposer_result.degraded
        refinement_result = None
```

5. Update `_build_result` to accept refinement fields and pass them through.

**Step 4: Run tests**

Run: `cd orchestrator && uv run pytest tests/unit/test_runner.py -v`
Expected: PASS (all existing + new tests)

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/pipeline/runner.py orchestrator/tests/unit/test_runner.py
git commit -m "feat: integrate RefinementLoop into PipelineRunner"
```

---

### Task 7: Add Config and Wire in __main__.py

**Files:**
- Modify: `orchestrator/src/orchestrator/config.py`
- Modify: `orchestrator/src/orchestrator/__main__.py`
- Modify: `orchestrator/tests/unit/test_config.py` (if exists, verify new fields)

**Step 1: Add config fields**

In `orchestrator/src/orchestrator/config.py`:

```python
    # Refinement Loop
    refinement_enabled: bool = False
    refinement_max_rounds: int = 2
```

**Step 2: Wire CriticAgent and RefinementLoop in create_app_components()**

In `orchestrator/src/orchestrator/__main__.py`:

```python
from orchestrator.agents.critic import CriticAgent
from orchestrator.pipeline.refinement import RefinementLoop

# After proposer_agent creation:
critic_agent = CriticAgent(client=llm_client, max_retries=llm_max_retries)

# After agents, before runner:
refinement_loop = None
if refinement_enabled:
    refinement_loop = RefinementLoop(
        proposer=proposer_agent,
        critic=critic_agent,
        max_rounds=refinement_max_rounds,
    )

# Pass to runner:
runner = PipelineRunner(
    ...,
    refinement_loop=refinement_loop,
)
```

Add `refinement_enabled` and `refinement_max_rounds` parameters to
`create_app_components()` and `_build_components()`.

**Step 3: Update scheduler to pass refinement flag**

In `orchestrator/src/orchestrator/pipeline/scheduler.py`, update `run_once()` to
accept `refinement: bool = False` and pass it to `runner.execute()`.

Update `_run_daily_premium()` to pass `refinement=True`:

```python
async def _run_daily_premium(self) -> None:
    await self.run_once(model_override=self.premium_model, refinement=True)
```

**Step 4: Run all tests**

Run: `cd orchestrator && uv run pytest -v`
Expected: PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/config.py orchestrator/src/orchestrator/__main__.py orchestrator/src/orchestrator/pipeline/scheduler.py
git commit -m "feat: wire CriticAgent and RefinementLoop with config"
```

---

### Task 8: Telegram /run Command Support

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/bot.py`

**Step 1: Update /run to support `refine` flag**

The `/run` command currently accepts `symbol` and optional `model`. Add `refine` as
an optional keyword:

```
/run BTC opus refine  → refinement=True
/run BTC              → refinement=False
```

Pass `refinement=True` to `scheduler.run_once()` when `refine` is in args.

**Step 2: Update formatters for refinement metadata**

In `orchestrator/src/orchestrator/telegram/formatters.py`, add refinement info to the
proposal message when `refinement_rounds > 0`:

```
🔄 Refined: 2 rounds (passed)
```

or:

```
⚠️ Refined: 2 rounds (exhausted)
```

**Step 3: Run all tests**

Run: `cd orchestrator && uv run pytest -v`
Expected: PASS

**Step 4: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/bot.py orchestrator/src/orchestrator/telegram/formatters.py
git commit -m "feat: /run supports refine flag, formatters show refinement status"
```

---

### Task 9: Final Verification

**Step 1: Run full test suite with coverage**

```bash
cd orchestrator && uv run pytest -v --cov=orchestrator --cov-report=term-missing
```

Expected: All tests pass, coverage ≥ 80%.

**Step 2: Run linter**

```bash
cd orchestrator && uv run ruff check src/ tests/
```

Expected: No errors.

**Step 3: Verify ruff format**

```bash
cd orchestrator && uv run ruff format --check src/ tests/
```

Expected: No formatting issues.
