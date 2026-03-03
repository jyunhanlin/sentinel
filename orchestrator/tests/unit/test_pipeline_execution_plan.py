from __future__ import annotations

import pytest

from orchestrator.execution.planner import ExecutionPlanner
from orchestrator.models import EntryOrder, Side, TakeProfit, TradeProposal


class FakeEquityProvider:
    async def get_equity(self) -> float:
        return 10_000.0

    async def get_available_margin(self) -> float:
        return 9_000.0


class FakeSettings:
    trade_margin_amount: float = 500.0
    paper_taker_fee_rate: float = 0.0005


@pytest.mark.asyncio
async def test_planner_integrates_with_proposal():
    """Integration test: real planner with real proposal model."""
    planner = ExecutionPlanner(
        equity_provider=FakeEquityProvider(),
        config=FakeSettings(),  # type: ignore[arg-type]
    )
    proposal = TradeProposal(
        symbol="BTC/USDT:USDT",
        side=Side.LONG,
        entry=EntryOrder(type="market"),
        position_size_risk_pct=1.0,
        stop_loss=93_000.0,
        take_profit=[
            TakeProfit(price=97_000.0, close_pct=50),
            TakeProfit(price=99_000.0, close_pct=100),
        ],
        suggested_leverage=10,
        time_horizon="4h",
        confidence=0.82,
        invalid_if=[],
        rationale="Test rationale.",
    )

    plan = await planner.create_plan(proposal, current_price=95_000.0)

    # Verify plan is complete and consistent
    assert plan.proposal_id == proposal.proposal_id
    assert plan.symbol == "BTC/USDT:USDT"
    assert plan.side == "long"
    assert plan.entry_order.side == "buy"
    assert plan.sl_order is not None
    assert plan.sl_order.side == "sell"
    assert len(plan.tp_orders) == 2
    assert plan.max_loss > 0
    assert plan.risk_reward_ratio > 0
    assert plan.liquidation_price < plan.entry_price
