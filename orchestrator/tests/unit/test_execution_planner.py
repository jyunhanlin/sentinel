from __future__ import annotations

import pytest

from orchestrator.execution.planner import ExecutionPlanner
from orchestrator.models import EntryOrder, Side, TakeProfit, TradeProposal


class FakeEquityProvider:
    def __init__(self, equity: float = 10_000.0, available: float = 9_000.0):
        self._equity = equity
        self._available = available

    async def get_equity(self) -> float:
        return self._equity

    async def get_available_margin(self) -> float:
        return self._available


class FakeSettings:
    trade_margin_amount: float = 500.0
    paper_taker_fee_rate: float = 0.0005


def _make_proposal(
    *,
    side: Side = Side.LONG,
    entry_price: float | None = None,
    entry_type: str = "market",
    stop_loss: float = 93_000.0,
    take_profit: list[TakeProfit] | None = None,
    leverage: int = 10,
    risk_pct: float = 1.0,
) -> TradeProposal:
    return TradeProposal(
        symbol="BTC/USDT:USDT",
        side=side,
        entry=EntryOrder(type=entry_type, price=entry_price),
        position_size_risk_pct=risk_pct,
        stop_loss=stop_loss,
        take_profit=take_profit or [
            TakeProfit(price=97_000.0, close_pct=50),
            TakeProfit(price=99_000.0, close_pct=100),
        ],
        suggested_leverage=leverage,
        time_horizon="4h",
        confidence=0.82,
        invalid_if=[],
        rationale="Test rationale",
    )


@pytest.mark.asyncio
async def test_create_plan_long_market():
    planner = ExecutionPlanner(
        equity_provider=FakeEquityProvider(),
        config=FakeSettings(),  # type: ignore[arg-type]
    )
    proposal = _make_proposal()
    plan = await planner.create_plan(proposal, current_price=95_000.0)

    # quantity = margin * leverage / price = 500 * 10 / 95000
    expected_qty = 500.0 * 10 / 95_000.0
    assert plan.quantity == pytest.approx(expected_qty, rel=1e-6)
    assert plan.entry_price == 95_000.0
    assert plan.leverage == 10
    assert plan.margin_mode == "isolated"
    assert plan.margin_required == pytest.approx(500.0)
    assert plan.notional_value == pytest.approx(expected_qty * 95_000.0, rel=1e-4)
    assert plan.equity_snapshot == 10_000.0
    assert plan.entry_order.side == "buy"
    assert plan.entry_order.order_type == "market"


@pytest.mark.asyncio
async def test_create_plan_short():
    planner = ExecutionPlanner(
        equity_provider=FakeEquityProvider(),
        config=FakeSettings(),  # type: ignore[arg-type]
    )
    proposal = _make_proposal(
        side=Side.SHORT,
        stop_loss=97_000.0,
        take_profit=[TakeProfit(price=91_000.0, close_pct=100)],
    )
    plan = await planner.create_plan(proposal, current_price=95_000.0)

    assert plan.entry_order.side == "sell"
    assert plan.sl_order is not None
    assert plan.sl_order.side == "buy"  # opposite of entry


@pytest.mark.asyncio
async def test_create_plan_computes_max_loss():
    planner = ExecutionPlanner(
        equity_provider=FakeEquityProvider(),
        config=FakeSettings(),  # type: ignore[arg-type]
    )
    proposal = _make_proposal(stop_loss=93_000.0)
    plan = await planner.create_plan(proposal, current_price=95_000.0)

    # max_loss = quantity * |entry - SL|
    expected_loss = plan.quantity * abs(95_000.0 - 93_000.0)
    assert plan.max_loss == pytest.approx(expected_loss, rel=1e-4)
    assert plan.max_loss_pct == pytest.approx(expected_loss / 10_000.0 * 100, rel=1e-4)


@pytest.mark.asyncio
async def test_create_plan_computes_tp_profits():
    planner = ExecutionPlanner(
        equity_provider=FakeEquityProvider(),
        config=FakeSettings(),  # type: ignore[arg-type]
    )
    proposal = _make_proposal(
        take_profit=[
            TakeProfit(price=97_000.0, close_pct=50),
            TakeProfit(price=99_000.0, close_pct=100),
        ],
    )
    plan = await planner.create_plan(proposal, current_price=95_000.0)

    assert len(plan.tp_profits) == 2
    # TP1: quantity * 50% * (97000 - 95000)
    assert plan.tp_profits[0] == pytest.approx(
        plan.quantity * 0.50 * 2_000.0, rel=1e-4,
    )


@pytest.mark.asyncio
async def test_create_plan_computes_liquidation_price_long():
    planner = ExecutionPlanner(
        equity_provider=FakeEquityProvider(),
        config=FakeSettings(),  # type: ignore[arg-type]
    )
    proposal = _make_proposal(leverage=10)
    plan = await planner.create_plan(proposal, current_price=95_000.0)

    # liq_price for long = entry * (1 - 1/leverage)
    expected_liq = 95_000.0 * (1 - 1 / 10)
    assert plan.liquidation_price == pytest.approx(expected_liq, rel=1e-4)


@pytest.mark.asyncio
async def test_create_plan_computes_risk_reward():
    planner = ExecutionPlanner(
        equity_provider=FakeEquityProvider(),
        config=FakeSettings(),  # type: ignore[arg-type]
    )
    proposal = _make_proposal(
        stop_loss=93_000.0,
        take_profit=[TakeProfit(price=99_000.0, close_pct=100)],
    )
    plan = await planner.create_plan(proposal, current_price=95_000.0)

    # risk = |95000 - 93000| = 2000, reward = |99000 - 95000| = 4000
    assert plan.risk_reward_ratio == pytest.approx(2.0, rel=1e-4)


@pytest.mark.asyncio
async def test_create_plan_limit_order():
    planner = ExecutionPlanner(
        equity_provider=FakeEquityProvider(),
        config=FakeSettings(),  # type: ignore[arg-type]
    )
    proposal = _make_proposal(entry_type="limit", entry_price=94_500.0)
    plan = await planner.create_plan(proposal, current_price=95_000.0)

    assert plan.entry_order.order_type == "limit"
    assert plan.entry_order.price == 94_500.0
    assert plan.entry_price == 94_500.0  # uses limit price, not current


@pytest.mark.asyncio
async def test_create_plan_estimated_fees():
    planner = ExecutionPlanner(
        equity_provider=FakeEquityProvider(),
        config=FakeSettings(),  # type: ignore[arg-type]
    )
    proposal = _make_proposal()
    plan = await planner.create_plan(proposal, current_price=95_000.0)

    # fees = notional * fee_rate (for entry)
    expected_fees = plan.notional_value * 0.0005
    assert plan.estimated_fees == pytest.approx(expected_fees, rel=1e-4)
