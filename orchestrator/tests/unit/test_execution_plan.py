from __future__ import annotations

import pytest

from orchestrator.execution.plan import ExecutionPlan, OrderInstruction


def test_order_instruction_is_frozen():
    oi = OrderInstruction(
        symbol="BTC/USDT:USDT",
        side="buy",
        order_type="market",
        quantity=0.05,
    )
    with pytest.raises(Exception):
        oi.quantity = 1.0  # type: ignore[misc]


def test_execution_plan_is_frozen():
    oi = OrderInstruction(
        symbol="BTC/USDT:USDT",
        side="buy",
        order_type="market",
        quantity=0.05,
    )
    plan = ExecutionPlan(
        proposal_id="test-id",
        symbol="BTC/USDT:USDT",
        side="long",
        entry_order=oi,
        sl_order=None,
        tp_orders=[],
        margin_mode="isolated",
        leverage=10,
        quantity=0.05,
        entry_price=95_000.0,
        notional_value=4_750.0,
        margin_required=475.0,
        liquidation_price=85_950.0,
        estimated_fees=4.75,
        max_loss=100.0,
        max_loss_pct=1.0,
        tp_profits=[],
        risk_reward_ratio=2.0,
        equity_snapshot=10_000.0,
    )
    with pytest.raises(Exception):
        plan.quantity = 1.0  # type: ignore[misc]


def test_execution_plan_has_all_required_fields():
    oi = OrderInstruction(
        symbol="BTC/USDT:USDT",
        side="buy",
        order_type="market",
        quantity=0.05,
    )
    plan = ExecutionPlan(
        proposal_id="test-id",
        symbol="BTC/USDT:USDT",
        side="long",
        entry_order=oi,
        sl_order=None,
        tp_orders=[],
        margin_mode="isolated",
        leverage=10,
        quantity=0.05,
        entry_price=95_000.0,
        notional_value=4_750.0,
        margin_required=475.0,
        liquidation_price=85_950.0,
        estimated_fees=4.75,
        max_loss=100.0,
        max_loss_pct=1.0,
        tp_profits=[],
        risk_reward_ratio=2.0,
        equity_snapshot=10_000.0,
    )
    assert plan.margin_mode == "isolated"
    assert plan.leverage == 10
    assert plan.entry_price == 95_000.0
