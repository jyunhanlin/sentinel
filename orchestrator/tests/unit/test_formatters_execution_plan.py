from __future__ import annotations

from orchestrator.execution.plan import ExecutionPlan, OrderInstruction
from orchestrator.telegram.formatters import format_execution_plan


def _make_plan() -> ExecutionPlan:
    return ExecutionPlan(
        proposal_id="test-123",
        symbol="BTC/USDT:USDT",
        side="long",
        entry_order=OrderInstruction(
            symbol="BTC/USDT:USDT",
            side="buy",
            order_type="market",
            quantity=0.05,
        ),
        sl_order=OrderInstruction(
            symbol="BTC/USDT:USDT",
            side="sell",
            order_type="market",
            quantity=0.05,
            stop_price=93_000.0,
            reduce_only=True,
        ),
        tp_orders=[
            OrderInstruction(
                symbol="BTC/USDT:USDT",
                side="sell",
                order_type="market",
                quantity=0.025,
                stop_price=97_000.0,
                reduce_only=True,
            ),
            OrderInstruction(
                symbol="BTC/USDT:USDT",
                side="sell",
                order_type="market",
                quantity=0.025,
                stop_price=99_000.0,
                reduce_only=True,
            ),
        ],
        margin_mode="isolated",
        leverage=10,
        quantity=0.05,
        entry_price=95_000.0,
        notional_value=4_750.0,
        margin_required=475.0,
        liquidation_price=85_500.0,
        estimated_fees=2.375,
        max_loss=100.0,
        max_loss_pct=1.0,
        tp_profits=[50.0, 100.0],
        risk_reward_ratio=2.0,
        equity_snapshot=10_000.0,
    )


def test_format_upper_section_contains_trade_params():
    text = format_execution_plan(
        plan=_make_plan(),
        confidence=0.82,
        time_horizon="4h",
    )
    assert "LONG" in text
    assert "BTC/USDT" in text
    assert "95,000" in text        # entry price
    assert "0.05" in text           # quantity
    assert "475" in text            # margin
    assert "85,500" in text         # liq price
    assert "93,000" in text         # SL
    assert "97,000" in text         # TP1
    assert "99,000" in text         # TP2
    assert "Risk/Reward" in text


def test_format_upper_section_contains_loss_and_profit():
    text = format_execution_plan(
        plan=_make_plan(),
        confidence=0.82,
        time_horizon="4h",
    )
    assert "100" in text           # max loss $100
    assert "1.0%" in text          # max loss pct


def test_format_lower_section_contains_analysis():
    text = format_execution_plan(
        plan=_make_plan(),
        confidence=0.82,
        time_horizon="4h",
        analysis_summary={
            "technical": "UP, BULLISH momentum\nSupport: 93,200 / Resist: 97,500",
            "positioning": "Funding -0.01%\nOI +3.2%",
            "catalyst": "No high-impact events",
            "correlation": "DXY↓ S&P risk-on",
        },
        rationale="BTC breaking above resistance with low funding.",
    )
    assert "Technical" in text
    assert "Positioning" in text
    assert "Catalyst" in text
    assert "Correlation" in text
    assert "BTC breaking above" in text


def test_format_without_analysis_shows_upper_only():
    text = format_execution_plan(
        plan=_make_plan(),
        confidence=0.82,
        time_horizon="4h",
    )
    # Should not crash, just no lower section
    assert "LONG" in text
    assert "Analysis" not in text


def test_format_short_side():
    plan = _make_plan()
    short_plan = plan.model_copy(update={"side": "short"})
    text = format_execution_plan(
        plan=short_plan,
        confidence=0.75,
        time_horizon="1d",
    )
    assert "SHORT" in text
    assert "\U0001f534" in text  # red circle emoji
