"""Integration test: full paper trading lifecycle with leverage."""

from unittest.mock import MagicMock

import pytest

from orchestrator.exchange.paper_engine import PaperEngine
from orchestrator.models import EntryOrder, Side, TakeProfit, TradeProposal
from orchestrator.risk.position_sizer import RiskPercentSizer


def _make_proposal(*, stop_loss: float, take_profit: list[TakeProfit]) -> TradeProposal:
    return TradeProposal(
        symbol="BTC/USDT:USDT",
        side=Side.LONG,
        entry=EntryOrder(type="market"),
        position_size_risk_pct=1.0,
        stop_loss=stop_loss,
        take_profit=take_profit,
        time_horizon="4h",
        confidence=0.75,
        invalid_if=[],
        rationale="integration test",
    )


def _make_engine(*, initial_equity: float = 10000.0) -> PaperEngine:
    return PaperEngine(
        initial_equity=initial_equity,
        taker_fee_rate=0.0005,
        position_sizer=RiskPercentSizer(),
        trade_repo=MagicMock(),
        snapshot_repo=MagicMock(),
        maintenance_margin_rate=0.5,
    )


class TestFullPaperTradingFlow:
    def test_approve_add_reduce_close_flow(self):
        """Full lifecycle: open with leverage → add → reduce → close."""
        engine = _make_engine()

        # 1. Open position with 10x leverage
        proposal = _make_proposal(
            stop_loss=67000.0,
            take_profit=[TakeProfit(price=70000.0, close_pct=100)],
        )
        pos = engine.open_position(proposal, current_price=68000.0, leverage=10)
        assert pos.leverage == 10
        assert pos.margin > 0
        assert pos.liquidation_price > 0
        initial_qty = pos.quantity

        # 2. Add to position (1% risk at higher price)
        pos = engine.add_to_position(
            trade_id=pos.trade_id, risk_pct=1.0, current_price=69000.0,
        )
        assert pos.quantity > initial_qty
        assert 68000.0 < pos.entry_price < 69000.0  # weighted avg

        # 3. Reduce 50% at profit
        result = engine.reduce_position(
            trade_id=pos.trade_id, pct=50.0, current_price=69500.0,
        )
        assert result.pnl > 0  # price > avg entry
        remaining = engine.get_open_positions()
        assert len(remaining) == 1
        assert remaining[0].quantity < pos.quantity

        # 4. Close remaining at profit
        result = engine.close_position(
            trade_id=pos.trade_id, current_price=70000.0,
        )
        assert result.reason == "manual"
        assert result.pnl > 0
        assert len(engine.get_open_positions()) == 0
        assert engine.equity > 10000.0  # net profitable

    def test_liquidation_wipes_margin(self):
        """Position at high leverage gets liquidated, losing full margin."""
        engine = _make_engine()

        proposal = _make_proposal(
            stop_loss=60000.0,
            take_profit=[TakeProfit(price=80000.0, close_pct=100)],
        )
        pos = engine.open_position(proposal, current_price=68000.0, leverage=50)
        liq_price = pos.liquidation_price
        margin = pos.margin

        # Price drops to liquidation level
        closed = engine.check_sl_tp(
            symbol="BTC/USDT:USDT", current_price=liq_price - 1,
        )
        assert len(closed) == 1
        assert closed[0].reason == "liquidation"
        assert closed[0].pnl == -margin  # lose entire margin
        assert len(engine.get_open_positions()) == 0

    def test_stop_loss_before_liquidation(self):
        """SL triggers before liquidation at moderate leverage."""
        engine = _make_engine()

        proposal = _make_proposal(
            stop_loss=67000.0,
            take_profit=[TakeProfit(price=70000.0, close_pct=100)],
        )
        pos = engine.open_position(proposal, current_price=68000.0, leverage=10)

        # SL at 67000 is above liquidation price (~61540 at 10x)
        assert pos.stop_loss == 67000.0
        assert pos.liquidation_price < 67000.0

        # Price drops to SL
        closed = engine.check_sl_tp(
            symbol="BTC/USDT:USDT", current_price=66900.0,
        )
        assert len(closed) == 1
        assert closed[0].reason == "sl"
        assert closed[0].pnl < 0  # loss but not full margin

    def test_multiple_positions_margin_tracking(self):
        """Multiple positions correctly track used/available margin."""
        engine = _make_engine(initial_equity=100000.0)

        proposal1 = _make_proposal(
            stop_loss=67000.0,
            take_profit=[TakeProfit(price=70000.0, close_pct=100)],
        )
        pos1 = engine.open_position(proposal1, current_price=68000.0, leverage=10)
        margin1 = pos1.margin

        proposal2 = TradeProposal(
            symbol="ETH/USDT:USDT",
            side=Side.SHORT,
            entry=EntryOrder(type="market"),
            position_size_risk_pct=1.0,
            stop_loss=3600.0,
            take_profit=[TakeProfit(price=3200.0, close_pct=100)],
            time_horizon="4h",
            confidence=0.7,
            invalid_if=[],
            rationale="test short",
        )
        pos2 = engine.open_position(proposal2, current_price=3400.0, leverage=20)
        margin2 = pos2.margin

        assert len(engine.get_open_positions()) == 2
        assert engine.used_margin == pytest.approx(margin1 + margin2, rel=0.01)
        assert engine.available_balance < 100000.0

    def test_position_pnl_with_leverage(self):
        """get_position_with_pnl returns correct ROE% amplified by leverage."""
        engine = _make_engine()

        proposal = _make_proposal(
            stop_loss=67000.0,
            take_profit=[TakeProfit(price=70000.0, close_pct=100)],
        )
        pos = engine.open_position(proposal, current_price=68000.0, leverage=10)

        # Price up 1% = $680
        info = engine.get_position_with_pnl(
            trade_id=pos.trade_id, current_price=68680.0,
        )
        # PnL% ~1%, but ROE% ~10% due to 10x leverage
        assert info["pnl_pct"] == pytest.approx(1.0, abs=0.1)
        assert info["roe_pct"] == pytest.approx(10.0, abs=1.0)
