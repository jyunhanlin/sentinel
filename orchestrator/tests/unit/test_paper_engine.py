from unittest.mock import MagicMock

import pytest

from orchestrator.exchange.paper_engine import PaperEngine
from orchestrator.models import EntryOrder, Side, TradeProposal
from orchestrator.risk.position_sizer import RiskPercentSizer


def _make_proposal(
    *,
    side: Side = Side.LONG,
    risk_pct: float = 1.5,
    stop_loss: float = 93000.0,
    take_profit: list[float] | None = None,
) -> TradeProposal:
    return TradeProposal(
        symbol="BTC/USDT:USDT",
        side=side,
        entry=EntryOrder(type="market"),
        position_size_risk_pct=risk_pct,
        stop_loss=stop_loss,
        take_profit=take_profit or [97000.0],
        time_horizon="4h",
        confidence=0.7,
        invalid_if=[],
        rationale="test",
    )


class TestPaperEngine:
    def _make_engine(self, *, trade_repo=None, snapshot_repo=None):
        return PaperEngine(
            initial_equity=10000.0,
            taker_fee_rate=0.0005,
            position_sizer=RiskPercentSizer(),
            trade_repo=trade_repo or MagicMock(),
            snapshot_repo=snapshot_repo or MagicMock(),
        )

    def test_open_position(self):
        engine = self._make_engine()
        proposal = _make_proposal()
        position = engine.open_position(proposal, current_price=95000.0)
        assert position.side == Side.LONG
        assert position.entry_price == 95000.0
        assert position.quantity == pytest.approx(0.075)
        assert len(engine.get_open_positions()) == 1

    def test_open_multiple_same_symbol(self):
        engine = self._make_engine()
        engine.open_position(_make_proposal(), current_price=95000.0)
        engine.open_position(_make_proposal(risk_pct=1.0), current_price=94000.0)
        assert len(engine.get_open_positions()) == 2

    def test_check_sl_triggers_close(self):
        engine = self._make_engine()
        engine.open_position(_make_proposal(stop_loss=93000.0), current_price=95000.0)
        # Price drops below SL
        closed = engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=92500.0)
        assert len(closed) == 1
        assert closed[0].pnl < 0
        assert len(engine.get_open_positions()) == 0

    def test_check_tp_triggers_close(self):
        engine = self._make_engine()
        engine.open_position(
            _make_proposal(take_profit=[97000.0]),
            current_price=95000.0,
        )
        # Price rises above TP
        closed = engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=97500.0)
        assert len(closed) == 1
        assert closed[0].pnl > 0

    def test_check_sl_short_position(self):
        engine = self._make_engine()
        engine.open_position(
            _make_proposal(side=Side.SHORT, stop_loss=97000.0, take_profit=[93000.0]),
            current_price=95000.0,
        )
        # Price rises above SL for short
        closed = engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=97500.0)
        assert len(closed) == 1
        assert closed[0].pnl < 0

    def test_equity_after_loss(self):
        engine = self._make_engine()
        assert engine.equity == 10000.0
        engine.open_position(_make_proposal(stop_loss=93000.0), current_price=95000.0)
        engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=92500.0)
        # equity should decrease by PnL + fees
        assert engine.equity < 10000.0

    def test_open_positions_risk_pct(self):
        engine = self._make_engine()
        engine.open_position(_make_proposal(risk_pct=1.5), current_price=95000.0)
        engine.open_position(_make_proposal(risk_pct=2.0), current_price=95000.0)
        assert engine.open_positions_risk_pct == pytest.approx(3.5)

    def test_no_trigger_when_price_between_sl_tp(self):
        engine = self._make_engine()
        engine.open_position(
            _make_proposal(stop_loss=93000.0, take_profit=[97000.0]),
            current_price=95000.0,
        )
        closed = engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=95500.0)
        assert len(closed) == 0
        assert len(engine.get_open_positions()) == 1
