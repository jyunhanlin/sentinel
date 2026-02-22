from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from orchestrator.exchange.paper_engine import PaperEngine, Position
from orchestrator.models import EntryOrder, Side, TradeProposal
from orchestrator.risk.position_sizer import RiskPercentSizer
from orchestrator.stats.calculator import StatsCalculator


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
    def _make_engine(self, *, trade_repo=None, snapshot_repo=None, initial_equity=100000.0):
        return PaperEngine(
            initial_equity=initial_equity,
            taker_fee_rate=0.0005,
            position_sizer=RiskPercentSizer(),
            trade_repo=trade_repo or MagicMock(),
            snapshot_repo=snapshot_repo or MagicMock(),
        )

    def test_open_position(self):
        engine = self._make_engine(initial_equity=10000.0)
        proposal = _make_proposal()
        position = engine.open_position(proposal, current_price=95000.0)
        assert position.side == Side.LONG
        assert position.entry_price == 95000.0
        assert position.quantity == pytest.approx(0.075)
        assert len(engine.get_open_positions()) == 1

    def test_open_multiple_same_symbol(self):
        engine = self._make_engine()
        engine.open_position(_make_proposal(), current_price=95000.0, leverage=10)
        engine.open_position(_make_proposal(risk_pct=1.0), current_price=94000.0, leverage=10)
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
        initial = engine.equity
        engine.open_position(_make_proposal(stop_loss=93000.0), current_price=95000.0)
        engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=92500.0)
        # equity should decrease by PnL + fees
        assert engine.equity < initial

    def test_open_positions_risk_pct(self):
        engine = self._make_engine()
        engine.open_position(_make_proposal(risk_pct=1.5), current_price=95000.0, leverage=10)
        engine.open_position(_make_proposal(risk_pct=2.0), current_price=95000.0, leverage=10)
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


class TestPaperEngineStats:
    def test_close_position_saves_stats_snapshot(self):
        """When a position is closed, stats should be calculated and snapshot saved."""
        trade_repo = MagicMock()
        trade_repo.get_all_closed.return_value = []
        snapshot_repo = MagicMock()
        stats_calc = StatsCalculator()

        engine = PaperEngine(
            initial_equity=10000.0,
            taker_fee_rate=0.0005,
            position_sizer=RiskPercentSizer(),
            trade_repo=trade_repo,
            snapshot_repo=snapshot_repo,
            stats_calculator=stats_calc,
        )

        # Open and close a position
        proposal = TradeProposal(
            symbol="BTC/USDT:USDT", side=Side.LONG, entry=EntryOrder(type="market"),
            position_size_risk_pct=1.0, stop_loss=93000.0, take_profit=[97000.0],
            time_horizon="4h", confidence=0.7, invalid_if=[], rationale="test",
        )
        engine.open_position(proposal, current_price=95000.0)

        # Mock the closed trades for stats calculation
        closed_trade_mock = MagicMock()
        closed_trade_mock.pnl = -150.0
        closed_trade_mock.closed_at = MagicMock()
        closed_trade_mock.closed_at.date.return_value = "2026-02-19"
        trade_repo.get_all_closed.return_value = [closed_trade_mock]

        # Trigger SL
        results = engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=92000.0)
        assert len(results) == 1

        # Verify snapshot was saved with stats
        snapshot_repo.save_snapshot.assert_called()
        call_kwargs = snapshot_repo.save_snapshot.call_args[1]
        assert "total_pnl" in call_kwargs
        assert "win_rate" in call_kwargs
        assert "total_trades" in call_kwargs


def test_position_has_leverage_fields():
    pos = Position(
        trade_id="t1",
        proposal_id="p1",
        symbol="BTC/USDT:USDT",
        side=Side.LONG,
        entry_price=68000.0,
        quantity=0.1,
        stop_loss=67000.0,
        take_profit=[70000.0],
        opened_at=datetime.now(UTC),
        risk_pct=1.0,
        leverage=10,
        margin=680.0,
        liquidation_price=61880.0,
    )
    assert pos.leverage == 10
    assert pos.margin == 680.0
    assert pos.liquidation_price == 61880.0


class TestMarginCalculation:
    def _make_engine(self):
        return PaperEngine(
            initial_equity=10000.0,
            taker_fee_rate=0.0005,
            position_sizer=RiskPercentSizer(),
            trade_repo=MagicMock(),
            snapshot_repo=MagicMock(),
            maintenance_margin_rate=0.5,
        )

    def test_calculate_margin(self):
        engine = self._make_engine()
        margin = engine.calculate_margin(
            quantity=0.1,
            price=68000.0,
            leverage=10,
        )
        assert margin == pytest.approx(680.0)

    def test_calculate_liquidation_price_long(self):
        engine = self._make_engine()
        liq = engine.calculate_liquidation_price(
            entry_price=68000.0,
            leverage=10,
            side=Side.LONG,
        )
        # liq = 68000 * (1 - 1/10 + 0.005) = 68000 * 0.905 = 61540.0
        assert liq == pytest.approx(61540.0)

    def test_calculate_liquidation_price_short(self):
        engine = self._make_engine()
        liq = engine.calculate_liquidation_price(
            entry_price=68000.0,
            leverage=10,
            side=Side.SHORT,
        )
        # liq = 68000 * (1 + 1/10 - 0.005) = 68000 * 1.095 = 74460.0
        assert liq == pytest.approx(74460.0)

    def test_available_balance(self):
        engine = self._make_engine()
        assert engine.available_balance == 10000.0

    def test_used_margin(self):
        engine = self._make_engine()
        assert engine.used_margin == 0.0


class TestOpenPositionWithLeverage:
    def _make_engine(self, **kwargs):
        defaults = dict(
            initial_equity=10000.0,
            taker_fee_rate=0.0005,
            position_sizer=RiskPercentSizer(),
            trade_repo=MagicMock(),
            snapshot_repo=MagicMock(),
            maintenance_margin_rate=0.5,
        )
        defaults.update(kwargs)
        return PaperEngine(**defaults)

    def test_open_position_with_leverage(self):
        engine = self._make_engine()
        proposal = _make_proposal(stop_loss=67000.0, take_profit=[70000.0])
        pos = engine.open_position(proposal, current_price=68000.0, leverage=10)
        assert pos.leverage == 10
        assert pos.margin == pytest.approx(pos.quantity * 68000.0 / 10)
        assert pos.liquidation_price > 0

    def test_open_position_default_leverage_1(self):
        engine = self._make_engine()
        proposal = _make_proposal()
        pos = engine.open_position(proposal, current_price=95000.0)
        assert pos.leverage == 1

    def test_open_position_insufficient_margin(self):
        # With $10 equity, even 10x leverage on BTC can't fit: margin = qty * 68000 / 10
        # Position sizer: qty = (10 * 0.015) / 1000 = 0.00015, margin = 0.00015 * 68000 / 1 = 10.2
        # At leverage=1, margin > equity
        engine = self._make_engine(initial_equity=10.0)
        proposal = _make_proposal(stop_loss=67000.0, take_profit=[70000.0])
        with pytest.raises(ValueError, match="Insufficient margin"):
            engine.open_position(proposal, current_price=68000.0, leverage=1)
