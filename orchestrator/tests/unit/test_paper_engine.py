from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from orchestrator.exchange.paper_engine import PaperEngine, Position
from orchestrator.models import EntryOrder, Side, TakeProfit, TradeProposal
from orchestrator.risk.position_sizer import RiskPercentSizer
from orchestrator.stats.calculator import StatsCalculator


def _make_proposal(
    *,
    side: Side = Side.LONG,
    risk_pct: float = 1.5,
    stop_loss: float = 93000.0,
    take_profit: list[TakeProfit] | None = None,
) -> TradeProposal:
    return TradeProposal(
        symbol="BTC/USDT:USDT",
        side=side,
        entry=EntryOrder(type="market"),
        position_size_risk_pct=risk_pct,
        stop_loss=stop_loss,
        take_profit=take_profit or [TakeProfit(price=97000.0, close_pct=100)],
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
            _make_proposal(take_profit=[TakeProfit(price=97000.0, close_pct=100)]),
            current_price=95000.0,
        )
        # Price rises above TP
        closed = engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=97500.0)
        assert len(closed) == 1
        assert closed[0].pnl > 0

    def test_check_sl_short_position(self):
        engine = self._make_engine()
        engine.open_position(
            _make_proposal(
                side=Side.SHORT, stop_loss=97000.0,
                take_profit=[TakeProfit(price=93000.0, close_pct=100)],
            ),
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
            _make_proposal(
                stop_loss=93000.0,
                take_profit=[TakeProfit(price=97000.0, close_pct=100)],
            ),
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
            position_size_risk_pct=1.0, stop_loss=93000.0,
            take_profit=[TakeProfit(price=97000.0, close_pct=100)],
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
        take_profit=[TakeProfit(price=70000.0, close_pct=100)],
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
        proposal = _make_proposal(
            stop_loss=67000.0,
            take_profit=[TakeProfit(price=70000.0, close_pct=100)],
        )
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
        proposal = _make_proposal(
            stop_loss=67000.0,
            take_profit=[TakeProfit(price=70000.0, close_pct=100)],
        )
        with pytest.raises(ValueError, match="Insufficient margin"):
            engine.open_position(proposal, current_price=68000.0, leverage=1)


class TestLiquidation:
    def _make_engine(self):
        return PaperEngine(
            initial_equity=10000.0,
            taker_fee_rate=0.0005,
            position_sizer=RiskPercentSizer(),
            trade_repo=MagicMock(),
            snapshot_repo=MagicMock(),
            maintenance_margin_rate=0.5,
        )

    def test_liquidation_triggers_before_sl_long(self):
        engine = self._make_engine()
        proposal = _make_proposal(
            stop_loss=67000.0,
            take_profit=[TakeProfit(price=70000.0, close_pct=100)],
        )
        engine.open_position(proposal, current_price=68000.0, leverage=10)
        # Liq price ~61540, SL=67000. Price crashes below liq
        closed = engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=61000.0)
        assert len(closed) == 1
        assert closed[0].reason == "liquidation"

    def test_sl_triggers_when_above_liq_long(self):
        engine = self._make_engine()
        proposal = _make_proposal(
            stop_loss=67000.0,
            take_profit=[TakeProfit(price=70000.0, close_pct=100)],
        )
        engine.open_position(proposal, current_price=68000.0, leverage=10)
        # Price hits SL but above liq
        closed = engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=66500.0)
        assert len(closed) == 1
        assert closed[0].reason == "sl"

    def test_liquidation_short(self):
        engine = self._make_engine()
        proposal = _make_proposal(
            side=Side.SHORT, stop_loss=70000.0,
            take_profit=[TakeProfit(price=65000.0, close_pct=100)],
        )
        engine.open_position(proposal, current_price=68000.0, leverage=10)
        # Liq price ~74460, push above
        closed = engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=75000.0)
        assert len(closed) == 1
        assert closed[0].reason == "liquidation"

    def test_liquidation_pnl_is_negative_margin(self):
        engine = self._make_engine()
        proposal = _make_proposal(
            stop_loss=67000.0,
            take_profit=[TakeProfit(price=70000.0, close_pct=100)],
        )
        pos = engine.open_position(proposal, current_price=68000.0, leverage=10)
        margin = pos.margin
        closed = engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=61000.0)
        assert len(closed) == 1
        assert closed[0].pnl == pytest.approx(-margin)


class TestManualOperations:
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

    def test_add_to_position(self):
        engine = self._make_engine()
        proposal = _make_proposal(
            stop_loss=67000.0,
            take_profit=[TakeProfit(price=70000.0, close_pct=100)],
        )
        pos = engine.open_position(proposal, current_price=68000.0, leverage=10)
        old_qty = pos.quantity
        old_entry = pos.entry_price

        updated = engine.add_to_position(
            trade_id=pos.trade_id, risk_pct=1.0, current_price=69000.0,
        )
        assert updated.quantity > old_qty
        # Avg price should be between old and new
        assert old_entry < updated.entry_price < 69000.0
        assert updated.margin > pos.margin
        assert len(engine.get_open_positions()) == 1  # still one position

    def test_reduce_position_50pct(self):
        engine = self._make_engine()
        proposal = _make_proposal(
            stop_loss=67000.0,
            take_profit=[TakeProfit(price=70000.0, close_pct=100)],
        )
        pos = engine.open_position(proposal, current_price=68000.0, leverage=10)
        original_qty = pos.quantity

        result = engine.reduce_position(
            trade_id=pos.trade_id, pct=50.0, current_price=69000.0,
        )
        assert result.quantity == pytest.approx(original_qty * 0.5, rel=0.01)
        assert result.pnl > 0  # price went up for long
        remaining = engine.get_open_positions()
        assert len(remaining) == 1
        assert remaining[0].quantity == pytest.approx(original_qty * 0.5, rel=0.01)

    def test_close_position(self):
        engine = self._make_engine()
        proposal = _make_proposal(
            stop_loss=67000.0,
            take_profit=[TakeProfit(price=70000.0, close_pct=100)],
        )
        pos = engine.open_position(proposal, current_price=68000.0, leverage=10)

        result = engine.close_position(
            trade_id=pos.trade_id, current_price=69000.0,
        )
        assert result.reason == "manual"
        assert len(engine.get_open_positions()) == 0

    def test_reduce_100pct_closes_position(self):
        engine = self._make_engine()
        proposal = _make_proposal(
            stop_loss=67000.0,
            take_profit=[TakeProfit(price=70000.0, close_pct=100)],
        )
        pos = engine.open_position(proposal, current_price=68000.0, leverage=10)

        engine.reduce_position(
            trade_id=pos.trade_id, pct=100.0, current_price=69000.0,
        )
        assert len(engine.get_open_positions()) == 0

    def test_add_insufficient_margin(self):
        engine = PaperEngine(
            initial_equity=1000.0,
            taker_fee_rate=0.0005,
            position_sizer=RiskPercentSizer(),
            trade_repo=MagicMock(),
            snapshot_repo=MagicMock(),
            maintenance_margin_rate=0.5,
        )
        proposal = _make_proposal(
            stop_loss=67000.0,
            take_profit=[TakeProfit(price=70000.0, close_pct=100)],
            risk_pct=0.5,
        )
        pos = engine.open_position(proposal, current_price=68000.0, leverage=10)

        with pytest.raises(ValueError, match="Insufficient margin"):
            engine.add_to_position(
                trade_id=pos.trade_id, risk_pct=50.0, current_price=69000.0,
            )

    def test_position_not_found(self):
        engine = self._make_engine()
        with pytest.raises(ValueError, match="not found"):
            engine.close_position(trade_id="nonexistent", current_price=69000.0)

    def test_get_position_with_pnl_long(self):
        engine = self._make_engine()
        proposal = _make_proposal(
            stop_loss=67000.0,
            take_profit=[TakeProfit(price=70000.0, close_pct=100)],
        )
        pos = engine.open_position(proposal, current_price=68000.0, leverage=10)

        info = engine.get_position_with_pnl(trade_id=pos.trade_id, current_price=69000.0)
        assert info["unrealized_pnl"] > 0
        assert info["roe_pct"] > 0
        assert info["pnl_pct"] > 0


class TestPartialTakeProfit:
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

    def test_tp1_triggers_partial_close(self):
        """When price hits TP1 (close_pct=50), only 50% of position should close."""
        engine = self._make_engine()
        proposal = _make_proposal(
            stop_loss=67000.0,
            take_profit=[
                TakeProfit(price=69000.0, close_pct=50),
                TakeProfit(price=70000.0, close_pct=100),
            ],
        )
        pos = engine.open_position(proposal, current_price=68000.0, leverage=10)
        original_qty = pos.quantity

        # Price hits TP1
        results = engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=69500.0)
        assert len(results) == 1
        assert results[0].reason == "tp"
        assert results[0].partial is True
        assert results[0].quantity == pytest.approx(original_qty * 0.5, rel=0.01)

        # Position still open with 50% remaining
        remaining = engine.get_open_positions()
        assert len(remaining) == 1
        assert remaining[0].quantity == pytest.approx(original_qty * 0.5, rel=0.01)

    def test_tp2_triggers_full_close(self):
        """After TP1 triggered, TP2 (close_pct=100) closes remaining."""
        engine = self._make_engine()
        proposal = _make_proposal(
            stop_loss=67000.0,
            take_profit=[
                TakeProfit(price=69000.0, close_pct=50),
                TakeProfit(price=70000.0, close_pct=100),
            ],
        )
        engine.open_position(proposal, current_price=68000.0, leverage=10)

        # TP1 triggers
        engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=69500.0)
        assert len(engine.get_open_positions()) == 1

        # TP2 triggers
        results = engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=70500.0)
        assert len(results) == 1
        assert results[0].partial is False
        assert len(engine.get_open_positions()) == 0

    def test_sl_still_closes_full_position(self):
        """SL should close entire position regardless of TP levels."""
        engine = self._make_engine()
        proposal = _make_proposal(
            stop_loss=67000.0,
            take_profit=[
                TakeProfit(price=69000.0, close_pct=50),
                TakeProfit(price=70000.0, close_pct=100),
            ],
        )
        engine.open_position(proposal, current_price=68000.0, leverage=10)

        results = engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=66500.0)
        assert len(results) == 1
        assert results[0].reason == "sl"
        assert len(engine.get_open_positions()) == 0

    def test_tp1_moves_sl_to_breakeven(self):
        """After TP1, stop_loss should move to entry price (breakeven)."""
        engine = self._make_engine()
        proposal = _make_proposal(
            stop_loss=67000.0,
            take_profit=[
                TakeProfit(price=69000.0, close_pct=50),
                TakeProfit(price=70000.0, close_pct=100),
            ],
        )
        pos = engine.open_position(proposal, current_price=68000.0, leverage=10)

        # TP1 triggers
        engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=69500.0)
        remaining = engine.get_open_positions()
        assert len(remaining) == 1
        # SL moved to entry price (breakeven)
        assert remaining[0].stop_loss == pytest.approx(pos.entry_price)

    def test_single_tp_100pct_closes_all(self):
        """A single TP with close_pct=100 should close full position."""
        engine = self._make_engine()
        proposal = _make_proposal(
            stop_loss=67000.0,
            take_profit=[TakeProfit(price=70000.0, close_pct=100)],
        )
        engine.open_position(proposal, current_price=68000.0, leverage=10)

        results = engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=70500.0)
        assert len(results) == 1
        assert results[0].partial is False
        assert len(engine.get_open_positions()) == 0

    def test_short_partial_tp(self):
        """Partial TP works for short positions too."""
        engine = self._make_engine()
        proposal = _make_proposal(
            side=Side.SHORT,
            stop_loss=70000.0,
            take_profit=[
                TakeProfit(price=67000.0, close_pct=50),
                TakeProfit(price=65000.0, close_pct=100),
            ],
        )
        pos = engine.open_position(proposal, current_price=68000.0, leverage=10)
        original_qty = pos.quantity

        # TP1 triggers for short (price goes down)
        results = engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=66500.0)
        assert len(results) == 1
        assert results[0].partial is True
        assert results[0].quantity == pytest.approx(original_qty * 0.5, rel=0.01)
