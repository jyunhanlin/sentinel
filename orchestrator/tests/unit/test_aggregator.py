
from orchestrator.models import EntryOrder, Side, TradeProposal
from orchestrator.pipeline.aggregator import aggregate_proposal


class TestAggregateProposal:
    def test_valid_long_proposal(self):
        proposal = TradeProposal(
            symbol="BTC/USDT:USDT",
            side=Side.LONG,
            entry=EntryOrder(type="market"),
            position_size_risk_pct=1.5,
            stop_loss=93000.0,
            take_profit=[97000.0],
            time_horizon="4h",
            confidence=0.75,
            invalid_if=[],
            rationale="Bullish",
        )
        result = aggregate_proposal(proposal, current_price=95200.0)
        assert result.valid is True
        assert result.proposal == proposal

    def test_long_with_sl_above_entry_rejected(self):
        proposal = TradeProposal(
            symbol="BTC/USDT:USDT",
            side=Side.LONG,
            entry=EntryOrder(type="market"),
            position_size_risk_pct=1.5,
            stop_loss=97000.0,  # SL above current price for long = wrong
            take_profit=[99000.0],
            time_horizon="4h",
            confidence=0.75,
            invalid_if=[],
            rationale="Bad SL",
        )
        result = aggregate_proposal(proposal, current_price=95200.0)
        assert result.valid is False
        assert "stop_loss" in result.rejection_reason.lower()

    def test_short_with_sl_below_entry_rejected(self):
        proposal = TradeProposal(
            symbol="BTC/USDT:USDT",
            side=Side.SHORT,
            entry=EntryOrder(type="market"),
            position_size_risk_pct=1.5,
            stop_loss=93000.0,  # SL below current price for short = wrong
            take_profit=[91000.0],
            time_horizon="4h",
            confidence=0.75,
            invalid_if=[],
            rationale="Bad SL",
        )
        result = aggregate_proposal(proposal, current_price=95200.0)
        assert result.valid is False
        assert "stop_loss" in result.rejection_reason.lower()

    def test_flat_always_valid(self):
        proposal = TradeProposal(
            symbol="BTC/USDT:USDT",
            side=Side.FLAT,
            entry=EntryOrder(type="market"),
            position_size_risk_pct=0.0,
            stop_loss=None,
            take_profit=[],
            time_horizon="4h",
            confidence=0.5,
            invalid_if=[],
            rationale="No trade",
        )
        result = aggregate_proposal(proposal, current_price=95200.0)
        assert result.valid is True

    def test_long_without_sl_rejected(self):
        proposal = TradeProposal(
            symbol="BTC/USDT:USDT",
            side=Side.LONG,
            entry=EntryOrder(type="market"),
            position_size_risk_pct=1.5,
            stop_loss=None,  # No SL for a directional trade
            take_profit=[97000.0],
            time_horizon="4h",
            confidence=0.75,
            invalid_if=[],
            rationale="Missing SL",
        )
        result = aggregate_proposal(proposal, current_price=95200.0)
        assert result.valid is False
