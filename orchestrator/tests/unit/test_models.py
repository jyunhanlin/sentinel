import pytest
from pydantic import ValidationError

from orchestrator.models import (
    EntryOrder,
    KeyEvent,
    KeyLevel,
    MarketInterpretation,
    SentimentReport,
    Side,
    TakeProfit,
    TradeProposal,
    Trend,
    VolatilityRegime,
)


class TestSentimentReport:
    def test_valid_report(self):
        report = SentimentReport(
            sentiment_score=72,
            key_events=[
                KeyEvent(event="BTC ETF inflows", impact="positive", source="Bloomberg")
            ],
            sources=["twitter", "news"],
            confidence=0.8,
        )
        assert report.sentiment_score == 72
        assert len(report.key_events) == 1

    def test_score_out_of_range(self):
        with pytest.raises(ValidationError):
            SentimentReport(
                sentiment_score=101,
                key_events=[],
                sources=[],
                confidence=0.5,
            )

    def test_confidence_out_of_range(self):
        with pytest.raises(ValidationError):
            SentimentReport(
                sentiment_score=50,
                key_events=[],
                sources=[],
                confidence=1.5,
            )


class TestMarketInterpretation:
    def test_valid_interpretation(self):
        interp = MarketInterpretation(
            trend=Trend.UP,
            volatility_regime=VolatilityRegime.MEDIUM,
            key_levels=[KeyLevel(type="support", price=93000.0)],
            risk_flags=["funding_elevated"],
        )
        assert interp.trend == Trend.UP
        assert len(interp.key_levels) == 1


class TestTradeProposal:
    def test_valid_proposal(self):
        proposal = TradeProposal(
            symbol="BTC/USDT:USDT",
            side=Side.LONG,
            entry=EntryOrder(type="market"),
            position_size_risk_pct=1.5,
            stop_loss=93000.0,
            take_profit=[
                TakeProfit(price=95500.0, close_pct=50),
                TakeProfit(price=97000.0, close_pct=100),
            ],
            time_horizon="4h",
            confidence=0.75,
            invalid_if=["funding_rate > 0.05%"],
            rationale="Bullish momentum",
        )
        assert proposal.proposal_id is not None
        assert proposal.side == Side.LONG

    def test_flat_side_no_stop_loss_required(self):
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
        assert proposal.side == Side.FLAT

    def test_risk_pct_negative_rejected(self):
        with pytest.raises(ValidationError):
            TradeProposal(
                symbol="BTC/USDT:USDT",
                side=Side.LONG,
                entry=EntryOrder(type="market"),
                position_size_risk_pct=-1.0,
                stop_loss=93000.0,
                take_profit=[],
                time_horizon="4h",
                confidence=0.75,
                invalid_if=[],
                rationale="test",
            )


class TestTakeProfit:
    def test_valid_take_profit(self):
        tp = TakeProfit(price=65800.0, close_pct=50)
        assert tp.price == 65800.0
        assert tp.close_pct == 50

    def test_close_pct_out_of_range(self):
        with pytest.raises(ValidationError):
            TakeProfit(price=65800.0, close_pct=0)
        with pytest.raises(ValidationError):
            TakeProfit(price=65800.0, close_pct=101)


class TestMarketInterpretationVolatility:
    def test_volatility_pct_field(self):
        interp = MarketInterpretation(
            trend=Trend.UP,
            volatility_regime=VolatilityRegime.MEDIUM,
            volatility_pct=2.3,
            key_levels=[],
            risk_flags=[],
        )
        assert interp.volatility_pct == 2.3


class TestTradeProposalLeverage:
    def test_suggested_leverage_field(self):
        proposal = TradeProposal(
            symbol="BTC/USDT:USDT", side=Side.LONG,
            entry=EntryOrder(type="market"),
            position_size_risk_pct=1.0, stop_loss=64000.0,
            take_profit=[
                TakeProfit(price=65800.0, close_pct=50),
                TakeProfit(price=67000.0, close_pct=100),
            ],
            suggested_leverage=10,
            time_horizon="4h", confidence=0.72,
            invalid_if=[], rationale="test",
        )
        assert proposal.suggested_leverage == 10
        assert proposal.take_profit[0].close_pct == 50

    def test_suggested_leverage_default(self):
        proposal = TradeProposal(
            symbol="BTC/USDT:USDT", side=Side.FLAT,
            entry=EntryOrder(type="market"),
            position_size_risk_pct=0.0, stop_loss=None,
            take_profit=[], time_horizon="4h",
            confidence=0.5, invalid_if=[], rationale="no trade",
        )
        assert proposal.suggested_leverage == 10

    def test_suggested_leverage_validation(self):
        with pytest.raises(ValidationError):
            TradeProposal(
                symbol="BTC/USDT:USDT", side=Side.LONG,
                entry=EntryOrder(type="market"),
                position_size_risk_pct=1.0, stop_loss=64000.0,
                take_profit=[], suggested_leverage=100,
                time_horizon="4h", confidence=0.7,
                invalid_if=[], rationale="test",
            )
