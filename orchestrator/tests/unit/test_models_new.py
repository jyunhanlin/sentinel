import pytest

from orchestrator.models import (
    CatalystEvent,
    CatalystReport,
    CorrelationAnalysis,
    PositioningAnalysis,
    TechnicalAnalysis,
)


class TestTechnicalAnalysis:
    def test_create_short_term(self):
        ta = TechnicalAnalysis(
            label="short_term",
            trend="up",
            trend_strength=25.0,
            volatility_regime="medium",
            volatility_pct=2.3,
            momentum="bullish",
            rsi=65.0,
            key_levels=[],
            risk_flags=[],
        )
        assert ta.label == "short_term"
        assert ta.above_200w_ma is None
        assert ta.bull_support_band_status is None

    def test_create_long_term_with_macro(self):
        ta = TechnicalAnalysis(
            label="long_term",
            trend="up",
            trend_strength=30.0,
            volatility_regime="low",
            volatility_pct=1.2,
            momentum="bullish",
            rsi=58.0,
            key_levels=[],
            risk_flags=[],
            above_200w_ma=True,
            bull_support_band_status="above",
        )
        assert ta.above_200w_ma is True
        assert ta.bull_support_band_status == "above"

    def test_frozen(self):
        ta = TechnicalAnalysis(
            label="short_term", trend="up", trend_strength=25.0,
            volatility_regime="medium", volatility_pct=2.3,
            momentum="bullish", rsi=65.0, key_levels=[], risk_flags=[],
        )
        with pytest.raises(Exception):
            ta.trend = "down"


class TestPositioningAnalysis:
    def test_create(self):
        pa = PositioningAnalysis(
            funding_trend="rising",
            funding_extreme=False,
            oi_change_pct=5.2,
            retail_bias="long",
            smart_money_bias="short",
            squeeze_risk="long_squeeze",
            liquidity_assessment="normal",
            risk_flags=["funding_elevated"],
            confidence=0.7,
        )
        assert pa.funding_trend == "rising"
        assert pa.squeeze_risk == "long_squeeze"

    def test_frozen(self):
        pa = PositioningAnalysis(
            funding_trend="stable", funding_extreme=False, oi_change_pct=0.0,
            retail_bias="neutral", smart_money_bias="neutral", squeeze_risk="none",
            liquidity_assessment="normal", risk_flags=[], confidence=0.5,
        )
        with pytest.raises(Exception):
            pa.funding_trend = "falling"


class TestCatalystReport:
    def test_create_with_events(self):
        event = CatalystEvent(
            event="FOMC Rate Decision",
            time="2026-03-15T18:00:00Z",
            impact="high",
            direction_bias="uncertain",
        )
        report = CatalystReport(
            upcoming_events=[event],
            active_events=[],
            risk_level="high",
            recommendation="wait",
            confidence=0.8,
        )
        assert len(report.upcoming_events) == 1
        assert report.recommendation == "wait"

    def test_frozen(self):
        report = CatalystReport(
            upcoming_events=[], active_events=[],
            risk_level="low", recommendation="proceed", confidence=0.6,
        )
        with pytest.raises(Exception):
            report.risk_level = "high"


class TestCorrelationAnalysis:
    def test_create(self):
        ca = CorrelationAnalysis(
            dxy_trend="strengthening",
            dxy_impact="headwind",
            sp500_regime="risk_off",
            btc_dominance_trend="rising",
            cross_market_alignment="unfavorable",
            risk_flags=["dxy_headwind"],
            confidence=0.6,
        )
        assert ca.cross_market_alignment == "unfavorable"

    def test_frozen(self):
        ca = CorrelationAnalysis(
            dxy_trend="stable", dxy_impact="neutral", sp500_regime="neutral",
            btc_dominance_trend="stable", cross_market_alignment="mixed",
            risk_flags=[], confidence=0.5,
        )
        with pytest.raises(Exception):
            ca.dxy_trend = "weakening"
