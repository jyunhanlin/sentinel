from __future__ import annotations

from orchestrator.agents.base import BaseAgent
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.models import (
    CatalystReport,
    CorrelationAnalysis,
    EntryOrder,
    PositioningAnalysis,
    Side,
    TechnicalAnalysis,
    TradeProposal,
)


class ProposerAgent(BaseAgent[TradeProposal]):
    output_model = TradeProposal
    _skill_name = "proposer"

    def _build_prompt(self, **kwargs) -> str:
        snapshot: MarketSnapshot = kwargs["snapshot"]
        tech_short: TechnicalAnalysis = kwargs["technical_short"]
        tech_long: TechnicalAnalysis = kwargs["technical_long"]
        positioning: PositioningAnalysis = kwargs["positioning"]
        catalyst: CatalystReport = kwargs["catalyst"]
        correlation: CorrelationAnalysis = kwargs["correlation"]

        def _format_key_levels(ta: TechnicalAnalysis) -> str:
            return ", ".join(f"{kl.type}={kl.price}" for kl in ta.key_levels) or "none"

        def _format_risk_flags(flags: list[str]) -> str:
            return ", ".join(flags) or "none"

        macro_str = ""
        if tech_long.above_200w_ma is not None:
            macro_str = (
                f"Above 200W MA: {tech_long.above_200w_ma}\n"
                f"Bull Support Band: {tech_long.bull_support_band_status}\n"
            )

        catalyst_events = ", ".join(
            f"{e.event} ({e.impact})" for e in catalyst.upcoming_events
        ) or "none"

        data = (
            f"=== Market Data ===\n"
            f"Symbol: {snapshot.symbol}\n"
            f"Current Price: {snapshot.current_price}\n"
            f"24h Volume: {snapshot.volume_24h:,.0f}\n"
            f"Funding Rate: {snapshot.funding_rate:.6f}\n\n"
            f"=== Short-Term Technical ({tech_short.label}) ===\n"
            f"Trend: {tech_short.trend} (ADX: {tech_short.trend_strength:.1f})\n"
            f"Momentum: {tech_short.momentum} (RSI: {tech_short.rsi:.1f})\n"
            f"Volatility: {tech_short.volatility_regime} ({tech_short.volatility_pct:.1f}%)\n"
            f"Key Levels: {_format_key_levels(tech_short)}\n"
            f"Risk Flags: {_format_risk_flags(tech_short.risk_flags)}\n\n"
            f"=== Long-Term Technical ({tech_long.label}) ===\n"
            f"Trend: {tech_long.trend} (ADX: {tech_long.trend_strength:.1f})\n"
            f"Momentum: {tech_long.momentum} (RSI: {tech_long.rsi:.1f})\n"
            f"Volatility: {tech_long.volatility_regime} ({tech_long.volatility_pct:.1f}%)\n"
            f"Key Levels: {_format_key_levels(tech_long)}\n"
            f"Risk Flags: {_format_risk_flags(tech_long.risk_flags)}\n"
            f"{macro_str}\n"
            f"=== Positioning ===\n"
            f"Funding Trend: {positioning.funding_trend} (extreme: {positioning.funding_extreme})\n"
            f"OI Change: {positioning.oi_change_pct:+.1f}%\n"
            f"Retail Bias: {positioning.retail_bias}\n"
            f"Smart Money Bias: {positioning.smart_money_bias}\n"
            f"Squeeze Risk: {positioning.squeeze_risk}\n"
            f"Liquidity: {positioning.liquidity_assessment}\n"
            f"Risk Flags: {_format_risk_flags(positioning.risk_flags)}\n"
            f"Confidence: {positioning.confidence:.2f}\n\n"
            f"=== Catalyst ===\n"
            f"Upcoming Events: {catalyst_events}\n"
            f"Risk Level: {catalyst.risk_level}\n"
            f"Recommendation: {catalyst.recommendation}\n"
            f"Confidence: {catalyst.confidence:.2f}\n\n"
            f"=== Cross-Market Correlation ===\n"
            f"DXY: {correlation.dxy_trend} ({correlation.dxy_impact})\n"
            f"S&P 500: {correlation.sp500_regime}\n"
            f"BTC Dominance: {correlation.btc_dominance_trend}\n"
            f"Alignment: {correlation.cross_market_alignment}\n"
            f"Risk Flags: {_format_risk_flags(correlation.risk_flags)}\n"
            f"Confidence: {correlation.confidence:.2f}"
        )

        return (
            f"Use the {self._skill_name} skill.\n\n"
            f"{data}"
        )

    def _get_default_output(self) -> TradeProposal:
        return TradeProposal(
            symbol="unknown",
            side=Side.FLAT,
            entry=EntryOrder(type="market"),
            position_size_risk_pct=0.0,
            stop_loss=None,
            take_profit=[],
            time_horizon="4h",
            confidence=0.0,
            invalid_if=[],
            rationale="Analysis degraded — no trade signal generated",
        )
