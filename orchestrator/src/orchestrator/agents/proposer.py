from __future__ import annotations

from orchestrator.agents.base import BaseAgent
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.models import (
    EntryOrder,
    MarketInterpretation,
    SentimentReport,
    Side,
    TradeProposal,
)


class ProposerAgent(BaseAgent[TradeProposal]):
    output_model = TradeProposal
    _skill_name = "proposer"

    def _build_prompt(self, **kwargs) -> str:
        snapshot: MarketSnapshot = kwargs["snapshot"]
        sentiment: SentimentReport = kwargs["sentiment"]
        market: MarketInterpretation = kwargs["market"]

        key_levels_str = ", ".join(
            f"{kl.type}={kl.price}" for kl in market.key_levels
        ) or "none identified"

        risk_flags_str = ", ".join(market.risk_flags) or "none"

        data = (
            f"=== Market Data ===\n"
            f"Symbol: {snapshot.symbol}\n"
            f"Current Price: {snapshot.current_price}\n"
            f"24h Volume: {snapshot.volume_24h:,.0f}\n"
            f"Funding Rate: {snapshot.funding_rate:.6f}\n\n"
            f"=== Sentiment Analysis ===\n"
            f"Sentiment Score: {sentiment.sentiment_score}/100\n"
            f"Confidence: {sentiment.confidence}\n"
            f"Key Events: {', '.join(e.event for e in sentiment.key_events) or 'none'}\n\n"
            f"=== Technical Analysis ===\n"
            f"Trend: {market.trend}\n"
            f"Volatility: {market.volatility_regime} ({market.volatility_pct:.1f}%)\n"
            f"Key Levels: {key_levels_str}\n"
            f"Risk Flags: {risk_flags_str}"
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
