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

    def _build_messages(self, **kwargs) -> list[dict]:
        snapshot: MarketSnapshot = kwargs["snapshot"]
        sentiment: SentimentReport = kwargs["sentiment"]
        market: MarketInterpretation = kwargs["market"]

        system_prompt = (
            "You are a crypto trade proposal generator. "
            "Based on sentiment analysis, technical analysis, and current market data, "
            "generate a structured trade proposal.\n\n"
            "Rules:\n"
            "- If no clear edge, set side='flat' with position_size_risk_pct=0\n"
            "- stop_loss MUST be below entry for long, above entry for short\n"
            "- position_size_risk_pct: 0.5-2.0% typical range\n"
            "- confidence: be conservative, rarely above 0.8\n\n"
            "Respond with ONLY a JSON object matching this schema:\n"
            "{\n"
            '  "symbol": "<symbol>",\n'
            '  "side": "long" | "short" | "flat",\n'
            '  "entry": {"type": "market"} or {"type": "limit", "price": <number>},\n'
            '  "position_size_risk_pct": <float 0.0-2.0>,\n'
            '  "stop_loss": <number or null>,\n'
            '  "take_profit": [<number>, ...],\n'
            '  "time_horizon": "<e.g. 4h, 1d>",\n'
            '  "confidence": <float 0.0-1.0>,\n'
            '  "invalid_if": ["<condition>"],\n'
            '  "rationale": "<1-2 sentence explanation>"\n'
            "}"
        )

        key_levels_str = ", ".join(
            f"{kl.type}={kl.price}" for kl in market.key_levels
        ) or "none identified"

        risk_flags_str = ", ".join(market.risk_flags) or "none"

        user_prompt = (
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
            f"Volatility: {market.volatility_regime}\n"
            f"Key Levels: {key_levels_str}\n"
            f"Risk Flags: {risk_flags_str}\n\n"
            f"Generate a trade proposal for {snapshot.symbol}."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

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
