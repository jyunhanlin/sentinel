from __future__ import annotations

from orchestrator.agents.base import BaseAgent
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.models import MarketInterpretation, Trend, VolatilityRegime


class MarketAgent(BaseAgent[MarketInterpretation]):
    output_model = MarketInterpretation

    def _build_messages(self, **kwargs) -> list[dict]:
        snapshot: MarketSnapshot = kwargs["snapshot"]

        system_prompt = (
            "You are a crypto technical analyst. "
            "Analyze the provided OHLCV data, funding rate, and volume to determine "
            "market structure, trend, volatility regime, key price levels, and risk flags.\n\n"
            "Respond with ONLY a JSON object matching this schema:\n"
            "{\n"
            '  "trend": "up" | "down" | "range",\n'
            '  "volatility_regime": "low" | "medium" | "high",\n'
            '  "key_levels": [{"type": "support|resistance", "price": <number>}],\n'
            '  "risk_flags": ["<flag_name>"]  // e.g. funding_elevated, oi_near_ath, volume_declining\n'
            "}"
        )

        ohlcv_summary = self._summarize_ohlcv(snapshot)

        user_prompt = (
            f"Symbol: {snapshot.symbol}\n"
            f"Current Price: {snapshot.current_price}\n"
            f"24h Volume: {snapshot.volume_24h:,.0f}\n"
            f"Funding Rate: {snapshot.funding_rate:.6f}\n"
            f"Timeframe: {snapshot.timeframe}\n\n"
            f"OHLCV Data ({len(snapshot.ohlcv)} candles):\n{ohlcv_summary}\n\n"
            "Identify the trend direction, volatility regime, key support/resistance levels, "
            "and any risk flags from the data."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _get_default_output(self) -> MarketInterpretation:
        return MarketInterpretation(
            trend=Trend.RANGE,
            volatility_regime=VolatilityRegime.MEDIUM,
            key_levels=[],
            risk_flags=["analysis_degraded"],
        )

    @staticmethod
    def _summarize_ohlcv(snapshot: MarketSnapshot) -> str:
        if not snapshot.ohlcv:
            return "No OHLCV data available"

        lines = []
        for candle in snapshot.ohlcv[-20:]:  # last 20 candles for technical analysis
            o, h, l, c, v = candle[1], candle[2], candle[3], candle[4], candle[5]
            lines.append(f"  O={o:.1f} H={h:.1f} L={l:.1f} C={c:.1f} V={v:.0f}")
        return "\n".join(lines)
