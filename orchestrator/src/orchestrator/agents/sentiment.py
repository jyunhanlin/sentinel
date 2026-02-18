from __future__ import annotations

from orchestrator.agents.base import BaseAgent
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.models import SentimentReport


class SentimentAgent(BaseAgent[SentimentReport]):
    output_model = SentimentReport

    def _build_messages(self, **kwargs) -> list[dict]:
        snapshot: MarketSnapshot = kwargs["snapshot"]

        system_prompt = (
            "You are a crypto market sentiment analyst. "
            "Analyze the provided market data and infer the current market sentiment.\n\n"
            "Respond with ONLY a JSON object matching this schema:\n"
            "{\n"
            '  "sentiment_score": <int 0-100, 50=neutral, >50=bullish, <50=bearish>,\n'
            '  "key_events": [{"event": "<desc>", "impact": "positive|negative|neutral", '
            '"source": "<source>"}],\n'
            '  "sources": ["<list of data sources used>"],\n'
            '  "confidence": <float 0.0-1.0>\n'
            "}"
        )

        ohlcv_summary = self._summarize_ohlcv(snapshot)

        user_prompt = (
            f"Symbol: {snapshot.symbol}\n"
            f"Current Price: {snapshot.current_price}\n"
            f"24h Volume: {snapshot.volume_24h:,.0f}\n"
            f"Funding Rate: {snapshot.funding_rate:.6f}\n"
            f"Timeframe: {snapshot.timeframe}\n\n"
            f"Recent OHLCV ({len(snapshot.ohlcv)} candles):\n{ohlcv_summary}\n\n"
            "Based on this market data and your knowledge of crypto markets, "
            "analyze the current sentiment. Consider price action, volume trends, "
            "and funding rate implications."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _get_default_output(self) -> SentimentReport:
        return SentimentReport(
            sentiment_score=50,
            key_events=[],
            sources=["degraded"],
            confidence=0.1,
        )

    @staticmethod
    def _summarize_ohlcv(snapshot: MarketSnapshot) -> str:
        if not snapshot.ohlcv:
            return "No OHLCV data available"

        lines = []
        for candle in snapshot.ohlcv[-10:]:  # last 10 candles max
            o, h, lo, c, v = candle[1], candle[2], candle[3], candle[4], candle[5]
            lines.append(f"  O={o:.1f} H={h:.1f} L={lo:.1f} C={c:.1f} V={v:.0f}")
        return "\n".join(lines)
