from __future__ import annotations

from orchestrator.agents.base import BaseAgent
from orchestrator.agents.utils import summarize_ohlcv
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.models import SentimentReport


class SentimentAgent(BaseAgent[SentimentReport]):
    output_model = SentimentReport
    _skill_name = "sentiment"

    def _build_prompt(self, **kwargs) -> str:
        snapshot: MarketSnapshot = kwargs["snapshot"]
        ohlcv_summary = summarize_ohlcv(snapshot.ohlcv, max_candles=10)

        data = (
            f"Symbol: {snapshot.symbol}\n"
            f"Current Price: {snapshot.current_price}\n"
            f"24h Volume: {snapshot.volume_24h:,.0f}\n"
            f"Funding Rate: {snapshot.funding_rate:.6f}\n"
            f"Timeframe: {snapshot.timeframe}\n\n"
            f"Recent OHLCV ({len(snapshot.ohlcv)} candles):\n{ohlcv_summary}"
        )

        return (
            f"Use the {self._skill_name} skill to analyze the following market data.\n"
            f"Read .claude/skills/{self._skill_name}/SKILL.md for instructions.\n\n"
            f"=== Market Data ===\n{data}"
        )

    def _get_default_output(self) -> SentimentReport:
        return SentimentReport(
            sentiment_score=50,
            key_events=[],
            sources=["degraded"],
            confidence=0.1,
        )
