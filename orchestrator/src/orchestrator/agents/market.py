from __future__ import annotations

from orchestrator.agents.base import BaseAgent
from orchestrator.agents.utils import summarize_ohlcv
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.models import MarketInterpretation, Trend, VolatilityRegime


class MarketAgent(BaseAgent[MarketInterpretation]):
    output_model = MarketInterpretation
    _skill_name = "market"

    def _build_prompt(self, **kwargs) -> str:
        snapshot: MarketSnapshot = kwargs["snapshot"]
        ohlcv_summary = summarize_ohlcv(snapshot.ohlcv, max_candles=20)

        data = (
            f"Symbol: {snapshot.symbol}\n"
            f"Current Price: {snapshot.current_price}\n"
            f"24h Volume: {snapshot.volume_24h:,.0f}\n"
            f"Funding Rate: {snapshot.funding_rate:.6f}\n"
            f"Timeframe: {snapshot.timeframe}\n\n"
            f"OHLCV Data ({len(snapshot.ohlcv)} candles):\n{ohlcv_summary}"
        )

        return (
            f"Use the {self._skill_name} skill.\n\n"
            f"=== Market Data ===\n{data}"
        )

    def _get_default_output(self) -> MarketInterpretation:
        return MarketInterpretation(
            trend=Trend.RANGE,
            volatility_regime=VolatilityRegime.MEDIUM,
            volatility_pct=0.0,
            key_levels=[],
            risk_flags=["analysis_degraded"],
        )
