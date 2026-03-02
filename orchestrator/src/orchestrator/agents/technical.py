from __future__ import annotations

from typing import Any

from orchestrator.agents.base import BaseAgent
from orchestrator.agents.utils import summarize_ohlcv
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.models import (
    Momentum,
    TechnicalAnalysis,
    Trend,
    VolatilityRegime,
)


class TechnicalAgent(BaseAgent[TechnicalAnalysis]):
    output_model = TechnicalAnalysis
    _skill_name = "technical"

    def __init__(
        self,
        client,
        *,
        label: str = "short_term",
        candle_count: int = 50,
        max_retries: int = 1,
    ) -> None:
        super().__init__(client, max_retries=max_retries)
        self._label = label
        self._candle_count = candle_count

    def _build_prompt(self, **kwargs) -> str:
        snapshot: MarketSnapshot = kwargs["snapshot"]
        macro_data: dict[str, Any] | None = kwargs.get("macro_data")

        ohlcv_summary = summarize_ohlcv(snapshot.ohlcv, max_candles=self._candle_count)

        data = (
            f"Symbol: {snapshot.symbol}\n"
            f"Current Price: {snapshot.current_price}\n"
            f"24h Volume: {snapshot.volume_24h:,.0f}\n"
            f"Funding Rate: {snapshot.funding_rate:.6f}\n"
            f"Timeframe: {snapshot.timeframe}\n"
            f"Analysis Label: {self._label}\n\n"
            f"OHLCV Data ({len(snapshot.ohlcv)} candles):\n{ohlcv_summary}"
        )

        if macro_data:
            data += (
                f"\n\n=== Macro Indicators ===\n"
                f"200W MA: {macro_data['ma_200w']:.0f}\n"
                f"Bull Support Band: {macro_data['bull_support_upper']:.0f} - "
                f"{macro_data['bull_support_lower']:.0f}"
            )

        return (
            f"Use the {self._skill_name} skill.\n\n"
            f"=== Market Data ===\n{data}"
        )

    def _get_default_output(self) -> TechnicalAnalysis:
        return TechnicalAnalysis(
            label=self._label,
            trend=Trend.RANGE,
            trend_strength=0.0,
            volatility_regime=VolatilityRegime.MEDIUM,
            volatility_pct=0.0,
            momentum=Momentum.NEUTRAL,
            rsi=50.0,
            key_levels=[],
            risk_flags=["analysis_degraded"],
        )
