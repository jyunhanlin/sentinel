from __future__ import annotations

from typing import Any

from orchestrator.agents.base import BaseAgent
from orchestrator.models import CorrelationAnalysis


class CorrelationAgent(BaseAgent[CorrelationAnalysis]):
    output_model = CorrelationAnalysis
    _skill_name = "correlation"

    def _build_prompt(self, **kwargs) -> str:
        symbol: str = kwargs["symbol"]
        dxy: dict[str, Any] = kwargs["dxy_data"]
        sp500: dict[str, Any] = kwargs["sp500_data"]
        btc_dom: dict[str, Any] = kwargs["btc_dominance"]

        dxy_trend_str = ", ".join(f"{v:.1f}" for v in dxy.get("trend_5d", []))
        sp500_trend_str = ", ".join(f"{v:.0f}" for v in sp500.get("trend_5d", []))

        data = (
            f"Symbol: {symbol}\n\n"
            f"=== DXY (US Dollar Index) ===\n"
            f"Current: {dxy.get('current', 0):.1f}\n"
            f"Change: {dxy.get('change_pct', 0):+.2f}%\n"
            f"5-Day Trend: {dxy_trend_str or 'N/A'}\n\n"
            f"=== S&P 500 ===\n"
            f"Current: {sp500.get('current', 0):.0f}\n"
            f"Change: {sp500.get('change_pct', 0):+.2f}%\n"
            f"5-Day Trend: {sp500_trend_str or 'N/A'}\n\n"
            f"=== BTC Dominance ===\n"
            f"Current: {btc_dom.get('current', 0):.1f}%\n"
            f"7-Day Change: {btc_dom.get('change_7d', 0):+.1f}%"
        )

        return (
            f"Use the {self._skill_name} skill.\n\n"
            f"=== Cross-Market Data ===\n{data}"
        )

    def _get_default_output(self) -> CorrelationAnalysis:
        return CorrelationAnalysis(
            dxy_trend="stable",
            dxy_impact="neutral",
            sp500_regime="neutral",
            btc_dominance_trend="stable",
            cross_market_alignment="mixed",
            risk_flags=["analysis_degraded"],
            confidence=0.1,
        )
