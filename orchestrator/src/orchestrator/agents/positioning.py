from __future__ import annotations

from typing import Any

from orchestrator.agents.base import BaseAgent
from orchestrator.models import PositioningAnalysis


class PositioningAgent(BaseAgent[PositioningAnalysis]):
    output_model = PositioningAnalysis
    _skill_name = "positioning"

    def _build_prompt(self, **kwargs) -> str:
        symbol: str = kwargs["symbol"]
        current_price: float = kwargs["current_price"]
        funding_history: list[float] = kwargs["funding_rate_history"]
        open_interest: float = kwargs["open_interest"]
        oi_change_pct: float = kwargs["oi_change_pct"]
        ls_ratio: float = kwargs["long_short_ratio"]
        top_ls_ratio: float = kwargs["top_trader_long_short_ratio"]
        order_book: dict[str, Any] = kwargs["order_book_summary"]

        funding_str = ", ".join(f"{r:.6f}" for r in funding_history[-10:])

        data = (
            f"Symbol: {symbol}\n"
            f"Current Price: {current_price}\n\n"
            f"=== Funding Rate History (last {len(funding_history[-10:])} periods) ===\n"
            f"{funding_str}\n\n"
            f"=== Open Interest ===\n"
            f"Current OI: {open_interest:,.0f}\n"
            f"OI Change: {oi_change_pct:+.1f}%\n\n"
            f"=== Long/Short Ratios ===\n"
            f"Retail L/S Ratio: {ls_ratio:.2f}\n"
            f"Top Trader L/S Ratio: {top_ls_ratio:.2f}\n\n"
            f"=== Order Book ===\n"
            f"Bid Depth: {order_book.get('bid_depth', 0):.0f}\n"
            f"Ask Depth: {order_book.get('ask_depth', 0):.0f}"
        )

        return (
            f"Use the {self._skill_name} skill.\n\n"
            f"=== Positioning Data ===\n{data}"
        )

    def _get_default_output(self) -> PositioningAnalysis:
        return PositioningAnalysis(
            funding_trend="stable",
            funding_extreme=False,
            oi_change_pct=0.0,
            retail_bias="neutral",
            smart_money_bias="neutral",
            squeeze_risk="none",
            liquidity_assessment="normal",
            risk_flags=["analysis_degraded"],
            confidence=0.1,
        )
