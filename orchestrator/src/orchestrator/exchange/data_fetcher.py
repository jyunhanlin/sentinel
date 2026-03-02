from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel

from orchestrator.exchange.client import ExchangeClient


class MarketSnapshot(BaseModel, frozen=True):
    symbol: str
    timeframe: str
    current_price: float
    volume_24h: float
    funding_rate: float
    ohlcv: list[list[float]]


class TickerSummary(BaseModel, frozen=True):
    symbol: str
    price: float
    change_24h_pct: float


class DataFetcher:
    def __init__(self, client: ExchangeClient) -> None:
        self._client = client

    async def fetch_snapshot(
        self, symbol: str, *, timeframe: str = "1h", limit: int = 100
    ) -> MarketSnapshot:
        ohlcv, funding, ticker = await self._parallel_fetch(symbol, timeframe, limit)

        return MarketSnapshot(
            symbol=symbol,
            timeframe=timeframe,
            current_price=ticker.get("last", 0.0),
            volume_24h=ticker.get("quoteVolume", 0.0),
            funding_rate=funding,
            ohlcv=ohlcv,
        )

    async def fetch_current_price(self, symbol: str) -> float:
        """Fetch the latest price for a symbol."""
        ticker = await self._client.fetch_ticker(symbol)
        return ticker.get("last", 0.0)

    async def fetch_ticker_summary(self, symbol: str) -> TickerSummary:
        """Fetch price + 24h change % for price board display."""
        ticker = await self._client.fetch_ticker(symbol)
        return TickerSummary(
            symbol=symbol,
            price=ticker.get("last", 0.0),
            change_24h_pct=ticker.get("percentage", 0.0) or 0.0,
        )

    async def fetch_positioning_data(self, symbol: str) -> dict[str, Any]:
        """Fetch all positioning-related data for the Positioning agent."""
        funding_hist, oi, ls_ratio, top_ls_ratio, order_book = await asyncio.gather(
            self._client.fetch_funding_rate_history(symbol),
            self._client.fetch_open_interest(symbol),
            self._client.fetch_long_short_ratio(symbol),
            self._client.fetch_top_trader_long_short_ratio(symbol),
            self._client.fetch_order_book(symbol),
        )

        bid_depth = sum(bid[1] for bid in order_book.get("bids", []))
        ask_depth = sum(ask[1] for ask in order_book.get("asks", []))

        return {
            "funding_rate_history": funding_hist,
            "open_interest": oi,
            "oi_change_pct": 0.0,  # TODO: needs OI history for real calculation
            "long_short_ratio": ls_ratio,
            "top_trader_long_short_ratio": top_ls_ratio,
            "order_book_summary": {"bid_depth": bid_depth, "ask_depth": ask_depth},
        }

    async def fetch_macro_indicators(self, symbol: str) -> dict[str, Any]:
        """Fetch weekly candles and calculate 200W MA + Bull Market Support Band."""
        weekly_ohlcv = await self._client.fetch_ohlcv(symbol, "1w", limit=210)

        closes = [candle[4] for candle in weekly_ohlcv]

        # 200W SMA
        ma_200w = sum(closes[-200:]) / min(len(closes), 200) if closes else 0.0

        # Bull Market Support Band: 20W SMA + 21W EMA
        sma_20w = sum(closes[-20:]) / min(len(closes), 20) if closes else 0.0

        # 21W EMA
        ema_21w = _calculate_ema(closes, 21)

        return {
            "ma_200w": ma_200w,
            "bull_support_upper": max(sma_20w, ema_21w),
            "bull_support_lower": min(sma_20w, ema_21w),
        }

    async def _parallel_fetch(
        self, symbol: str, timeframe: str, limit: int
    ) -> tuple[list[list[float]], float, dict[str, Any]]:
        ohlcv_task = self._client.fetch_ohlcv(symbol, timeframe, limit=limit)
        funding_task = self._client.fetch_funding_rate(symbol)
        ticker_task = self._client.fetch_ticker(symbol)

        ohlcv, funding, ticker = await asyncio.gather(
            ohlcv_task, funding_task, ticker_task
        )
        return ohlcv, funding, ticker


def _calculate_ema(values: list[float], period: int) -> float:
    """Calculate EMA for the given period."""
    if not values or period <= 0:
        return 0.0
    if len(values) < period:
        return sum(values) / len(values)

    multiplier = 2 / (period + 1)
    ema = sum(values[:period]) / period

    for value in values[period:]:
        ema = (value - ema) * multiplier + ema

    return ema
