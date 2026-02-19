from __future__ import annotations

import asyncio

from pydantic import BaseModel

from orchestrator.exchange.client import ExchangeClient


class MarketSnapshot(BaseModel, frozen=True):
    symbol: str
    timeframe: str
    current_price: float
    volume_24h: float
    funding_rate: float
    ohlcv: list[list]


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

    async def _parallel_fetch(
        self, symbol: str, timeframe: str, limit: int
    ) -> tuple[list[list], float, dict]:
        ohlcv_task = self._client.fetch_ohlcv(symbol, timeframe, limit=limit)
        funding_task = self._client.fetch_funding_rate(symbol)
        ticker_task = self._client.fetch_ticker(symbol)

        ohlcv, funding, ticker = await asyncio.gather(
            ohlcv_task, funding_task, ticker_task
        )
        return ohlcv, funding, ticker
