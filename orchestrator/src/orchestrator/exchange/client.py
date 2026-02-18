from __future__ import annotations

import ccxt.async_support as ccxt


class ExchangeClient:
    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: str = "",
        api_secret: str = "",
    ) -> None:
        self.exchange_id = exchange_id
        exchange_class = getattr(ccxt, exchange_id)
        self._exchange = exchange_class(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "options": {"defaultType": "swap"},
            }
        )

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1h", *, limit: int = 100
    ) -> list[list]:
        return await self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    async def fetch_funding_rate(self, symbol: str) -> float:
        result = await self._exchange.fetch_funding_rate(symbol)
        return result.get("fundingRate", 0.0)

    async def fetch_ticker(self, symbol: str) -> dict:
        return await self._exchange.fetch_ticker(symbol)

    async def close(self) -> None:
        await self._exchange.close()
