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

    async def create_market_order(
        self, symbol: str, side: str, amount: float
    ) -> dict:
        return await self._exchange.create_order(symbol, "market", side, amount)

    async def create_stop_order(
        self, symbol: str, side: str, amount: float, *, stop_price: float
    ) -> dict:
        return await self._exchange.create_order(
            symbol,
            "stop_market",
            side,
            amount,
            params={"stopPrice": stop_price},
        )

    async def cancel_order(self, order_id: str, symbol: str) -> dict:
        return await self._exchange.cancel_order(order_id, symbol)

    async def fetch_order(self, order_id: str, symbol: str) -> dict:
        return await self._exchange.fetch_order(order_id, symbol)

    async def close(self) -> None:
        await self._exchange.close()
