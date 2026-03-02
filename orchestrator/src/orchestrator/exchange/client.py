from __future__ import annotations

from typing import Any

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
    ) -> list[list[float]]:
        return await self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    async def fetch_funding_rate(self, symbol: str) -> float:
        result = await self._exchange.fetch_funding_rate(symbol)
        return result.get("fundingRate", 0.0)

    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        return await self._exchange.fetch_ticker(symbol)

    async def create_market_order(
        self, symbol: str, side: str, amount: float
    ) -> dict[str, Any]:
        return await self._exchange.create_order(symbol, "market", side, amount)

    async def create_stop_order(
        self, symbol: str, side: str, amount: float, *, stop_price: float
    ) -> dict[str, Any]:
        return await self._exchange.create_order(
            symbol,
            "stop_market",
            side,
            amount,
            params={"stopPrice": stop_price},
        )

    async def cancel_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        return await self._exchange.cancel_order(order_id, symbol)

    async def fetch_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        return await self._exchange.fetch_order(order_id, symbol)

    async def fetch_funding_rate_history(
        self, symbol: str, *, limit: int = 30
    ) -> list[float]:
        """Fetch recent funding rate history via Binance API."""
        rates = await self._exchange.fetch_funding_rate_history(symbol, limit=limit)
        return [r.get("fundingRate", 0.0) for r in rates]

    async def fetch_open_interest(self, symbol: str) -> float:
        """Fetch current open interest."""
        result = await self._exchange.fetch_open_interest(symbol)
        return result.get("openInterestAmount", 0.0)

    async def fetch_long_short_ratio(self, symbol: str) -> float:
        """Fetch global long/short account ratio."""
        result = await self._exchange.fetch_long_short_ratio_history(symbol, limit=1)
        if result:
            return result[0].get("longShortRatio", 1.0)
        return 1.0

    async def fetch_top_trader_long_short_ratio(self, symbol: str) -> float:
        """Fetch top trader long/short ratio."""
        result = await self._exchange.fetch_long_short_ratio_history(
            symbol, limit=1, params={"traderType": "top"}
        )
        if result:
            return result[0].get("longShortRatio", 1.0)
        return 1.0

    async def fetch_order_book(
        self, symbol: str, *, limit: int = 20
    ) -> dict[str, Any]:
        """Fetch order book."""
        return await self._exchange.fetch_order_book(symbol, limit=limit)

    async def close(self) -> None:
        await self._exchange.close()
