from __future__ import annotations

import aiohttp
import structlog

logger = structlog.get_logger(__name__)


class ExternalDataFetcher:
    """Fetches cross-market data from free external APIs."""

    async def fetch_dxy_data(self) -> dict:
        """Fetch DXY data from Yahoo Finance."""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB?range=5d&interval=1d"
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return self._default_market_data()
                    data = await resp.json()
                    closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
                    closes = [c for c in closes if c is not None]
                    current = closes[-1] if closes else 0.0
                    change_pct = ((closes[-1] / closes[0]) - 1) * 100 if len(closes) >= 2 else 0.0
                    return {"current": current, "change_pct": change_pct, "trend_5d": closes}
        except Exception:
            logger.warning("dxy_fetch_failed")
            return self._default_market_data()

    async def fetch_sp500_data(self) -> dict:
        """Fetch S&P 500 data from Yahoo Finance."""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?range=5d&interval=1d"
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return self._default_market_data()
                    data = await resp.json()
                    closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
                    closes = [c for c in closes if c is not None]
                    current = closes[-1] if closes else 0.0
                    change_pct = ((closes[-1] / closes[0]) - 1) * 100 if len(closes) >= 2 else 0.0
                    return {"current": current, "change_pct": change_pct, "trend_5d": closes}
        except Exception:
            logger.warning("sp500_fetch_failed")
            return self._default_market_data()

    async def fetch_btc_dominance(self) -> dict:
        """Fetch BTC dominance from CoinGecko (free)."""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.coingecko.com/api/v3/global"
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return {"current": 0.0, "change_7d": 0.0}
                    data = await resp.json()
                    btc_dom = data["data"]["market_cap_percentage"].get("btc", 0.0)
                    change = data["data"].get("market_cap_change_percentage_24h_usd", 0.0)
                    return {"current": btc_dom, "change_7d": change}
        except Exception:
            logger.warning("btc_dominance_fetch_failed")
            return {"current": 0.0, "change_7d": 0.0}

    async def fetch_economic_calendar(self) -> list[dict]:
        """Fetch upcoming economic events. Returns empty list as placeholder.

        TODO: integrate with a free economic calendar API.
        """
        return []

    async def fetch_exchange_announcements(self, exchange_id: str = "binance") -> list[str]:
        """Fetch exchange announcements. Returns empty list as placeholder.

        TODO: integrate with Binance announcement API.
        """
        return []

    @staticmethod
    def _default_market_data() -> dict:
        return {"current": 0.0, "change_pct": 0.0, "trend_5d": []}
