from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import structlog

from orchestrator.exchange.paper_engine import CloseResult

if TYPE_CHECKING:
    from orchestrator.exchange.data_fetcher import DataFetcher, TickerSummary
    from orchestrator.exchange.paper_engine import PaperEngine

logger = structlog.get_logger(__name__)

CloseCallback = Callable[["CloseResult"], Awaitable[None]]
TickCallback = Callable[[list["TickerSummary"]], Awaitable[None]]


class PriceMonitor:
    """Lightweight SL/TP/Liquidation checker that runs independently of the LLM pipeline."""

    def __init__(
        self,
        *,
        paper_engine: PaperEngine,
        data_fetcher: DataFetcher,
        on_close: CloseCallback | None = None,
        on_tick: TickCallback | None = None,
        symbols: list[str] | None = None,
    ) -> None:
        self._paper_engine = paper_engine
        self._data_fetcher = data_fetcher
        self._on_close = on_close
        self._on_tick = on_tick
        self._symbols = symbols or []

    async def check(self) -> list[CloseResult]:
        """Check SL/TP/Liquidation for all open positions + update price board."""
        # Phase 1: Fetch ticker summaries for all monitored symbols (price board)
        await self._fetch_and_broadcast_tickers()

        # Phase 2: Check SL/TP on open positions
        positions = self._paper_engine.get_open_positions()
        if not positions:
            return []

        # Deduplicate symbols
        pos_symbols = list({p.symbol for p in positions})
        all_closed: list[CloseResult] = []

        for symbol in pos_symbols:
            try:
                current_price = await self._data_fetcher.fetch_current_price(symbol)
            except Exception:
                logger.exception("price_fetch_failed", symbol=symbol)
                continue

            closed = self._paper_engine.check_sl_tp(
                symbol=symbol, current_price=current_price,
            )

            for result in closed:
                logger.info(
                    "monitor_position_closed",
                    trade_id=result.trade_id,
                    symbol=result.symbol,
                    reason=result.reason,
                    pnl=result.pnl,
                )
                if self._on_close is not None:
                    try:
                        await self._on_close(result)
                    except Exception:
                        logger.exception("on_close_callback_failed", trade_id=result.trade_id)

            all_closed.extend(closed)

        return all_closed

    async def _fetch_and_broadcast_tickers(self) -> None:
        """Fetch ticker summaries for all monitored symbols and call on_tick."""
        if not self._symbols:
            return

        summaries: list[TickerSummary] = []
        for symbol in self._symbols:
            try:
                summary = await self._data_fetcher.fetch_ticker_summary(symbol)
                summaries.append(summary)
            except Exception:
                logger.exception("ticker_summary_fetch_failed", symbol=symbol)

        if self._on_tick is not None and summaries:
            try:
                await self._on_tick(summaries)
            except Exception:
                logger.exception("on_tick_callback_failed")
