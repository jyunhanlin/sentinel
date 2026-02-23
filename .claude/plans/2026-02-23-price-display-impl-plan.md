# Price Display Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Show current prices in two places: (A) all confirmation cards display `Now: $XX,XXX` before execution, (B) a pinned price board message updated every 60s with all monitored symbols + 24h change %.

**Architecture:** Add `TickerSummary` model and `fetch_ticker_summary()` to DataFetcher. Extend PriceMonitor with an `on_tick` callback that fires every cycle with ticker data for all monitored symbols. Add `update_price_board()` to SentinelBot that edits the pinned message. Unify confirmation card labels.

**Tech Stack:** Python 3.12+, python-telegram-bot, CCXT (fetch_ticker → `percentage` field), structlog

---

### Task 1: Add TickerSummary Model + fetch_ticker_summary

**Files:**
- Modify: `orchestrator/src/orchestrator/exchange/data_fetcher.py:1-53`
- Test: `orchestrator/tests/unit/test_data_fetcher.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_data_fetcher.py (create or add)
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.exchange.data_fetcher import DataFetcher, TickerSummary


class TestFetchTickerSummary:
    @pytest.mark.asyncio
    async def test_returns_ticker_summary(self):
        client = AsyncMock()
        client.fetch_ticker.return_value = {
            "last": 69123.5,
            "percentage": 1.2,
        }
        fetcher = DataFetcher(client)

        result = await fetcher.fetch_ticker_summary("BTC/USDT:USDT")

        assert isinstance(result, TickerSummary)
        assert result.symbol == "BTC/USDT:USDT"
        assert result.price == 69123.5
        assert result.change_24h_pct == 1.2

    @pytest.mark.asyncio
    async def test_handles_missing_percentage(self):
        client = AsyncMock()
        client.fetch_ticker.return_value = {
            "last": 2530.0,
        }
        fetcher = DataFetcher(client)

        result = await fetcher.fetch_ticker_summary("ETH/USDT:USDT")

        assert result.price == 2530.0
        assert result.change_24h_pct == 0.0
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_data_fetcher.py::TestFetchTickerSummary -v`
Expected: FAIL — `ImportError: cannot import name 'TickerSummary'`

**Step 3: Implement**

Add `TickerSummary` and `fetch_ticker_summary()` to `data_fetcher.py`:

```python
# Add after MarketSnapshot class (after line 18)

class TickerSummary(BaseModel, frozen=True):
    symbol: str
    price: float
    change_24h_pct: float


# Add after fetch_current_price method (after line 41)

    async def fetch_ticker_summary(self, symbol: str) -> TickerSummary:
        """Fetch price + 24h change % for price board display."""
        ticker = await self._client.fetch_ticker(symbol)
        return TickerSummary(
            symbol=symbol,
            price=ticker.get("last", 0.0),
            change_24h_pct=ticker.get("percentage", 0.0) or 0.0,
        )
```

**Step 4: Run test to verify it passes**

Run: `cd orchestrator && uv run pytest tests/unit/test_data_fetcher.py::TestFetchTickerSummary -v`
Expected: PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/exchange/data_fetcher.py orchestrator/tests/unit/test_data_fetcher.py
git commit -m "feat: add TickerSummary model and fetch_ticker_summary to DataFetcher"
```

---

### Task 2: Extend PriceMonitor with on_tick Callback

**Files:**
- Modify: `orchestrator/src/orchestrator/exchange/price_monitor.py:1-70`
- Modify: `orchestrator/tests/unit/test_price_monitor.py:1-93`

**Step 1: Write the failing tests**

```python
# Add to tests/unit/test_price_monitor.py

from orchestrator.exchange.data_fetcher import TickerSummary


@pytest.fixture
def monitor_with_tick():
    engine = MagicMock(spec=PaperEngine)
    data_fetcher = AsyncMock()
    on_close = AsyncMock()
    on_tick = AsyncMock()
    return PriceMonitor(
        paper_engine=engine,
        data_fetcher=data_fetcher,
        on_close=on_close,
        on_tick=on_tick,
        symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
    )


class TestPriceMonitorOnTick:
    @pytest.mark.asyncio
    async def test_check_calls_on_tick_with_summaries(self, monitor_with_tick):
        monitor = monitor_with_tick
        monitor._paper_engine.get_open_positions.return_value = []
        btc_summary = TickerSummary(symbol="BTC/USDT:USDT", price=69000.0, change_24h_pct=1.2)
        eth_summary = TickerSummary(symbol="ETH/USDT:USDT", price=2530.0, change_24h_pct=-0.5)
        monitor._data_fetcher.fetch_ticker_summary.side_effect = [btc_summary, eth_summary]

        await monitor.check()

        monitor._on_tick.assert_called_once()
        summaries = monitor._on_tick.call_args[0][0]
        assert len(summaries) == 2
        assert summaries[0].symbol == "BTC/USDT:USDT"
        assert summaries[1].symbol == "ETH/USDT:USDT"

    @pytest.mark.asyncio
    async def test_check_skips_on_tick_when_no_callback(self):
        engine = MagicMock(spec=PaperEngine)
        data_fetcher = AsyncMock()
        monitor = PriceMonitor(
            paper_engine=engine,
            data_fetcher=data_fetcher,
            symbols=["BTC/USDT:USDT"],
        )
        engine.get_open_positions.return_value = []
        data_fetcher.fetch_ticker_summary.return_value = TickerSummary(
            symbol="BTC/USDT:USDT", price=69000.0, change_24h_pct=1.0,
        )

        # Should not raise
        await monitor.check()

    @pytest.mark.asyncio
    async def test_on_tick_handles_partial_fetch_failure(self, monitor_with_tick):
        monitor = monitor_with_tick
        monitor._paper_engine.get_open_positions.return_value = []
        btc_summary = TickerSummary(symbol="BTC/USDT:USDT", price=69000.0, change_24h_pct=1.2)
        monitor._data_fetcher.fetch_ticker_summary.side_effect = [
            btc_summary,
            Exception("API down"),
        ]

        await monitor.check()

        # on_tick still called with partial results
        monitor._on_tick.assert_called_once()
        summaries = monitor._on_tick.call_args[0][0]
        assert len(summaries) == 1
        assert summaries[0].symbol == "BTC/USDT:USDT"

    @pytest.mark.asyncio
    async def test_on_tick_uses_symbols_not_positions(self, monitor_with_tick):
        """on_tick fetches all monitored symbols, not just open position symbols."""
        monitor = monitor_with_tick
        # Only BTC position, but both BTC and ETH in symbols
        pos = MagicMock(symbol="BTC/USDT:USDT")
        monitor._paper_engine.get_open_positions.return_value = [pos]
        monitor._paper_engine.check_sl_tp.return_value = []

        btc_summary = TickerSummary(symbol="BTC/USDT:USDT", price=69000.0, change_24h_pct=1.2)
        eth_summary = TickerSummary(symbol="ETH/USDT:USDT", price=2530.0, change_24h_pct=-0.5)
        monitor._data_fetcher.fetch_ticker_summary.side_effect = [btc_summary, eth_summary]
        monitor._data_fetcher.fetch_current_price.return_value = 69000.0

        await monitor.check()

        # on_tick gets ALL symbols (BTC + ETH), not just position symbols
        assert monitor._data_fetcher.fetch_ticker_summary.call_count == 2
```

**Step 2: Run tests to verify they fail**

Run: `cd orchestrator && uv run pytest tests/unit/test_price_monitor.py::TestPriceMonitorOnTick -v`
Expected: FAIL — `unexpected keyword argument 'on_tick'` / `unexpected keyword argument 'symbols'`

**Step 3: Implement**

Update `price_monitor.py`:

```python
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

        from orchestrator.exchange.data_fetcher import TickerSummary

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
```

**Step 4: Run tests to verify they pass**

Run: `cd orchestrator && uv run pytest tests/unit/test_price_monitor.py -v`
Expected: All PASS (both old and new tests)

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/exchange/price_monitor.py orchestrator/tests/unit/test_price_monitor.py
git commit -m "feat: extend PriceMonitor with on_tick callback for price board"
```

---

### Task 3: Add format_price_board Formatter

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/formatters.py:1-378`
- Test: `orchestrator/tests/unit/test_formatters.py` (create or add)

**Step 1: Write the failing test**

```python
# tests/unit/test_formatters.py (create or add)
from orchestrator.exchange.data_fetcher import TickerSummary
from orchestrator.telegram.formatters import format_price_board


class TestFormatPriceBoard:
    def test_formats_multiple_symbols(self):
        summaries = [
            TickerSummary(symbol="BTC/USDT:USDT", price=69123.5, change_24h_pct=1.2),
            TickerSummary(symbol="ETH/USDT:USDT", price=2530.8, change_24h_pct=-0.5),
        ]
        result = format_price_board(summaries)

        assert "━━ Price Board ━━" in result
        assert "BTC/USDT" in result
        assert "$69,123.5" in result
        assert "+1.20%" in result
        assert "ETH/USDT" in result
        assert "$2,530.8" in result
        assert "-0.50%" in result
        assert "Updated:" in result

    def test_formats_empty_list(self):
        result = format_price_board([])
        assert "━━ Price Board ━━" in result
        assert "No symbols" in result

    def test_positive_change_has_plus_sign(self):
        summaries = [
            TickerSummary(symbol="SOL/USDT:USDT", price=142.35, change_24h_pct=3.1),
        ]
        result = format_price_board(summaries)
        assert "+3.10%" in result

    def test_negative_change_has_minus_sign(self):
        summaries = [
            TickerSummary(symbol="SOL/USDT:USDT", price=142.35, change_24h_pct=-2.5),
        ]
        result = format_price_board(summaries)
        assert "-2.50%" in result
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_formatters.py::TestFormatPriceBoard -v`
Expected: FAIL — `ImportError: cannot import name 'format_price_board'`

**Step 3: Implement**

Add to `formatters.py` (after existing functions):

```python
def format_price_board(summaries: list) -> str:
    """Format a price board for pinned message display."""
    from datetime import datetime, timezone

    lines = ["━━ Price Board ━━"]

    if not summaries:
        lines.append("No symbols configured.")
        return "\n".join(lines)

    for s in summaries:
        # Strip :USDT suffix for cleaner display
        display_symbol = s.symbol.replace(":USDT", "")
        sign = "+" if s.change_24h_pct >= 0 else ""
        lines.append(f"{display_symbol}  ${s.price:,.1f}  {sign}{s.change_24h_pct:.2f}%")

    now = datetime.now(timezone.utc).strftime("%H:%M:%S")
    lines.append(f"\nUpdated: {now} UTC")

    return "\n".join(lines)
```

**Step 4: Run test to verify it passes**

Run: `cd orchestrator && uv run pytest tests/unit/test_formatters.py::TestFormatPriceBoard -v`
Expected: PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/formatters.py orchestrator/tests/unit/test_formatters.py
git commit -m "feat: add format_price_board formatter for pinned price display"
```

---

### Task 4: Add Pinned Price Board to SentinelBot

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/bot.py:129-165, 233-258`
- Test: `orchestrator/tests/unit/test_bot_price_board.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_bot_price_board.py
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.exchange.data_fetcher import TickerSummary


class TestPriceBoardUpdate:
    @pytest.mark.asyncio
    async def test_update_price_board_sends_and_pins_first_time(self):
        """First call should send a new message and pin it."""
        from orchestrator.telegram.bot import SentinelBot

        bot = SentinelBot(token="test", admin_chat_ids=[123, 456])
        mock_app = MagicMock()
        mock_bot = AsyncMock()
        mock_app.bot = mock_bot
        bot._app = mock_app

        sent_msg = MagicMock()
        sent_msg.message_id = 99
        mock_bot.send_message.return_value = sent_msg

        summaries = [
            TickerSummary(symbol="BTC/USDT:USDT", price=69000.0, change_24h_pct=1.0),
        ]

        await bot.update_price_board(summaries)

        # Should send to both admin chats
        assert mock_bot.send_message.call_count == 2
        # Should pin both messages
        assert mock_bot.pin_chat_message.call_count == 2
        # Should store message ids
        assert bot._price_board_msg_ids[123] == 99
        assert bot._price_board_msg_ids[456] == 99

    @pytest.mark.asyncio
    async def test_update_price_board_edits_existing_message(self):
        """Subsequent calls should edit the existing pinned message."""
        from orchestrator.telegram.bot import SentinelBot

        bot = SentinelBot(token="test", admin_chat_ids=[123])
        mock_app = MagicMock()
        mock_bot = AsyncMock()
        mock_app.bot = mock_bot
        bot._app = mock_app

        # Simulate existing pinned message
        bot._price_board_msg_ids = {123: 99}

        summaries = [
            TickerSummary(symbol="BTC/USDT:USDT", price=69500.0, change_24h_pct=1.5),
        ]

        await bot.update_price_board(summaries)

        # Should edit, not send new
        mock_bot.send_message.assert_not_called()
        mock_bot.edit_message_text.assert_called_once()
        call_kwargs = mock_bot.edit_message_text.call_args.kwargs
        assert call_kwargs["chat_id"] == 123
        assert call_kwargs["message_id"] == 99

    @pytest.mark.asyncio
    async def test_update_price_board_resends_if_edit_fails(self):
        """If edit fails (message deleted), send a new one and re-pin."""
        from telegram.error import BadRequest

        from orchestrator.telegram.bot import SentinelBot

        bot = SentinelBot(token="test", admin_chat_ids=[123])
        mock_app = MagicMock()
        mock_bot = AsyncMock()
        mock_app.bot = mock_bot
        bot._app = mock_app

        bot._price_board_msg_ids = {123: 99}
        mock_bot.edit_message_text.side_effect = BadRequest("Message to edit not found")

        sent_msg = MagicMock()
        sent_msg.message_id = 200
        mock_bot.send_message.return_value = sent_msg

        summaries = [
            TickerSummary(symbol="BTC/USDT:USDT", price=69000.0, change_24h_pct=1.0),
        ]

        await bot.update_price_board(summaries)

        # Should fall back to send + pin
        mock_bot.send_message.assert_called_once()
        mock_bot.pin_chat_message.assert_called_once()
        assert bot._price_board_msg_ids[123] == 200

    @pytest.mark.asyncio
    async def test_update_price_board_noop_when_app_is_none(self):
        from orchestrator.telegram.bot import SentinelBot

        bot = SentinelBot(token="test", admin_chat_ids=[123])
        # _app is None by default

        summaries = [
            TickerSummary(symbol="BTC/USDT:USDT", price=69000.0, change_24h_pct=1.0),
        ]

        # Should not raise
        await bot.update_price_board(summaries)
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_bot_price_board.py -v`
Expected: FAIL — `AttributeError: 'SentinelBot' object has no attribute 'update_price_board'`

**Step 3: Implement**

Add to `SentinelBot.__init__` (after line 164 `self._leverage_options`):

```python
        self._price_board_msg_ids: dict[int, int] = {}  # chat_id → message_id
```

Add `update_price_board` method (after `push_risk_rejection`, around line 258):

```python
    async def update_price_board(self, summaries: list) -> None:
        """Update or create the pinned price board message in all admin chats."""
        if self._app is None:
            return

        from orchestrator.telegram.formatters import format_price_board

        text = format_price_board(summaries)

        for chat_id in self.admin_chat_ids:
            msg_id = self._price_board_msg_ids.get(chat_id)

            if msg_id is not None:
                # Try to edit existing message
                try:
                    await self._app.bot.edit_message_text(
                        text=text, chat_id=chat_id, message_id=msg_id,
                    )
                    continue
                except BadRequest:
                    # Message deleted or not found — fall through to send new
                    pass

            # Send new message and pin it
            try:
                sent = await self._app.bot.send_message(chat_id=chat_id, text=text)
                self._price_board_msg_ids[chat_id] = sent.message_id
                await self._app.bot.pin_chat_message(
                    chat_id=chat_id, message_id=sent.message_id, disable_notification=True,
                )
            except Exception:
                logger.exception("price_board_send_failed", chat_id=chat_id)
```

Ensure `BadRequest` is imported at the top of `bot.py` (check if it already is — it's used in `_safe_callback_reply`).

**Step 4: Run test to verify it passes**

Run: `cd orchestrator && uv run pytest tests/unit/test_bot_price_board.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/bot.py orchestrator/tests/unit/test_bot_price_board.py
git commit -m "feat: add pinned price board to SentinelBot"
```

---

### Task 5: Wire Price Board in App Bootstrap

**Files:**
- Modify: `orchestrator/src/orchestrator/__main__.py:142-150, 311-322`

**Step 1: Update PriceMonitor creation to pass symbols**

In `create_app_components()`, update the PriceMonitor creation (lines 142-150):

```python
    # Price Monitor
    from orchestrator.exchange.price_monitor import PriceMonitor

    price_monitor: PriceMonitor | None = None
    if price_monitor_enabled:
        price_monitor = PriceMonitor(
            paper_engine=paper_engine,
            data_fetcher=data_fetcher,
            symbols=symbols,
        )
```

**Step 2: Wire on_tick in `_run_bot`**

In `_run_bot()`, after the existing `price_monitor._on_close` wiring (line 321):

```python
        price_monitor._on_tick = bot.update_price_board
```

So the block becomes:

```python
    # Wire price monitor -> bot notification
    price_monitor = components.get("price_monitor")
    if price_monitor is not None:
        price_monitor._on_close = bot.push_close_report
        price_monitor._on_tick = bot.update_price_board
```

**Step 3: Run existing tests**

Run: `cd orchestrator && uv run pytest tests/unit/ -v --tb=short`
Expected: All PASS

**Step 4: Commit**

```bash
git add orchestrator/src/orchestrator/__main__.py
git commit -m "feat: wire price board callback in app bootstrap"
```

---

### Task 6: Unify Confirmation Card Labels

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/bot.py:768-774, 928-930, 1016-1020, 1113-1119`

This task is purely cosmetic — unify the current price label across all 4 confirmation cards.

**Step 1: Update leverage confirm card (line 768-774)**

Change:
```python
        text = (
            f"━━ CONFIRM ORDER ━━\n"
            f"{p.symbol}  {side_str} · {leverage}x\n\n"
            f"Entry: ~${current_price:,.1f} | Qty: {qty:.4f}\n"
            f"Margin: ${margin:,.2f} | Liq: ~${liq:,.1f}\n"
            f"SL: ${p.stop_loss:,.1f} | TP: {tp_str}\n"
        )
```

To:
```python
        text = (
            f"━━ CONFIRM ORDER ━━\n"
            f"{p.symbol}  {side_str} · {leverage}x\n\n"
            f"Now: ${current_price:,.1f} | Qty: {qty:.4f}\n"
            f"Margin: ${margin:,.2f} | Liq: ~${liq:,.1f}\n"
            f"SL: ${p.stop_loss:,.1f} | TP: {tp_str}\n"
        )
```

**Step 2: Update close confirm card (line 930)**

The close card uses `format_position_card()` which already shows `Entry:` price. Add a `Now:` line before the card.

Change:
```python
        text = f"━━ CLOSE POSITION? ━━\n\n{format_position_card(info)}"
```

To:
```python
        text = (
            f"━━ CLOSE POSITION? ━━\n"
            f"Now: ${current_price:,.1f}\n\n"
            f"{format_position_card(info)}"
        )
```

**Step 3: Update reduce confirm card (lines 1016-1020)**

Change:
```python
        text = (
            f"━━ CONFIRM REDUCE ━━\n"
            f"{pos.symbol}  {side_str} {pos.leverage}x\n\n"
            f"Close {pct:.0f}%: {close_qty:.4f} at ~${current_price:,.1f}\n"
            f"Est. PnL: {pnl_sign}${est_pnl:,.2f}"
        )
```

To:
```python
        text = (
            f"━━ CONFIRM REDUCE ━━\n"
            f"{pos.symbol}  {side_str} {pos.leverage}x\n\n"
            f"Now: ${current_price:,.1f}\n"
            f"Close {pct:.0f}%: {close_qty:.4f}\n"
            f"Est. PnL: {pnl_sign}${est_pnl:,.2f}"
        )
```

**Step 4: Update add confirm card (lines 1113-1119)**

Change:
```python
        text = (
            f"━━ CONFIRM ADD ━━\n"
            f"{pos.symbol}  {side_str} {pos.leverage}x\n\n"
            f"Current Price: ~${current_price:,.1f}\n"
            f"Add Qty: {add_qty:.4f} | Add Margin: ${add_margin:,.2f}\n"
            f"Risk: {risk_pct}%"
        )
```

To:
```python
        text = (
            f"━━ CONFIRM ADD ━━\n"
            f"{pos.symbol}  {side_str} {pos.leverage}x\n\n"
            f"Now: ${current_price:,.1f}\n"
            f"Add Qty: {add_qty:.4f} | Add Margin: ${add_margin:,.2f}\n"
            f"Risk: {risk_pct}%"
        )
```

**Step 5: Run all tests**

Run: `cd orchestrator && uv run pytest tests/unit/ -v --tb=short`
Expected: All PASS

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/bot.py
git commit -m "refactor: unify confirmation card labels to use 'Now:' prefix"
```

---

### Task 7: Final Verification

**Step 1: Run full test suite**

```bash
cd orchestrator && uv run pytest tests/unit/ -v --cov=orchestrator --cov-report=term-missing
```

Expected: All PASS, coverage maintained

**Step 2: Run linter**

```bash
cd orchestrator && uv run ruff check src/ tests/
```

Expected: No errors

**Step 3: Commit if any fixups**

```bash
git commit -m "chore: fix lint issues in price display feature"
```
