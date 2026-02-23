# Price Monitor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a lightweight price monitoring loop that checks SL/TP/Liquidation on open positions at a configurable interval, independent of the LLM pipeline.

**Architecture:** Add a `PriceMonitor` class that runs as an APScheduler job inside the existing `PipelineScheduler`. It fetches current prices via `DataFetcher.fetch_current_price()` (no LLM calls), runs `PaperEngine.check_sl_tp()`, and notifies admins via a callback when positions close. The monitor only runs when there are open positions.

**Tech Stack:** Python 3.12+, APScheduler (already used), CCXT (already used), structlog

**Current SL/TP check:** Only runs during pipeline execution (`pipeline/runner.py:88-95`), which happens every 720 minutes by default. This means SL/TP can be delayed by hours.

---

### Task 1: Add Price Monitor Config

**Files:**
- Modify: `orchestrator/src/orchestrator/config.py:46-52`
- Test: `orchestrator/tests/unit/test_config.py`

**Step 1: Write the failing test**

```python
# In test_config.py, add:
def test_price_monitor_defaults(monkeypatch, isolated_env):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")
    monkeypatch.setenv("TELEGRAM_ADMIN_CHAT_IDS", "[123]")
    s = Settings()
    assert s.price_monitor_interval_seconds == 60
    assert s.price_monitor_enabled is True
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_config.py::test_price_monitor_defaults -v`
Expected: FAIL — `AttributeError`

**Step 3: Implement**

Add to `config.py` in the Paper Trading section (after `paper_leverage_options`):

```python
    # Price Monitor
    price_monitor_interval_seconds: int = 60  # check every N seconds
    price_monitor_enabled: bool = True
```

**Step 4: Run test to verify it passes**

Run: `cd orchestrator && uv run pytest tests/unit/test_config.py::test_price_monitor_defaults -v`
Expected: PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/config.py orchestrator/tests/unit/test_config.py
git commit -m "feat: add price monitor config (interval_seconds, enabled)"
```

---

### Task 2: Create PriceMonitor Class

**Files:**
- Create: `orchestrator/src/orchestrator/exchange/price_monitor.py`
- Test: `orchestrator/tests/unit/test_price_monitor.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_price_monitor.py
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.exchange.paper_engine import CloseResult, PaperEngine, Position
from orchestrator.exchange.price_monitor import PriceMonitor
from orchestrator.models import Side


@pytest.fixture
def monitor():
    engine = MagicMock(spec=PaperEngine)
    data_fetcher = AsyncMock()
    on_close = AsyncMock()
    return PriceMonitor(
        paper_engine=engine,
        data_fetcher=data_fetcher,
        on_close=on_close,
    )


class TestPriceMonitor:
    @pytest.mark.asyncio
    async def test_check_skips_when_no_open_positions(self, monitor):
        monitor._paper_engine.get_open_positions.return_value = []
        await monitor.check()
        monitor._data_fetcher.fetch_current_price.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_fetches_price_per_symbol(self, monitor):
        pos_btc = MagicMock(symbol="BTC/USDT:USDT")
        pos_eth = MagicMock(symbol="ETH/USDT:USDT")
        monitor._paper_engine.get_open_positions.return_value = [pos_btc, pos_eth]
        monitor._paper_engine.check_sl_tp.return_value = []
        monitor._data_fetcher.fetch_current_price.return_value = 68000.0

        await monitor.check()

        # Should fetch price for each unique symbol
        assert monitor._data_fetcher.fetch_current_price.call_count == 2

    @pytest.mark.asyncio
    async def test_check_calls_sl_tp_per_symbol(self, monitor):
        pos = MagicMock(symbol="BTC/USDT:USDT")
        monitor._paper_engine.get_open_positions.return_value = [pos]
        monitor._paper_engine.check_sl_tp.return_value = []
        monitor._data_fetcher.fetch_current_price.return_value = 68000.0

        await monitor.check()

        monitor._paper_engine.check_sl_tp.assert_called_once_with(
            symbol="BTC/USDT:USDT", current_price=68000.0,
        )

    @pytest.mark.asyncio
    async def test_check_notifies_on_close(self, monitor):
        pos = MagicMock(symbol="BTC/USDT:USDT")
        monitor._paper_engine.get_open_positions.return_value = [pos]
        monitor._data_fetcher.fetch_current_price.return_value = 66000.0

        close_result = MagicMock(spec=CloseResult)
        close_result.trade_id = "t1"
        close_result.reason = "sl"
        monitor._paper_engine.check_sl_tp.return_value = [close_result]

        await monitor.check()

        monitor._on_close.assert_called_once_with(close_result)

    @pytest.mark.asyncio
    async def test_check_handles_fetch_error_gracefully(self, monitor):
        pos = MagicMock(symbol="BTC/USDT:USDT")
        monitor._paper_engine.get_open_positions.return_value = [pos]
        monitor._data_fetcher.fetch_current_price.side_effect = Exception("API down")

        # Should not raise, just log
        await monitor.check()
        monitor._paper_engine.check_sl_tp.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_deduplicates_symbols(self, monitor):
        """Two positions on same symbol should only fetch price once."""
        pos1 = MagicMock(symbol="BTC/USDT:USDT")
        pos2 = MagicMock(symbol="BTC/USDT:USDT")
        monitor._paper_engine.get_open_positions.return_value = [pos1, pos2]
        monitor._paper_engine.check_sl_tp.return_value = []
        monitor._data_fetcher.fetch_current_price.return_value = 68000.0

        await monitor.check()

        monitor._data_fetcher.fetch_current_price.assert_called_once_with("BTC/USDT:USDT")
        monitor._paper_engine.check_sl_tp.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_price_monitor.py -v`
Expected: FAIL — module not found

**Step 3: Implement PriceMonitor**

Create `orchestrator/src/orchestrator/exchange/price_monitor.py`:

```python
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from orchestrator.exchange.data_fetcher import DataFetcher
    from orchestrator.exchange.paper_engine import CloseResult, PaperEngine

logger = structlog.get_logger(__name__)

CloseCallback = Callable[["CloseResult"], Awaitable[None]]


class PriceMonitor:
    """Lightweight SL/TP/Liquidation checker that runs independently of the LLM pipeline."""

    def __init__(
        self,
        *,
        paper_engine: PaperEngine,
        data_fetcher: DataFetcher,
        on_close: CloseCallback | None = None,
    ) -> None:
        self._paper_engine = paper_engine
        self._data_fetcher = data_fetcher
        self._on_close = on_close

    async def check(self) -> list[CloseResult]:
        """Check SL/TP/Liquidation for all open positions. Returns closed positions."""
        from orchestrator.exchange.paper_engine import CloseResult

        positions = self._paper_engine.get_open_positions()
        if not positions:
            return []

        # Deduplicate symbols
        symbols = list({p.symbol for p in positions})
        all_closed: list[CloseResult] = []

        for symbol in symbols:
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
```

**Step 4: Run test to verify it passes**

Run: `cd orchestrator && uv run pytest tests/unit/test_price_monitor.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/exchange/price_monitor.py orchestrator/tests/unit/test_price_monitor.py
git commit -m "feat: add PriceMonitor for lightweight SL/TP/Liq checking"
```

---

### Task 3: Integrate PriceMonitor into Scheduler

**Files:**
- Modify: `orchestrator/src/orchestrator/pipeline/scheduler.py:23-126`
- Test: `orchestrator/tests/unit/test_scheduler.py` (existing or create)

**Step 1: Write failing test**

```python
# tests/unit/test_scheduler.py (add or create)
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.pipeline.scheduler import PipelineScheduler


class TestPriceMonitorJob:
    def test_start_adds_price_monitor_job(self):
        runner = MagicMock()
        monitor = AsyncMock()
        scheduler = PipelineScheduler(
            runner=runner, symbols=["BTC/USDT:USDT"],
            price_monitor=monitor, price_monitor_interval_seconds=30,
        )
        with patch("orchestrator.pipeline.scheduler.AsyncIOScheduler") as MockSched:
            mock_instance = MagicMock()
            MockSched.return_value = mock_instance
            scheduler.start()

            # Verify price monitor job was added
            job_ids = [call.kwargs["id"] for call in mock_instance.add_job.call_args_list]
            assert "price_monitor" in job_ids

    def test_start_skips_monitor_when_none(self):
        runner = MagicMock()
        scheduler = PipelineScheduler(
            runner=runner, symbols=["BTC/USDT:USDT"],
        )
        with patch("orchestrator.pipeline.scheduler.AsyncIOScheduler") as MockSched:
            mock_instance = MagicMock()
            MockSched.return_value = mock_instance
            scheduler.start()

            job_ids = [call.kwargs.get("id", "") for call in mock_instance.add_job.call_args_list]
            assert "price_monitor" not in job_ids
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_scheduler.py::TestPriceMonitorJob -v`
Expected: FAIL — `unexpected keyword argument 'price_monitor'`

**Step 3: Implement**

Add to `PipelineScheduler.__init__` (after `on_result` param):

```python
    price_monitor: PriceMonitor | None = None,
    price_monitor_interval_seconds: int = 60,
```

Store as `self._price_monitor` and `self._price_monitor_interval`.

Add to `start()` method (after approval expiry job, around line 104):

```python
        # Price monitor — lightweight SL/TP check
        if self._price_monitor is not None:
            self._scheduler.add_job(
                self._price_monitor.check,
                trigger=IntervalTrigger(seconds=self._price_monitor_interval),
                id="price_monitor",
                name="Price Monitor (SL/TP/Liq)",
                replace_existing=True,
            )
```

Update `TYPE_CHECKING` imports to include `PriceMonitor`.

**Step 4: Run test to verify it passes**

Run: `cd orchestrator && uv run pytest tests/unit/test_scheduler.py::TestPriceMonitorJob -v`
Expected: PASS

**Step 5: Run all tests**

Run: `cd orchestrator && uv run pytest tests/unit/ -v --tb=short`
Expected: All PASS

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/pipeline/scheduler.py orchestrator/tests/unit/test_scheduler.py
git commit -m "feat: integrate PriceMonitor as APScheduler job in PipelineScheduler"
```

---

### Task 4: Wire PriceMonitor in App Bootstrap

**Files:**
- Modify: `orchestrator/src/orchestrator/__main__.py:51-217, 294-320`
- Test: manual verification (integration)

**Step 1: Update `create_app_components`**

Add parameters (after `price_deviation_threshold`):

```python
    # Price Monitor
    price_monitor_interval_seconds: int = 60,
    price_monitor_enabled: bool = True,
```

After creating `data_fetcher` (line 108) and `paper_engine` (line 136), create PriceMonitor:

```python
    # Price Monitor
    from orchestrator.exchange.price_monitor import PriceMonitor

    price_monitor: PriceMonitor | None = None
    if price_monitor_enabled:
        price_monitor = PriceMonitor(
            paper_engine=paper_engine,
            data_fetcher=data_fetcher,
        )
```

Pass to scheduler (around line 171):

```python
    scheduler = PipelineScheduler(
        runner=runner,
        symbols=symbols,
        interval_minutes=pipeline_interval_minutes,
        default_model=llm_model,
        premium_model=llm_model_premium,
        approval_manager=approval_manager,
        price_monitor=price_monitor,
        price_monitor_interval_seconds=price_monitor_interval_seconds,
    )
```

Add to return dict:

```python
    return {
        ...
        "price_monitor": price_monitor,
    }
```

**Step 2: Update `_build_components`**

Pass new settings:

```python
    price_monitor_interval_seconds=settings.price_monitor_interval_seconds,
    price_monitor_enabled=settings.price_monitor_enabled,
```

**Step 3: Wire close callback in `_run_bot`**

In `_run_bot()` (around line 299), wire the close callback:

```python
    # Wire price monitor -> bot notification
    price_monitor = components.get("price_monitor")
    if price_monitor is not None:
        price_monitor._on_close = bot.push_close_report
```

**Step 4: Verify**

Run: `cd orchestrator && uv run pytest tests/unit/ -v --tb=short`
Expected: All PASS (no new tests needed, just wiring)

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/__main__.py
git commit -m "feat: wire PriceMonitor into app bootstrap and bot close notification"
```

---

### Task 5: Final Verification

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

**Step 3: Verify logging output**

The PriceMonitor should log:
- `price_fetch_failed` — when exchange API is unreachable
- `monitor_position_closed` — when a position hits SL/TP/Liq
- `on_close_callback_failed` — when TG notification fails

**Step 4: Commit if any fixups**

```bash
git commit -m "chore: fix lint issues in price monitor integration"
```
