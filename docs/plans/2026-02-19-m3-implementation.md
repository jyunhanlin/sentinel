# M3: Eval Framework + Performance Statistics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add observability layer — performance metrics (5 indicators), golden eval framework with LLM-as-judge, self-consistency checker, TG/CLI commands, and bring test coverage to 80%+.

**Architecture:** Three modules: `stats/` for trading performance calculation, `eval/` for LLM output evaluation, plus TG/CLI integration. Stats are computed on every position close and stored in extended `account_snapshots`. Eval runs on-demand via TG `/eval` or CLI `python -m orchestrator eval`.

**Tech Stack:** Python 3.12, Pydantic v2, SQLModel, PyYAML (new dep), pytest, structlog

---

## Pre-flight

Before starting, run from `orchestrator/` directory:
```bash
export PATH="$HOME/.local/bin:$PATH"
uv run pytest -v --tb=short   # expect 115 passed
uv run ruff check src/ tests/ # expect clean
```

All commands in this plan assume you're in `orchestrator/` and `PATH` includes `~/.local/bin`.

---

### Task 1: Add PyYAML dependency

**Files:**
- Modify: `orchestrator/pyproject.toml:6-16`

**Step 1: Add pyyaml to dependencies**

In `pyproject.toml`, add `"pyyaml>=6.0"` to the `dependencies` list:

```toml
dependencies = [
    "aiosqlite>=0.22.1",
    "apscheduler>=3.11.2",
    "ccxt>=4.5.38",
    "litellm>=1.81.13",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "python-telegram-bot>=22.6",
    "pyyaml>=6.0",
    "sqlmodel>=0.0.34",
    "structlog>=24.0",
]
```

**Step 2: Sync dependencies**

Run: `uv sync --all-extras`
Expected: Resolves and installs pyyaml.

**Step 3: Commit**

```bash
git add orchestrator/pyproject.toml uv.lock
git commit -m "chore: add pyyaml dependency for eval dataset loading"
```

---

### Task 2: Extend AccountSnapshotRecord with stats fields

**Files:**
- Modify: `orchestrator/src/orchestrator/storage/models.py:65-73`
- Test: `orchestrator/tests/unit/test_storage.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_storage.py`:

```python
class TestAccountSnapshotStatsFields:
    def test_snapshot_has_stats_fields(self, session):
        """AccountSnapshotRecord should have performance stats fields."""
        from orchestrator.storage.models import AccountSnapshotRecord

        snapshot = AccountSnapshotRecord(
            equity=10500.0,
            open_positions_count=2,
            daily_pnl=150.0,
            total_pnl=500.0,
            win_rate=0.625,
            profit_factor=1.85,
            max_drawdown_pct=4.2,
            sharpe_ratio=1.32,
            total_trades=16,
        )
        assert snapshot.total_pnl == 500.0
        assert snapshot.win_rate == 0.625
        assert snapshot.profit_factor == 1.85
        assert snapshot.max_drawdown_pct == 4.2
        assert snapshot.sharpe_ratio == 1.32
        assert snapshot.total_trades == 16
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_storage.py::TestAccountSnapshotStatsFields -v`
Expected: FAIL — `AccountSnapshotRecord` doesn't have those fields.

**Step 3: Add stats fields to AccountSnapshotRecord**

In `storage/models.py`, update `AccountSnapshotRecord`:

```python
class AccountSnapshotRecord(SQLModel, table=True):
    __tablename__ = "account_snapshots"

    id: int | None = Field(default=None, primary_key=True)
    equity: float
    open_positions_count: int = 0
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    total_trades: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_storage.py::TestAccountSnapshotStatsFields -v`
Expected: PASS

**Step 5: Update AccountSnapshotRepository.save_snapshot**

In `storage/repository.py`, update `save_snapshot` to accept stats fields:

```python
def save_snapshot(
    self,
    *,
    equity: float,
    open_count: int,
    daily_pnl: float,
    total_pnl: float = 0.0,
    win_rate: float = 0.0,
    profit_factor: float = 0.0,
    max_drawdown_pct: float = 0.0,
    sharpe_ratio: float = 0.0,
    total_trades: int = 0,
) -> AccountSnapshotRecord:
    record = AccountSnapshotRecord(
        equity=equity,
        open_positions_count=open_count,
        daily_pnl=daily_pnl,
        total_pnl=total_pnl,
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown_pct=max_drawdown_pct,
        sharpe_ratio=sharpe_ratio,
        total_trades=total_trades,
    )
    self._session.add(record)
    self._session.commit()
    self._session.refresh(record)
    return record
```

**Step 6: Run full test suite**

Run: `uv run pytest -v --tb=short`
Expected: All tests pass (existing `save_snapshot` callers use kwargs, so backward compatible).

**Step 7: Commit**

```bash
git add orchestrator/src/orchestrator/storage/models.py orchestrator/src/orchestrator/storage/repository.py orchestrator/tests/unit/test_storage.py
git commit -m "feat: extend account snapshots with performance stats fields"
```

---

### Task 3: Add PaperTradeRepository.get_all_closed method

**Files:**
- Modify: `orchestrator/src/orchestrator/storage/repository.py`
- Test: `orchestrator/tests/unit/test_storage.py`

We need a method to get ALL closed trades (not just recent N) for stats calculation.

**Step 1: Write the failing test**

Add to `tests/unit/test_storage.py`:

```python
class TestPaperTradeRepositoryAllClosed:
    def test_get_all_closed_returns_all(self, session):
        repo = PaperTradeRepository(session)
        # Create 3 closed trades
        for i in range(3):
            repo.save_trade(
                trade_id=f"t-{i}", proposal_id=f"p-{i}",
                symbol="BTC/USDT:USDT", side="long",
                entry_price=95000.0, quantity=0.01,
            )
            repo.update_trade_closed(
                f"t-{i}", exit_price=96000.0, pnl=10.0, fees=1.0
            )
        # Create 1 open trade
        repo.save_trade(
            trade_id="t-open", proposal_id="p-open",
            symbol="BTC/USDT:USDT", side="long",
            entry_price=95000.0, quantity=0.01,
        )
        result = repo.get_all_closed()
        assert len(result) == 3
        assert all(t.status == "closed" for t in result)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_storage.py::TestPaperTradeRepositoryAllClosed -v`
Expected: FAIL — `get_all_closed` doesn't exist.

**Step 3: Implement get_all_closed**

Add to `PaperTradeRepository` in `repository.py`:

```python
def get_all_closed(self) -> list[PaperTradeRecord]:
    statement = (
        select(PaperTradeRecord)
        .where(PaperTradeRecord.status == "closed")
        .order_by(PaperTradeRecord.closed_at.asc())
    )
    return list(self._session.exec(statement).all())
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_storage.py::TestPaperTradeRepositoryAllClosed -v`
Expected: PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/storage/repository.py orchestrator/tests/unit/test_storage.py
git commit -m "feat: add get_all_closed method to PaperTradeRepository"
```

---

### Task 4: Implement StatsCalculator

**Files:**
- Create: `orchestrator/src/orchestrator/stats/__init__.py`
- Create: `orchestrator/src/orchestrator/stats/calculator.py`
- Create: `orchestrator/tests/unit/test_stats_calculator.py`

**Step 1: Create `stats/__init__.py`**

```python
```
(empty file)

**Step 2: Write the failing tests**

Create `tests/unit/test_stats_calculator.py`:

```python
import pytest

from orchestrator.stats.calculator import PerformanceStats, StatsCalculator


class TestStatsCalculator:
    def _make_trade(self, *, pnl: float, closed_at_str: str = "2026-02-19"):
        """Helper to create a minimal trade-like object."""
        from datetime import datetime, UTC
        from unittest.mock import MagicMock

        trade = MagicMock()
        trade.pnl = pnl
        trade.fees = abs(pnl) * 0.001  # tiny fee
        trade.closed_at = datetime.fromisoformat(f"{closed_at_str}T12:00:00+00:00")
        return trade

    def test_no_trades_returns_zeros(self):
        calc = StatsCalculator()
        stats = calc.calculate(closed_trades=[], initial_equity=10000.0)
        assert stats.total_pnl == 0.0
        assert stats.win_rate == 0.0
        assert stats.profit_factor == 0.0
        assert stats.max_drawdown_pct == 0.0
        assert stats.sharpe_ratio == 0.0
        assert stats.total_trades == 0

    def test_all_winning_trades(self):
        trades = [self._make_trade(pnl=100.0), self._make_trade(pnl=200.0)]
        calc = StatsCalculator()
        stats = calc.calculate(closed_trades=trades, initial_equity=10000.0)
        assert stats.total_pnl == 300.0
        assert stats.total_pnl_pct == pytest.approx(3.0, rel=0.01)
        assert stats.win_rate == 1.0
        assert stats.winning_trades == 2
        assert stats.losing_trades == 0
        assert stats.profit_factor == float("inf")

    def test_mixed_trades(self):
        trades = [
            self._make_trade(pnl=200.0),
            self._make_trade(pnl=-100.0),
            self._make_trade(pnl=150.0),
            self._make_trade(pnl=-50.0),
        ]
        calc = StatsCalculator()
        stats = calc.calculate(closed_trades=trades, initial_equity=10000.0)
        assert stats.total_pnl == 200.0
        assert stats.total_trades == 4
        assert stats.winning_trades == 2
        assert stats.losing_trades == 2
        assert stats.win_rate == 0.5
        # profit_factor = 350 / 150 = 2.333...
        assert stats.profit_factor == pytest.approx(2.333, rel=0.01)

    def test_all_losing_trades(self):
        trades = [self._make_trade(pnl=-100.0), self._make_trade(pnl=-50.0)]
        calc = StatsCalculator()
        stats = calc.calculate(closed_trades=trades, initial_equity=10000.0)
        assert stats.win_rate == 0.0
        assert stats.profit_factor == 0.0

    def test_max_drawdown(self):
        """Drawdown should capture peak-to-trough decline."""
        trades = [
            self._make_trade(pnl=500.0, closed_at_str="2026-02-01"),
            self._make_trade(pnl=-800.0, closed_at_str="2026-02-02"),
            self._make_trade(pnl=200.0, closed_at_str="2026-02-03"),
        ]
        calc = StatsCalculator()
        stats = calc.calculate(closed_trades=trades, initial_equity=10000.0)
        # Peak = 10000 + 500 = 10500, trough = 10500 - 800 = 9700
        # Drawdown = (10500 - 9700) / 10500 = 7.619%
        assert stats.max_drawdown_pct == pytest.approx(7.619, rel=0.01)

    def test_sharpe_ratio_insufficient_data(self):
        """With < 2 unique days, Sharpe should be 0."""
        trades = [self._make_trade(pnl=100.0)]
        calc = StatsCalculator()
        stats = calc.calculate(closed_trades=trades, initial_equity=10000.0)
        assert stats.sharpe_ratio == 0.0

    def test_performance_stats_is_frozen(self):
        stats = PerformanceStats(
            total_pnl=0.0, total_pnl_pct=0.0, win_rate=0.0,
            total_trades=0, winning_trades=0, losing_trades=0,
            profit_factor=0.0, max_drawdown_pct=0.0, sharpe_ratio=0.0,
        )
        with pytest.raises(Exception):
            stats.total_pnl = 999.0
```

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_stats_calculator.py -v`
Expected: FAIL — module doesn't exist.

**Step 4: Implement StatsCalculator**

Create `stats/calculator.py`:

```python
from __future__ import annotations

import math
from collections import defaultdict
from typing import Protocol

from pydantic import BaseModel


class ClosedTrade(Protocol):
    pnl: float
    closed_at: object  # datetime with .date()


class PerformanceStats(BaseModel, frozen=True):
    total_pnl: float
    total_pnl_pct: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float


class StatsCalculator:
    def calculate(
        self, *, closed_trades: list, initial_equity: float
    ) -> PerformanceStats:
        if not closed_trades:
            return PerformanceStats(
                total_pnl=0.0, total_pnl_pct=0.0, win_rate=0.0,
                total_trades=0, winning_trades=0, losing_trades=0,
                profit_factor=0.0, max_drawdown_pct=0.0, sharpe_ratio=0.0,
            )

        total_pnl = sum(t.pnl for t in closed_trades)
        total_trades = len(closed_trades)
        winning_trades = sum(1 for t in closed_trades if t.pnl > 0)
        losing_trades = sum(1 for t in closed_trades if t.pnl < 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        gross_profit = sum(t.pnl for t in closed_trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in closed_trades if t.pnl < 0))
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        max_drawdown_pct = self._calc_max_drawdown(closed_trades, initial_equity)
        sharpe_ratio = self._calc_sharpe(closed_trades, initial_equity)

        return PerformanceStats(
            total_pnl=total_pnl,
            total_pnl_pct=(total_pnl / initial_equity) * 100 if initial_equity > 0 else 0.0,
            win_rate=win_rate,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            profit_factor=profit_factor,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
        )

    def _calc_max_drawdown(self, trades: list, initial_equity: float) -> float:
        equity = initial_equity
        peak = equity
        max_dd = 0.0
        for t in trades:
            equity += t.pnl
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd * 100  # as percentage

    def _calc_sharpe(self, trades: list, initial_equity: float) -> float:
        # Group PnL by date
        daily_pnl: dict[object, float] = defaultdict(float)
        for t in trades:
            if t.closed_at is not None:
                day = t.closed_at.date()
                daily_pnl[day] += t.pnl

        if len(daily_pnl) < 2:
            return 0.0

        daily_returns = [pnl / initial_equity for pnl in daily_pnl.values()]
        mean_return = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean_return) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
        std_return = math.sqrt(variance) if variance > 0 else 0.0

        if std_return == 0.0:
            return 0.0

        return (mean_return / std_return) * math.sqrt(365)
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_stats_calculator.py -v`
Expected: All PASS

**Step 6: Lint check**

Run: `uv run ruff check src/orchestrator/stats/ tests/unit/test_stats_calculator.py`

**Step 7: Commit**

```bash
git add orchestrator/src/orchestrator/stats/ orchestrator/tests/unit/test_stats_calculator.py
git commit -m "feat: add StatsCalculator with 5 performance metrics"
```

---

### Task 5: Integrate StatsCalculator into PaperEngine

**Files:**
- Modify: `orchestrator/src/orchestrator/exchange/paper_engine.py`
- Modify: `orchestrator/tests/unit/test_paper_engine.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_paper_engine.py`:

```python
class TestPaperEngineStats:
    def test_close_position_saves_stats_snapshot(self):
        """When a position is closed, stats should be calculated and snapshot saved."""
        from unittest.mock import MagicMock, call
        from orchestrator.exchange.paper_engine import PaperEngine, Position
        from orchestrator.models import Side, TradeProposal, EntryOrder
        from orchestrator.risk.position_sizer import RiskPercentSizer
        from orchestrator.stats.calculator import StatsCalculator

        trade_repo = MagicMock()
        trade_repo.get_all_closed.return_value = []
        snapshot_repo = MagicMock()
        stats_calc = StatsCalculator()

        engine = PaperEngine(
            initial_equity=10000.0,
            taker_fee_rate=0.0005,
            position_sizer=RiskPercentSizer(),
            trade_repo=trade_repo,
            snapshot_repo=snapshot_repo,
            stats_calculator=stats_calc,
        )

        # Open and close a position
        proposal = TradeProposal(
            symbol="BTC/USDT:USDT", side=Side.LONG, entry=EntryOrder(type="market"),
            position_size_risk_pct=1.0, stop_loss=93000.0, take_profit=[97000.0],
            time_horizon="4h", confidence=0.7, invalid_if=[], rationale="test",
        )
        engine.open_position(proposal, current_price=95000.0)

        # Mock the closed trades for stats calculation
        closed_trade_mock = MagicMock()
        closed_trade_mock.pnl = -150.0
        closed_trade_mock.closed_at = MagicMock()
        closed_trade_mock.closed_at.date.return_value = "2026-02-19"
        trade_repo.get_all_closed.return_value = [closed_trade_mock]

        # Trigger SL
        results = engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=92000.0)
        assert len(results) == 1

        # Verify snapshot was saved with stats
        snapshot_repo.save_snapshot.assert_called()
        call_kwargs = snapshot_repo.save_snapshot.call_args[1]
        assert "total_pnl" in call_kwargs
        assert "win_rate" in call_kwargs
        assert "total_trades" in call_kwargs
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_paper_engine.py::TestPaperEngineStats -v`
Expected: FAIL — PaperEngine doesn't accept `stats_calculator` param.

**Step 3: Update PaperEngine**

Modify `exchange/paper_engine.py`:

1. Add `stats_calculator` to `__init__`:
```python
from orchestrator.stats.calculator import StatsCalculator

class PaperEngine:
    def __init__(
        self,
        *,
        initial_equity: float,
        taker_fee_rate: float,
        position_sizer: PositionSizer,
        trade_repo: PaperTradeRepository,
        snapshot_repo: AccountSnapshotRepository,
        stats_calculator: StatsCalculator | None = None,
    ) -> None:
        # ... existing init ...
        self._stats_calculator = stats_calculator
```

2. Add `_save_stats_snapshot` method:
```python
def _save_stats_snapshot(self) -> None:
    """Calculate performance stats and save snapshot."""
    if self._stats_calculator is None:
        return
    from datetime import date, UTC, datetime

    closed_trades = self._trade_repo.get_all_closed()
    stats = self._stats_calculator.calculate(
        closed_trades=closed_trades, initial_equity=self._initial_equity
    )
    daily_pnl = self._trade_repo.get_daily_pnl(datetime.now(UTC).date())
    self._snapshot_repo.save_snapshot(
        equity=self.equity,
        open_count=len(self._positions),
        daily_pnl=daily_pnl,
        total_pnl=stats.total_pnl,
        win_rate=stats.win_rate,
        profit_factor=stats.profit_factor,
        max_drawdown_pct=stats.max_drawdown_pct,
        sharpe_ratio=stats.sharpe_ratio,
        total_trades=stats.total_trades,
    )
    logger.info("stats_snapshot_saved", total_pnl=stats.total_pnl, win_rate=stats.win_rate)
```

3. Call `_save_stats_snapshot()` at the end of `_close()` method:
```python
def _close(self, pos: Position, *, exit_price: float, reason: str) -> CloseResult:
    # ... existing close logic ...
    self._save_stats_snapshot()
    return CloseResult(...)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_paper_engine.py::TestPaperEngineStats -v`
Expected: PASS

**Step 5: Update existing PaperEngine tests**

Existing tests don't pass `stats_calculator`, which is fine (it's optional/None). Verify all pass:

Run: `uv run pytest tests/unit/test_paper_engine.py -v`
Expected: All PASS

**Step 6: Update `__main__.py` to wire StatsCalculator**

In `__main__.py`, import and pass `StatsCalculator` to `PaperEngine`:

```python
from orchestrator.stats.calculator import StatsCalculator

# In create_app_components:
stats_calculator = StatsCalculator()

paper_engine = PaperEngine(
    initial_equity=paper_initial_equity,
    taker_fee_rate=paper_taker_fee_rate,
    position_sizer=RiskPercentSizer(),
    trade_repo=paper_trade_repo,
    snapshot_repo=account_snapshot_repo,
    stats_calculator=stats_calculator,
)
```

Also add `stats_calculator` to the returned dict.

**Step 7: Run full tests**

Run: `uv run pytest -v --tb=short`
Expected: All pass

**Step 8: Commit**

```bash
git add orchestrator/src/orchestrator/exchange/paper_engine.py orchestrator/src/orchestrator/__main__.py orchestrator/tests/unit/test_paper_engine.py
git commit -m "feat: integrate StatsCalculator into PaperEngine on position close"
```

---

### Task 6: Add format_perf_report and format_eval_report formatters

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/formatters.py`
- Modify: `orchestrator/tests/unit/test_telegram.py`

**Step 1: Write the failing tests**

Add to `tests/unit/test_telegram.py`:

```python
from orchestrator.stats.calculator import PerformanceStats


class TestFormatPerfReport:
    def test_format_perf_report_positive(self):
        from orchestrator.telegram.formatters import format_perf_report

        stats = PerformanceStats(
            total_pnl=1250.0, total_pnl_pct=12.5, win_rate=0.625,
            total_trades=16, winning_trades=10, losing_trades=6,
            profit_factor=1.85, max_drawdown_pct=4.2, sharpe_ratio=1.32,
        )
        text = format_perf_report(stats)
        assert "+$1,250.00" in text
        assert "12.5%" in text
        assert "62.5%" in text
        assert "10/16" in text
        assert "1.85" in text
        assert "4.2%" in text
        assert "1.32" in text

    def test_format_perf_report_no_trades(self):
        from orchestrator.telegram.formatters import format_perf_report

        stats = PerformanceStats(
            total_pnl=0.0, total_pnl_pct=0.0, win_rate=0.0,
            total_trades=0, winning_trades=0, losing_trades=0,
            profit_factor=0.0, max_drawdown_pct=0.0, sharpe_ratio=0.0,
        )
        text = format_perf_report(stats)
        assert "No trades" in text or "0" in text


class TestFormatEvalReport:
    def test_format_eval_report_with_failures(self):
        from orchestrator.telegram.formatters import format_eval_report

        report = {
            "dataset_name": "golden_v1",
            "total_cases": 5,
            "passed_cases": 4,
            "failed_cases": 1,
            "accuracy": 0.8,
            "consistency_score": 0.933,
            "failures": [{"case_id": "bear_divergence", "reason": "expected SHORT, got LONG"}],
        }
        text = format_eval_report(report)
        assert "golden_v1" in text
        assert "80" in text
        assert "bear_divergence" in text
        assert "93" in text

    def test_format_eval_report_all_passed(self):
        from orchestrator.telegram.formatters import format_eval_report

        report = {
            "dataset_name": "golden_v1",
            "total_cases": 5,
            "passed_cases": 5,
            "failed_cases": 0,
            "accuracy": 1.0,
            "consistency_score": 1.0,
            "failures": [],
        }
        text = format_eval_report(report)
        assert "100" in text
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_telegram.py::TestFormatPerfReport tests/unit/test_telegram.py::TestFormatEvalReport -v`
Expected: FAIL — functions don't exist.

**Step 3: Implement formatters**

Add to `telegram/formatters.py`:

```python
from orchestrator.stats.calculator import PerformanceStats


def format_perf_report(stats: PerformanceStats) -> str:
    if stats.total_trades == 0:
        return "No trades yet. Performance report will be available after closing positions."

    pnl_sign = "+" if stats.total_pnl >= 0 else "-"
    pnl_str = f"{pnl_sign}${abs(stats.total_pnl):,.2f}"
    pnl_pct_sign = "+" if stats.total_pnl_pct >= 0 else ""
    pf_str = "inf" if stats.profit_factor == float("inf") else f"{stats.profit_factor:.2f}"

    lines = [
        "Performance Report",
        "─────────────────────",
        f"Total PnL:      {pnl_str} ({pnl_pct_sign}{stats.total_pnl_pct:.1f}%)",
        f"Win Rate:       {stats.win_rate:.1%} ({stats.winning_trades}/{stats.total_trades})",
        f"Profit Factor:  {pf_str}",
        f"Max Drawdown:   {stats.max_drawdown_pct:.1f}%",
        f"Sharpe Ratio:   {stats.sharpe_ratio:.2f}",
        "─────────────────────",
    ]
    return "\n".join(lines)


def format_eval_report(report: dict) -> str:
    lines = [
        f"Eval Report ({report['dataset_name']})",
        "──────────────────────────",
        f"Cases: {report['total_cases']} | Passed: {report['passed_cases']} | Failed: {report['failed_cases']}",
        f"Accuracy: {report['accuracy']:.0%}",
    ]
    if report.get("consistency_score") is not None:
        lines.append(f"Consistency: {report['consistency_score']:.1%}")
    lines.append("──────────────────────────")

    for f in report.get("failures", []):
        lines.append(f"  {f['case_id']}: {f['reason']}")

    return "\n".join(lines)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_telegram.py::TestFormatPerfReport tests/unit/test_telegram.py::TestFormatEvalReport -v`
Expected: All PASS

**Step 5: Update format_help to include new commands**

Update `format_help()` to add `/perf` and `/eval`:

```python
def format_help() -> str:
    return (
        "Available commands:\n\n"
        "/start - Welcome message\n"
        "/status - Account overview & latest proposals\n"
        "/coin <symbol> - Detailed analysis for a symbol (e.g. /coin BTC)\n"
        "/run - Trigger pipeline for all symbols\n"
        "/run <symbol> - Trigger pipeline for specific symbol\n"
        "/run <symbol> sonnet|opus - Trigger with specific model\n"
        "/history - Recent trade records\n"
        "/perf - Performance report (PnL, win rate, Sharpe, etc.)\n"
        "/eval - Run LLM evaluation and show results\n"
        "/resume - Un-pause pipeline after risk pause\n"
        "/help - Show this message"
    )
```

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/formatters.py orchestrator/tests/unit/test_telegram.py
git commit -m "feat: add performance report and eval report formatters"
```

---

### Task 7: Add /perf TG command

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/bot.py`
- Modify: `orchestrator/tests/unit/test_telegram.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_telegram.py`:

```python
from unittest.mock import AsyncMock, MagicMock, patch

class TestPerfHandler:
    @pytest.mark.asyncio
    async def test_perf_handler_returns_stats(self):
        """The /perf command should show performance stats."""
        from orchestrator.stats.calculator import PerformanceStats

        bot = SentinelBot(token="test-token", admin_chat_ids=[123])
        mock_snapshot = MagicMock()
        mock_snapshot.total_pnl = 500.0
        mock_snapshot.win_rate = 0.6
        mock_snapshot.profit_factor = 1.5
        mock_snapshot.max_drawdown_pct = 3.0
        mock_snapshot.sharpe_ratio = 1.1
        mock_snapshot.total_trades = 10
        mock_snapshot.equity = 10500.0
        mock_snapshot.total_pnl_pct = 5.0

        mock_snapshot_repo = MagicMock()
        mock_snapshot_repo.get_latest.return_value = mock_snapshot
        bot.set_snapshot_repo(mock_snapshot_repo)

        update = MagicMock()
        update.effective_chat.id = 123
        update.message.reply_text = AsyncMock()
        context = MagicMock()

        await bot._perf_handler(update, context)
        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "Win Rate" in text or "win" in text.lower()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_telegram.py::TestPerfHandler -v`
Expected: FAIL — `set_snapshot_repo` / `_perf_handler` don't exist.

**Step 3: Implement /perf in bot.py**

Update `bot.py`:

1. Add import: `from orchestrator.telegram.formatters import format_perf_report`
2. Add `AccountSnapshotRepository` to TYPE_CHECKING imports
3. Add `_snapshot_repo` to `__init__` and `set_snapshot_repo()` method
4. Register handler in `build()`: `self._app.add_handler(CommandHandler("perf", self._perf_handler))`
5. Add handler:

```python
def set_snapshot_repo(self, repo: AccountSnapshotRepository) -> None:
    self._snapshot_repo = repo

async def _perf_handler(
    self, update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    if not await self._check_admin(update):
        return
    if self._snapshot_repo is None:
        await update.message.reply_text("Stats not configured.")
        return
    snapshot = self._snapshot_repo.get_latest()
    if snapshot is None or snapshot.total_trades == 0:
        await update.message.reply_text("No performance data yet. Close some positions first.")
        return
    from orchestrator.stats.calculator import PerformanceStats
    stats = PerformanceStats(
        total_pnl=snapshot.total_pnl,
        total_pnl_pct=(snapshot.total_pnl / snapshot.equity * 100) if snapshot.equity > 0 else 0.0,
        win_rate=snapshot.win_rate,
        total_trades=snapshot.total_trades,
        winning_trades=int(snapshot.win_rate * snapshot.total_trades),
        losing_trades=snapshot.total_trades - int(snapshot.win_rate * snapshot.total_trades),
        profit_factor=snapshot.profit_factor,
        max_drawdown_pct=snapshot.max_drawdown_pct,
        sharpe_ratio=snapshot.sharpe_ratio,
    )
    await update.message.reply_text(format_perf_report(stats))
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_telegram.py::TestPerfHandler -v`
Expected: PASS

**Step 5: Wire snapshot_repo in `__main__.py`**

Add `bot.set_snapshot_repo(account_snapshot_repo)` after existing `set_*` calls.

**Step 6: Run full tests**

Run: `uv run pytest -v --tb=short`
Expected: All pass

**Step 7: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/bot.py orchestrator/src/orchestrator/__main__.py orchestrator/tests/unit/test_telegram.py
git commit -m "feat: add /perf TG command for performance report"
```

---

### Task 8: Implement eval dataset models and loader

**Files:**
- Create: `orchestrator/src/orchestrator/eval/__init__.py`
- Create: `orchestrator/src/orchestrator/eval/dataset.py`
- Create: `orchestrator/tests/unit/test_eval_dataset.py`

**Step 1: Create empty `__init__.py`**

**Step 2: Write the failing tests**

Create `tests/unit/test_eval_dataset.py`:

```python
import pytest
import yaml
import tempfile
import os

from orchestrator.eval.dataset import EvalCase, ExpectedRange, load_dataset


class TestEvalDataset:
    def _write_yaml(self, cases: list[dict]) -> str:
        """Write cases to a temp YAML file and return path."""
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            yaml.dump(cases, f)
        return path

    def test_load_single_case(self):
        cases = [{
            "id": "bull_breakout",
            "description": "Strong uptrend",
            "snapshot": {
                "symbol": "BTC/USDT:USDT",
                "current_price": 95000.0,
                "ohlcv": [[1, 94000, 95500, 93500, 95200, 1000]],
                "funding_rate": 0.01,
                "open_interest": 15000000000,
            },
            "expected": {
                "proposal": {
                    "side": ["long"],
                    "confidence": {"min": 0.5},
                },
            },
        }]
        path = self._write_yaml(cases)
        try:
            dataset = load_dataset(path)
            assert len(dataset) == 1
            assert dataset[0].id == "bull_breakout"
            assert dataset[0].expected.proposal is not None
            assert dataset[0].expected.proposal.side == ["long"]
        finally:
            os.unlink(path)

    def test_load_with_range_constraints(self):
        cases = [{
            "id": "test",
            "description": "test",
            "snapshot": {"symbol": "BTC/USDT:USDT", "current_price": 95000.0, "ohlcv": []},
            "expected": {
                "sentiment": {
                    "sentiment_score": {"min": 60, "max": 90},
                    "confidence": {"min": 0.5},
                },
            },
        }]
        path = self._write_yaml(cases)
        try:
            dataset = load_dataset(path)
            assert dataset[0].expected.sentiment is not None
            assert dataset[0].expected.sentiment.sentiment_score.min == 60
            assert dataset[0].expected.sentiment.sentiment_score.max == 90
        finally:
            os.unlink(path)

    def test_expected_range_is_frozen(self):
        r = ExpectedRange(min=1.0, max=10.0)
        with pytest.raises(Exception):
            r.min = 5.0
```

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_eval_dataset.py -v`
Expected: FAIL — module doesn't exist.

**Step 4: Implement dataset models and loader**

Create `eval/dataset.py`:

```python
from __future__ import annotations

import yaml
from pydantic import BaseModel


class ExpectedRange(BaseModel, frozen=True):
    min: float | None = None
    max: float | None = None


class ExpectedSentiment(BaseModel, frozen=True):
    sentiment_score: ExpectedRange | None = None
    confidence: ExpectedRange | None = None


class ExpectedMarket(BaseModel, frozen=True):
    trend: list[str] | None = None
    volatility_regime: list[str] | None = None


class ExpectedProposal(BaseModel, frozen=True):
    side: list[str] | None = None
    confidence: ExpectedRange | None = None
    has_stop_loss: bool | None = None
    sl_correct_side: bool | None = None


class ExpectedOutputs(BaseModel, frozen=True):
    sentiment: ExpectedSentiment | None = None
    market: ExpectedMarket | None = None
    proposal: ExpectedProposal | None = None


class EvalCase(BaseModel, frozen=True):
    id: str
    description: str
    snapshot: dict
    expected: ExpectedOutputs


def load_dataset(path: str) -> list[EvalCase]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return [EvalCase.model_validate(case) for case in raw]
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_eval_dataset.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/eval/ orchestrator/tests/unit/test_eval_dataset.py
git commit -m "feat: add eval dataset models and YAML loader"
```

---

### Task 9: Implement rule-based scorer

**Files:**
- Create: `orchestrator/src/orchestrator/eval/scorers.py`
- Create: `orchestrator/tests/unit/test_eval_scorers.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_eval_scorers.py`:

```python
import pytest

from orchestrator.eval.dataset import ExpectedProposal, ExpectedRange, ExpectedSentiment
from orchestrator.eval.scorers import RuleScorer, ScoreResult
from orchestrator.models import (
    EntryOrder, SentimentReport, Side, TradeProposal,
)


class TestRuleScorer:
    def test_score_proposal_side_pass(self):
        scorer = RuleScorer()
        proposal = TradeProposal(
            symbol="BTC/USDT:USDT", side=Side.LONG, entry=EntryOrder(type="market"),
            position_size_risk_pct=1.0, stop_loss=93000.0, take_profit=[97000.0],
            time_horizon="4h", confidence=0.7, invalid_if=[], rationale="test",
        )
        expected = ExpectedProposal(side=["long"])
        results = scorer.score_proposal(proposal, expected)
        side_result = next(r for r in results if r.field == "side")
        assert side_result.passed is True

    def test_score_proposal_side_fail(self):
        scorer = RuleScorer()
        proposal = TradeProposal(
            symbol="BTC/USDT:USDT", side=Side.LONG, entry=EntryOrder(type="market"),
            position_size_risk_pct=1.0, stop_loss=93000.0, take_profit=[97000.0],
            time_horizon="4h", confidence=0.7, invalid_if=[], rationale="test",
        )
        expected = ExpectedProposal(side=["short", "flat"])
        results = scorer.score_proposal(proposal, expected)
        side_result = next(r for r in results if r.field == "side")
        assert side_result.passed is False

    def test_score_sentiment_range_pass(self):
        scorer = RuleScorer()
        sentiment = SentimentReport(
            sentiment_score=75, key_events=[], sources=["test"], confidence=0.8
        )
        expected = ExpectedSentiment(
            sentiment_score=ExpectedRange(min=60, max=90),
            confidence=ExpectedRange(min=0.5),
        )
        results = scorer.score_sentiment(sentiment, expected)
        assert all(r.passed for r in results)

    def test_score_sentiment_range_fail(self):
        scorer = RuleScorer()
        sentiment = SentimentReport(
            sentiment_score=30, key_events=[], sources=["test"], confidence=0.3
        )
        expected = ExpectedSentiment(
            sentiment_score=ExpectedRange(min=60, max=90),
            confidence=ExpectedRange(min=0.5),
        )
        results = scorer.score_sentiment(sentiment, expected)
        failed = [r for r in results if not r.passed]
        assert len(failed) == 2  # both score and confidence fail

    def test_score_result_is_frozen(self):
        result = ScoreResult(field="test", passed=True, expected="x", actual="x")
        with pytest.raises(Exception):
            result.passed = False
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_eval_scorers.py -v`
Expected: FAIL — module doesn't exist.

**Step 3: Implement RuleScorer**

Create `eval/scorers.py`:

```python
from __future__ import annotations

from pydantic import BaseModel

from orchestrator.eval.dataset import (
    ExpectedMarket,
    ExpectedProposal,
    ExpectedRange,
    ExpectedSentiment,
)
from orchestrator.models import MarketInterpretation, SentimentReport, TradeProposal


class ScoreResult(BaseModel, frozen=True):
    field: str
    passed: bool
    expected: str
    actual: str
    reason: str = ""


class RuleScorer:
    def score_proposal(
        self, proposal: TradeProposal, expected: ExpectedProposal
    ) -> list[ScoreResult]:
        results: list[ScoreResult] = []

        if expected.side is not None:
            passed = proposal.side.value in expected.side
            results.append(ScoreResult(
                field="side", passed=passed,
                expected=str(expected.side), actual=proposal.side.value,
                reason="" if passed else f"expected one of {expected.side}, got {proposal.side.value}",
            ))

        if expected.confidence is not None:
            results.append(self._check_range(
                "confidence", proposal.confidence, expected.confidence
            ))

        if expected.has_stop_loss is not None:
            has_sl = proposal.stop_loss is not None
            passed = has_sl == expected.has_stop_loss
            results.append(ScoreResult(
                field="has_stop_loss", passed=passed,
                expected=str(expected.has_stop_loss), actual=str(has_sl),
            ))

        if expected.sl_correct_side is not None and proposal.stop_loss is not None:
            if proposal.side.value == "long":
                correct = proposal.stop_loss < (proposal.entry.price or 0)
            elif proposal.side.value == "short":
                correct = proposal.stop_loss > (proposal.entry.price or float("inf"))
            else:
                correct = True
            results.append(ScoreResult(
                field="sl_correct_side", passed=correct == expected.sl_correct_side,
                expected=str(expected.sl_correct_side), actual=str(correct),
            ))

        return results

    def score_sentiment(
        self, output: SentimentReport, expected: ExpectedSentiment
    ) -> list[ScoreResult]:
        results: list[ScoreResult] = []

        if expected.sentiment_score is not None:
            results.append(self._check_range(
                "sentiment_score", float(output.sentiment_score), expected.sentiment_score
            ))

        if expected.confidence is not None:
            results.append(self._check_range(
                "confidence", output.confidence, expected.confidence
            ))

        return results

    def score_market(
        self, output: MarketInterpretation, expected: ExpectedMarket
    ) -> list[ScoreResult]:
        results: list[ScoreResult] = []

        if expected.trend is not None:
            passed = output.trend.value in expected.trend
            results.append(ScoreResult(
                field="trend", passed=passed,
                expected=str(expected.trend), actual=output.trend.value,
                reason="" if passed else f"expected one of {expected.trend}, got {output.trend.value}",
            ))

        if expected.volatility_regime is not None:
            passed = output.volatility_regime.value in expected.volatility_regime
            results.append(ScoreResult(
                field="volatility_regime", passed=passed,
                expected=str(expected.volatility_regime),
                actual=output.volatility_regime.value,
            ))

        return results

    def _check_range(
        self, field: str, value: float, expected: ExpectedRange
    ) -> ScoreResult:
        passed = True
        reasons: list[str] = []

        if expected.min is not None and value < expected.min:
            passed = False
            reasons.append(f"{value} < min {expected.min}")
        if expected.max is not None and value > expected.max:
            passed = False
            reasons.append(f"{value} > max {expected.max}")

        return ScoreResult(
            field=field, passed=passed,
            expected=f"[{expected.min}, {expected.max}]", actual=str(value),
            reason="; ".join(reasons),
        )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_eval_scorers.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/eval/scorers.py orchestrator/tests/unit/test_eval_scorers.py
git commit -m "feat: add rule-based eval scorer for proposal, sentiment, and market"
```

---

### Task 10: Implement eval runner

**Files:**
- Create: `orchestrator/src/orchestrator/eval/runner.py`
- Create: `orchestrator/tests/unit/test_eval_runner.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_eval_runner.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

from orchestrator.eval.dataset import (
    EvalCase, ExpectedOutputs, ExpectedProposal,
)
from orchestrator.eval.runner import EvalRunner, EvalReport


class TestEvalRunner:
    @pytest.mark.asyncio
    async def test_run_single_passing_case(self):
        from orchestrator.models import (
            EntryOrder, Side, TradeProposal, SentimentReport,
            MarketInterpretation, Trend, VolatilityRegime,
        )
        from orchestrator.agents.base import AgentResult
        from orchestrator.llm.client import LLMCallResult

        case = EvalCase(
            id="bull_breakout",
            description="Strong uptrend",
            snapshot={"symbol": "BTC/USDT:USDT", "current_price": 95000.0, "ohlcv": []},
            expected=ExpectedOutputs(
                proposal=ExpectedProposal(side=["long"]),
            ),
        )

        # Mock agents to return expected outputs
        sentiment_agent = AsyncMock()
        sentiment_agent.analyze.return_value = MagicMock(
            output=SentimentReport(
                sentiment_score=75, key_events=[], sources=["test"], confidence=0.8
            ),
            degraded=False, llm_calls=[], messages=[],
        )
        market_agent = AsyncMock()
        market_agent.analyze.return_value = MagicMock(
            output=MarketInterpretation(
                trend=Trend.UP, volatility_regime=VolatilityRegime.MEDIUM,
                key_levels=[], risk_flags=[],
            ),
            degraded=False, llm_calls=[], messages=[],
        )
        proposer_agent = AsyncMock()
        proposer_agent.analyze.return_value = MagicMock(
            output=TradeProposal(
                symbol="BTC/USDT:USDT", side=Side.LONG, entry=EntryOrder(type="market"),
                position_size_risk_pct=1.0, stop_loss=93000.0, take_profit=[97000.0],
                time_horizon="4h", confidence=0.7, invalid_if=[], rationale="Bullish",
            ),
            degraded=False, llm_calls=[], messages=[],
        )

        runner = EvalRunner(
            sentiment_agent=sentiment_agent,
            market_agent=market_agent,
            proposer_agent=proposer_agent,
        )

        report = await runner.run(cases=[case], dataset_name="test")
        assert isinstance(report, EvalReport)
        assert report.total_cases == 1
        assert report.passed_cases == 1
        assert report.accuracy == 1.0

    @pytest.mark.asyncio
    async def test_run_single_failing_case(self):
        from orchestrator.models import (
            EntryOrder, Side, TradeProposal, SentimentReport,
            MarketInterpretation, Trend, VolatilityRegime,
        )

        case = EvalCase(
            id="bear_divergence",
            description="Should be short",
            snapshot={"symbol": "BTC/USDT:USDT", "current_price": 95000.0, "ohlcv": []},
            expected=ExpectedOutputs(
                proposal=ExpectedProposal(side=["short"]),
            ),
        )

        sentiment_agent = AsyncMock()
        sentiment_agent.analyze.return_value = MagicMock(
            output=SentimentReport(
                sentiment_score=50, key_events=[], sources=[], confidence=0.5
            ),
            degraded=False, llm_calls=[], messages=[],
        )
        market_agent = AsyncMock()
        market_agent.analyze.return_value = MagicMock(
            output=MarketInterpretation(
                trend=Trend.UP, volatility_regime=VolatilityRegime.LOW,
                key_levels=[], risk_flags=[],
            ),
            degraded=False, llm_calls=[], messages=[],
        )
        proposer_agent = AsyncMock()
        proposer_agent.analyze.return_value = MagicMock(
            output=TradeProposal(
                symbol="BTC/USDT:USDT", side=Side.LONG, entry=EntryOrder(type="market"),
                position_size_risk_pct=1.0, stop_loss=93000.0, take_profit=[97000.0],
                time_horizon="4h", confidence=0.7, invalid_if=[], rationale="wrong",
            ),
            degraded=False, llm_calls=[], messages=[],
        )

        runner = EvalRunner(
            sentiment_agent=sentiment_agent,
            market_agent=market_agent,
            proposer_agent=proposer_agent,
        )

        report = await runner.run(cases=[case], dataset_name="test")
        assert report.passed_cases == 0
        assert report.failed_cases == 1
        assert len(report.case_results) == 1
        assert report.case_results[0].passed is False

    def test_eval_report_is_frozen(self):
        report = EvalReport(
            dataset_name="test", total_cases=0, passed_cases=0,
            failed_cases=0, accuracy=0.0, case_results=[],
        )
        with pytest.raises(Exception):
            report.total_cases = 5
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_eval_runner.py -v`
Expected: FAIL — module doesn't exist.

**Step 3: Implement EvalRunner**

Create `eval/runner.py`:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel

from orchestrator.eval.dataset import EvalCase
from orchestrator.eval.scorers import RuleScorer, ScoreResult
from orchestrator.exchange.data_fetcher import MarketSnapshot

if TYPE_CHECKING:
    from orchestrator.agents.base import BaseAgent

logger = structlog.get_logger(__name__)


class CaseResult(BaseModel, frozen=True):
    case_id: str
    passed: bool
    scores: list[ScoreResult]
    consistency: float | None = None


class EvalReport(BaseModel, frozen=True):
    dataset_name: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    accuracy: float
    consistency_score: float | None = None
    case_results: list[CaseResult]


class EvalRunner:
    def __init__(
        self,
        *,
        sentiment_agent: BaseAgent,
        market_agent: BaseAgent,
        proposer_agent: BaseAgent,
    ) -> None:
        self._sentiment_agent = sentiment_agent
        self._market_agent = market_agent
        self._proposer_agent = proposer_agent
        self._scorer = RuleScorer()

    async def run(
        self, *, cases: list[EvalCase], dataset_name: str
    ) -> EvalReport:
        case_results: list[CaseResult] = []

        for case in cases:
            logger.info("eval_case_start", case_id=case.id)
            result = await self._evaluate_case(case)
            case_results.append(result)
            logger.info("eval_case_done", case_id=case.id, passed=result.passed)

        passed = sum(1 for r in case_results if r.passed)
        total = len(case_results)

        return EvalReport(
            dataset_name=dataset_name,
            total_cases=total,
            passed_cases=passed,
            failed_cases=total - passed,
            accuracy=passed / total if total > 0 else 0.0,
            case_results=case_results,
        )

    async def _evaluate_case(self, case: EvalCase) -> CaseResult:
        snapshot = self._build_snapshot(case.snapshot)
        all_scores: list[ScoreResult] = []

        # Run agents
        sentiment_result = await self._sentiment_agent.analyze(snapshot=snapshot)
        market_result = await self._market_agent.analyze(snapshot=snapshot)
        proposer_result = await self._proposer_agent.analyze(
            snapshot=snapshot,
            sentiment=sentiment_result.output,
            market=market_result.output,
        )

        # Score each agent output
        if case.expected.sentiment is not None:
            all_scores.extend(
                self._scorer.score_sentiment(sentiment_result.output, case.expected.sentiment)
            )

        if case.expected.market is not None:
            all_scores.extend(
                self._scorer.score_market(market_result.output, case.expected.market)
            )

        if case.expected.proposal is not None:
            all_scores.extend(
                self._scorer.score_proposal(proposer_result.output, case.expected.proposal)
            )

        passed = all(s.passed for s in all_scores) if all_scores else True

        return CaseResult(case_id=case.id, passed=passed, scores=all_scores)

    def _build_snapshot(self, raw: dict) -> MarketSnapshot:
        return MarketSnapshot(
            symbol=raw.get("symbol", "BTC/USDT:USDT"),
            timeframe=raw.get("timeframe", "4h"),
            current_price=raw.get("current_price", 0.0),
            volume_24h=raw.get("volume_24h", 0.0),
            funding_rate=raw.get("funding_rate", 0.0),
            open_interest=raw.get("open_interest", 0.0),
            ohlcv=raw.get("ohlcv", []),
        )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_eval_runner.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/eval/runner.py orchestrator/tests/unit/test_eval_runner.py
git commit -m "feat: add eval runner with case evaluation and report generation"
```

---

### Task 11: Implement self-consistency checker

**Files:**
- Create: `orchestrator/src/orchestrator/eval/consistency.py`
- Create: `orchestrator/tests/unit/test_eval_consistency.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_eval_consistency.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

from orchestrator.eval.consistency import ConsistencyChecker
from orchestrator.eval.dataset import EvalCase, ExpectedOutputs
from orchestrator.models import (
    EntryOrder, Side, TradeProposal,
)


class TestConsistencyChecker:
    @pytest.mark.asyncio
    async def test_fully_consistent(self):
        """All runs produce same side → consistency 1.0."""
        case = EvalCase(
            id="test", description="test",
            snapshot={"symbol": "BTC/USDT:USDT", "current_price": 95000.0, "ohlcv": []},
            expected=ExpectedOutputs(),
        )

        proposer = AsyncMock()
        proposer.analyze.return_value = MagicMock(
            output=TradeProposal(
                symbol="BTC/USDT:USDT", side=Side.LONG, entry=EntryOrder(type="market"),
                position_size_risk_pct=1.0, stop_loss=93000.0, take_profit=[97000.0],
                time_horizon="4h", confidence=0.7, invalid_if=[], rationale="test",
            ),
        )

        checker = ConsistencyChecker(proposer_agent=proposer)
        score = await checker.check(case, runs=3)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_partially_consistent(self):
        """2 out of 3 agree → consistency 2/3."""
        case = EvalCase(
            id="test", description="test",
            snapshot={"symbol": "BTC/USDT:USDT", "current_price": 95000.0, "ohlcv": []},
            expected=ExpectedOutputs(),
        )

        proposer = AsyncMock()
        long_proposal = MagicMock()
        long_proposal.output = TradeProposal(
            symbol="BTC/USDT:USDT", side=Side.LONG, entry=EntryOrder(type="market"),
            position_size_risk_pct=1.0, stop_loss=93000.0, take_profit=[97000.0],
            time_horizon="4h", confidence=0.7, invalid_if=[], rationale="test",
        )
        short_proposal = MagicMock()
        short_proposal.output = TradeProposal(
            symbol="BTC/USDT:USDT", side=Side.SHORT, entry=EntryOrder(type="market"),
            position_size_risk_pct=1.0, stop_loss=97000.0, take_profit=[93000.0],
            time_horizon="4h", confidence=0.6, invalid_if=[], rationale="test",
        )
        proposer.analyze.side_effect = [long_proposal, long_proposal, short_proposal]

        checker = ConsistencyChecker(proposer_agent=proposer)
        score = await checker.check(case, runs=3)
        assert score == pytest.approx(2 / 3)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_eval_consistency.py -v`
Expected: FAIL

**Step 3: Implement ConsistencyChecker**

Create `eval/consistency.py`:

```python
from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import structlog

from orchestrator.eval.dataset import EvalCase
from orchestrator.exchange.data_fetcher import MarketSnapshot

if TYPE_CHECKING:
    from orchestrator.agents.base import BaseAgent

logger = structlog.get_logger(__name__)


class ConsistencyChecker:
    def __init__(self, *, proposer_agent: BaseAgent) -> None:
        self._proposer_agent = proposer_agent

    async def check(self, case: EvalCase, *, runs: int = 3) -> float:
        snapshot = MarketSnapshot(
            symbol=case.snapshot.get("symbol", "BTC/USDT:USDT"),
            timeframe=case.snapshot.get("timeframe", "4h"),
            current_price=case.snapshot.get("current_price", 0.0),
            volume_24h=case.snapshot.get("volume_24h", 0.0),
            funding_rate=case.snapshot.get("funding_rate", 0.0),
            open_interest=case.snapshot.get("open_interest", 0.0),
            ohlcv=case.snapshot.get("ohlcv", []),
        )

        sides: list[str] = []
        for i in range(runs):
            result = await self._proposer_agent.analyze(
                snapshot=snapshot,
                sentiment=None,
                market=None,
            )
            sides.append(result.output.side.value)
            logger.info("consistency_run", case_id=case.id, run=i + 1, side=sides[-1])

        if not sides:
            return 0.0

        counter = Counter(sides)
        most_common_count = counter.most_common(1)[0][1]
        return most_common_count / len(sides)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_eval_consistency.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/eval/consistency.py orchestrator/tests/unit/test_eval_consistency.py
git commit -m "feat: add self-consistency checker for eval framework"
```

---

### Task 12: Create golden dataset v1 (5 cases)

**Files:**
- Create: `orchestrator/src/orchestrator/eval/datasets/golden_v1.yaml`
- Modify: `orchestrator/tests/unit/test_eval_dataset.py` (add load test for golden_v1)

**Step 1: Create the golden dataset**

Create `eval/datasets/golden_v1.yaml` with 5 representative market scenarios. Each case should have realistic OHLCV data (at least 5 candles) and cover bull, bear, sideways, high-vol, and funding-anomaly scenarios.

Refer to `.claude/plans/2026-02-19-m3-eval-perf-design.md` for the case definitions.

**Step 2: Write test to validate loading**

Add to `tests/unit/test_eval_dataset.py`:

```python
import os

class TestGoldenDatasetV1:
    def test_load_golden_v1(self):
        """The golden_v1 dataset should load and have 5 cases."""
        dataset_path = os.path.join(
            os.path.dirname(__file__),
            "../../src/orchestrator/eval/datasets/golden_v1.yaml",
        )
        dataset = load_dataset(dataset_path)
        assert len(dataset) == 5
        ids = {c.id for c in dataset}
        assert "bull_breakout" in ids
        assert "bear_divergence" in ids
        assert "sideways_range" in ids
        assert "high_volatility" in ids
        assert "funding_anomaly" in ids
```

**Step 3: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_eval_dataset.py::TestGoldenDatasetV1 -v`
Expected: PASS

**Step 4: Commit**

```bash
git add orchestrator/src/orchestrator/eval/datasets/ orchestrator/tests/unit/test_eval_dataset.py
git commit -m "feat: add golden_v1 eval dataset with 5 market scenarios"
```

---

### Task 13: Add /eval TG command and CLI subcommands

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/bot.py`
- Modify: `orchestrator/src/orchestrator/__main__.py`
- Modify: `orchestrator/tests/unit/test_telegram.py`
- Create: `orchestrator/tests/unit/test_main_cli.py`

**Step 1: Write the failing test for /eval handler**

Add to `tests/unit/test_telegram.py`:

```python
class TestEvalHandler:
    @pytest.mark.asyncio
    async def test_eval_handler_no_runner(self):
        """Without eval runner, /eval should say not configured."""
        bot = SentinelBot(token="test-token", admin_chat_ids=[123])
        update = MagicMock()
        update.effective_chat.id = 123
        update.message.reply_text = AsyncMock()
        context = MagicMock()

        await bot._eval_handler(update, context)
        text = update.message.reply_text.call_args[0][0]
        assert "not configured" in text.lower() or "not available" in text.lower()
```

**Step 2: Implement /eval in bot.py**

Add to `SentinelBot`:
- `_eval_runner` attribute (optional)
- `set_eval_runner()` method
- Register handler in `build()`
- `_eval_handler` method that runs eval and sends report

```python
async def _eval_handler(
    self, update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    if not await self._check_admin(update):
        return
    if self._eval_runner is None:
        await update.message.reply_text("Eval not configured.")
        return
    await update.message.reply_text("Running evaluation...")
    report = await self._eval_runner.run_default()
    await update.message.reply_text(format_eval_report(report.to_dict()))
```

**Step 3: Write CLI subcommand test**

Create `tests/unit/test_main_cli.py`:

```python
from orchestrator.__main__ import parse_args


class TestParseArgs:
    def test_default_mode_is_run(self):
        args = parse_args([])
        assert args.command is None or args.command == "run"

    def test_eval_subcommand(self):
        args = parse_args(["eval"])
        assert args.command == "eval"

    def test_perf_subcommand(self):
        args = parse_args(["perf"])
        assert args.command == "perf"
```

**Step 4: Implement CLI subcommands in `__main__.py`**

Add `argparse` based subcommand parsing. The `main()` function should:
- `python -m orchestrator` → default (run bot + scheduler)
- `python -m orchestrator eval` → run eval
- `python -m orchestrator perf` → print performance report

**Step 5: Run tests**

Run: `uv run pytest tests/unit/test_telegram.py::TestEvalHandler tests/unit/test_main_cli.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/bot.py orchestrator/src/orchestrator/__main__.py orchestrator/tests/unit/test_telegram.py orchestrator/tests/unit/test_main_cli.py
git commit -m "feat: add /eval TG command and CLI subcommands (eval, perf)"
```

---

### Task 14: Wire eval components in __main__.py

**Files:**
- Modify: `orchestrator/src/orchestrator/__main__.py`
- Modify: `orchestrator/tests/unit/test_main.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_main.py`:

```python
class TestAppComponentsM3:
    def test_create_app_components_includes_m3(self):
        """create_app_components should include stats and eval components."""
        components = create_app_components(
            telegram_bot_token="test", telegram_admin_chat_ids=[123],
            exchange_id="binance", database_url="sqlite:///:memory:",
            anthropic_api_key="test-key",
        )
        assert "stats_calculator" in components
        assert "eval_runner" in components
```

**Step 2: Wire components**

In `create_app_components()`:
- Create `StatsCalculator` and `EvalRunner`
- Pass `StatsCalculator` to `PaperEngine`
- Pass `EvalRunner` to `SentinelBot` via `set_eval_runner()`
- Add both to returned dict

**Step 3: Run test**

Run: `uv run pytest tests/unit/test_main.py::TestAppComponentsM3 -v`
Expected: PASS

**Step 4: Run full tests**

Run: `uv run pytest -v --tb=short`
Expected: All pass

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/__main__.py orchestrator/tests/unit/test_main.py
git commit -m "feat: wire M3 eval and stats components into app entrypoint"
```

---

### Task 15: Improve bot.py test coverage to 80%+

**Files:**
- Modify: `orchestrator/tests/unit/test_telegram.py`

This task adds tests for bot handler methods that are currently untested (handlers for /start, /help, /status, /coin, /run, /history, /resume).

**Step 1: Check current coverage**

Run: `uv run pytest --cov=orchestrator.telegram --cov-report=term-missing tests/unit/test_telegram.py -v`

**Step 2: Add handler tests**

Add tests that mock `Update` and `ContextTypes.DEFAULT_TYPE` objects. Test each handler's logic branches:

- `_start_handler`: admin check → reply with welcome
- `_help_handler`: admin check → reply with help
- `_status_handler`: with in-memory results / with DB fallback / empty
- `_coin_handler`: no args / matching symbol / no match / DB fallback
- `_run_handler`: no scheduler / with symbols / with model alias
- `_history_handler`: no trade repo / with trades
- `_resume_handler`: no engine / engine paused

Pattern for mocking TG Update:
```python
def _make_update(chat_id: int = 123):
    update = MagicMock(spec=Update)
    update.effective_chat = MagicMock()
    update.effective_chat.id = chat_id
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()
    return update

def _make_context(args=None):
    ctx = MagicMock()
    ctx.args = args or []
    return ctx
```

**Step 3: Run and verify coverage >= 80%**

Run: `uv run pytest --cov=orchestrator.telegram --cov-report=term-missing tests/unit/test_telegram.py -v`
Expected: Coverage >= 80% for both `bot.py` and `formatters.py`

**Step 4: Commit**

```bash
git add orchestrator/tests/unit/test_telegram.py
git commit -m "test: improve bot.py handler test coverage to 80%+"
```

---

### Task 16: Improve scheduler.py test coverage to 80%+

**Files:**
- Modify: `orchestrator/tests/unit/test_scheduler.py`

**Step 1: Check current coverage**

Run: `uv run pytest --cov=orchestrator.pipeline.scheduler --cov-report=term-missing tests/unit/test_scheduler.py -v`

**Step 2: Add start/stop tests**

Test `start()` and `stop()` methods by mocking `AsyncIOScheduler`:

```python
from unittest.mock import MagicMock, patch

class TestSchedulerLifecycle:
    def test_start_creates_scheduler(self):
        runner = MagicMock()
        scheduler = PipelineScheduler(
            runner=runner, symbols=["BTC/USDT:USDT"], interval_minutes=15,
        )
        with patch("orchestrator.pipeline.scheduler.AsyncIOScheduler") as mock_cls:
            scheduler.start()
            mock_cls.return_value.start.assert_called_once()
            mock_cls.return_value.add_job.assert_called()

    def test_stop_shuts_down_scheduler(self):
        runner = MagicMock()
        scheduler = PipelineScheduler(
            runner=runner, symbols=["BTC/USDT:USDT"], interval_minutes=15,
        )
        with patch("orchestrator.pipeline.scheduler.AsyncIOScheduler") as mock_cls:
            scheduler.start()
            scheduler.stop()
            mock_cls.return_value.shutdown.assert_called_once_with(wait=False)

    def test_stop_without_start_is_safe(self):
        runner = MagicMock()
        scheduler = PipelineScheduler(
            runner=runner, symbols=["BTC/USDT:USDT"], interval_minutes=15,
        )
        scheduler.stop()  # Should not raise

    def test_start_with_premium_model_adds_daily_job(self):
        runner = MagicMock()
        scheduler = PipelineScheduler(
            runner=runner, symbols=["BTC/USDT:USDT"],
            interval_minutes=15, premium_model="anthropic/claude-opus-4-6",
        )
        with patch("orchestrator.pipeline.scheduler.AsyncIOScheduler") as mock_cls:
            scheduler.start()
            # Should have 2 jobs: interval + daily
            assert mock_cls.return_value.add_job.call_count == 2
```

**Step 3: Run and verify coverage >= 80%**

Run: `uv run pytest --cov=orchestrator.pipeline.scheduler --cov-report=term-missing tests/unit/test_scheduler.py -v`
Expected: Coverage >= 80%

**Step 4: Commit**

```bash
git add orchestrator/tests/unit/test_scheduler.py
git commit -m "test: improve scheduler.py test coverage to 80%+"
```

---

### Task 17: Final lint fix and full verification

**Files:**
- Any files with lint issues

**Step 1: Run ruff**

Run: `uv run ruff check src/ tests/ --fix`

**Step 2: Run full test suite with coverage**

Run: `uv run pytest -v --cov=orchestrator --cov-report=term-missing --tb=short`

Expected: All tests pass, overall coverage >= 80%.

**Step 3: Commit any fixes**

```bash
git add -A
git commit -m "chore: fix all ruff lint errors for M3 code"
```

---

## Review Checkpoint

After completing all tasks, run the full verification:

```bash
uv run pytest -v --cov=orchestrator --cov-report=term-missing --tb=short
uv run ruff check src/ tests/
```

Expected outcomes:
- All tests pass (115 existing + ~40 new ≈ 155+)
- Coverage >= 80% overall
- Lint clean
- New modules: `stats/calculator.py`, `eval/dataset.py`, `eval/scorers.py`, `eval/runner.py`, `eval/consistency.py`, `eval/datasets/golden_v1.yaml`
- Modified: `storage/models.py`, `storage/repository.py`, `exchange/paper_engine.py`, `telegram/bot.py`, `telegram/formatters.py`, `__main__.py`
