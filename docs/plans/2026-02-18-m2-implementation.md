# M2: Risk Management + Paper Trading Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add risk gate and paper trading engine after the M1 pipeline, completing the proposal → risk check → simulated execution → TG report loop.

**Architecture:** RiskChecker (pure code, no LLM) validates proposals against account-level constraints. PaperEngine maintains an in-memory ledger with DB persistence, executing approved proposals and checking SL/TP on each pipeline run. PositionSizer uses strategy pattern (Risk % mode for MVP). Pipeline runner integrates both new components between Aggregator and TG push.

**Tech Stack:** Python 3.12+, uv, Pydantic v2, SQLModel, structlog, pytest, pytest-asyncio

**Design doc:** `.claude/plans/2026-02-18-m2-risk-paper-design.md`

**Important:** All commands must run from `orchestrator/` directory. Use `export PATH="$HOME/.local/bin:$PATH"` before any `uv` command.

---

### Task 1: Add Paper Trading Config Fields

**Files:**
- Modify: `orchestrator/src/orchestrator/config.py:32-37`
- Test: `orchestrator/tests/unit/test_config.py`

**Step 1: Write the failing test**

Add to `orchestrator/tests/unit/test_config.py`:

```python
def test_settings_paper_trading_defaults(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")
    monkeypatch.setenv("TELEGRAM_ADMIN_CHAT_IDS", "[123]")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    settings = Settings()
    assert settings.paper_initial_equity == 10000.0
    assert settings.paper_taker_fee_rate == 0.0005
    assert settings.paper_maker_fee_rate == 0.0002
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_config.py::test_settings_paper_trading_defaults -v`
Expected: FAIL with `AttributeError`

**Step 3: Write minimal implementation**

Add to `orchestrator/src/orchestrator/config.py`, after the Risk section (line 37):

```python
    # Paper Trading
    paper_initial_equity: float = 10000.0
    paper_taker_fee_rate: float = 0.0005   # 0.05%
    paper_maker_fee_rate: float = 0.0002   # 0.02%
```

**Step 4: Run test to verify it passes**

Run: `cd orchestrator && uv run pytest tests/unit/test_config.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/config.py orchestrator/tests/unit/test_config.py
git commit -m "feat: add paper trading config fields"
```

---

### Task 2: Position Sizer (Strategy Pattern)

**Files:**
- Create: `orchestrator/src/orchestrator/risk/__init__.py` (ensure exists)
- Create: `orchestrator/src/orchestrator/risk/position_sizer.py`
- Create: `orchestrator/tests/unit/test_position_sizer.py`

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_position_sizer.py
import pytest

from orchestrator.risk.position_sizer import RiskPercentSizer


class TestRiskPercentSizer:
    def test_basic_calculation(self):
        sizer = RiskPercentSizer()
        # equity=$10000, risk=1.5%, entry=$95000, sl=$93000
        # risk_amount = 10000 * 0.015 = $150
        # distance = abs(95000 - 93000) = $2000
        # quantity = 150 / 2000 = 0.075
        qty = sizer.calculate(
            equity=10000.0,
            risk_pct=1.5,
            entry_price=95000.0,
            stop_loss=93000.0,
        )
        assert qty == pytest.approx(0.075)

    def test_short_position(self):
        sizer = RiskPercentSizer()
        # entry=$95000, sl=$97000 (short)
        qty = sizer.calculate(
            equity=10000.0,
            risk_pct=1.0,
            entry_price=95000.0,
            stop_loss=97000.0,
        )
        # risk_amount = 100, distance = 2000, qty = 0.05
        assert qty == pytest.approx(0.05)

    def test_zero_distance_raises(self):
        sizer = RiskPercentSizer()
        with pytest.raises(ValueError, match="stop_loss cannot equal entry_price"):
            sizer.calculate(
                equity=10000.0,
                risk_pct=1.0,
                entry_price=95000.0,
                stop_loss=95000.0,
            )

    def test_zero_risk_returns_zero(self):
        sizer = RiskPercentSizer()
        qty = sizer.calculate(
            equity=10000.0,
            risk_pct=0.0,
            entry_price=95000.0,
            stop_loss=93000.0,
        )
        assert qty == 0.0
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_position_sizer.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# orchestrator/src/orchestrator/risk/position_sizer.py
from __future__ import annotations

from abc import ABC, abstractmethod


class PositionSizer(ABC):
    @abstractmethod
    def calculate(
        self, *, equity: float, risk_pct: float, entry_price: float, stop_loss: float
    ) -> float:
        """Return quantity in base currency units."""


class RiskPercentSizer(PositionSizer):
    """quantity = (equity * risk_pct / 100) / abs(entry - stop_loss)"""

    def calculate(
        self, *, equity: float, risk_pct: float, entry_price: float, stop_loss: float
    ) -> float:
        if entry_price == stop_loss:
            raise ValueError("stop_loss cannot equal entry_price")
        if risk_pct == 0.0:
            return 0.0
        risk_amount = equity * (risk_pct / 100)
        price_distance = abs(entry_price - stop_loss)
        return risk_amount / price_distance
```

Ensure `orchestrator/src/orchestrator/risk/__init__.py` exists (it should from M0).

**Step 4: Run test to verify it passes**

Run: `cd orchestrator && uv run pytest tests/unit/test_position_sizer.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/risk/position_sizer.py orchestrator/tests/unit/test_position_sizer.py
git commit -m "feat: add position sizer with Risk% strategy pattern"
```

---

### Task 3: Risk Checker

**Files:**
- Create: `orchestrator/src/orchestrator/risk/checker.py`
- Create: `orchestrator/tests/unit/test_risk_checker.py`

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_risk_checker.py
import pytest

from orchestrator.models import EntryOrder, Side, TradeProposal
from orchestrator.risk.checker import RiskChecker, RiskResult


def _make_proposal(
    *,
    side: Side = Side.LONG,
    risk_pct: float = 1.0,
    stop_loss: float = 93000.0,
    symbol: str = "BTC/USDT:USDT",
    invalid_if: list[str] | None = None,
) -> TradeProposal:
    return TradeProposal(
        symbol=symbol,
        side=side,
        entry=EntryOrder(type="market"),
        position_size_risk_pct=risk_pct,
        stop_loss=stop_loss,
        take_profit=[97000.0],
        time_horizon="4h",
        confidence=0.7,
        invalid_if=invalid_if or [],
        rationale="test",
    )


class TestRiskChecker:
    def test_approved_proposal(self):
        checker = RiskChecker(
            max_single_risk_pct=2.0,
            max_total_exposure_pct=20.0,
            max_consecutive_losses=5,
            max_daily_loss_pct=5.0,
        )
        result = checker.check(
            proposal=_make_proposal(risk_pct=1.0),
            open_positions_risk_pct=5.0,
            consecutive_losses=0,
            daily_loss_pct=0.0,
        )
        assert result.approved is True

    def test_reject_single_risk_too_high(self):
        checker = RiskChecker(
            max_single_risk_pct=2.0,
            max_total_exposure_pct=20.0,
            max_consecutive_losses=5,
            max_daily_loss_pct=5.0,
        )
        result = checker.check(
            proposal=_make_proposal(risk_pct=3.0),
            open_positions_risk_pct=0.0,
            consecutive_losses=0,
            daily_loss_pct=0.0,
        )
        assert result.approved is False
        assert result.rule_violated == "max_single_risk"
        assert result.action == "reject"

    def test_reject_total_exposure_exceeded(self):
        checker = RiskChecker(
            max_single_risk_pct=2.0,
            max_total_exposure_pct=20.0,
            max_consecutive_losses=5,
            max_daily_loss_pct=5.0,
        )
        result = checker.check(
            proposal=_make_proposal(risk_pct=1.5),
            open_positions_risk_pct=19.0,
            consecutive_losses=0,
            daily_loss_pct=0.0,
        )
        assert result.approved is False
        assert result.rule_violated == "max_total_exposure"
        assert result.action == "reject"

    def test_pause_consecutive_losses(self):
        checker = RiskChecker(
            max_single_risk_pct=2.0,
            max_total_exposure_pct=20.0,
            max_consecutive_losses=5,
            max_daily_loss_pct=5.0,
        )
        result = checker.check(
            proposal=_make_proposal(risk_pct=1.0),
            open_positions_risk_pct=0.0,
            consecutive_losses=5,
            daily_loss_pct=0.0,
        )
        assert result.approved is False
        assert result.rule_violated == "max_consecutive_losses"
        assert result.action == "pause"

    def test_pause_daily_loss(self):
        checker = RiskChecker(
            max_single_risk_pct=2.0,
            max_total_exposure_pct=20.0,
            max_consecutive_losses=5,
            max_daily_loss_pct=5.0,
        )
        result = checker.check(
            proposal=_make_proposal(risk_pct=1.0),
            open_positions_risk_pct=0.0,
            consecutive_losses=0,
            daily_loss_pct=6.0,
        )
        assert result.approved is False
        assert result.rule_violated == "max_daily_loss"
        assert result.action == "pause"

    def test_flat_proposal_always_approved(self):
        checker = RiskChecker(
            max_single_risk_pct=2.0,
            max_total_exposure_pct=20.0,
            max_consecutive_losses=5,
            max_daily_loss_pct=5.0,
        )
        result = checker.check(
            proposal=_make_proposal(side=Side.FLAT, risk_pct=0.0, stop_loss=None),
            open_positions_risk_pct=100.0,
            consecutive_losses=100,
            daily_loss_pct=100.0,
        )
        assert result.approved is True
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_risk_checker.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# orchestrator/src/orchestrator/risk/checker.py
from __future__ import annotations

import structlog
from pydantic import BaseModel

from orchestrator.models import Side, TradeProposal

logger = structlog.get_logger(__name__)


class RiskResult(BaseModel, frozen=True):
    approved: bool
    rule_violated: str = ""
    reason: str = ""
    action: str = "reject"  # "reject" | "pause"


class RiskChecker:
    def __init__(
        self,
        *,
        max_single_risk_pct: float,
        max_total_exposure_pct: float,
        max_consecutive_losses: int,
        max_daily_loss_pct: float,
    ) -> None:
        self._max_single_risk_pct = max_single_risk_pct
        self._max_total_exposure_pct = max_total_exposure_pct
        self._max_consecutive_losses = max_consecutive_losses
        self._max_daily_loss_pct = max_daily_loss_pct

    def check(
        self,
        *,
        proposal: TradeProposal,
        open_positions_risk_pct: float,
        consecutive_losses: int,
        daily_loss_pct: float,
    ) -> RiskResult:
        if proposal.side == Side.FLAT:
            return RiskResult(approved=True)

        # Rule 1: Max single risk
        if proposal.position_size_risk_pct > self._max_single_risk_pct:
            reason = (
                f"Single risk {proposal.position_size_risk_pct}% "
                f"exceeds {self._max_single_risk_pct}% limit"
            )
            logger.warning("risk_rejected", rule="max_single_risk", reason=reason)
            return RiskResult(
                approved=False,
                rule_violated="max_single_risk",
                reason=reason,
                action="reject",
            )

        # Rule 2: Max total exposure
        total = open_positions_risk_pct + proposal.position_size_risk_pct
        if total > self._max_total_exposure_pct:
            reason = (
                f"Total exposure {total}% "
                f"exceeds {self._max_total_exposure_pct}% limit"
            )
            logger.warning("risk_rejected", rule="max_total_exposure", reason=reason)
            return RiskResult(
                approved=False,
                rule_violated="max_total_exposure",
                reason=reason,
                action="reject",
            )

        # Rule 3: Max consecutive losses
        if consecutive_losses >= self._max_consecutive_losses:
            reason = (
                f"{consecutive_losses} consecutive losses "
                f"reached {self._max_consecutive_losses} limit"
            )
            logger.warning("risk_paused", rule="max_consecutive_losses", reason=reason)
            return RiskResult(
                approved=False,
                rule_violated="max_consecutive_losses",
                reason=reason,
                action="pause",
            )

        # Rule 4: Max daily loss
        if daily_loss_pct > self._max_daily_loss_pct:
            reason = (
                f"Daily loss {daily_loss_pct}% "
                f"exceeds {self._max_daily_loss_pct}% limit"
            )
            logger.warning("risk_paused", rule="max_daily_loss", reason=reason)
            return RiskResult(
                approved=False,
                rule_violated="max_daily_loss",
                reason=reason,
                action="pause",
            )

        return RiskResult(approved=True)
```

**Step 4: Run test to verify it passes**

Run: `cd orchestrator && uv run pytest tests/unit/test_risk_checker.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/risk/checker.py orchestrator/tests/unit/test_risk_checker.py
git commit -m "feat: add risk checker with account-level rule engine"
```

---

### Task 4: Paper Trade & Account Snapshot Repositories

**Files:**
- Modify: `orchestrator/src/orchestrator/storage/repository.py`
- Modify: `orchestrator/tests/unit/test_storage.py`

**Step 1: Write the failing tests**

Add to `orchestrator/tests/unit/test_storage.py`:

```python
from orchestrator.storage.models import AccountSnapshotRecord, PaperTradeRecord
from orchestrator.storage.repository import AccountSnapshotRepository, PaperTradeRepository


class TestPaperTradeRepository:
    def test_save_and_get_open(self, session):
        repo = PaperTradeRepository(session)
        repo.save_trade(
            trade_id="t-001",
            proposal_id="p-001",
            symbol="BTC/USDT:USDT",
            side="long",
            entry_price=95000.0,
            quantity=0.075,
            risk_pct=1.5,
        )
        open_positions = repo.get_open_positions()
        assert len(open_positions) == 1
        assert open_positions[0].trade_id == "t-001"
        assert open_positions[0].status == "open"

    def test_close_trade(self, session):
        repo = PaperTradeRepository(session)
        repo.save_trade(
            trade_id="t-002",
            proposal_id="p-002",
            symbol="BTC/USDT:USDT",
            side="long",
            entry_price=95000.0,
            quantity=0.075,
            risk_pct=1.5,
        )
        repo.update_trade_closed(
            trade_id="t-002",
            exit_price=93000.0,
            pnl=-150.0,
            fees=7.13,
        )
        trade = repo.get_by_trade_id("t-002")
        assert trade.status == "closed"
        assert trade.exit_price == 93000.0
        assert trade.pnl == -150.0

    def test_count_consecutive_losses(self, session):
        repo = PaperTradeRepository(session)
        # 3 losses in a row
        for i in range(3):
            repo.save_trade(
                trade_id=f"t-loss-{i}",
                proposal_id=f"p-{i}",
                symbol="BTC/USDT:USDT",
                side="long",
                entry_price=95000.0,
                quantity=0.075,
                risk_pct=1.0,
            )
            repo.update_trade_closed(
                trade_id=f"t-loss-{i}",
                exit_price=93000.0,
                pnl=-150.0,
                fees=7.0,
            )
        assert repo.count_consecutive_losses() == 3

    def test_consecutive_losses_reset_on_win(self, session):
        repo = PaperTradeRepository(session)
        # 1 loss then 1 win
        repo.save_trade(
            trade_id="t-l1", proposal_id="p-1", symbol="BTC/USDT:USDT",
            side="long", entry_price=95000.0, quantity=0.075, risk_pct=1.0,
        )
        repo.update_trade_closed(trade_id="t-l1", exit_price=93000.0, pnl=-150.0, fees=7.0)
        repo.save_trade(
            trade_id="t-w1", proposal_id="p-2", symbol="BTC/USDT:USDT",
            side="long", entry_price=93000.0, quantity=0.08, risk_pct=1.0,
        )
        repo.update_trade_closed(trade_id="t-w1", exit_price=95000.0, pnl=160.0, fees=7.0)
        assert repo.count_consecutive_losses() == 0

    def test_get_daily_pnl(self, session):
        from datetime import UTC, datetime
        repo = PaperTradeRepository(session)
        repo.save_trade(
            trade_id="t-d1", proposal_id="p-1", symbol="BTC/USDT:USDT",
            side="long", entry_price=95000.0, quantity=0.075, risk_pct=1.0,
        )
        repo.update_trade_closed(trade_id="t-d1", exit_price=93000.0, pnl=-150.0, fees=7.0)
        today = datetime.now(UTC).date()
        assert repo.get_daily_pnl(today) == pytest.approx(-150.0)

    def test_get_recent_closed(self, session):
        repo = PaperTradeRepository(session)
        repo.save_trade(
            trade_id="t-r1", proposal_id="p-1", symbol="BTC/USDT:USDT",
            side="long", entry_price=95000.0, quantity=0.075, risk_pct=1.0,
        )
        repo.update_trade_closed(trade_id="t-r1", exit_price=97000.0, pnl=150.0, fees=7.0)
        recent = repo.get_recent_closed(limit=5)
        assert len(recent) == 1
        assert recent[0].trade_id == "t-r1"


class TestAccountSnapshotRepository:
    def test_save_and_get_latest(self, session):
        repo = AccountSnapshotRepository(session)
        repo.save_snapshot(equity=10000.0, open_count=2, daily_pnl=-50.0)
        latest = repo.get_latest()
        assert latest is not None
        assert latest.equity == 10000.0
        assert latest.open_positions_count == 2
```

**Step 2: Run tests to verify they fail**

Run: `cd orchestrator && uv run pytest tests/unit/test_storage.py::TestPaperTradeRepository -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Add to `orchestrator/src/orchestrator/storage/repository.py`:

Add imports at the top:

```python
from datetime import date

from orchestrator.storage.models import AccountSnapshotRecord, PaperTradeRecord
```

Add the `PaperTradeRecord` model needs a `risk_pct` field. Modify `orchestrator/src/orchestrator/storage/models.py` — add `risk_pct: float = 0.0` after the `fees` field (line 58):

```python
    fees: float = 0.0
    risk_pct: float = 0.0
```

Then add the repository classes to `orchestrator/src/orchestrator/storage/repository.py`:

```python
class PaperTradeRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def save_trade(
        self,
        *,
        trade_id: str,
        proposal_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        risk_pct: float = 0.0,
    ) -> PaperTradeRecord:
        record = PaperTradeRecord(
            trade_id=trade_id,
            proposal_id=proposal_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            risk_pct=risk_pct,
        )
        self._session.add(record)
        self._session.commit()
        self._session.refresh(record)
        return record

    def get_by_trade_id(self, trade_id: str) -> PaperTradeRecord | None:
        statement = select(PaperTradeRecord).where(PaperTradeRecord.trade_id == trade_id)
        return self._session.exec(statement).first()

    def get_open_positions(self) -> list[PaperTradeRecord]:
        statement = select(PaperTradeRecord).where(PaperTradeRecord.status == "open")
        return list(self._session.exec(statement).all())

    def update_trade_closed(
        self,
        trade_id: str,
        *,
        exit_price: float,
        pnl: float,
        fees: float,
    ) -> PaperTradeRecord:
        from datetime import UTC, datetime

        trade = self.get_by_trade_id(trade_id)
        if trade is None:
            raise ValueError(f"Trade {trade_id} not found")
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.fees = fees
        trade.status = "closed"
        trade.closed_at = datetime.now(UTC)
        self._session.add(trade)
        self._session.commit()
        self._session.refresh(trade)
        return trade

    def get_recent_closed(self, *, limit: int = 10) -> list[PaperTradeRecord]:
        statement = (
            select(PaperTradeRecord)
            .where(PaperTradeRecord.status == "closed")
            .order_by(PaperTradeRecord.closed_at.desc())
            .limit(limit)
        )
        return list(self._session.exec(statement).all())

    def count_consecutive_losses(self) -> int:
        """Count consecutive losses from most recent closed trade backwards."""
        statement = (
            select(PaperTradeRecord)
            .where(PaperTradeRecord.status == "closed")
            .order_by(PaperTradeRecord.closed_at.desc())
        )
        trades = list(self._session.exec(statement).all())
        count = 0
        for trade in trades:
            if trade.pnl < 0:
                count += 1
            else:
                break
        return count

    def get_daily_pnl(self, day: date) -> float:
        """Sum PnL for all closed trades on a given date."""
        statement = (
            select(PaperTradeRecord)
            .where(PaperTradeRecord.status == "closed")
        )
        trades = list(self._session.exec(statement).all())
        return sum(t.pnl for t in trades if t.closed_at and t.closed_at.date() == day)


class AccountSnapshotRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def save_snapshot(
        self,
        *,
        equity: float,
        open_count: int,
        daily_pnl: float,
    ) -> AccountSnapshotRecord:
        record = AccountSnapshotRecord(
            equity=equity,
            open_positions_count=open_count,
            daily_pnl=daily_pnl,
        )
        self._session.add(record)
        self._session.commit()
        self._session.refresh(record)
        return record

    def get_latest(self) -> AccountSnapshotRecord | None:
        statement = (
            select(AccountSnapshotRecord)
            .order_by(AccountSnapshotRecord.created_at.desc())
        )
        return self._session.exec(statement).first()
```

**Step 4: Run tests to verify they pass**

Run: `cd orchestrator && uv run pytest tests/unit/test_storage.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/storage/models.py orchestrator/src/orchestrator/storage/repository.py orchestrator/tests/unit/test_storage.py
git commit -m "feat: add paper trade and account snapshot repositories"
```

---

### Task 5: Paper Trading Engine

**Files:**
- Create: `orchestrator/src/orchestrator/exchange/paper_engine.py`
- Create: `orchestrator/tests/unit/test_paper_engine.py`

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_paper_engine.py
import pytest
from unittest.mock import MagicMock

from orchestrator.exchange.paper_engine import PaperEngine, Position
from orchestrator.models import EntryOrder, Side, TradeProposal
from orchestrator.risk.position_sizer import RiskPercentSizer


def _make_proposal(
    *,
    side: Side = Side.LONG,
    risk_pct: float = 1.5,
    stop_loss: float = 93000.0,
    take_profit: list[float] | None = None,
) -> TradeProposal:
    return TradeProposal(
        symbol="BTC/USDT:USDT",
        side=side,
        entry=EntryOrder(type="market"),
        position_size_risk_pct=risk_pct,
        stop_loss=stop_loss,
        take_profit=take_profit or [97000.0],
        time_horizon="4h",
        confidence=0.7,
        invalid_if=[],
        rationale="test",
    )


class TestPaperEngine:
    def _make_engine(self, *, trade_repo=None, snapshot_repo=None):
        return PaperEngine(
            initial_equity=10000.0,
            taker_fee_rate=0.0005,
            position_sizer=RiskPercentSizer(),
            trade_repo=trade_repo or MagicMock(),
            snapshot_repo=snapshot_repo or MagicMock(),
        )

    def test_open_position(self):
        engine = self._make_engine()
        proposal = _make_proposal()
        position = engine.open_position(proposal, current_price=95000.0)
        assert position.side == Side.LONG
        assert position.entry_price == 95000.0
        assert position.quantity == pytest.approx(0.075)
        assert len(engine.get_open_positions()) == 1

    def test_open_multiple_same_symbol(self):
        engine = self._make_engine()
        engine.open_position(_make_proposal(), current_price=95000.0)
        engine.open_position(_make_proposal(risk_pct=1.0), current_price=94000.0)
        assert len(engine.get_open_positions()) == 2

    def test_check_sl_triggers_close(self):
        engine = self._make_engine()
        engine.open_position(_make_proposal(stop_loss=93000.0), current_price=95000.0)
        # Price drops below SL
        closed = engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=92500.0)
        assert len(closed) == 1
        assert closed[0].pnl < 0
        assert len(engine.get_open_positions()) == 0

    def test_check_tp_triggers_close(self):
        engine = self._make_engine()
        engine.open_position(
            _make_proposal(take_profit=[97000.0]),
            current_price=95000.0,
        )
        # Price rises above TP
        closed = engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=97500.0)
        assert len(closed) == 1
        assert closed[0].pnl > 0

    def test_check_sl_short_position(self):
        engine = self._make_engine()
        engine.open_position(
            _make_proposal(side=Side.SHORT, stop_loss=97000.0, take_profit=[93000.0]),
            current_price=95000.0,
        )
        # Price rises above SL for short
        closed = engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=97500.0)
        assert len(closed) == 1
        assert closed[0].pnl < 0

    def test_equity_after_loss(self):
        engine = self._make_engine()
        assert engine.equity == 10000.0
        engine.open_position(_make_proposal(stop_loss=93000.0), current_price=95000.0)
        engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=92500.0)
        # equity should decrease by PnL + fees
        assert engine.equity < 10000.0

    def test_open_positions_risk_pct(self):
        engine = self._make_engine()
        engine.open_position(_make_proposal(risk_pct=1.5), current_price=95000.0)
        engine.open_position(_make_proposal(risk_pct=2.0), current_price=95000.0)
        assert engine.open_positions_risk_pct == pytest.approx(3.5)

    def test_no_trigger_when_price_between_sl_tp(self):
        engine = self._make_engine()
        engine.open_position(
            _make_proposal(stop_loss=93000.0, take_profit=[97000.0]),
            current_price=95000.0,
        )
        closed = engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=95500.0)
        assert len(closed) == 0
        assert len(engine.get_open_positions()) == 1
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_paper_engine.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# orchestrator/src/orchestrator/exchange/paper_engine.py
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel

from orchestrator.models import Side, TradeProposal
from orchestrator.risk.position_sizer import PositionSizer

if TYPE_CHECKING:
    from orchestrator.storage.repository import AccountSnapshotRepository, PaperTradeRepository

logger = structlog.get_logger(__name__)


class Position(BaseModel, frozen=True):
    trade_id: str
    proposal_id: str
    symbol: str
    side: Side
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: list[float]
    opened_at: datetime
    risk_pct: float


class CloseResult(BaseModel, frozen=True):
    trade_id: str
    symbol: str
    side: Side
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    fees: float
    reason: str  # "sl" | "tp"


class PaperEngine:
    def __init__(
        self,
        *,
        initial_equity: float,
        taker_fee_rate: float,
        position_sizer: PositionSizer,
        trade_repo: PaperTradeRepository,
        snapshot_repo: AccountSnapshotRepository,
    ) -> None:
        self._initial_equity = initial_equity
        self._taker_fee_rate = taker_fee_rate
        self._position_sizer = position_sizer
        self._trade_repo = trade_repo
        self._snapshot_repo = snapshot_repo
        self._positions: list[Position] = []
        self._closed_pnl: float = 0.0
        self._total_fees: float = 0.0
        self._paused: bool = False

    @property
    def equity(self) -> float:
        return self._initial_equity + self._closed_pnl - self._total_fees

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def open_positions_risk_pct(self) -> float:
        return sum(p.risk_pct for p in self._positions)

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        logger.info("engine_pause_state", paused=paused)

    def get_open_positions(self) -> list[Position]:
        return list(self._positions)

    def open_position(self, proposal: TradeProposal, current_price: float) -> Position:
        quantity = self._position_sizer.calculate(
            equity=self.equity,
            risk_pct=proposal.position_size_risk_pct,
            entry_price=current_price,
            stop_loss=proposal.stop_loss,
        )

        open_fee = quantity * current_price * self._taker_fee_rate
        self._total_fees += open_fee

        position = Position(
            trade_id=str(uuid.uuid4()),
            proposal_id=proposal.proposal_id,
            symbol=proposal.symbol,
            side=proposal.side,
            entry_price=current_price,
            quantity=quantity,
            stop_loss=proposal.stop_loss,
            take_profit=proposal.take_profit,
            opened_at=datetime.now(UTC),
            risk_pct=proposal.position_size_risk_pct,
        )
        self._positions.append(position)

        self._trade_repo.save_trade(
            trade_id=position.trade_id,
            proposal_id=position.proposal_id,
            symbol=position.symbol,
            side=position.side.value,
            entry_price=position.entry_price,
            quantity=position.quantity,
            risk_pct=position.risk_pct,
        )

        logger.info(
            "position_opened",
            trade_id=position.trade_id,
            symbol=position.symbol,
            side=position.side,
            quantity=position.quantity,
            entry_price=current_price,
            fee=open_fee,
        )

        return position

    def check_sl_tp(self, *, symbol: str, current_price: float) -> list[CloseResult]:
        closed: list[CloseResult] = []
        remaining: list[Position] = []

        for pos in self._positions:
            if pos.symbol != symbol:
                remaining.append(pos)
                continue

            trigger = self._check_trigger(pos, current_price)
            if trigger is not None:
                exit_price, reason = trigger
                result = self._close(pos, exit_price=exit_price, reason=reason)
                closed.append(result)
            else:
                remaining.append(pos)

        self._positions = remaining
        return closed

    def rebuild_from_db(self) -> None:
        open_trades = self._trade_repo.get_open_positions()
        self._positions = [
            Position(
                trade_id=t.trade_id,
                proposal_id=t.proposal_id,
                symbol=t.symbol,
                side=Side(t.side),
                entry_price=t.entry_price,
                quantity=t.quantity,
                stop_loss=0.0,  # not stored in DB, positions will rely on next check
                take_profit=[],
                opened_at=t.opened_at,
                risk_pct=t.risk_pct,
            )
            for t in open_trades
        ]
        # Rebuild closed PnL and fees
        closed_trades = self._trade_repo.get_recent_closed(limit=1000)
        self._closed_pnl = sum(t.pnl for t in closed_trades)
        self._total_fees = sum(t.fees for t in closed_trades)
        logger.info(
            "engine_rebuilt",
            open_positions=len(self._positions),
            closed_pnl=self._closed_pnl,
            total_fees=self._total_fees,
        )

    def _check_trigger(
        self, pos: Position, current_price: float
    ) -> tuple[float, str] | None:
        if pos.side == Side.LONG:
            if current_price <= pos.stop_loss:
                return pos.stop_loss, "sl"
            if pos.take_profit and current_price >= pos.take_profit[0]:
                return pos.take_profit[0], "tp"
        elif pos.side == Side.SHORT:
            if current_price >= pos.stop_loss:
                return pos.stop_loss, "sl"
            if pos.take_profit and current_price <= pos.take_profit[0]:
                return pos.take_profit[0], "tp"
        return None

    def _close(self, pos: Position, *, exit_price: float, reason: str) -> CloseResult:
        close_fee = pos.quantity * exit_price * self._taker_fee_rate
        self._total_fees += close_fee

        if pos.side == Side.LONG:
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        self._closed_pnl += pnl

        self._trade_repo.update_trade_closed(
            trade_id=pos.trade_id,
            exit_price=exit_price,
            pnl=pnl,
            fees=close_fee,
        )

        logger.info(
            "position_closed",
            trade_id=pos.trade_id,
            symbol=pos.symbol,
            reason=reason,
            pnl=pnl,
            fee=close_fee,
        )

        return CloseResult(
            trade_id=pos.trade_id,
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            pnl=pnl,
            fees=close_fee,
            reason=reason,
        )
```

**Step 4: Run test to verify it passes**

Run: `cd orchestrator && uv run pytest tests/unit/test_paper_engine.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/exchange/paper_engine.py orchestrator/tests/unit/test_paper_engine.py
git commit -m "feat: add paper trading engine with in-memory ledger"
```

---

### Task 6: Telegram Formatters (trade report + risk rejection)

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/formatters.py`
- Modify: `orchestrator/tests/unit/test_telegram.py`

**Step 1: Write the failing tests**

Add to `orchestrator/tests/unit/test_telegram.py`:

```python
from orchestrator.exchange.paper_engine import CloseResult
from orchestrator.models import Side
from orchestrator.risk.checker import RiskResult
from orchestrator.telegram.formatters import format_risk_rejection, format_trade_report


class TestFormatTradeReport:
    def test_format_long_close_sl(self):
        result = CloseResult(
            trade_id="t-001",
            symbol="BTC/USDT:USDT",
            side=Side.LONG,
            entry_price=95000.0,
            exit_price=93000.0,
            quantity=0.075,
            pnl=-150.0,
            fees=7.13,
            reason="sl",
        )
        text = format_trade_report(result)
        assert "[CLOSED]" in text
        assert "BTC/USDT:USDT" in text
        assert "LONG" in text
        assert "93,000.0" in text
        assert "-$150.00" in text
        assert "SL" in text

    def test_format_short_close_tp(self):
        result = CloseResult(
            trade_id="t-002",
            symbol="ETH/USDT:USDT",
            side=Side.SHORT,
            entry_price=3000.0,
            exit_price=2800.0,
            quantity=1.0,
            pnl=200.0,
            fees=2.90,
            reason="tp",
        )
        text = format_trade_report(result)
        assert "SHORT" in text
        assert "TP" in text
        assert "$200.00" in text


class TestFormatRiskRejection:
    def test_format_rejection(self):
        risk_result = RiskResult(
            approved=False,
            rule_violated="max_total_exposure",
            reason="Total exposure 22% exceeds 20% limit",
            action="reject",
        )
        text = format_risk_rejection(
            symbol="BTC/USDT:USDT",
            side="LONG",
            entry_price=95000.0,
            risk_result=risk_result,
        )
        assert "[RISK REJECTED]" in text
        assert "BTC/USDT:USDT" in text
        assert "max_total_exposure" in text

    def test_format_pause(self):
        risk_result = RiskResult(
            approved=False,
            rule_violated="max_consecutive_losses",
            reason="5 consecutive losses reached 5 limit",
            action="pause",
        )
        text = format_risk_rejection(
            symbol="BTC/USDT:USDT",
            side="LONG",
            entry_price=95000.0,
            risk_result=risk_result,
        )
        assert "[RISK PAUSED]" in text
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_telegram.py::TestFormatTradeReport -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Add to `orchestrator/src/orchestrator/telegram/formatters.py`:

```python
from orchestrator.exchange.paper_engine import CloseResult
from orchestrator.risk.checker import RiskResult


def format_trade_report(result: CloseResult) -> str:
    reason_label = {"sl": "SL", "tp": "TP"}.get(result.reason, result.reason.upper())
    pnl_str = f"${result.pnl:,.2f}" if result.pnl >= 0 else f"-${abs(result.pnl):,.2f}"

    lines = [
        f"[CLOSED] {result.symbol}",
        f"Side: {result.side.value.upper()}",
        f"Entry: {result.entry_price:,.1f} → Exit: {result.exit_price:,.1f} ({reason_label})",
        f"Quantity: {result.quantity:.4f}",
        f"PnL: {pnl_str} (fees: ${result.fees:,.2f})",
    ]
    return "\n".join(lines)


def format_risk_rejection(
    *, symbol: str, side: str, entry_price: float, risk_result: RiskResult
) -> str:
    label = "RISK PAUSED" if risk_result.action == "pause" else "RISK REJECTED"
    lines = [
        f"[{label}] {symbol}",
        f"Proposed: {side} @ {entry_price:,.1f}",
        f"Rule: {risk_result.rule_violated}",
        f"Reason: {risk_result.reason}",
    ]
    return "\n".join(lines)
```

Note: Move the `CloseResult` and `RiskResult` imports from `TYPE_CHECKING` to regular imports since formatters need them at runtime. Keep the existing `PipelineResult` import under `TYPE_CHECKING`.

**Step 4: Run tests to verify they pass**

Run: `cd orchestrator && uv run pytest tests/unit/test_telegram.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/formatters.py orchestrator/tests/unit/test_telegram.py
git commit -m "feat: add trade report and risk rejection formatters"
```

---

### Task 7: Integrate Risk + Paper Engine into Pipeline Runner

**Files:**
- Modify: `orchestrator/src/orchestrator/pipeline/runner.py`
- Modify: `orchestrator/tests/unit/test_runner.py`

**Step 1: Write the failing tests**

Add to `orchestrator/tests/unit/test_runner.py`:

```python
class TestPipelineRunnerWithRisk:
    """Tests for M2 risk + paper engine integration."""

    @pytest.mark.asyncio
    async def test_approved_proposal_opens_position(self):
        """When risk check passes, paper engine opens a position."""
        # Setup mocks for all dependencies
        data_fetcher = AsyncMock()
        data_fetcher.fetch_snapshot.return_value = MagicMock(
            symbol="BTC/USDT:USDT",
            current_price=95000.0,
        )

        sentiment_agent = AsyncMock()
        sentiment_agent.analyze.return_value = MagicMock(
            output=MagicMock(), degraded=False, llm_calls=[],
        )
        market_agent = AsyncMock()
        market_agent.analyze.return_value = MagicMock(
            output=MagicMock(), degraded=False, llm_calls=[],
        )
        proposer_agent = AsyncMock()
        proposer_agent.analyze.return_value = MagicMock(
            output=_make_proposal(side=Side.LONG, risk_pct=1.0),
            degraded=False, llm_calls=[],
        )

        risk_checker = MagicMock()
        risk_checker.check.return_value = RiskResult(approved=True)

        paper_engine = MagicMock()
        paper_engine.check_sl_tp.return_value = []
        paper_engine.open_positions_risk_pct = 0.0
        paper_engine.paused = False
        paper_engine.open_position.return_value = MagicMock(trade_id="t-001")

        runner = PipelineRunner(
            data_fetcher=data_fetcher,
            sentiment_agent=sentiment_agent,
            market_agent=market_agent,
            proposer_agent=proposer_agent,
            pipeline_repo=MagicMock(),
            llm_call_repo=MagicMock(),
            proposal_repo=MagicMock(),
            risk_checker=risk_checker,
            paper_engine=paper_engine,
        )

        result = await runner.execute("BTC/USDT:USDT")
        assert result.status == "completed"
        paper_engine.open_position.assert_called_once()

    @pytest.mark.asyncio
    async def test_risk_rejected_does_not_open_position(self):
        data_fetcher = AsyncMock()
        data_fetcher.fetch_snapshot.return_value = MagicMock(
            symbol="BTC/USDT:USDT",
            current_price=95000.0,
        )

        sentiment_agent = AsyncMock()
        sentiment_agent.analyze.return_value = MagicMock(
            output=MagicMock(), degraded=False, llm_calls=[],
        )
        market_agent = AsyncMock()
        market_agent.analyze.return_value = MagicMock(
            output=MagicMock(), degraded=False, llm_calls=[],
        )
        proposer_agent = AsyncMock()
        proposer_agent.analyze.return_value = MagicMock(
            output=_make_proposal(side=Side.LONG, risk_pct=3.0),
            degraded=False, llm_calls=[],
        )

        risk_checker = MagicMock()
        risk_checker.check.return_value = RiskResult(
            approved=False, rule_violated="max_single_risk",
            reason="too high", action="reject",
        )

        paper_engine = MagicMock()
        paper_engine.check_sl_tp.return_value = []
        paper_engine.open_positions_risk_pct = 0.0
        paper_engine.paused = False

        runner = PipelineRunner(
            data_fetcher=data_fetcher,
            sentiment_agent=sentiment_agent,
            market_agent=market_agent,
            proposer_agent=proposer_agent,
            pipeline_repo=MagicMock(),
            llm_call_repo=MagicMock(),
            proposal_repo=MagicMock(),
            risk_checker=risk_checker,
            paper_engine=paper_engine,
        )

        result = await runner.execute("BTC/USDT:USDT")
        assert result.status == "risk_rejected"
        paper_engine.open_position.assert_not_called()
```

You'll need these imports at the top of the test file:

```python
from orchestrator.models import EntryOrder, Side, TradeProposal
from orchestrator.risk.checker import RiskResult
```

And a helper:

```python
def _make_proposal(*, side=Side.LONG, risk_pct=1.0):
    return TradeProposal(
        symbol="BTC/USDT:USDT", side=side, entry=EntryOrder(type="market"),
        position_size_risk_pct=risk_pct, stop_loss=93000.0,
        take_profit=[97000.0], time_horizon="4h", confidence=0.7,
        invalid_if=[], rationale="test",
    )
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_runner.py::TestPipelineRunnerWithRisk -v`
Expected: FAIL (PipelineRunner doesn't accept risk_checker/paper_engine yet)

**Step 3: Modify PipelineRunner**

Update `orchestrator/src/orchestrator/pipeline/runner.py`:

1. Add `risk_checker` and `paper_engine` to `__init__` (optional params with `None` default for backward compat)
2. Add `PipelineResult` fields: `risk_result`, `close_results`
3. In `execute()`:
   - After fetch_snapshot: call `paper_engine.check_sl_tp()` for the symbol
   - After aggregator validates: call `risk_checker.check()`
   - If risk approved + not FLAT: call `paper_engine.open_position()`
   - If risk rejected: save with `risk_check_result="risk_rejected"`

Key changes to `PipelineResult`:

```python
class PipelineResult(BaseModel, frozen=True):
    run_id: str
    symbol: str
    status: str  # completed, rejected, risk_rejected, risk_paused, failed
    model_used: str = ""
    proposal: TradeProposal | None = None
    rejection_reason: str = ""
    sentiment_degraded: bool = False
    market_degraded: bool = False
    proposer_degraded: bool = False
    risk_result: RiskResult | None = None
    close_results: list[CloseResult] = []
```

Key changes to `execute()` flow — insert between aggregation and final return:

```python
            # --- Step ②: Check SL/TP on existing positions ---
            close_results = []
            if self._paper_engine is not None:
                close_results = self._paper_engine.check_sl_tp(
                    symbol=symbol, current_price=snapshot.current_price
                )
                for cr in close_results:
                    log.info("position_closed_sltp", trade_id=cr.trade_id, reason=cr.reason)

            # ... (existing LLM + aggregator code) ...

            # --- Step ⑥: Risk check ---
            risk_result: RiskResult | None = None
            if aggregation.valid and self._risk_checker is not None and self._paper_engine is not None:
                from datetime import UTC, datetime
                risk_result = self._risk_checker.check(
                    proposal=aggregation.proposal,
                    open_positions_risk_pct=self._paper_engine.open_positions_risk_pct,
                    consecutive_losses=...,  # from paper_trade_repo
                    daily_loss_pct=...,      # from paper_trade_repo
                )

                if risk_result.approved and aggregation.proposal.side != Side.FLAT:
                    # Step ⑦: Open position
                    self._paper_engine.open_position(
                        aggregation.proposal, current_price=snapshot.current_price
                    )
                elif not risk_result.approved:
                    # Risk rejected or paused
                    status = "risk_paused" if risk_result.action == "pause" else "risk_rejected"
                    if risk_result.action == "pause":
                        self._paper_engine.set_paused(True)
                    ...
```

The implementer should read the full existing `execute()` method and restructure accordingly, keeping backward compatibility (risk_checker and paper_engine are optional).

**Step 4: Run tests to verify they pass**

Run: `cd orchestrator && uv run pytest tests/unit/test_runner.py -v`
Expected: ALL PASS (both old and new tests)

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/pipeline/runner.py orchestrator/tests/unit/test_runner.py
git commit -m "feat: integrate risk checker and paper engine into pipeline runner"
```

---

### Task 8: Telegram Bot — /history, /resume, Push Notifications

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/bot.py`
- Modify: `orchestrator/src/orchestrator/telegram/formatters.py` (add `format_history`)
- Modify: `orchestrator/tests/unit/test_telegram.py`

**Step 1: Write the failing tests**

Add to `orchestrator/tests/unit/test_telegram.py`:

```python
from orchestrator.telegram.formatters import format_help, format_history


class TestFormatHistory:
    def test_format_empty_history(self):
        text = format_history([])
        assert "No closed trades" in text

    def test_format_history_with_trades(self):
        from orchestrator.storage.models import PaperTradeRecord
        from datetime import UTC, datetime
        trade = PaperTradeRecord(
            trade_id="t-001", proposal_id="p-001",
            symbol="BTC/USDT:USDT", side="long",
            entry_price=95000.0, exit_price=93000.0,
            quantity=0.075, pnl=-150.0, fees=7.13,
            status="closed", risk_pct=1.5,
            opened_at=datetime.now(UTC), closed_at=datetime.now(UTC),
        )
        text = format_history([trade])
        assert "BTC/USDT:USDT" in text
        assert "-$150.00" in text


class TestFormatHelpUpdated:
    def test_help_includes_history(self):
        text = format_help()
        assert "/history" in text

    def test_help_includes_resume(self):
        text = format_help()
        assert "/resume" in text
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_telegram.py::TestFormatHistory -v`
Expected: FAIL with `ImportError`

**Step 3: Implement**

Add `format_history` to `formatters.py`:

```python
def format_history(trades: list) -> str:
    if not trades:
        return "No closed trades yet."

    lines = ["Recent closed trades:\n"]
    for t in trades:
        pnl_str = f"${t.pnl:,.2f}" if t.pnl >= 0 else f"-${abs(t.pnl):,.2f}"
        lines.append(
            f"  {t.symbol} {t.side.upper()} | "
            f"{t.entry_price:,.1f} → {t.exit_price:,.1f} | "
            f"PnL: {pnl_str}"
        )
    return "\n".join(lines)
```

Update `format_help()` to include `/resume`:

```python
"/resume - Un-pause pipeline after risk pause\n"
```

Add handlers to `SentinelBot`:
- `_history_handler`: calls `paper_trade_repo.get_recent_closed()` + `format_history()`
- `_resume_handler`: calls `paper_engine.set_paused(False)` + replies confirmation
- Register both in `build()`

Add `set_paper_engine` and `set_trade_repo` methods to `SentinelBot`, or pass them through `__init__`.

Update `push_to_admins` to also handle `CloseResult` and `RiskResult` pushes by adding:
- `push_close_report(result: CloseResult)` — sends `format_trade_report()` to all admins
- `push_risk_rejection(...)` — sends `format_risk_rejection()` to all admins

**Step 4: Run tests to verify they pass**

Run: `cd orchestrator && uv run pytest tests/unit/test_telegram.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/bot.py orchestrator/src/orchestrator/telegram/formatters.py orchestrator/tests/unit/test_telegram.py
git commit -m "feat: add /history, /resume commands and push notifications for trades"
```

---

### Task 9: Extract Shared OHLCV Utility (M1 Fix #1)

**Files:**
- Create: `orchestrator/src/orchestrator/agents/utils.py`
- Modify: `orchestrator/src/orchestrator/agents/sentiment.py`
- Modify: `orchestrator/src/orchestrator/agents/market.py`
- Create: `orchestrator/tests/unit/test_agent_utils.py`

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_agent_utils.py
from orchestrator.agents.utils import summarize_ohlcv


class TestSummarizeOhlcv:
    def test_empty_ohlcv(self):
        assert summarize_ohlcv([], max_candles=10) == "No OHLCV data available"

    def test_summarizes_last_n(self):
        candles = [
            [1700000000000, 95000.0, 95500.0, 94800.0, 95200.0, 1000.0],
            [1700003600000, 95200.0, 95800.0, 95100.0, 95600.0, 800.0],
        ]
        text = summarize_ohlcv(candles, max_candles=5)
        assert "O=95000.0" in text
        assert "O=95200.0" in text

    def test_limits_to_max_candles(self):
        candles = [[i, float(i), float(i), float(i), float(i), 100.0] for i in range(20)]
        text = summarize_ohlcv(candles, max_candles=3)
        lines = [l for l in text.strip().split("\n") if l.strip()]
        assert len(lines) == 3
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_agent_utils.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement and refactor**

```python
# orchestrator/src/orchestrator/agents/utils.py
from __future__ import annotations


def summarize_ohlcv(ohlcv: list[list], *, max_candles: int = 10) -> str:
    if not ohlcv:
        return "No OHLCV data available"

    lines = []
    for candle in ohlcv[-max_candles:]:
        o, h, lo, c, v = candle[1], candle[2], candle[3], candle[4], candle[5]
        lines.append(f"  O={o:.1f} H={h:.1f} L={lo:.1f} C={c:.1f} V={v:.0f}")
    return "\n".join(lines)
```

Then in `sentiment.py`, replace `_summarize_ohlcv` static method (lines 54-63):
- Remove the static method
- Change line 27 from `ohlcv_summary = self._summarize_ohlcv(snapshot)` to:
  ```python
  from orchestrator.agents.utils import summarize_ohlcv
  ohlcv_summary = summarize_ohlcv(snapshot.ohlcv, max_candles=10)
  ```

Same in `market.py` (lines 54-63), but with `max_candles=20`.

**Step 4: Run ALL agent tests to verify no breakage**

Run: `cd orchestrator && uv run pytest tests/unit/test_agent_utils.py tests/unit/test_agent_sentiment.py tests/unit/test_agent_market.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/agents/utils.py orchestrator/src/orchestrator/agents/sentiment.py orchestrator/src/orchestrator/agents/market.py orchestrator/tests/unit/test_agent_utils.py
git commit -m "refactor: extract shared summarize_ohlcv utility from agents"
```

---

### Task 10: Fix Prompt Storage (M1 Fix #5)

**Files:**
- Modify: `orchestrator/src/orchestrator/pipeline/runner.py:146-157`
- Modify: `orchestrator/tests/unit/test_runner.py`

**Step 1: Write the failing test**

Add to `orchestrator/tests/unit/test_runner.py`:

```python
class TestSaveLLMCalls:
    def test_saves_full_messages_json(self):
        """_save_llm_calls should store full messages, not placeholder."""
        llm_call_repo = MagicMock()
        runner = PipelineRunner(
            data_fetcher=MagicMock(),
            sentiment_agent=MagicMock(),
            market_agent=MagicMock(),
            proposer_agent=MagicMock(),
            pipeline_repo=MagicMock(),
            llm_call_repo=llm_call_repo,
            proposal_repo=MagicMock(),
        )
        mock_result = MagicMock()
        mock_result.llm_calls = [MagicMock(
            content='{"test": true}', model="test", latency_ms=100,
            input_tokens=50, output_tokens=25,
        )]
        mock_result.messages = [{"role": "user", "content": "analyze"}]

        runner._save_llm_calls("run-1", "sentiment", mock_result)

        call_kwargs = llm_call_repo.save_call.call_args[1]
        assert call_kwargs["prompt"] != "(see messages)"
        assert "analyze" in call_kwargs["prompt"]
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_runner.py::TestSaveLLMCalls -v`
Expected: FAIL (prompt still contains placeholder)

**Step 3: Implement**

In `runner.py`, update `_save_llm_calls`:

```python
    def _save_llm_calls(self, run_id: str, agent_type: str, result: AgentResult) -> None:
        import json
        prompt_json = json.dumps(result.messages, ensure_ascii=False) if result.messages else ""
        for call in result.llm_calls:
            self._llm_call_repo.save_call(
                run_id=run_id,
                agent_type=agent_type,
                prompt=prompt_json,
                response=call.content,
                model=call.model,
                latency_ms=call.latency_ms,
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
            )
```

Also need to add `messages` field to `AgentResult` in `agents/base.py`:

```python
class AgentResult[T: BaseModel](BaseModel):
    output: T
    degraded: bool = False
    llm_calls: list[LLMCallResult] = []
    messages: list[dict] = []
```

And in `BaseAgent.analyze()`, include messages in the returned result:

```python
return AgentResult(
    output=validation.value,
    degraded=False,
    llm_calls=llm_calls,
    messages=messages,
)
```

(Same for the degraded return.)

**Step 4: Run tests to verify they pass**

Run: `cd orchestrator && uv run pytest tests/unit/test_runner.py tests/unit/test_agent_base.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/pipeline/runner.py orchestrator/src/orchestrator/agents/base.py orchestrator/tests/unit/test_runner.py
git commit -m "fix: store full LLM messages instead of placeholder in call records"
```

---

### Task 11: Wire M2 Components into Application Entrypoint

**Files:**
- Modify: `orchestrator/src/orchestrator/__main__.py`
- Modify: `orchestrator/tests/unit/test_main.py`

**Step 1: Write the failing test**

Update `orchestrator/tests/unit/test_main.py`:

```python
def test_create_app_components_includes_m2():
    """Verify M2 components (risk_checker, paper_engine) are in output."""
    components = create_app_components(
        telegram_bot_token="test-token",
        telegram_admin_chat_ids=[123],
        exchange_id="binance",
        database_url="sqlite:///:memory:",
        anthropic_api_key="test-key",
    )
    assert "paper_engine" in components
    assert "risk_checker" in components
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_main.py -v`
Expected: FAIL (no paper_engine key)

**Step 3: Implement wiring**

Add to `create_app_components()` in `__main__.py`:

```python
from orchestrator.risk.checker import RiskChecker
from orchestrator.risk.position_sizer import RiskPercentSizer
from orchestrator.exchange.paper_engine import PaperEngine
from orchestrator.storage.repository import PaperTradeRepository, AccountSnapshotRepository

# ... inside create_app_components(), after existing repos:

    paper_trade_repo = PaperTradeRepository(session)
    account_snapshot_repo = AccountSnapshotRepository(session)

    # Risk
    risk_checker = RiskChecker(
        max_single_risk_pct=max_single_risk_pct,
        max_total_exposure_pct=max_total_exposure_pct,
        max_consecutive_losses=max_consecutive_losses,
        max_daily_loss_pct=max_daily_loss_pct,
    )

    # Paper Engine
    paper_engine = PaperEngine(
        initial_equity=paper_initial_equity,
        taker_fee_rate=paper_taker_fee_rate,
        position_sizer=RiskPercentSizer(),
        trade_repo=paper_trade_repo,
        snapshot_repo=account_snapshot_repo,
    )
    paper_engine.rebuild_from_db()
```

Add the new params to `create_app_components()` signature (with defaults from config). Pass `risk_checker` and `paper_engine` to `PipelineRunner`. Add them to the return dict.

Also add the new settings to `main()` function, passing `paper_initial_equity`, `paper_taker_fee_rate`, risk params, etc.

**Step 4: Run tests to verify they pass**

Run: `cd orchestrator && uv run pytest tests/unit/test_main.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/__main__.py orchestrator/tests/unit/test_main.py
git commit -m "feat: wire M2 risk checker and paper engine into app entrypoint"
```

---

### Task 12: Fix /status and /coin to Read from DB (M1 Fix #4)

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/bot.py`
- Modify: `orchestrator/tests/unit/test_telegram.py`

**Step 1: Write the failing test**

```python
class TestBotStatusFromDB:
    def test_status_reads_from_proposal_repo(self):
        """Verify /status reads from DB, not in-memory dict."""
        # This test verifies that the bot has a proposal_repo attribute
        # and status_handler uses it
        bot = SentinelBot(
            token="test-token",
            admin_chat_ids=[123],
        )
        # bot should accept a proposal_repo
        assert hasattr(bot, 'set_proposal_repo') or hasattr(bot, '_proposal_repo')
```

**Step 2-3: Implement**

Add `set_proposal_repo(repo)` to `SentinelBot`. Modify `_status_handler` and `_coin_handler` to read from `proposal_repo.get_recent()` instead of `self._latest_results`. Remove `_latest_results` dict.

Keep `_latest_results` as a cache that gets updated on each run, but fall back to DB if empty (e.g., after restart).

**Step 4: Run tests**

Run: `cd orchestrator && uv run pytest tests/unit/test_telegram.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/bot.py orchestrator/tests/unit/test_telegram.py
git commit -m "fix: read /status and /coin data from DB instead of in-memory only"
```

---

### Task 13: Full Integration Verification

**Files:**
- No new files

**Step 1: Run full test suite with coverage**

```bash
cd orchestrator && uv run pytest -v --cov=orchestrator --cov-report=term-missing
```

Expected: ALL PASS, coverage >= 80%

**Step 2: Run lint**

```bash
cd orchestrator && uv run ruff check src/ tests/
```

Expected: All checks passed

**Step 3: Fix any issues**

If lint errors: `uv run ruff check --fix src/ tests/`
If coverage below 80%: add tests for uncovered lines.

**Step 4: Commit fixes if any**

```bash
git add -A
git commit -m "chore: fix lint errors and ensure M2 test coverage"
```

---

### Task 14: Final Ruff + Test Verification

**Step 1: Run complete verification**

```bash
cd orchestrator && uv run ruff check src/ tests/ && uv run pytest -v --cov=orchestrator --cov-report=term-missing
```

Expected: All checks passed, 80%+ coverage, 0 failures.

If everything passes, M2 is complete.

```bash
git log --oneline | head -20
```

Review all M2 commits are clean and atomic.
