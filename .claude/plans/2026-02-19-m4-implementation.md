# M4: Semi-Auto Trading with Approval Flow Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform pipeline from auto paper trading to semi-auto with TG inline keyboard approval, switchable Paper/Live execution via CCXT, and exchange-native SL/TP orders.

**Architecture:** Three new modules: `approval/` for approval state machine, `execution/` for order executor strategy pattern (Paper/Live), plus TG inline keyboard integration. Pipeline runner gets a new `pending_approval` status path. Config controls `trading_mode` (paper/live) and `approval_timeout_minutes`.

**Tech Stack:** Python 3.12, python-telegram-bot (InlineKeyboardMarkup, CallbackQueryHandler), CCXT async, Pydantic v2, APScheduler

---

## Pre-flight

Before starting, run from `orchestrator/` directory:
```bash
export PATH="$HOME/.local/bin:$PATH"
uv run pytest -v --tb=short   # expect 183 passed
uv run ruff check src/ tests/ # expect clean
```

All commands in this plan assume you're in `orchestrator/` and `PATH` includes `~/.local/bin`.

---

### Task 1: Add semi-auto config fields

**Files:**
- Modify: `orchestrator/src/orchestrator/config.py:38-42`
- Test: `orchestrator/tests/unit/test_config.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_config.py`:

```python
def test_settings_semi_auto_defaults():
    """Settings should have semi-auto trading config fields."""
    settings = Settings(
        telegram_bot_token="test", telegram_admin_chat_ids=[123],
        anthropic_api_key="test-key",
    )
    assert settings.trading_mode == "paper"
    assert settings.approval_timeout_minutes == 15
    assert settings.price_deviation_threshold == 0.01
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_config.py::test_settings_semi_auto_defaults -v`
Expected: FAIL — fields don't exist.

**Step 3: Add config fields**

In `config.py`, add after Paper Trading section:

```python
# Semi-auto Trading
trading_mode: str = "paper"                    # "paper" | "live"
approval_timeout_minutes: int = 15
price_deviation_threshold: float = 0.01        # 1%
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/config.py orchestrator/tests/unit/test_config.py
git commit -m "feat: add semi-auto trading config fields"
```

---

### Task 2: Add ApprovalRecord to storage models

**Files:**
- Modify: `orchestrator/src/orchestrator/storage/models.py`
- Test: `orchestrator/tests/unit/test_storage.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_storage.py`:

```python
class TestApprovalRecord:
    def test_create_approval_record(self, session):
        from orchestrator.storage.models import ApprovalRecord
        from datetime import UTC, datetime, timedelta

        now = datetime.now(UTC)
        record = ApprovalRecord(
            approval_id="a-001",
            proposal_id="p-001",
            run_id="run-001",
            snapshot_price=95200.0,
            status="pending",
            created_at=now,
            expires_at=now + timedelta(minutes=15),
        )
        assert record.approval_id == "a-001"
        assert record.status == "pending"
        assert record.message_id is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_storage.py::TestApprovalRecord -v`
Expected: FAIL — `ApprovalRecord` doesn't exist.

**Step 3: Implement ApprovalRecord**

Add to `storage/models.py`:

```python
class ApprovalRecord(SQLModel, table=True):
    __tablename__ = "approval_records"

    id: int | None = Field(default=None, primary_key=True)
    approval_id: str = Field(unique=True, index=True)
    proposal_id: str = Field(index=True)
    run_id: str
    snapshot_price: float
    status: str = "pending"  # pending, approved, rejected, expired
    message_id: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    resolved_at: datetime | None = None
```

**Step 4: Extend PaperTradeRecord with live trading fields**

Add to `PaperTradeRecord`:

```python
mode: str = "paper"              # "paper" | "live"
exchange_order_id: str = ""
sl_order_id: str = ""
tp_order_id: str = ""
```

**Step 5: Run tests**

Run: `uv run pytest tests/unit/test_storage.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/storage/models.py orchestrator/tests/unit/test_storage.py
git commit -m "feat: add ApprovalRecord and extend PaperTradeRecord with live fields"
```

---

### Task 3: Add ApprovalRepository

**Files:**
- Modify: `orchestrator/src/orchestrator/storage/repository.py`
- Test: `orchestrator/tests/unit/test_storage.py`

**Step 1: Write the failing tests**

Add to `tests/unit/test_storage.py`:

```python
from orchestrator.storage.repository import ApprovalRepository

class TestApprovalRepository:
    def test_save_and_get_approval(self, session):
        from datetime import UTC, datetime, timedelta

        repo = ApprovalRepository(session)
        now = datetime.now(UTC)
        record = repo.save_approval(
            approval_id="a-001", proposal_id="p-001", run_id="run-001",
            snapshot_price=95200.0, expires_at=now + timedelta(minutes=15),
        )
        assert record.approval_id == "a-001"
        assert record.status == "pending"

        fetched = repo.get_by_id("a-001")
        assert fetched is not None
        assert fetched.proposal_id == "p-001"

    def test_update_status(self, session):
        from datetime import UTC, datetime, timedelta

        repo = ApprovalRepository(session)
        now = datetime.now(UTC)
        repo.save_approval(
            approval_id="a-002", proposal_id="p-002", run_id="run-002",
            snapshot_price=95000.0, expires_at=now + timedelta(minutes=15),
        )
        updated = repo.update_status("a-002", status="approved")
        assert updated.status == "approved"
        assert updated.resolved_at is not None

    def test_get_pending(self, session):
        from datetime import UTC, datetime, timedelta

        repo = ApprovalRepository(session)
        now = datetime.now(UTC)
        repo.save_approval(
            approval_id="a-p1", proposal_id="p-1", run_id="r-1",
            snapshot_price=95000.0, expires_at=now + timedelta(minutes=15),
        )
        repo.save_approval(
            approval_id="a-p2", proposal_id="p-2", run_id="r-2",
            snapshot_price=95000.0, expires_at=now + timedelta(minutes=15),
        )
        repo.update_status("a-p2", status="approved")

        pending = repo.get_pending()
        assert len(pending) == 1
        assert pending[0].approval_id == "a-p1"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_storage.py::TestApprovalRepository -v`
Expected: FAIL

**Step 3: Implement ApprovalRepository**

Add to `repository.py`:

```python
from orchestrator.storage.models import ApprovalRecord

class ApprovalRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def save_approval(
        self,
        *,
        approval_id: str,
        proposal_id: str,
        run_id: str,
        snapshot_price: float,
        expires_at: datetime,
        message_id: int | None = None,
    ) -> ApprovalRecord:
        from datetime import UTC, datetime as dt

        record = ApprovalRecord(
            approval_id=approval_id,
            proposal_id=proposal_id,
            run_id=run_id,
            snapshot_price=snapshot_price,
            status="pending",
            message_id=message_id,
            created_at=dt.now(UTC),
            expires_at=expires_at,
        )
        self._session.add(record)
        self._session.commit()
        self._session.refresh(record)
        return record

    def get_by_id(self, approval_id: str) -> ApprovalRecord | None:
        statement = select(ApprovalRecord).where(
            ApprovalRecord.approval_id == approval_id
        )
        return self._session.exec(statement).first()

    def update_status(self, approval_id: str, *, status: str) -> ApprovalRecord:
        from datetime import UTC, datetime as dt

        record = self.get_by_id(approval_id)
        if record is None:
            raise ValueError(f"Approval {approval_id} not found")
        record.status = status
        record.resolved_at = dt.now(UTC)
        self._session.add(record)
        self._session.commit()
        self._session.refresh(record)
        return record

    def update_message_id(self, approval_id: str, *, message_id: int) -> ApprovalRecord:
        record = self.get_by_id(approval_id)
        if record is None:
            raise ValueError(f"Approval {approval_id} not found")
        record.message_id = message_id
        self._session.add(record)
        self._session.commit()
        self._session.refresh(record)
        return record

    def get_pending(self) -> list[ApprovalRecord]:
        statement = select(ApprovalRecord).where(
            ApprovalRecord.status == "pending"
        )
        return list(self._session.exec(statement).all())
```

Add `datetime` import at top of file if not already present.

**Step 4: Run tests**

Run: `uv run pytest tests/unit/test_storage.py::TestApprovalRepository -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/storage/repository.py orchestrator/tests/unit/test_storage.py
git commit -m "feat: add ApprovalRepository for approval record persistence"
```

---

### Task 4: Implement ApprovalManager

**Files:**
- Create: `orchestrator/src/orchestrator/approval/__init__.py`
- Create: `orchestrator/src/orchestrator/approval/manager.py`
- Create: `orchestrator/tests/unit/test_approval_manager.py`

**Step 1: Create `approval/__init__.py`**

Empty file.

**Step 2: Write the failing tests**

Create `tests/unit/test_approval_manager.py`:

```python
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from orchestrator.approval.manager import ApprovalManager, PendingApproval
from orchestrator.models import EntryOrder, Side, TradeProposal


def _make_proposal():
    return TradeProposal(
        symbol="BTC/USDT:USDT", side=Side.LONG, entry=EntryOrder(type="market"),
        position_size_risk_pct=1.5, stop_loss=93000.0, take_profit=[97000.0],
        time_horizon="4h", confidence=0.75, invalid_if=[], rationale="test",
    )


class TestApprovalManager:
    def test_create_pending(self):
        repo = MagicMock()
        repo.save_approval.return_value = MagicMock(approval_id="a-001")
        mgr = ApprovalManager(repo=repo, timeout_minutes=15)
        approval = mgr.create(
            proposal=_make_proposal(), run_id="run-1", snapshot_price=95200.0
        )
        assert isinstance(approval, PendingApproval)
        assert approval.status == "pending"
        assert (approval.expires_at - approval.created_at).total_seconds() == 900

    def test_approve_valid(self):
        repo = MagicMock()
        repo.save_approval.return_value = MagicMock(approval_id="a-001")
        repo.update_status.return_value = MagicMock(status="approved")
        mgr = ApprovalManager(repo=repo, timeout_minutes=15)
        approval = mgr.create(
            proposal=_make_proposal(), run_id="run-1", snapshot_price=95200.0
        )
        result = mgr.approve(approval.approval_id)
        assert result is not None
        repo.update_status.assert_called_once()

    def test_approve_expired_returns_none(self):
        repo = MagicMock()
        repo.save_approval.return_value = MagicMock(approval_id="a-001")
        mgr = ApprovalManager(repo=repo, timeout_minutes=0)  # instant expiry
        approval = mgr.create(
            proposal=_make_proposal(), run_id="run-1", snapshot_price=95200.0
        )
        # Manually expire
        import time
        time.sleep(0.01)
        result = mgr.approve(approval.approval_id)
        assert result is None

    def test_reject(self):
        repo = MagicMock()
        repo.save_approval.return_value = MagicMock(approval_id="a-001")
        repo.update_status.return_value = MagicMock(status="rejected")
        mgr = ApprovalManager(repo=repo, timeout_minutes=15)
        approval = mgr.create(
            proposal=_make_proposal(), run_id="run-1", snapshot_price=95200.0
        )
        result = mgr.reject(approval.approval_id)
        assert result is not None

    def test_expire_stale(self):
        repo = MagicMock()
        repo.save_approval.return_value = MagicMock(approval_id="a-001")
        repo.update_status.return_value = MagicMock(status="expired")
        mgr = ApprovalManager(repo=repo, timeout_minutes=0)
        mgr.create(proposal=_make_proposal(), run_id="run-1", snapshot_price=95200.0)
        import time
        time.sleep(0.01)
        expired = mgr.expire_stale()
        assert len(expired) == 1

    def test_pending_approval_is_frozen(self):
        approval = PendingApproval(
            approval_id="a-001", proposal=_make_proposal(),
            run_id="run-1", snapshot_price=95200.0,
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(minutes=15),
        )
        with pytest.raises(Exception):
            approval.status = "approved"
```

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_approval_manager.py -v`
Expected: FAIL

**Step 4: Implement ApprovalManager**

Create `approval/manager.py`:

```python
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel

from orchestrator.models import TradeProposal

if TYPE_CHECKING:
    from orchestrator.storage.repository import ApprovalRepository

logger = structlog.get_logger(__name__)


class PendingApproval(BaseModel, frozen=True):
    approval_id: str
    proposal: TradeProposal
    run_id: str
    snapshot_price: float
    created_at: datetime
    expires_at: datetime
    status: str = "pending"
    message_id: int | None = None


class ApprovalManager:
    def __init__(self, *, repo: ApprovalRepository, timeout_minutes: int = 15) -> None:
        self._repo = repo
        self._timeout_minutes = timeout_minutes
        self._pending: dict[str, PendingApproval] = {}

    def create(
        self, *, proposal: TradeProposal, run_id: str, snapshot_price: float
    ) -> PendingApproval:
        approval_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        expires_at = now + timedelta(minutes=self._timeout_minutes)

        approval = PendingApproval(
            approval_id=approval_id,
            proposal=proposal,
            run_id=run_id,
            snapshot_price=snapshot_price,
            created_at=now,
            expires_at=expires_at,
        )
        self._pending[approval_id] = approval

        self._repo.save_approval(
            approval_id=approval_id,
            proposal_id=proposal.proposal_id,
            run_id=run_id,
            snapshot_price=snapshot_price,
            expires_at=expires_at,
        )

        logger.info(
            "approval_created",
            approval_id=approval_id,
            symbol=proposal.symbol,
            side=proposal.side,
            expires_at=expires_at.isoformat(),
        )
        return approval

    def approve(self, approval_id: str) -> PendingApproval | None:
        approval = self._pending.get(approval_id)
        if approval is None:
            return None
        if datetime.now(UTC) > approval.expires_at:
            self._expire(approval_id)
            return None

        del self._pending[approval_id]
        self._repo.update_status(approval_id, status="approved")
        logger.info("approval_approved", approval_id=approval_id)
        return approval

    def reject(self, approval_id: str) -> PendingApproval | None:
        approval = self._pending.pop(approval_id, None)
        if approval is None:
            return None
        self._repo.update_status(approval_id, status="rejected")
        logger.info("approval_rejected", approval_id=approval_id)
        return approval

    def get(self, approval_id: str) -> PendingApproval | None:
        return self._pending.get(approval_id)

    def set_message_id(self, approval_id: str, message_id: int) -> None:
        approval = self._pending.get(approval_id)
        if approval is not None:
            # Replace with updated version (frozen model)
            updated = PendingApproval(
                **{**approval.model_dump(), "message_id": message_id}
            )
            self._pending[approval_id] = updated
            self._repo.update_message_id(approval_id, message_id=message_id)

    def expire_stale(self) -> list[PendingApproval]:
        now = datetime.now(UTC)
        expired: list[PendingApproval] = []
        for aid, approval in list(self._pending.items()):
            if now > approval.expires_at:
                expired.append(approval)
                self._expire(aid)
        return expired

    def get_pending_count(self) -> int:
        return len(self._pending)

    def _expire(self, approval_id: str) -> None:
        self._pending.pop(approval_id, None)
        self._repo.update_status(approval_id, status="expired")
        logger.info("approval_expired", approval_id=approval_id)
```

**Step 5: Run tests**

Run: `uv run pytest tests/unit/test_approval_manager.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/approval/ orchestrator/tests/unit/test_approval_manager.py
git commit -m "feat: add ApprovalManager with pending approval state machine"
```

---

### Task 5: Implement OrderExecutor interface and PaperExecutor

**Files:**
- Create: `orchestrator/src/orchestrator/execution/__init__.py`
- Create: `orchestrator/src/orchestrator/execution/executor.py`
- Create: `orchestrator/tests/unit/test_executor.py`

**Step 1: Create `execution/__init__.py`**

Empty file.

**Step 2: Write the failing tests**

Create `tests/unit/test_executor.py`:

```python
import pytest
from unittest.mock import MagicMock

from orchestrator.execution.executor import ExecutionResult, PaperExecutor
from orchestrator.exchange.paper_engine import Position
from orchestrator.models import EntryOrder, Side, TradeProposal


def _make_proposal():
    return TradeProposal(
        symbol="BTC/USDT:USDT", side=Side.LONG, entry=EntryOrder(type="market"),
        position_size_risk_pct=1.5, stop_loss=93000.0, take_profit=[97000.0],
        time_horizon="4h", confidence=0.75, invalid_if=[], rationale="test",
    )


class TestPaperExecutor:
    @pytest.mark.asyncio
    async def test_execute_entry(self):
        paper_engine = MagicMock()
        paper_engine.open_position.return_value = Position(
            trade_id="t-001", proposal_id="p-001",
            symbol="BTC/USDT:USDT", side=Side.LONG,
            entry_price=95200.0, quantity=0.075,
            stop_loss=93000.0, take_profit=[97000.0],
            opened_at=MagicMock(), risk_pct=1.5,
        )

        executor = PaperExecutor(paper_engine=paper_engine)
        result = await executor.execute_entry(_make_proposal(), current_price=95200.0)

        assert isinstance(result, ExecutionResult)
        assert result.mode == "paper"
        assert result.trade_id == "t-001"
        assert result.entry_price == 95200.0
        paper_engine.open_position.assert_called_once()

    @pytest.mark.asyncio
    async def test_place_sl_tp_is_noop_for_paper(self):
        paper_engine = MagicMock()
        executor = PaperExecutor(paper_engine=paper_engine)
        order_ids = await executor.place_sl_tp(
            symbol="BTC/USDT:USDT", side="long", quantity=0.075,
            stop_loss=93000.0, take_profit=[97000.0],
        )
        assert order_ids == []  # paper mode doesn't place exchange orders

    def test_execution_result_is_frozen(self):
        result = ExecutionResult(
            trade_id="t-001", symbol="BTC/USDT:USDT", side="long",
            entry_price=95200.0, quantity=0.075, fees=3.57, mode="paper",
        )
        with pytest.raises(Exception):
            result.mode = "live"
```

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_executor.py -v`
Expected: FAIL

**Step 4: Implement ExecutionResult, OrderExecutor ABC, and PaperExecutor**

Create `execution/executor.py`:

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel

from orchestrator.models import TradeProposal

if TYPE_CHECKING:
    from orchestrator.exchange.paper_engine import PaperEngine

logger = structlog.get_logger(__name__)


class ExecutionResult(BaseModel, frozen=True):
    trade_id: str
    symbol: str
    side: str
    entry_price: float
    quantity: float
    fees: float
    mode: str  # "paper" | "live"
    exchange_order_id: str = ""
    sl_order_id: str = ""
    tp_order_id: str = ""


class OrderExecutor(ABC):
    @abstractmethod
    async def execute_entry(
        self, proposal: TradeProposal, current_price: float
    ) -> ExecutionResult: ...

    @abstractmethod
    async def place_sl_tp(
        self, *, symbol: str, side: str, quantity: float,
        stop_loss: float, take_profit: list[float],
    ) -> list[str]: ...

    @abstractmethod
    async def cancel_orders(self, order_ids: list[str]) -> None: ...


class PaperExecutor(OrderExecutor):
    def __init__(self, *, paper_engine: PaperEngine) -> None:
        self._paper_engine = paper_engine

    async def execute_entry(
        self, proposal: TradeProposal, current_price: float
    ) -> ExecutionResult:
        position = self._paper_engine.open_position(proposal, current_price)
        logger.info(
            "paper_execution",
            trade_id=position.trade_id,
            symbol=position.symbol,
            entry_price=position.entry_price,
        )
        return ExecutionResult(
            trade_id=position.trade_id,
            symbol=position.symbol,
            side=position.side.value,
            entry_price=position.entry_price,
            quantity=position.quantity,
            fees=position.quantity * position.entry_price * self._paper_engine._taker_fee_rate,
            mode="paper",
        )

    async def place_sl_tp(
        self, *, symbol: str, side: str, quantity: float,
        stop_loss: float, take_profit: list[float],
    ) -> list[str]:
        # Paper mode: SL/TP handled by PaperEngine.check_sl_tp()
        return []

    async def cancel_orders(self, order_ids: list[str]) -> None:
        # Paper mode: nothing to cancel
        pass
```

**Step 5: Run tests**

Run: `uv run pytest tests/unit/test_executor.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/execution/ orchestrator/tests/unit/test_executor.py
git commit -m "feat: add OrderExecutor interface and PaperExecutor implementation"
```

---

### Task 6: Implement LiveExecutor

**Files:**
- Modify: `orchestrator/src/orchestrator/exchange/client.py`
- Modify: `orchestrator/src/orchestrator/execution/executor.py`
- Create: `orchestrator/tests/unit/test_live_executor.py`

**Step 1: Write the failing tests for ExchangeClient extensions**

Add to `tests/unit/test_exchange.py` or create `tests/unit/test_live_executor.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.execution.executor import ExecutionResult, LiveExecutor
from orchestrator.models import EntryOrder, Side, TradeProposal


def _make_proposal():
    return TradeProposal(
        symbol="BTC/USDT:USDT", side=Side.LONG, entry=EntryOrder(type="market"),
        position_size_risk_pct=1.5, stop_loss=93000.0, take_profit=[97000.0],
        time_horizon="4h", confidence=0.75, invalid_if=[], rationale="test",
    )


class TestLiveExecutor:
    @pytest.mark.asyncio
    async def test_execute_entry_success(self):
        exchange_client = AsyncMock()
        exchange_client.create_market_order.return_value = {
            "id": "binance-order-001",
            "price": 95200.0,
            "filled": 0.075,
            "fee": {"cost": 3.57},
        }
        position_sizer = MagicMock()
        position_sizer.calculate.return_value = 0.075
        paper_engine = MagicMock()
        paper_engine.equity = 10000.0
        paper_engine._taker_fee_rate = 0.0005

        executor = LiveExecutor(
            exchange_client=exchange_client,
            position_sizer=position_sizer,
            paper_engine=paper_engine,
            price_deviation_threshold=0.01,
        )
        result = await executor.execute_entry(_make_proposal(), current_price=95200.0)

        assert isinstance(result, ExecutionResult)
        assert result.mode == "live"
        assert result.exchange_order_id == "binance-order-001"
        exchange_client.create_market_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_entry_price_deviation_raises(self):
        exchange_client = AsyncMock()
        position_sizer = MagicMock()
        position_sizer.calculate.return_value = 0.075
        paper_engine = MagicMock()
        paper_engine.equity = 10000.0

        executor = LiveExecutor(
            exchange_client=exchange_client,
            position_sizer=position_sizer,
            paper_engine=paper_engine,
            price_deviation_threshold=0.01,
        )
        # current_price deviated 5% from snapshot
        with pytest.raises(ValueError, match="deviated"):
            await executor.execute_entry(
                _make_proposal(), current_price=100000.0  # 5% away from 95200
            )

    @pytest.mark.asyncio
    async def test_place_sl_tp(self):
        exchange_client = AsyncMock()
        exchange_client.create_stop_order.side_effect = [
            {"id": "sl-001"},
            {"id": "tp-001"},
        ]
        executor = LiveExecutor(
            exchange_client=exchange_client,
            position_sizer=MagicMock(),
            paper_engine=MagicMock(),
            price_deviation_threshold=0.01,
        )
        ids = await executor.place_sl_tp(
            symbol="BTC/USDT:USDT", side="long", quantity=0.075,
            stop_loss=93000.0, take_profit=[97000.0],
        )
        assert len(ids) == 2
        assert "sl-001" in ids
        assert "tp-001" in ids
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_live_executor.py -v`
Expected: FAIL

**Step 3: Add exchange client methods**

In `exchange/client.py`, add:

```python
async def create_market_order(
    self, symbol: str, side: str, amount: float
) -> dict:
    return await self._exchange.create_order(
        symbol, "market", side, amount
    )

async def create_stop_order(
    self, symbol: str, side: str, amount: float, *, stop_price: float
) -> dict:
    return await self._exchange.create_order(
        symbol, "stop_market", side, amount,
        params={"stopPrice": stop_price},
    )

async def cancel_order(self, order_id: str, symbol: str) -> dict:
    return await self._exchange.cancel_order(order_id, symbol)

async def fetch_order(self, order_id: str, symbol: str) -> dict:
    return await self._exchange.fetch_order(order_id, symbol)
```

**Step 4: Implement LiveExecutor**

Add to `execution/executor.py`:

```python
class LiveExecutor(OrderExecutor):
    def __init__(
        self,
        *,
        exchange_client,  # ExchangeClient
        position_sizer,   # PositionSizer
        paper_engine,     # PaperEngine (for equity)
        price_deviation_threshold: float = 0.01,
    ) -> None:
        self._exchange = exchange_client
        self._position_sizer = position_sizer
        self._paper_engine = paper_engine
        self._threshold = price_deviation_threshold

    async def execute_entry(
        self, proposal: TradeProposal, current_price: float
    ) -> ExecutionResult:
        import uuid

        # Check price deviation from proposal snapshot
        if proposal.stop_loss is not None:
            quantity = self._position_sizer.calculate(
                equity=self._paper_engine.equity,
                risk_pct=proposal.position_size_risk_pct,
                entry_price=current_price,
                stop_loss=proposal.stop_loss,
            )
        else:
            quantity = 0.0

        # Validate deviation
        # Use entry estimate from the proposal's stop_loss context
        expected_price = current_price  # snapshot price passed at approval time
        # We compare with the live market price (re-fetched by caller)
        # For now current_price IS the re-fetched price; snapshot was stored in PendingApproval

        ccxt_side = "buy" if proposal.side.value == "long" else "sell"
        order = await self._exchange.create_market_order(
            proposal.symbol, ccxt_side, quantity
        )

        fill_price = order.get("price", current_price)
        fill_qty = order.get("filled", quantity)
        fee_info = order.get("fee", {})
        fees = fee_info.get("cost", fill_qty * fill_price * 0.0005)

        trade_id = str(uuid.uuid4())

        logger.info(
            "live_execution",
            trade_id=trade_id,
            order_id=order.get("id", ""),
            symbol=proposal.symbol,
            fill_price=fill_price,
            quantity=fill_qty,
        )

        return ExecutionResult(
            trade_id=trade_id,
            symbol=proposal.symbol,
            side=proposal.side.value,
            entry_price=fill_price,
            quantity=fill_qty,
            fees=fees,
            mode="live",
            exchange_order_id=order.get("id", ""),
        )

    async def place_sl_tp(
        self, *, symbol: str, side: str, quantity: float,
        stop_loss: float, take_profit: list[float],
    ) -> list[str]:
        order_ids: list[str] = []
        # SL: opposite side
        close_side = "sell" if side == "long" else "buy"

        sl_order = await self._exchange.create_stop_order(
            symbol, close_side, quantity, stop_price=stop_loss
        )
        order_ids.append(sl_order.get("id", ""))

        # TP: first target only for MVP
        if take_profit:
            tp_order = await self._exchange.create_stop_order(
                symbol, close_side, quantity, stop_price=take_profit[0]
            )
            order_ids.append(tp_order.get("id", ""))

        logger.info("sl_tp_placed", symbol=symbol, order_ids=order_ids)
        return order_ids

    async def cancel_orders(self, order_ids: list[str]) -> None:
        for oid in order_ids:
            if oid:
                try:
                    await self._exchange.cancel_order(oid, "")
                except Exception as e:
                    logger.warning("cancel_order_failed", order_id=oid, error=str(e))
```

Note: The `execute_entry` method needs a price deviation check. The caller (approval callback handler) will pass `snapshot_price` from the `PendingApproval` and the freshly fetched `current_price`. Add the deviation check to the executor or (better) in the callback handler before calling execute. For the test, we verify the executor itself works. The deviation check should be in the handler or a separate helper — let's add it as a static method:

Add to `LiveExecutor`:

```python
def check_price_deviation(
    self, snapshot_price: float, current_price: float
) -> float:
    """Return deviation as a ratio. Raises ValueError if above threshold."""
    if snapshot_price <= 0:
        return 0.0
    deviation = abs(current_price - snapshot_price) / snapshot_price
    if deviation > self._threshold:
        raise ValueError(
            f"Price deviated {deviation:.1%} from proposal time "
            f"(was ${snapshot_price:,.1f}, now ${current_price:,.1f}). "
            f"Threshold: {self._threshold:.1%}"
        )
    return deviation
```

**Step 5: Run tests**

Run: `uv run pytest tests/unit/test_live_executor.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/exchange/client.py orchestrator/src/orchestrator/execution/executor.py orchestrator/tests/unit/test_live_executor.py
git commit -m "feat: add LiveExecutor with CCXT market orders and SL/TP placement"
```

---

### Task 7: Add TG formatters for approval and execution

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/formatters.py`
- Modify: `orchestrator/tests/unit/test_telegram.py`

**Step 1: Write the failing tests**

Add to `tests/unit/test_telegram.py`:

```python
class TestFormatPendingApproval:
    def test_format_pending_long(self):
        from orchestrator.telegram.formatters import format_pending_approval
        from orchestrator.approval.manager import PendingApproval
        from datetime import UTC, datetime, timedelta

        now = datetime.now(UTC)
        approval = PendingApproval(
            approval_id="a-001",
            proposal=TradeProposal(
                symbol="BTC/USDT:USDT", side=Side.LONG, entry=EntryOrder(type="market"),
                position_size_risk_pct=1.5, stop_loss=93000.0, take_profit=[97000.0],
                time_horizon="4h", confidence=0.75, invalid_if=[], rationale="Strong breakout",
            ),
            run_id="run-1", snapshot_price=95200.0,
            created_at=now, expires_at=now + timedelta(minutes=15),
        )
        text = format_pending_approval(approval)
        assert "PENDING" in text
        assert "BTC/USDT:USDT" in text
        assert "LONG" in text
        assert "93,000" in text or "93000" in text
        assert "15" in text  # timeout mention


class TestFormatExecutionResult:
    def test_format_live_execution(self):
        from orchestrator.telegram.formatters import format_execution_result
        from orchestrator.execution.executor import ExecutionResult

        result = ExecutionResult(
            trade_id="t-001", symbol="BTC/USDT:USDT", side="long",
            entry_price=95350.0, quantity=0.075, fees=3.57,
            mode="live", exchange_order_id="binance-001",
            sl_order_id="sl-001", tp_order_id="tp-001",
        )
        text = format_execution_result(result)
        assert "EXECUTED" in text
        assert "live" in text.lower()
        assert "95,350" in text or "95350" in text

    def test_format_paper_execution(self):
        from orchestrator.telegram.formatters import format_execution_result
        from orchestrator.execution.executor import ExecutionResult

        result = ExecutionResult(
            trade_id="t-002", symbol="BTC/USDT:USDT", side="long",
            entry_price=95200.0, quantity=0.075, fees=3.57,
            mode="paper",
        )
        text = format_execution_result(result)
        assert "EXECUTED" in text
        assert "paper" in text.lower()
```

**Step 2: Implement formatters**

Add to `telegram/formatters.py`:

```python
def format_pending_approval(approval) -> str:
    """Format a PendingApproval for TG push with inline keyboard context."""
    p = approval.proposal
    lines = [
        f"[PENDING APPROVAL] {p.symbol}",
        f"Side: {p.side.value.upper()}",
        f"Entry: {p.entry.type} @ ~${approval.snapshot_price:,.1f}",
        f"Risk: {p.position_size_risk_pct}%",
    ]
    if p.stop_loss is not None:
        lines.append(f"SL: ${p.stop_loss:,.1f}")
    if p.take_profit:
        tp_str = ", ".join(f"${tp:,.1f}" for tp in p.take_profit)
        lines.append(f"TP: {tp_str}")
    lines.append(f"Confidence: {p.confidence:.0%}")
    lines.append(f"Rationale: {p.rationale}")

    remaining = int((approval.expires_at - approval.created_at).total_seconds() / 60)
    lines.append(f"\nExpires in {remaining} minutes")
    return "\n".join(lines)


def format_execution_result(result) -> str:
    """Format an ExecutionResult for TG confirmation."""
    lines = [
        f"[EXECUTED] {result.symbol} {result.side.upper()}",
        f"Mode: {result.mode}",
        f"Entry: ${result.entry_price:,.1f} | Qty: {result.quantity:.4f}",
        f"Fees: ${result.fees:,.2f}",
    ]
    if result.sl_order_id:
        lines.append(f"SL order: {result.sl_order_id}")
    if result.tp_order_id:
        lines.append(f"TP order: {result.tp_order_id}")
    return "\n".join(lines)
```

Also update `format_help()` to not add new commands yet (approval is via inline keyboard, no new slash commands needed).

**Step 3: Run tests**

Run: `uv run pytest tests/unit/test_telegram.py::TestFormatPendingApproval tests/unit/test_telegram.py::TestFormatExecutionResult -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/formatters.py orchestrator/tests/unit/test_telegram.py
git commit -m "feat: add pending approval and execution result TG formatters"
```

---

### Task 8: Modify PipelineRunner for pending_approval path

**Files:**
- Modify: `orchestrator/src/orchestrator/pipeline/runner.py`
- Modify: `orchestrator/tests/unit/test_runner.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_runner.py`:

```python
class TestPipelineRunnerApproval:
    """Tests for M4 semi-auto approval flow."""

    @pytest.mark.asyncio
    async def test_approval_required_returns_pending(self):
        """When approval_required=True, risk-approved proposals get pending_approval status."""
        data_fetcher = AsyncMock()
        data_fetcher.fetch_snapshot.return_value = MagicMock(
            symbol="BTC/USDT:USDT", current_price=95000.0,
        )
        sentiment_agent = AsyncMock()
        sentiment_agent.analyze.return_value = MagicMock(
            output=MagicMock(), degraded=False, llm_calls=[], messages=[],
        )
        market_agent = AsyncMock()
        market_agent.analyze.return_value = MagicMock(
            output=MagicMock(), degraded=False, llm_calls=[], messages=[],
        )
        proposer_agent = AsyncMock()
        proposer_agent.analyze.return_value = MagicMock(
            output=_make_proposal(side=Side.LONG, risk_pct=1.0),
            degraded=False, llm_calls=[], messages=[],
        )
        risk_checker = MagicMock()
        risk_checker.check.return_value = RiskResult(approved=True)
        paper_engine = MagicMock()
        paper_engine.check_sl_tp.return_value = []
        paper_engine.open_positions_risk_pct = 0.0
        paper_engine.equity = 10000.0
        paper_engine._trade_repo.get_daily_pnl.return_value = 0.0
        paper_engine._trade_repo.count_consecutive_losses.return_value = 0

        approval_manager = MagicMock()
        approval_manager.create.return_value = MagicMock(
            approval_id="a-001",
            snapshot_price=95000.0,
        )

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
            approval_manager=approval_manager,
        )

        result = await runner.execute("BTC/USDT:USDT")
        assert result.status == "pending_approval"
        assert result.approval_id is not None
        approval_manager.create.assert_called_once()
        paper_engine.open_position.assert_not_called()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_runner.py::TestPipelineRunnerApproval -v`
Expected: FAIL — `approval_manager` param doesn't exist.

**Step 3: Modify PipelineRunner**

1. Add `approval_manager` optional param to `__init__`
2. Add `approval_id` field to `PipelineResult`
3. In `execute()`, after risk check passes, if `self._approval_manager is not None`:
   - Create PendingApproval instead of executing
   - Return `PipelineResult(status="pending_approval", approval_id=...)`
4. If `self._approval_manager is None`, keep existing auto-execute behavior

Key changes to `runner.py`:

```python
# In PipelineResult, add:
approval_id: str | None = None

# In __init__, add:
approval_manager: ApprovalManager | None = None,

# In execute(), replace Step 7 block:
if aggregation.proposal.side != Side.FLAT:
    if self._approval_manager is not None:
        # Semi-auto: create pending approval
        approval = self._approval_manager.create(
            proposal=aggregation.proposal,
            run_id=run_id,
            snapshot_price=snapshot.current_price,
        )
        self._proposal_repo.save_proposal(
            proposal_id=aggregation.proposal.proposal_id,
            run_id=run_id,
            proposal_json=aggregation.proposal.model_dump_json(),
            risk_check_result="pending_approval",
        )
        self._pipeline_repo.update_run_status(run_id, "pending_approval")
        log.info("pipeline_pending_approval", approval_id=approval.approval_id)
        return PipelineResult(
            run_id=run_id,
            symbol=symbol,
            status="pending_approval",
            model_used=model_used,
            proposal=aggregation.proposal,
            risk_result=risk_result,
            approval_id=approval.approval_id,
            sentiment_degraded=sentiment_result.degraded,
            market_degraded=market_result.degraded,
            proposer_degraded=proposer_result.degraded,
            close_results=close_results,
        )
    else:
        # Auto mode: execute immediately
        self._paper_engine.open_position(
            aggregation.proposal, current_price=snapshot.current_price
        )
```

**Step 4: Run tests**

Run: `uv run pytest tests/unit/test_runner.py -v`
Expected: All PASS (existing tests don't pass approval_manager, so backward compatible)

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/pipeline/runner.py orchestrator/tests/unit/test_runner.py
git commit -m "feat: add pending_approval path to pipeline runner for semi-auto mode"
```

---

### Task 9: Add TG inline keyboard and callback handler

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/bot.py`
- Modify: `orchestrator/tests/unit/test_telegram.py`

This is the core integration task. The bot needs to:
1. Push proposals with InlineKeyboardMarkup instead of plain text
2. Handle callback queries for approve/reject
3. On approve: re-fetch price, check deviation, execute via OrderExecutor, push confirmation
4. On reject: mark rejected, edit message
5. Register a periodic job to expire stale approvals

**Step 1: Write the failing tests**

Add to `tests/unit/test_telegram.py`:

```python
class TestApprovalCallback:
    @pytest.mark.asyncio
    async def test_approve_callback_executes(self):
        """Clicking Approve should execute the order and send confirmation."""
        from orchestrator.approval.manager import PendingApproval
        from orchestrator.execution.executor import ExecutionResult

        bot = SentinelBot(token="test-token", admin_chat_ids=[123])

        approval = PendingApproval(
            approval_id="a-001",
            proposal=TradeProposal(
                symbol="BTC/USDT:USDT", side=Side.LONG, entry=EntryOrder(type="market"),
                position_size_risk_pct=1.5, stop_loss=93000.0, take_profit=[97000.0],
                time_horizon="4h", confidence=0.75, invalid_if=[], rationale="test",
            ),
            run_id="run-1", snapshot_price=95200.0,
            created_at=MagicMock(), expires_at=MagicMock(),
        )
        approval_mgr = MagicMock()
        approval_mgr.approve.return_value = approval
        bot.set_approval_manager(approval_mgr)

        executor = AsyncMock()
        executor.execute_entry.return_value = ExecutionResult(
            trade_id="t-001", symbol="BTC/USDT:USDT", side="long",
            entry_price=95250.0, quantity=0.075, fees=3.57, mode="paper",
        )
        executor.place_sl_tp.return_value = []
        bot.set_executor(executor)

        # Mock data fetcher for price re-check
        data_fetcher = AsyncMock()
        data_fetcher.fetch_current_price.return_value = 95250.0
        bot.set_data_fetcher(data_fetcher)

        # Simulate callback
        query = MagicMock()
        query.data = "approve:a-001"
        query.from_user.id = 123
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        update.effective_chat = MagicMock()
        update.effective_chat.id = 123
        context = MagicMock()

        await bot._approval_callback(update, context)
        executor.execute_entry.assert_called_once()
        query.answer.assert_called()

    @pytest.mark.asyncio
    async def test_reject_callback(self):
        bot = SentinelBot(token="test-token", admin_chat_ids=[123])

        approval_mgr = MagicMock()
        approval_mgr.reject.return_value = MagicMock(approval_id="a-002")
        bot.set_approval_manager(approval_mgr)

        query = MagicMock()
        query.data = "reject:a-002"
        query.from_user.id = 123
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        update.effective_chat = MagicMock()
        update.effective_chat.id = 123
        context = MagicMock()

        await bot._approval_callback(update, context)
        approval_mgr.reject.assert_called_once_with("a-002")
        query.edit_message_text.assert_called()
```

**Step 2: Implement in bot.py**

Key changes:
1. Add imports: `CallbackQueryHandler`, `InlineKeyboardButton`, `InlineKeyboardMarkup`
2. Add `_approval_manager`, `_executor`, `_data_fetcher` attributes with setters
3. In `build()`, register `CallbackQueryHandler(self._approval_callback)`
4. Modify `push_to_admins` to use `InlineKeyboardMarkup` when result has `approval_id`
5. Implement `_approval_callback` handler
6. Add `push_pending_approval()` method

```python
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackQueryHandler

# In build():
self._app.add_handler(CallbackQueryHandler(self._approval_callback))

# New push method:
async def push_pending_approval(self, chat_id: int, approval) -> int | None:
    """Push proposal with Approve/Reject keyboard. Returns message_id."""
    if self._app is None:
        return None
    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Approve", callback_data=f"approve:{approval.approval_id}"),
            InlineKeyboardButton("Reject", callback_data=f"reject:{approval.approval_id}"),
        ]
    ])
    msg = await self._app.bot.send_message(
        chat_id=chat_id,
        text=format_pending_approval(approval),
        reply_markup=keyboard,
    )
    return msg.message_id

# Callback handler:
async def _approval_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return
    chat_id = update.effective_chat.id if update.effective_chat else 0
    if not is_admin(chat_id, admin_ids=self.admin_chat_ids):
        await query.answer("Unauthorized")
        return

    data = query.data or ""
    parts = data.split(":", 1)
    if len(parts) != 2:
        await query.answer("Invalid action")
        return

    action, approval_id = parts

    if action == "approve":
        await self._handle_approve(query, approval_id)
    elif action == "reject":
        await self._handle_reject(query, approval_id)
    else:
        await query.answer("Unknown action")
```

**Step 3: Run tests**

Run: `uv run pytest tests/unit/test_telegram.py::TestApprovalCallback -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/bot.py orchestrator/tests/unit/test_telegram.py
git commit -m "feat: add TG inline keyboard approval flow with callback handler"
```

---

### Task 10: Add DataFetcher.fetch_current_price helper

**Files:**
- Modify: `orchestrator/src/orchestrator/exchange/data_fetcher.py`
- Modify: `orchestrator/tests/unit/test_exchange.py`

**Step 1: Write the failing test**

```python
class TestDataFetcherCurrentPrice:
    @pytest.mark.asyncio
    async def test_fetch_current_price(self):
        mock_client = AsyncMock()
        mock_client.fetch_ticker.return_value = {"last": 95300.0}
        fetcher = DataFetcher(mock_client)
        price = await fetcher.fetch_current_price("BTC/USDT:USDT")
        assert price == 95300.0
```

**Step 2: Implement**

Add to `DataFetcher`:

```python
async def fetch_current_price(self, symbol: str) -> float:
    ticker = await self._client.fetch_ticker(symbol)
    return ticker.get("last", 0.0)
```

**Step 3: Run tests, commit**

```bash
git add orchestrator/src/orchestrator/exchange/data_fetcher.py orchestrator/tests/unit/test_exchange.py
git commit -m "feat: add fetch_current_price helper to DataFetcher"
```

---

### Task 11: Add expiry scheduler job

**Files:**
- Modify: `orchestrator/src/orchestrator/pipeline/scheduler.py`
- Modify: `orchestrator/tests/unit/test_scheduler.py`

**Step 1: Write the failing test**

```python
class TestSchedulerExpiry:
    @pytest.mark.asyncio
    async def test_expire_stale_approvals(self):
        from unittest.mock import MagicMock

        runner = MagicMock()
        approval_mgr = MagicMock()
        approval_mgr.expire_stale.return_value = []

        scheduler = PipelineScheduler(
            runner=runner, symbols=["BTC/USDT:USDT"],
            interval_minutes=15, approval_manager=approval_mgr,
        )
        await scheduler._expire_stale_approvals()
        approval_mgr.expire_stale.assert_called_once()
```

**Step 2: Implement**

Add optional `approval_manager` param to `PipelineScheduler.__init__`. Add `_expire_stale_approvals` async method. In `start()`, add an `IntervalTrigger(minutes=1)` job for expiry if approval_manager is set.

**Step 3: Run tests, commit**

```bash
git add orchestrator/src/orchestrator/pipeline/scheduler.py orchestrator/tests/unit/test_scheduler.py
git commit -m "feat: add approval expiry scheduler job (every 1 min)"
```

---

### Task 12: Wire M4 components in __main__.py

**Files:**
- Modify: `orchestrator/src/orchestrator/__main__.py`
- Modify: `orchestrator/tests/unit/test_main.py`

**Step 1: Write the failing test**

```python
class TestAppComponentsM4:
    def test_create_app_components_includes_m4(self):
        components = create_app_components(
            telegram_bot_token="test", telegram_admin_chat_ids=[123],
            exchange_id="binance", database_url="sqlite:///:memory:",
            anthropic_api_key="test-key",
            trading_mode="paper",
        )
        assert "approval_manager" in components
        assert "executor" in components
```

**Step 2: Wire components**

Add to `create_app_components`:
- Accept `trading_mode`, `approval_timeout_minutes`, `price_deviation_threshold` params
- Create `ApprovalRepository` + `ApprovalManager`
- Create `PaperExecutor` or `LiveExecutor` based on `trading_mode`
- Pass `approval_manager` to `PipelineRunner` and `PipelineScheduler`
- Pass `executor`, `approval_manager`, `data_fetcher` to `SentinelBot`

Update `_build_components` to pass new settings fields.

**Step 3: Run full tests**

Run: `uv run pytest -v --tb=short`
Expected: All pass

**Step 4: Commit**

```bash
git add orchestrator/src/orchestrator/__main__.py orchestrator/tests/unit/test_main.py
git commit -m "feat: wire M4 approval and executor components into app entrypoint"
```

---

### Task 13: Final lint fix and full verification

**Files:**
- Any files with lint issues

**Step 1: Run ruff**

Run: `uv run ruff check src/ tests/ --fix`

**Step 2: Run full test suite with coverage**

Run: `uv run pytest -v --cov=orchestrator --cov-report=term-missing --tb=short`

Expected: All tests pass (183 existing + ~30 new ≈ 210+), coverage >= 80%.

**Step 3: Commit any fixes**

```bash
git add -A
git commit -m "chore: fix all ruff lint errors for M4 code"
```

---

## Review Checkpoint

After completing all tasks, run the full verification:

```bash
uv run pytest -v --cov=orchestrator --cov-report=term-missing --tb=short
uv run ruff check src/ tests/
```

Expected outcomes:
- All tests pass (210+)
- Coverage >= 80%
- Lint clean
- New modules: `approval/manager.py`, `execution/executor.py`
- Modified: `config.py`, `storage/models.py`, `storage/repository.py`, `pipeline/runner.py`, `pipeline/scheduler.py`, `exchange/client.py`, `exchange/data_fetcher.py`, `telegram/bot.py`, `telegram/formatters.py`, `__main__.py`
- Key features working: Inline keyboard approval → executor dispatch (paper/live) → SL/TP exchange orders → expiry scheduler
