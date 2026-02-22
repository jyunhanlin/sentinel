# Paper Trading Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add leverage/margin simulation, manual position management (add/reduce/close), and history pagination to the paper trading engine.

**Architecture:** Extend the existing PaperEngine with leverage-aware position sizing and margin tracking. Add new Telegram inline button flows for manual operations. Keep the single-position-per-trade model with average price updates on add. All models remain frozen (create new instances, never mutate).

**Tech Stack:** Python 3.12+, Pydantic v2 (frozen), SQLModel/SQLite, python-telegram-bot, structlog, pytest

---

### Task 1: Add Leverage Config

**Files:**
- Modify: `orchestrator/src/orchestrator/config.py:46-54`
- Test: `orchestrator/tests/unit/test_config.py` (existing)

**Step 1: Write the failing test**

```python
# In test_config.py, add:
def test_paper_leverage_defaults(monkeypatch, isolated_env):
    """New leverage config fields have sensible defaults."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")
    monkeypatch.setenv("TELEGRAM_ADMIN_CHAT_IDS", "[123]")
    s = Settings()
    assert s.paper_default_leverage == 10
    assert s.paper_maintenance_margin_rate == 0.5
    assert s.paper_leverage_options == [5, 10, 20, 50]
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_config.py::test_paper_leverage_defaults -v`
Expected: FAIL — `AttributeError: 'Settings' has no attribute 'paper_default_leverage'`

**Step 3: Write minimal implementation**

Add to `config.py` after line 49 (`paper_maker_fee_rate`):

```python
    paper_default_leverage: int = 10
    paper_maintenance_margin_rate: float = 0.5  # %
    paper_leverage_options: list[int] = Field(default=[5, 10, 20, 50])
```

**Step 4: Run test to verify it passes**

Run: `cd orchestrator && uv run pytest tests/unit/test_config.py::test_paper_leverage_defaults -v`
Expected: PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/config.py orchestrator/tests/unit/test_config.py
git commit -m "feat: add leverage config options for paper trading"
```

---

### Task 2: Extend PaperTradeRecord with Leverage Fields

**Files:**
- Modify: `orchestrator/src/orchestrator/storage/models.py:46-67`

**Step 1: Add new fields to PaperTradeRecord**

After line 64 (`tp_order_id`), add:

```python
    leverage: int = 1
    margin: float = 0.0
    liquidation_price: float = 0.0
    close_reason: str = ""  # "sl" | "tp" | "liquidation" | "manual" | "partial_reduce"
    stop_loss: float = 0.0
    take_profit_json: str = "[]"  # JSON-encoded list[float]
```

Note: `stop_loss` and `take_profit_json` fix the existing bug where SL/TP data is lost on `rebuild_from_db()` (lines 170-171 of paper_engine.py).

**Step 2: Verify existing tests still pass**

Run: `cd orchestrator && uv run pytest tests/unit/ -v --tb=short`
Expected: All existing tests PASS (new fields have defaults, so backward-compatible)

**Step 3: Commit**

```bash
git add orchestrator/src/orchestrator/storage/models.py
git commit -m "feat: add leverage, margin, SL/TP fields to PaperTradeRecord"
```

---

### Task 3: Extend Repository with Leverage-Aware Save and Paginated Query

**Files:**
- Modify: `orchestrator/src/orchestrator/storage/repository.py:123-224`
- Test: `orchestrator/tests/unit/test_repository.py` (existing or create)

**Step 1: Write failing tests**

```python
# tests/unit/test_repository.py
import json
from datetime import UTC, datetime

import pytest
from sqlmodel import Session, SQLModel, create_engine

from orchestrator.storage.models import PaperTradeRecord
from orchestrator.storage.repository import PaperTradeRepository


@pytest.fixture
def repo():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    session = Session(engine)
    return PaperTradeRepository(session)


class TestPaperTradeRepoLeverage:
    def test_save_trade_with_leverage(self, repo):
        record = repo.save_trade(
            trade_id="t1", proposal_id="p1", symbol="BTC/USDT:USDT",
            side="long", entry_price=68000.0, quantity=0.1, risk_pct=1.0,
            leverage=10, margin=680.0, liquidation_price=61880.0,
            stop_loss=67000.0, take_profit=[70000.0],
        )
        assert record.leverage == 10
        assert record.margin == 680.0
        assert record.liquidation_price == 61880.0
        assert record.stop_loss == 67000.0
        assert json.loads(record.take_profit_json) == [70000.0]

    def test_get_closed_paginated_basic(self, repo):
        # Create 7 closed trades
        for i in range(7):
            repo.save_trade(
                trade_id=f"t{i}", proposal_id=f"p{i}", symbol="BTC/USDT:USDT",
                side="long", entry_price=68000.0, quantity=0.1,
            )
            repo.update_trade_closed(f"t{i}", exit_price=69000.0, pnl=100.0, fees=3.4)

        trades, total = repo.get_closed_paginated(offset=0, limit=5)
        assert len(trades) == 5
        assert total == 7

    def test_get_closed_paginated_with_symbol_filter(self, repo):
        repo.save_trade(
            trade_id="t1", proposal_id="p1", symbol="BTC/USDT:USDT",
            side="long", entry_price=68000.0, quantity=0.1,
        )
        repo.save_trade(
            trade_id="t2", proposal_id="p2", symbol="ETH/USDT:USDT",
            side="short", entry_price=2400.0, quantity=1.0,
        )
        repo.update_trade_closed("t1", exit_price=69000.0, pnl=100.0, fees=3.4)
        repo.update_trade_closed("t2", exit_price=2300.0, pnl=100.0, fees=1.2)

        trades, total = repo.get_closed_paginated(offset=0, limit=5, symbol="BTC/USDT:USDT")
        assert len(trades) == 1
        assert total == 1
        assert trades[0].symbol == "BTC/USDT:USDT"

    def test_update_trade_position_modified(self, repo):
        repo.save_trade(
            trade_id="t1", proposal_id="p1", symbol="BTC/USDT:USDT",
            side="long", entry_price=68000.0, quantity=0.1,
            leverage=10, margin=680.0, liquidation_price=61880.0,
        )
        updated = repo.update_trade_position(
            trade_id="t1", entry_price=68500.0, quantity=0.15,
            margin=1027.5, liquidation_price=61700.0,
        )
        assert updated.entry_price == 68500.0
        assert updated.quantity == 0.15
        assert updated.margin == 1027.5

    def test_update_trade_partial_close(self, repo):
        repo.save_trade(
            trade_id="t1", proposal_id="p1", symbol="BTC/USDT:USDT",
            side="long", entry_price=68000.0, quantity=0.1,
            leverage=10, margin=680.0, liquidation_price=61880.0,
        )
        updated = repo.update_trade_partial_close(
            trade_id="t1", remaining_qty=0.05, remaining_margin=340.0,
        )
        assert updated.quantity == 0.05
        assert updated.margin == 340.0
        assert updated.status == "open"  # still open
```

**Step 2: Run tests to verify they fail**

Run: `cd orchestrator && uv run pytest tests/unit/test_repository.py::TestPaperTradeRepoLeverage -v`
Expected: FAIL

**Step 3: Implement repository changes**

Update `save_trade` signature in `repository.py` to accept new fields:

```python
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
    leverage: int = 1,
    margin: float = 0.0,
    liquidation_price: float = 0.0,
    stop_loss: float = 0.0,
    take_profit: list[float] | None = None,
) -> PaperTradeRecord:
    import json
    record = PaperTradeRecord(
        trade_id=trade_id, proposal_id=proposal_id, symbol=symbol,
        side=side, entry_price=entry_price, quantity=quantity,
        risk_pct=risk_pct, leverage=leverage, margin=margin,
        liquidation_price=liquidation_price, stop_loss=stop_loss,
        take_profit_json=json.dumps(take_profit or []),
    )
    self._session.add(record)
    self._session.commit()
    self._session.refresh(record)
    return record
```

Add new methods:

```python
def get_closed_paginated(
    self,
    offset: int = 0,
    limit: int = 5,
    symbol: str | None = None,
) -> tuple[list[PaperTradeRecord], int]:
    """Return (trades, total_count) with pagination and optional symbol filter."""
    from sqlmodel import func
    base = select(PaperTradeRecord).where(PaperTradeRecord.status == "closed")
    count_stmt = select(func.count()).select_from(PaperTradeRecord).where(
        PaperTradeRecord.status == "closed"
    )
    if symbol:
        base = base.where(PaperTradeRecord.symbol == symbol)
        count_stmt = count_stmt.where(PaperTradeRecord.symbol == symbol)
    total = self._session.exec(count_stmt).one()
    trades = list(
        self._session.exec(
            base.order_by(PaperTradeRecord.closed_at.desc()).offset(offset).limit(limit)
        ).all()
    )
    return trades, total

def update_trade_position(
    self,
    trade_id: str,
    *,
    entry_price: float,
    quantity: float,
    margin: float,
    liquidation_price: float,
) -> PaperTradeRecord:
    """Update position after add (avg price, qty, margin)."""
    trade = self.get_by_trade_id(trade_id)
    if trade is None:
        raise ValueError(f"Trade {trade_id} not found")
    trade.entry_price = entry_price
    trade.quantity = quantity
    trade.margin = margin
    trade.liquidation_price = liquidation_price
    self._session.add(trade)
    self._session.commit()
    self._session.refresh(trade)
    return trade

def update_trade_partial_close(
    self,
    trade_id: str,
    *,
    remaining_qty: float,
    remaining_margin: float,
) -> PaperTradeRecord:
    """Update position after partial reduce."""
    trade = self.get_by_trade_id(trade_id)
    if trade is None:
        raise ValueError(f"Trade {trade_id} not found")
    trade.quantity = remaining_qty
    trade.margin = remaining_margin
    self._session.add(trade)
    self._session.commit()
    self._session.refresh(trade)
    return trade

def update_trade_close_reason(
    self,
    trade_id: str,
    *,
    reason: str,
) -> PaperTradeRecord:
    """Set close_reason on a trade."""
    trade = self.get_by_trade_id(trade_id)
    if trade is None:
        raise ValueError(f"Trade {trade_id} not found")
    trade.close_reason = reason
    self._session.add(trade)
    self._session.commit()
    self._session.refresh(trade)
    return trade
```

**Step 4: Run tests to verify they pass**

Run: `cd orchestrator && uv run pytest tests/unit/test_repository.py::TestPaperTradeRepoLeverage -v`
Expected: PASS

**Step 5: Run all tests**

Run: `cd orchestrator && uv run pytest tests/unit/ -v --tb=short`
Expected: All PASS

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/storage/repository.py orchestrator/tests/unit/test_repository.py
git commit -m "feat: add leverage-aware save and paginated query to trade repository"
```

---

### Task 4: Extend Position Model with Leverage Fields

**Files:**
- Modify: `orchestrator/src/orchestrator/exchange/paper_engine.py:20-31`
- Test: `orchestrator/tests/unit/test_paper_engine.py`

**Step 1: Write failing test**

```python
# Add to test_paper_engine.py
def test_position_has_leverage_fields():
    from orchestrator.exchange.paper_engine import Position
    pos = Position(
        trade_id="t1", proposal_id="p1", symbol="BTC/USDT:USDT",
        side=Side.LONG, entry_price=68000.0, quantity=0.1,
        stop_loss=67000.0, take_profit=[70000.0],
        opened_at=datetime.now(UTC), risk_pct=1.0,
        leverage=10, margin=680.0, liquidation_price=61880.0,
    )
    assert pos.leverage == 10
    assert pos.margin == 680.0
    assert pos.liquidation_price == 61880.0
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_paper_engine.py::test_position_has_leverage_fields -v`
Expected: FAIL — `unexpected keyword argument 'leverage'`

**Step 3: Implement**

Add to `Position` class in `paper_engine.py` (after `risk_pct` field):

```python
    leverage: int = 1
    margin: float = 0.0
    liquidation_price: float = 0.0
```

**Step 4: Run test to verify it passes**

Run: `cd orchestrator && uv run pytest tests/unit/test_paper_engine.py::test_position_has_leverage_fields -v`
Expected: PASS

**Step 5: Run all existing tests**

Run: `cd orchestrator && uv run pytest tests/unit/test_paper_engine.py -v`
Expected: All PASS (defaults preserve backward compat)

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/exchange/paper_engine.py orchestrator/tests/unit/test_paper_engine.py
git commit -m "feat: add leverage, margin, liquidation_price to Position model"
```

---

### Task 5: Add Margin Calculation Helpers to PaperEngine

**Files:**
- Modify: `orchestrator/src/orchestrator/exchange/paper_engine.py`
- Test: `orchestrator/tests/unit/test_paper_engine.py`

**Step 1: Write failing tests**

```python
# Add to test_paper_engine.py
class TestMarginCalculation:
    def _make_engine(self):
        return PaperEngine(
            initial_equity=10000.0,
            taker_fee_rate=0.0005,
            position_sizer=RiskPercentSizer(),
            trade_repo=MagicMock(),
            snapshot_repo=MagicMock(),
            maintenance_margin_rate=0.5,
        )

    def test_calculate_margin(self):
        engine = self._make_engine()
        margin = engine.calculate_margin(
            quantity=0.1, price=68000.0, leverage=10,
        )
        assert margin == pytest.approx(680.0)

    def test_calculate_liquidation_price_long(self):
        engine = self._make_engine()
        liq = engine.calculate_liquidation_price(
            entry_price=68000.0, leverage=10, side=Side.LONG,
        )
        # liq = 68000 * (1 - 1/10 + 0.005) = 68000 * 0.905 = 61540.0
        assert liq == pytest.approx(61540.0)

    def test_calculate_liquidation_price_short(self):
        engine = self._make_engine()
        liq = engine.calculate_liquidation_price(
            entry_price=68000.0, leverage=10, side=Side.SHORT,
        )
        # liq = 68000 * (1 + 1/10 - 0.005) = 68000 * 1.095 = 74460.0
        assert liq == pytest.approx(74460.0)

    def test_available_balance(self):
        engine = self._make_engine()
        assert engine.available_balance == 10000.0
        # After opening, available should decrease by margin

    def test_used_margin(self):
        engine = self._make_engine()
        assert engine.used_margin == 0.0
```

**Step 2: Run tests to verify they fail**

Run: `cd orchestrator && uv run pytest tests/unit/test_paper_engine.py::TestMarginCalculation -v`
Expected: FAIL

**Step 3: Implement**

Add `maintenance_margin_rate` param to `PaperEngine.__init__` and helper methods:

```python
def __init__(
    self,
    *,
    initial_equity: float,
    taker_fee_rate: float,
    position_sizer: PositionSizer,
    trade_repo: PaperTradeRepository,
    snapshot_repo: AccountSnapshotRepository,
    stats_calculator: StatsCalculator | None = None,
    maintenance_margin_rate: float = 0.5,  # NEW
) -> None:
    # ... existing init ...
    self._maintenance_margin_rate = maintenance_margin_rate

def calculate_margin(self, *, quantity: float, price: float, leverage: int) -> float:
    return quantity * price / leverage

def calculate_liquidation_price(
    self, *, entry_price: float, leverage: int, side: Side,
) -> float:
    mmr = self._maintenance_margin_rate / 100
    if side == Side.LONG:
        return entry_price * (1 - 1 / leverage + mmr)
    return entry_price * (1 + 1 / leverage - mmr)

@property
def used_margin(self) -> float:
    return sum(p.margin for p in self._positions)

@property
def available_balance(self) -> float:
    return self.equity - self.used_margin
```

**Step 4: Run tests to verify they pass**

Run: `cd orchestrator && uv run pytest tests/unit/test_paper_engine.py::TestMarginCalculation -v`
Expected: PASS

**Step 5: Run all tests**

Run: `cd orchestrator && uv run pytest tests/unit/ -v --tb=short`
Expected: All PASS

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/exchange/paper_engine.py orchestrator/tests/unit/test_paper_engine.py
git commit -m "feat: add margin calculation helpers and available_balance to PaperEngine"
```

---

### Task 6: Update open_position to Support Leverage

**Files:**
- Modify: `orchestrator/src/orchestrator/exchange/paper_engine.py:86-138`
- Test: `orchestrator/tests/unit/test_paper_engine.py`

**Step 1: Write failing test**

```python
class TestOpenPositionWithLeverage:
    def _make_engine(self, **kwargs):
        defaults = dict(
            initial_equity=10000.0,
            taker_fee_rate=0.0005,
            position_sizer=RiskPercentSizer(),
            trade_repo=MagicMock(),
            snapshot_repo=MagicMock(),
            maintenance_margin_rate=0.5,
        )
        defaults.update(kwargs)
        return PaperEngine(**defaults)

    def test_open_position_with_leverage(self):
        engine = self._make_engine()
        proposal = _make_proposal(stop_loss=67000.0, take_profit=[70000.0])
        pos = engine.open_position(proposal, current_price=68000.0, leverage=10)
        assert pos.leverage == 10
        assert pos.margin == pytest.approx(pos.quantity * 68000.0 / 10)
        assert pos.liquidation_price > 0

    def test_open_position_default_leverage_1(self):
        engine = self._make_engine()
        proposal = _make_proposal()
        pos = engine.open_position(proposal, current_price=95000.0)
        assert pos.leverage == 1

    def test_open_position_insufficient_margin(self):
        engine = self._make_engine(initial_equity=100.0)
        proposal = _make_proposal(stop_loss=67000.0, take_profit=[70000.0])
        with pytest.raises(ValueError, match="Insufficient margin"):
            engine.open_position(proposal, current_price=68000.0, leverage=2)
```

**Step 2: Run tests to verify they fail**

Run: `cd orchestrator && uv run pytest tests/unit/test_paper_engine.py::TestOpenPositionWithLeverage -v`

**Step 3: Implement**

Update `open_position` signature to accept `leverage: int = 1` and compute margin/liq:

```python
def open_position(
    self, proposal: TradeProposal, current_price: float, leverage: int = 1,
) -> Position:
    if proposal.stop_loss is None:
        raise ValueError(...)

    stop_loss = proposal.stop_loss
    quantity = self._position_sizer.calculate(...)

    margin = self.calculate_margin(quantity=quantity, price=current_price, leverage=leverage)
    if margin > self.available_balance:
        raise ValueError(
            f"Insufficient margin: need ${margin:,.2f}, available ${self.available_balance:,.2f}"
        )

    liquidation_price = self.calculate_liquidation_price(
        entry_price=current_price, leverage=leverage, side=proposal.side,
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
        stop_loss=stop_loss,
        take_profit=proposal.take_profit,
        opened_at=datetime.now(UTC),
        risk_pct=proposal.position_size_risk_pct,
        leverage=leverage,
        margin=margin,
        liquidation_price=liquidation_price,
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
        leverage=leverage,
        margin=margin,
        liquidation_price=liquidation_price,
        stop_loss=stop_loss,
        take_profit=proposal.take_profit,
    )
    # ... logging ...
    return position
```

Also update `rebuild_from_db` to restore leverage/margin/liq/SL/TP from DB.

**Step 4: Run tests**

Run: `cd orchestrator && uv run pytest tests/unit/test_paper_engine.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/exchange/paper_engine.py orchestrator/tests/unit/test_paper_engine.py
git commit -m "feat: add leverage support to open_position with margin validation"
```

---

### Task 7: Add Liquidation Check to check_sl_tp

**Files:**
- Modify: `orchestrator/src/orchestrator/exchange/paper_engine.py:140-201`
- Test: `orchestrator/tests/unit/test_paper_engine.py`

**Step 1: Write failing tests**

```python
class TestLiquidation:
    def _make_engine(self):
        return PaperEngine(
            initial_equity=10000.0,
            taker_fee_rate=0.0005,
            position_sizer=RiskPercentSizer(),
            trade_repo=MagicMock(),
            snapshot_repo=MagicMock(),
            maintenance_margin_rate=0.5,
        )

    def test_liquidation_triggers_before_sl_long(self):
        engine = self._make_engine()
        proposal = _make_proposal(stop_loss=67000.0, take_profit=[70000.0])
        pos = engine.open_position(proposal, current_price=68000.0, leverage=10)
        # Liq price ~61540, SL=67000. Price crashes below liq
        closed = engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=61000.0)
        assert len(closed) == 1
        assert closed[0].reason == "liquidation"

    def test_sl_triggers_when_above_liq_long(self):
        engine = self._make_engine()
        proposal = _make_proposal(stop_loss=67000.0, take_profit=[70000.0])
        engine.open_position(proposal, current_price=68000.0, leverage=10)
        # Price hits SL but above liq
        closed = engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=66500.0)
        assert len(closed) == 1
        assert closed[0].reason == "sl"

    def test_liquidation_short(self):
        engine = self._make_engine()
        proposal = _make_proposal(
            side=Side.SHORT, stop_loss=70000.0, take_profit=[65000.0],
        )
        pos = engine.open_position(proposal, current_price=68000.0, leverage=10)
        # Liq price ~74460, push above
        closed = engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=75000.0)
        assert len(closed) == 1
        assert closed[0].reason == "liquidation"
```

**Step 2: Run tests**

Expected: FAIL — `reason` won't be "liquidation" yet

**Step 3: Implement**

Update `_check_trigger` to check liquidation first:

```python
def _check_trigger(
    self, pos: Position, current_price: float
) -> tuple[float, str] | None:
    # Liquidation check (highest priority)
    if pos.leverage > 1:
        if pos.side == Side.LONG and current_price <= pos.liquidation_price:
            return current_price, "liquidation"
        if pos.side == Side.SHORT and current_price >= pos.liquidation_price:
            return current_price, "liquidation"

    # SL/TP checks (existing logic)
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
```

Update `_close` to handle `reason="liquidation"` — PnL = negative margin (total loss of margin).

**Step 4: Run tests**

Run: `cd orchestrator && uv run pytest tests/unit/test_paper_engine.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/exchange/paper_engine.py orchestrator/tests/unit/test_paper_engine.py
git commit -m "feat: add liquidation check with highest priority in check_sl_tp"
```

---

### Task 8: Add Manual Position Operations (add/reduce/close)

**Files:**
- Modify: `orchestrator/src/orchestrator/exchange/paper_engine.py`
- Test: `orchestrator/tests/unit/test_paper_engine.py`

**Step 1: Write failing tests**

```python
class TestManualOperations:
    def _make_engine(self):
        return PaperEngine(
            initial_equity=10000.0,
            taker_fee_rate=0.0005,
            position_sizer=RiskPercentSizer(),
            trade_repo=MagicMock(),
            snapshot_repo=MagicMock(),
            maintenance_margin_rate=0.5,
        )

    def test_add_to_position(self):
        engine = self._make_engine()
        proposal = _make_proposal(stop_loss=67000.0, take_profit=[70000.0])
        pos = engine.open_position(proposal, current_price=68000.0, leverage=10)
        old_qty = pos.quantity
        old_entry = pos.entry_price

        updated = engine.add_to_position(
            trade_id=pos.trade_id, risk_pct=1.0,
            current_price=69000.0,
        )
        assert updated.quantity > old_qty
        # Avg price should be between old and new
        assert old_entry < updated.entry_price < 69000.0
        assert updated.margin > pos.margin
        assert len(engine.get_open_positions()) == 1  # still one position

    def test_reduce_position_50pct(self):
        engine = self._make_engine()
        proposal = _make_proposal(stop_loss=67000.0, take_profit=[70000.0])
        pos = engine.open_position(proposal, current_price=68000.0, leverage=10)
        original_qty = pos.quantity

        result = engine.reduce_position(
            trade_id=pos.trade_id, pct=50.0, current_price=69000.0,
        )
        assert result.quantity == pytest.approx(original_qty * 0.5, rel=0.01)
        assert result.pnl > 0  # price went up for long
        remaining = engine.get_open_positions()
        assert len(remaining) == 1
        assert remaining[0].quantity == pytest.approx(original_qty * 0.5, rel=0.01)

    def test_close_position(self):
        engine = self._make_engine()
        proposal = _make_proposal(stop_loss=67000.0, take_profit=[70000.0])
        pos = engine.open_position(proposal, current_price=68000.0, leverage=10)

        result = engine.close_position(
            trade_id=pos.trade_id, current_price=69000.0,
        )
        assert result.reason == "manual"
        assert len(engine.get_open_positions()) == 0

    def test_reduce_100pct_closes_position(self):
        engine = self._make_engine()
        proposal = _make_proposal(stop_loss=67000.0, take_profit=[70000.0])
        pos = engine.open_position(proposal, current_price=68000.0, leverage=10)

        result = engine.reduce_position(
            trade_id=pos.trade_id, pct=100.0, current_price=69000.0,
        )
        assert len(engine.get_open_positions()) == 0

    def test_add_insufficient_margin(self):
        engine = PaperEngine(
            initial_equity=1000.0,
            taker_fee_rate=0.0005,
            position_sizer=RiskPercentSizer(),
            trade_repo=MagicMock(),
            snapshot_repo=MagicMock(),
            maintenance_margin_rate=0.5,
        )
        proposal = _make_proposal(stop_loss=67000.0, take_profit=[70000.0], risk_pct=0.5)
        pos = engine.open_position(proposal, current_price=68000.0, leverage=10)

        with pytest.raises(ValueError, match="Insufficient margin"):
            engine.add_to_position(
                trade_id=pos.trade_id, risk_pct=50.0, current_price=69000.0,
            )

    def test_position_not_found(self):
        engine = self._make_engine()
        with pytest.raises(ValueError, match="not found"):
            engine.close_position(trade_id="nonexistent", current_price=69000.0)

    def test_get_position_with_pnl_long(self):
        engine = self._make_engine()
        proposal = _make_proposal(stop_loss=67000.0, take_profit=[70000.0])
        pos = engine.open_position(proposal, current_price=68000.0, leverage=10)

        info = engine.get_position_with_pnl(trade_id=pos.trade_id, current_price=69000.0)
        assert info["unrealized_pnl"] > 0
        assert info["roe_pct"] > 0
        assert info["pnl_pct"] > 0
```

**Step 2: Run tests**

Expected: FAIL — methods don't exist

**Step 3: Implement all three methods**

```python
def add_to_position(
    self, *, trade_id: str, risk_pct: float, current_price: float,
) -> Position:
    pos = self._find_position(trade_id)
    add_qty = self._position_sizer.calculate(
        equity=self.equity, risk_pct=risk_pct,
        entry_price=current_price, stop_loss=pos.stop_loss,
    )
    add_margin = self.calculate_margin(
        quantity=add_qty, price=current_price, leverage=pos.leverage,
    )
    if add_margin > self.available_balance:
        raise ValueError(
            f"Insufficient margin: need ${add_margin:,.2f}, "
            f"available ${self.available_balance:,.2f}"
        )

    # Fee
    fee = add_qty * current_price * self._taker_fee_rate
    self._total_fees += fee

    # Average price
    total_qty = pos.quantity + add_qty
    new_entry = (pos.quantity * pos.entry_price + add_qty * current_price) / total_qty
    new_margin = pos.margin + add_margin
    new_liq = self.calculate_liquidation_price(
        entry_price=new_entry, leverage=pos.leverage, side=pos.side,
    )

    new_pos = Position(
        trade_id=pos.trade_id,
        proposal_id=pos.proposal_id,
        symbol=pos.symbol,
        side=pos.side,
        entry_price=new_entry,
        quantity=total_qty,
        stop_loss=pos.stop_loss,
        take_profit=pos.take_profit,
        opened_at=pos.opened_at,
        risk_pct=pos.risk_pct + risk_pct,
        leverage=pos.leverage,
        margin=new_margin,
        liquidation_price=new_liq,
    )
    self._replace_position(pos.trade_id, new_pos)

    self._trade_repo.update_trade_position(
        trade_id=pos.trade_id, entry_price=new_entry,
        quantity=total_qty, margin=new_margin, liquidation_price=new_liq,
    )
    logger.info(
        "position_added", trade_id=pos.trade_id, add_qty=add_qty,
        new_avg_entry=new_entry, new_total_qty=total_qty, fee=fee,
    )
    return new_pos

def reduce_position(
    self, *, trade_id: str, pct: float, current_price: float,
) -> CloseResult:
    if pct >= 100.0:
        return self.close_position(trade_id=trade_id, current_price=current_price)

    pos = self._find_position(trade_id)
    close_qty = pos.quantity * pct / 100
    remaining_qty = pos.quantity - close_qty

    # PnL for closed portion
    if pos.side == Side.LONG:
        pnl = (current_price - pos.entry_price) * close_qty
    else:
        pnl = (pos.entry_price - current_price) * close_qty

    fee = close_qty * current_price * self._taker_fee_rate
    self._total_fees += fee
    self._closed_pnl += pnl

    # Release proportional margin
    remaining_margin = pos.margin * (remaining_qty / pos.quantity)

    new_pos = Position(
        trade_id=pos.trade_id,
        proposal_id=pos.proposal_id,
        symbol=pos.symbol,
        side=pos.side,
        entry_price=pos.entry_price,  # avg price unchanged
        quantity=remaining_qty,
        stop_loss=pos.stop_loss,
        take_profit=pos.take_profit,
        opened_at=pos.opened_at,
        risk_pct=pos.risk_pct * (remaining_qty / pos.quantity),
        leverage=pos.leverage,
        margin=remaining_margin,
        liquidation_price=pos.liquidation_price,
    )
    self._replace_position(pos.trade_id, new_pos)

    self._trade_repo.update_trade_partial_close(
        trade_id=pos.trade_id,
        remaining_qty=remaining_qty,
        remaining_margin=remaining_margin,
    )

    # Save partial close as separate record for history
    self._trade_repo.save_trade(
        trade_id=str(uuid.uuid4()),
        proposal_id=pos.proposal_id,
        symbol=pos.symbol,
        side=pos.side.value,
        entry_price=pos.entry_price,
        quantity=close_qty,
        risk_pct=0.0,
        leverage=pos.leverage,
        margin=0.0,
        liquidation_price=0.0,
    )
    # immediately close the partial record
    # (or alternatively, use close_reason on the main record)

    logger.info(
        "position_reduced", trade_id=pos.trade_id,
        close_qty=close_qty, remaining_qty=remaining_qty, pnl=pnl,
    )

    self._save_stats_snapshot()

    return CloseResult(
        trade_id=pos.trade_id,
        symbol=pos.symbol,
        side=pos.side,
        entry_price=pos.entry_price,
        exit_price=current_price,
        quantity=close_qty,
        pnl=pnl,
        fees=fee,
        reason="partial_reduce",
    )

def close_position(self, *, trade_id: str, current_price: float) -> CloseResult:
    pos = self._find_position(trade_id)
    self._positions = [p for p in self._positions if p.trade_id != trade_id]
    result = self._close(pos, exit_price=current_price, reason="manual")
    self._trade_repo.update_trade_close_reason(trade_id=trade_id, reason="manual")
    return result

def get_position_with_pnl(
    self, *, trade_id: str, current_price: float,
) -> dict:
    pos = self._find_position(trade_id)
    direction = 1 if pos.side == Side.LONG else -1
    unrealized_pnl = (current_price - pos.entry_price) * pos.quantity * direction
    notional = pos.entry_price * pos.quantity
    pnl_pct = (unrealized_pnl / notional * 100) if notional else 0
    roe_pct = (unrealized_pnl / pos.margin * 100) if pos.margin else pnl_pct
    return {
        "position": pos,
        "unrealized_pnl": unrealized_pnl,
        "pnl_pct": pnl_pct,
        "roe_pct": roe_pct,
    }

def _find_position(self, trade_id: str) -> Position:
    for p in self._positions:
        if p.trade_id == trade_id:
            return p
    raise ValueError(f"Position {trade_id} not found")

def _replace_position(self, trade_id: str, new_pos: Position) -> None:
    self._positions = [
        new_pos if p.trade_id == trade_id else p for p in self._positions
    ]
```

**Step 4: Run tests**

Run: `cd orchestrator && uv run pytest tests/unit/test_paper_engine.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/exchange/paper_engine.py orchestrator/tests/unit/test_paper_engine.py
git commit -m "feat: add manual position operations (add/reduce/close) to PaperEngine"
```

---

### Task 9: Update PaperExecutor for Leverage

**Files:**
- Modify: `orchestrator/src/orchestrator/execution/executor.py:54-78`
- Test: `orchestrator/tests/unit/test_executor.py` (create if needed)

**Step 1: Write failing test**

```python
# tests/unit/test_executor.py
import pytest
from unittest.mock import MagicMock, AsyncMock
from orchestrator.execution.executor import PaperExecutor, ExecutionResult
from orchestrator.models import EntryOrder, Side, TradeProposal

@pytest.mark.asyncio
async def test_paper_executor_passes_leverage():
    engine = MagicMock()
    position = MagicMock()
    position.trade_id = "t1"
    position.symbol = "BTC/USDT:USDT"
    position.side = Side.LONG
    position.entry_price = 68000.0
    position.quantity = 0.1
    position.leverage = 10
    position.margin = 680.0
    engine.open_position.return_value = position
    engine._taker_fee_rate = 0.0005

    executor = PaperExecutor(paper_engine=engine)
    proposal = TradeProposal(
        symbol="BTC/USDT:USDT", side=Side.LONG, entry=EntryOrder(type="market"),
        position_size_risk_pct=1.0, stop_loss=67000.0, take_profit=[70000.0],
        time_horizon="4h", confidence=0.7, invalid_if=[], rationale="test",
    )
    result = await executor.execute_entry(proposal, current_price=68000.0, leverage=10)
    engine.open_position.assert_called_once_with(proposal, 68000.0, leverage=10)
    assert result.mode == "paper"
```

**Step 2: Implement**

Update `PaperExecutor.execute_entry` signature:

```python
async def execute_entry(
    self, proposal: TradeProposal, current_price: float, leverage: int = 1,
) -> ExecutionResult:
    position = self._paper_engine.open_position(proposal, current_price, leverage=leverage)
    ...
```

Also update `OrderExecutor` ABC to include `leverage` parameter.

**Step 3: Run tests**

Run: `cd orchestrator && uv run pytest tests/unit/test_executor.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add orchestrator/src/orchestrator/execution/executor.py orchestrator/tests/unit/test_executor.py
git commit -m "feat: pass leverage through PaperExecutor to PaperEngine"
```

---

### Task 10: Update Formatters for Leverage Display

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/formatters.py`
- Test: `orchestrator/tests/unit/test_telegram.py`

**Step 1: Write failing tests**

```python
# Add to test_telegram.py
class TestLeverageFormatting:
    def test_format_position_card(self):
        from orchestrator.telegram.formatters import format_position_card
        info = {
            "position": MagicMock(
                symbol="BTC/USDT:USDT", side=Side.LONG, leverage=10,
                entry_price=68000.0, quantity=0.1, margin=680.0,
                liquidation_price=61540.0, stop_loss=67000.0,
                take_profit=[70000.0], opened_at=datetime.now(UTC),
                trade_id="t1",
            ),
            "unrealized_pnl": 125.0,
            "pnl_pct": 1.84,
            "roe_pct": 18.4,
        }
        text = format_position_card(info)
        assert "10x" in text
        assert "Margin" in text
        assert "Liq" in text
        assert "ROE" in text

    def test_format_status_with_positions(self):
        from orchestrator.telegram.formatters import format_account_overview
        text = format_account_overview(
            equity=10150.0, available=7320.0, used_margin=2680.0,
            initial_equity=10000.0, position_cards=["...card..."],
        )
        assert "Available" in text
        assert "Used Margin" in text

    def test_format_history_paginated(self):
        from orchestrator.telegram.formatters import format_history_paginated
        trade = MagicMock(
            symbol="BTC/USDT:USDT", side="long", leverage=10,
            entry_price=68000.0, exit_price=67000.0,
            pnl=-100.0, fees=3.4, opened_at=datetime.now(UTC),
            closed_at=datetime.now(UTC), close_reason="sl", margin=680.0,
        )
        text = format_history_paginated([trade], page=1, total_pages=3)
        assert "Page 1/3" in text
        assert "10x" in text
        assert "ROE" in text
```

**Step 2: Implement new formatter functions**

Add `format_position_card()`, `format_account_overview()`, `format_history_paginated()` to `formatters.py`.

**Step 3: Run tests**

Run: `cd orchestrator && uv run pytest tests/unit/test_telegram.py::TestLeverageFormatting -v`
Expected: PASS

**Step 4: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/formatters.py orchestrator/tests/unit/test_telegram.py
git commit -m "feat: add leverage-aware formatters for position cards and history"
```

---

### Task 11: Add Telegram Callback Handlers for Position Operations

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/bot.py`
- Test: `orchestrator/tests/unit/test_telegram.py`

This is the largest task. It adds callback handlers for:
1. Leverage selection during approve flow
2. Add/Reduce/Close buttons on position cards
3. Confirmation steps for all operations
4. History pagination callbacks

**Step 1: Write failing tests for leverage selection in approve flow**

```python
class TestApproveWithLeverage:
    @pytest.mark.asyncio
    async def test_approve_shows_leverage_selection(self):
        """When user clicks Approve, bot should show leverage options."""
        # Setup bot with mocks...
        # Simulate callback with "approve:{approval_id}"
        # Assert bot edits message with leverage buttons [5x] [10x] [20x] [50x]

    @pytest.mark.asyncio
    async def test_leverage_selection_shows_confirmation(self):
        """After selecting leverage, show order details for confirmation."""
        # Simulate callback "leverage:{approval_id}:10"
        # Assert confirmation message with margin, liq price info

    @pytest.mark.asyncio
    async def test_confirm_leverage_executes_trade(self):
        """Confirming leverage executes the trade with selected leverage."""
        # Simulate callback "confirm_leverage:{approval_id}:10"
        # Assert executor.execute_entry called with leverage=10
```

**Step 2: Write failing tests for position operation callbacks**

```python
class TestPositionOperationCallbacks:
    @pytest.mark.asyncio
    async def test_add_shows_risk_options(self):
        # Click "add:{trade_id}" → shows [0.5%] [1%] [2%]

    @pytest.mark.asyncio
    async def test_confirm_add_executes(self):
        # Click "confirm_add:{trade_id}:1.0" → calls engine.add_to_position

    @pytest.mark.asyncio
    async def test_reduce_shows_pct_options(self):
        # Click "reduce:{trade_id}" → shows [25%] [50%] [75%]

    @pytest.mark.asyncio
    async def test_confirm_reduce_executes(self):
        # Click "confirm_reduce:{trade_id}:50" → calls engine.reduce_position

    @pytest.mark.asyncio
    async def test_close_shows_confirmation(self):
        # Click "close:{trade_id}" → shows confirm/cancel

    @pytest.mark.asyncio
    async def test_confirm_close_executes(self):
        # Click "confirm_close:{trade_id}" → calls engine.close_position
```

**Step 3: Implement callback routing**

Update `_callback_router` in `bot.py` to handle new callback data patterns. Add handler methods:
- `_handle_leverage_select(query, approval_id, leverage)`
- `_handle_confirm_leverage(query, approval_id, leverage)`
- `_handle_add(query, trade_id)`
- `_handle_confirm_add(query, trade_id, risk_pct)`
- `_handle_reduce(query, trade_id)`
- `_handle_confirm_reduce(query, trade_id, pct)`
- `_handle_close(query, trade_id)`
- `_handle_confirm_close(query, trade_id)`
- `_handle_cancel(query)`

Modify existing `_handle_approve` to show leverage selection instead of immediately executing.

**Step 4: Run tests**

Run: `cd orchestrator && uv run pytest tests/unit/test_telegram.py -v`

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/bot.py orchestrator/tests/unit/test_telegram.py
git commit -m "feat: add TG callback handlers for leverage selection and position operations"
```

---

### Task 12: Update /status Handler with Position Cards

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/bot.py:288-308`
- Test: `orchestrator/tests/unit/test_telegram.py`

**Step 1: Write failing test**

```python
class TestStatusWithPositions:
    @pytest.mark.asyncio
    async def test_status_shows_position_cards_with_buttons(self):
        """Status should show each position with PnL and action buttons."""
        # Setup bot with paper_engine that has open positions
        # data_fetcher returns current prices
        # Assert reply contains position cards with inline keyboards
```

**Step 2: Implement**

Update `_status_handler` to:
1. Get open positions from `_paper_engine.get_open_positions()`
2. For each position, fetch current price via `_data_fetcher`
3. Call `_paper_engine.get_position_with_pnl()` for each
4. Format using `format_account_overview()` and `format_position_card()`
5. Send with inline keyboard buttons per position

**Step 3: Run tests & commit**

```bash
git commit -m "feat: show position cards with PnL and action buttons in /status"
```

---

### Task 13: Update /history Handler with Pagination and Filtering

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/bot.py:419-428`
- Test: `orchestrator/tests/unit/test_telegram.py`

**Step 1: Write failing tests**

```python
class TestHistoryPagination:
    @pytest.mark.asyncio
    async def test_history_shows_paginated(self):
        # trade_repo.get_closed_paginated returns (trades, total)
        # Assert reply has [Prev] [Page 1/N] [Next] buttons

    @pytest.mark.asyncio
    async def test_history_filter_by_symbol(self):
        # Callback "history:filter:BTC/USDT:USDT" re-renders with filter

    @pytest.mark.asyncio
    async def test_history_next_page(self):
        # Callback "history:page:2" shows page 2
```

**Step 2: Implement**

Update `_history_handler` to use `get_closed_paginated()` with pagination buttons. Add callback handlers for `history:page:{n}` and `history:filter:{symbol}`.

**Step 3: Run tests & commit**

```bash
git commit -m "feat: add pagination and symbol filtering to /history"
```

---

### Task 14: Integration Test — Full Approve-to-Close Flow

**Files:**
- Create: `orchestrator/tests/unit/test_paper_trading_flow.py`

**Step 1: Write integration test**

```python
class TestFullPaperTradingFlow:
    def test_approve_add_reduce_close_flow(self):
        """Full lifecycle: open with leverage → add → reduce → close."""
        engine = PaperEngine(
            initial_equity=10000.0,
            taker_fee_rate=0.0005,
            position_sizer=RiskPercentSizer(),
            trade_repo=MagicMock(),
            snapshot_repo=MagicMock(),
            maintenance_margin_rate=0.5,
        )

        # 1. Open position with 10x leverage
        proposal = _make_proposal(stop_loss=67000.0, take_profit=[70000.0])
        pos = engine.open_position(proposal, current_price=68000.0, leverage=10)
        assert pos.leverage == 10
        initial_qty = pos.quantity

        # 2. Add to position (1% risk)
        pos = engine.add_to_position(
            trade_id=pos.trade_id, risk_pct=1.0, current_price=69000.0,
        )
        assert pos.quantity > initial_qty
        assert 68000.0 < pos.entry_price < 69000.0  # avg price

        # 3. Reduce 50%
        result = engine.reduce_position(
            trade_id=pos.trade_id, pct=50.0, current_price=69500.0,
        )
        assert result.pnl > 0
        remaining = engine.get_open_positions()
        assert len(remaining) == 1

        # 4. Close remaining
        result = engine.close_position(
            trade_id=pos.trade_id, current_price=70000.0,
        )
        assert result.reason == "manual"
        assert len(engine.get_open_positions()) == 0
        assert engine.equity > 10000.0  # profitable trade
```

**Step 2: Run test**

Run: `cd orchestrator && uv run pytest tests/unit/test_paper_trading_flow.py -v`
Expected: PASS (if all prior tasks implemented correctly)

**Step 3: Commit**

```bash
git add orchestrator/tests/unit/test_paper_trading_flow.py
git commit -m "test: add full paper trading lifecycle integration test"
```

---

### Task 15: Final Verification

**Step 1: Run full test suite**

```bash
cd orchestrator && uv run pytest tests/unit/ -v --cov=orchestrator --cov-report=term-missing
```

Expected: All PASS, coverage >= 80%

**Step 2: Run linter**

```bash
cd orchestrator && uv run ruff check src/ tests/
```

Expected: No errors

**Step 3: Verify no regressions**

```bash
cd orchestrator && uv run pytest tests/unit/test_paper_engine.py tests/unit/test_telegram.py -v
```

Expected: All existing + new tests PASS

**Step 4: Final commit if any fixups needed**

```bash
git commit -m "chore: fix lint issues and test coverage gaps"
```
