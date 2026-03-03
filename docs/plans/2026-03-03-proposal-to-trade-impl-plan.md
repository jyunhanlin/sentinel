# Proposal → Trade → Position Management Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an ExecutionPlan middle layer that computes concrete trade numbers from proposals, redesign Telegram messages with a two-section format, enable parameter adjustment before execution, and add full position management via Telegram.

**Architecture:** An ExecutionPlanner (pure Python, no LLM) sits between the Proposer and execution. It reads equity from an EquityProvider interface and computes quantity, margin, liquidation price, and fees using fixed-amount margin sizing. The Telegram bot gains a parameter adjustment flow (hybrid buttons + text) and a position management menu (move SL, adjust TP, add/reduce/close).

**Tech Stack:** Python 3.12, Pydantic v2 (frozen models), python-telegram-bot, pytest + pytest-asyncio

---

### Task 1: EquityProvider Protocol + PaperEquityProvider

**Files:**
- Create: `orchestrator/src/orchestrator/execution/equity.py`
- Test: `orchestrator/tests/unit/test_equity_provider.py`

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_equity_provider.py
from __future__ import annotations

import pytest

from orchestrator.execution.equity import PaperEquityProvider


@pytest.mark.asyncio
async def test_get_equity_returns_engine_equity():
    class FakeEngine:
        @property
        def equity(self) -> float:
            return 10_000.0

    provider = PaperEquityProvider(engine=FakeEngine())
    assert await provider.get_equity() == 10_000.0


@pytest.mark.asyncio
async def test_get_available_margin_returns_engine_available():
    class FakeEngine:
        @property
        def equity(self) -> float:
            return 10_000.0

        @property
        def used_margin(self) -> float:
            return 1_000.0

    provider = PaperEquityProvider(engine=FakeEngine())
    assert await provider.get_available_margin() == 9_000.0
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_equity_provider.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.execution.equity'`

**Step 3: Write minimal implementation**

```python
# orchestrator/src/orchestrator/execution/equity.py
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EquityProvider(Protocol):
    async def get_equity(self) -> float: ...
    async def get_available_margin(self) -> float: ...


class PaperEquityProvider:
    """Reads equity from PaperEngine's simulated account."""

    def __init__(self, engine: object) -> None:
        self._engine = engine

    async def get_equity(self) -> float:
        return self._engine.equity  # type: ignore[attr-defined]

    async def get_available_margin(self) -> float:
        equity: float = self._engine.equity  # type: ignore[attr-defined]
        used: float = self._engine.used_margin  # type: ignore[attr-defined]
        return equity - used
```

**Step 4: Run test to verify it passes**

Run: `cd orchestrator && uv run pytest tests/unit/test_equity_provider.py -v`
Expected: PASS (2 tests)

**Step 5: Lint**

Run: `cd orchestrator && uv run ruff check src/orchestrator/execution/equity.py tests/unit/test_equity_provider.py`
Expected: No errors

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/execution/equity.py orchestrator/tests/unit/test_equity_provider.py
git commit -m "feat: add EquityProvider protocol and PaperEquityProvider"
```

---

### Task 2: Add `trade_margin_amount` to Settings

**Files:**
- Modify: `orchestrator/src/orchestrator/config.py:46-52` (paper trading section)
- Modify: `orchestrator/tests/conftest.py` (env cleanup fixture)

**Step 1: Add field to Settings**

In `config.py`, add `trade_margin_amount` to the paper trading section (after line 52):

```python
trade_margin_amount: float = 500.0
```

**Step 2: Update conftest.py env cleanup**

The `_clean_settings_env` fixture dynamically removes all Settings fields from env. Since it uses `Settings.model_fields`, the new field is auto-cleaned. Verify no manual change needed.

**Step 3: Run existing tests to verify nothing breaks**

Run: `cd orchestrator && uv run pytest tests/ -v --tb=short`
Expected: All existing tests PASS

**Step 4: Commit**

```bash
git add orchestrator/src/orchestrator/config.py
git commit -m "feat: add trade_margin_amount to Settings"
```

---

### Task 3: ExecutionPlan Model + OrderInstruction

**Files:**
- Create: `orchestrator/src/orchestrator/execution/plan.py`
- Test: `orchestrator/tests/unit/test_execution_plan.py`

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_execution_plan.py
from __future__ import annotations

import pytest

from orchestrator.execution.plan import ExecutionPlan, OrderInstruction


def test_order_instruction_is_frozen():
    oi = OrderInstruction(
        symbol="BTC/USDT:USDT",
        side="buy",
        order_type="market",
        quantity=0.05,
    )
    with pytest.raises(Exception):
        oi.quantity = 1.0  # type: ignore[misc]


def test_execution_plan_is_frozen():
    oi = OrderInstruction(
        symbol="BTC/USDT:USDT",
        side="buy",
        order_type="market",
        quantity=0.05,
    )
    plan = ExecutionPlan(
        proposal_id="test-id",
        symbol="BTC/USDT:USDT",
        side="long",
        entry_order=oi,
        sl_order=None,
        tp_orders=[],
        margin_mode="isolated",
        leverage=10,
        quantity=0.05,
        entry_price=95_000.0,
        notional_value=4_750.0,
        margin_required=475.0,
        liquidation_price=85_950.0,
        estimated_fees=4.75,
        max_loss=100.0,
        max_loss_pct=1.0,
        tp_profits=[],
        risk_reward_ratio=2.0,
        equity_snapshot=10_000.0,
    )
    with pytest.raises(Exception):
        plan.quantity = 1.0  # type: ignore[misc]


def test_execution_plan_has_all_required_fields():
    oi = OrderInstruction(
        symbol="BTC/USDT:USDT",
        side="buy",
        order_type="market",
        quantity=0.05,
    )
    plan = ExecutionPlan(
        proposal_id="test-id",
        symbol="BTC/USDT:USDT",
        side="long",
        entry_order=oi,
        sl_order=None,
        tp_orders=[],
        margin_mode="isolated",
        leverage=10,
        quantity=0.05,
        entry_price=95_000.0,
        notional_value=4_750.0,
        margin_required=475.0,
        liquidation_price=85_950.0,
        estimated_fees=4.75,
        max_loss=100.0,
        max_loss_pct=1.0,
        tp_profits=[],
        risk_reward_ratio=2.0,
        equity_snapshot=10_000.0,
    )
    assert plan.margin_mode == "isolated"
    assert plan.leverage == 10
    assert plan.entry_price == 95_000.0
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_execution_plan.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# orchestrator/src/orchestrator/execution/plan.py
from __future__ import annotations

from pydantic import BaseModel


class OrderInstruction(BaseModel, frozen=True):
    symbol: str
    side: str                    # "buy" / "sell"
    order_type: str              # "market" / "limit"
    quantity: float
    price: float | None = None
    reduce_only: bool = False
    stop_price: float | None = None


class ExecutionPlan(BaseModel, frozen=True):
    proposal_id: str
    symbol: str
    side: str                     # "long" / "short"
    entry_order: OrderInstruction
    sl_order: OrderInstruction | None
    tp_orders: list[OrderInstruction]
    margin_mode: str = "isolated"
    leverage: int
    quantity: float
    entry_price: float
    notional_value: float         # quantity × entry_price
    margin_required: float
    liquidation_price: float
    estimated_fees: float
    max_loss: float               # dollar amount if SL hit
    max_loss_pct: float           # as % of equity
    tp_profits: list[float]       # estimated profit per TP level
    risk_reward_ratio: float
    equity_snapshot: float        # equity at computation time
```

**Step 4: Run test to verify it passes**

Run: `cd orchestrator && uv run pytest tests/unit/test_execution_plan.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/execution/plan.py orchestrator/tests/unit/test_execution_plan.py
git commit -m "feat: add ExecutionPlan and OrderInstruction models"
```

---

### Task 4: ExecutionPlanner — Core Computation

**Files:**
- Create: `orchestrator/src/orchestrator/execution/planner.py`
- Test: `orchestrator/tests/unit/test_execution_planner.py`

**Step 1: Write the failing tests**

```python
# orchestrator/tests/unit/test_execution_planner.py
from __future__ import annotations

import pytest

from orchestrator.execution.planner import ExecutionPlanner
from orchestrator.models import EntryOrder, Side, TakeProfit, TradeProposal


class FakeEquityProvider:
    def __init__(self, equity: float = 10_000.0, available: float = 9_000.0):
        self._equity = equity
        self._available = available

    async def get_equity(self) -> float:
        return self._equity

    async def get_available_margin(self) -> float:
        return self._available


class FakeSettings:
    trade_margin_amount: float = 500.0
    paper_taker_fee_rate: float = 0.0005


def _make_proposal(
    *,
    side: Side = Side.LONG,
    entry_price: float | None = None,
    entry_type: str = "market",
    stop_loss: float = 93_000.0,
    take_profit: list[TakeProfit] | None = None,
    leverage: int = 10,
    risk_pct: float = 1.0,
) -> TradeProposal:
    return TradeProposal(
        symbol="BTC/USDT:USDT",
        side=side,
        entry=EntryOrder(type=entry_type, price=entry_price),
        position_size_risk_pct=risk_pct,
        stop_loss=stop_loss,
        take_profit=take_profit or [
            TakeProfit(price=97_000.0, close_pct=50),
            TakeProfit(price=99_000.0, close_pct=100),
        ],
        suggested_leverage=leverage,
        time_horizon="4h",
        confidence=0.82,
        invalid_if=[],
        rationale="Test rationale",
    )


@pytest.mark.asyncio
async def test_create_plan_long_market():
    planner = ExecutionPlanner(
        equity_provider=FakeEquityProvider(),
        config=FakeSettings(),  # type: ignore[arg-type]
    )
    proposal = _make_proposal()
    plan = await planner.create_plan(proposal, current_price=95_000.0)

    # quantity = margin * leverage / price = 500 * 10 / 95000
    expected_qty = 500.0 * 10 / 95_000.0
    assert plan.quantity == pytest.approx(expected_qty, rel=1e-6)
    assert plan.entry_price == 95_000.0
    assert plan.leverage == 10
    assert plan.margin_mode == "isolated"
    assert plan.margin_required == pytest.approx(500.0)
    assert plan.notional_value == pytest.approx(expected_qty * 95_000.0, rel=1e-4)
    assert plan.equity_snapshot == 10_000.0
    assert plan.entry_order.side == "buy"
    assert plan.entry_order.order_type == "market"


@pytest.mark.asyncio
async def test_create_plan_short():
    planner = ExecutionPlanner(
        equity_provider=FakeEquityProvider(),
        config=FakeSettings(),  # type: ignore[arg-type]
    )
    proposal = _make_proposal(
        side=Side.SHORT,
        stop_loss=97_000.0,
        take_profit=[TakeProfit(price=91_000.0, close_pct=100)],
    )
    plan = await planner.create_plan(proposal, current_price=95_000.0)

    assert plan.entry_order.side == "sell"
    assert plan.sl_order is not None
    assert plan.sl_order.side == "buy"  # opposite of entry


@pytest.mark.asyncio
async def test_create_plan_computes_max_loss():
    planner = ExecutionPlanner(
        equity_provider=FakeEquityProvider(),
        config=FakeSettings(),  # type: ignore[arg-type]
    )
    proposal = _make_proposal(stop_loss=93_000.0)
    plan = await planner.create_plan(proposal, current_price=95_000.0)

    # max_loss = quantity * |entry - SL|
    expected_loss = plan.quantity * abs(95_000.0 - 93_000.0)
    assert plan.max_loss == pytest.approx(expected_loss, rel=1e-4)
    assert plan.max_loss_pct == pytest.approx(expected_loss / 10_000.0 * 100, rel=1e-4)


@pytest.mark.asyncio
async def test_create_plan_computes_tp_profits():
    planner = ExecutionPlanner(
        equity_provider=FakeEquityProvider(),
        config=FakeSettings(),  # type: ignore[arg-type]
    )
    proposal = _make_proposal(
        take_profit=[
            TakeProfit(price=97_000.0, close_pct=50),
            TakeProfit(price=99_000.0, close_pct=100),
        ],
    )
    plan = await planner.create_plan(proposal, current_price=95_000.0)

    assert len(plan.tp_profits) == 2
    # TP1: quantity * 50% * (97000 - 95000)
    assert plan.tp_profits[0] == pytest.approx(
        plan.quantity * 0.50 * 2_000.0, rel=1e-4,
    )


@pytest.mark.asyncio
async def test_create_plan_computes_liquidation_price_long():
    planner = ExecutionPlanner(
        equity_provider=FakeEquityProvider(),
        config=FakeSettings(),  # type: ignore[arg-type]
    )
    proposal = _make_proposal(leverage=10)
    plan = await planner.create_plan(proposal, current_price=95_000.0)

    # liq_price for long = entry * (1 - 1/leverage)
    expected_liq = 95_000.0 * (1 - 1 / 10)
    assert plan.liquidation_price == pytest.approx(expected_liq, rel=1e-4)


@pytest.mark.asyncio
async def test_create_plan_computes_risk_reward():
    planner = ExecutionPlanner(
        equity_provider=FakeEquityProvider(),
        config=FakeSettings(),  # type: ignore[arg-type]
    )
    proposal = _make_proposal(
        stop_loss=93_000.0,
        take_profit=[TakeProfit(price=99_000.0, close_pct=100)],
    )
    plan = await planner.create_plan(proposal, current_price=95_000.0)

    # risk = |95000 - 93000| = 2000, reward = |99000 - 95000| = 4000
    assert plan.risk_reward_ratio == pytest.approx(2.0, rel=1e-4)


@pytest.mark.asyncio
async def test_create_plan_limit_order():
    planner = ExecutionPlanner(
        equity_provider=FakeEquityProvider(),
        config=FakeSettings(),  # type: ignore[arg-type]
    )
    proposal = _make_proposal(entry_type="limit", entry_price=94_500.0)
    plan = await planner.create_plan(proposal, current_price=95_000.0)

    assert plan.entry_order.order_type == "limit"
    assert plan.entry_order.price == 94_500.0
    assert plan.entry_price == 94_500.0  # uses limit price, not current


@pytest.mark.asyncio
async def test_create_plan_estimated_fees():
    planner = ExecutionPlanner(
        equity_provider=FakeEquityProvider(),
        config=FakeSettings(),  # type: ignore[arg-type]
    )
    proposal = _make_proposal()
    plan = await planner.create_plan(proposal, current_price=95_000.0)

    # fees = notional * fee_rate (for entry)
    expected_fees = plan.notional_value * 0.0005
    assert plan.estimated_fees == pytest.approx(expected_fees, rel=1e-4)
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_execution_planner.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# orchestrator/src/orchestrator/execution/planner.py
from __future__ import annotations

from typing import TYPE_CHECKING

from orchestrator.execution.plan import ExecutionPlan, OrderInstruction
from orchestrator.models import Side

if TYPE_CHECKING:
    from orchestrator.execution.equity import EquityProvider
    from orchestrator.models import TradeProposal


class ExecutionPlanner:
    """Computes concrete trade numbers from a TradeProposal.

    Pure Python — no LLM calls. Reads equity from an EquityProvider
    and uses fixed-amount margin sizing from config.
    """

    def __init__(self, equity_provider: EquityProvider, config: object) -> None:
        self._equity = equity_provider
        self._config = config

    async def create_plan(
        self, proposal: TradeProposal, current_price: float,
    ) -> ExecutionPlan:
        equity = await self._equity.get_equity()
        margin = self._config.trade_margin_amount  # type: ignore[attr-defined]
        fee_rate: float = getattr(self._config, "paper_taker_fee_rate", 0.0005)
        leverage = proposal.suggested_leverage

        # Entry price: use limit price if provided, else current market price
        entry_price = (
            proposal.entry.price
            if proposal.entry.type == "limit" and proposal.entry.price is not None
            else current_price
        )

        # Quantity from fixed margin
        quantity = margin * leverage / entry_price

        # Notional value
        notional = quantity * entry_price

        # Order sides
        entry_side = "buy" if proposal.side == Side.LONG else "sell"
        exit_side = "sell" if proposal.side == Side.LONG else "buy"

        # Entry order
        entry_order = OrderInstruction(
            symbol=proposal.symbol,
            side=entry_side,
            order_type=proposal.entry.type,
            quantity=quantity,
            price=proposal.entry.price if proposal.entry.type == "limit" else None,
        )

        # SL order
        sl_order = None
        if proposal.stop_loss is not None:
            sl_order = OrderInstruction(
                symbol=proposal.symbol,
                side=exit_side,
                order_type="market",
                quantity=quantity,
                stop_price=proposal.stop_loss,
                reduce_only=True,
            )

        # TP orders
        tp_orders: list[OrderInstruction] = []
        remaining_qty = quantity
        for tp in proposal.take_profit:
            close_qty = quantity * tp.close_pct / 100
            close_qty = min(close_qty, remaining_qty)
            tp_orders.append(
                OrderInstruction(
                    symbol=proposal.symbol,
                    side=exit_side,
                    order_type="market",
                    quantity=close_qty,
                    stop_price=tp.price,
                    reduce_only=True,
                ),
            )
            remaining_qty -= close_qty

        # Max loss
        max_loss = 0.0
        if proposal.stop_loss is not None:
            max_loss = quantity * abs(entry_price - proposal.stop_loss)
        max_loss_pct = (max_loss / equity * 100) if equity > 0 else 0.0

        # TP profits
        tp_profits: list[float] = []
        remaining_for_profit = quantity
        for tp in proposal.take_profit:
            portion = quantity * tp.close_pct / 100
            portion = min(portion, remaining_for_profit)
            profit = portion * abs(tp.price - entry_price)
            tp_profits.append(profit)
            remaining_for_profit -= portion

        # Risk/Reward ratio
        risk_reward = 0.0
        if proposal.stop_loss is not None and proposal.take_profit:
            risk_dist = abs(entry_price - proposal.stop_loss)
            reward_dist = abs(proposal.take_profit[-1].price - entry_price)
            if risk_dist > 0:
                risk_reward = reward_dist / risk_dist

        # Liquidation price (simplified)
        if proposal.side == Side.LONG:
            liq_price = entry_price * (1 - 1 / leverage)
        else:
            liq_price = entry_price * (1 + 1 / leverage)

        # Estimated fees (entry only)
        estimated_fees = notional * fee_rate

        return ExecutionPlan(
            proposal_id=proposal.proposal_id,
            symbol=proposal.symbol,
            side=proposal.side.value,
            entry_order=entry_order,
            sl_order=sl_order,
            tp_orders=tp_orders,
            margin_mode="isolated",
            leverage=leverage,
            quantity=quantity,
            entry_price=entry_price,
            notional_value=notional,
            margin_required=margin,
            liquidation_price=liq_price,
            estimated_fees=estimated_fees,
            max_loss=max_loss,
            max_loss_pct=max_loss_pct,
            tp_profits=tp_profits,
            risk_reward_ratio=risk_reward,
            equity_snapshot=equity,
        )
```

**Step 4: Run test to verify it passes**

Run: `cd orchestrator && uv run pytest tests/unit/test_execution_planner.py -v`
Expected: PASS (8 tests)

**Step 5: Lint**

Run: `cd orchestrator && uv run ruff check src/orchestrator/execution/planner.py`
Expected: No errors

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/execution/planner.py orchestrator/tests/unit/test_execution_planner.py
git commit -m "feat: add ExecutionPlanner with fixed-margin position sizing"
```

---

### Task 5: Two-Section Telegram Formatter

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/formatters.py`
- Test: `orchestrator/tests/unit/test_formatters_execution_plan.py`

**Context:** The existing `format_pending_approval()` (line 168) formats proposals for Telegram. We need a new `format_execution_plan()` function that produces the two-section format from the design doc. The existing function stays for backward compatibility; the new one will be used by the updated approval flow.

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_formatters_execution_plan.py
from __future__ import annotations

from orchestrator.execution.plan import ExecutionPlan, OrderInstruction
from orchestrator.telegram.formatters import format_execution_plan


def _make_plan() -> ExecutionPlan:
    return ExecutionPlan(
        proposal_id="test-123",
        symbol="BTC/USDT:USDT",
        side="long",
        entry_order=OrderInstruction(
            symbol="BTC/USDT:USDT",
            side="buy",
            order_type="market",
            quantity=0.05,
        ),
        sl_order=OrderInstruction(
            symbol="BTC/USDT:USDT",
            side="sell",
            order_type="market",
            quantity=0.05,
            stop_price=93_000.0,
            reduce_only=True,
        ),
        tp_orders=[
            OrderInstruction(
                symbol="BTC/USDT:USDT",
                side="sell",
                order_type="market",
                quantity=0.025,
                stop_price=97_000.0,
                reduce_only=True,
            ),
            OrderInstruction(
                symbol="BTC/USDT:USDT",
                side="sell",
                order_type="market",
                quantity=0.025,
                stop_price=99_000.0,
                reduce_only=True,
            ),
        ],
        margin_mode="isolated",
        leverage=10,
        quantity=0.05,
        entry_price=95_000.0,
        notional_value=4_750.0,
        margin_required=475.0,
        liquidation_price=85_500.0,
        estimated_fees=2.375,
        max_loss=100.0,
        max_loss_pct=1.0,
        tp_profits=[50.0, 100.0],
        risk_reward_ratio=2.0,
        equity_snapshot=10_000.0,
    )


def test_format_upper_section_contains_trade_params():
    text = format_execution_plan(
        plan=_make_plan(),
        confidence=0.82,
        time_horizon="4h",
    )
    assert "LONG" in text
    assert "BTC/USDT" in text
    assert "95,000" in text        # entry price
    assert "0.05" in text           # quantity
    assert "475" in text            # margin
    assert "85,500" in text         # liq price
    assert "93,000" in text         # SL
    assert "97,000" in text         # TP1
    assert "99,000" in text         # TP2
    assert "Risk/Reward" in text


def test_format_upper_section_contains_loss_and_profit():
    text = format_execution_plan(
        plan=_make_plan(),
        confidence=0.82,
        time_horizon="4h",
    )
    assert "100" in text           # max loss $100
    assert "1.0%" in text          # max loss pct


def test_format_lower_section_contains_analysis():
    text = format_execution_plan(
        plan=_make_plan(),
        confidence=0.82,
        time_horizon="4h",
        analysis_summary={
            "technical": "UP, BULLISH momentum\nSupport: 93,200 / Resist: 97,500",
            "positioning": "Funding -0.01%\nOI +3.2%",
            "catalyst": "No high-impact events",
            "correlation": "DXY↓ S&P risk-on",
        },
        rationale="BTC breaking above resistance with low funding.",
    )
    assert "分析摘要" in text
    assert "Technical" in text
    assert "Positioning" in text
    assert "Catalyst" in text
    assert "Correlation" in text
    assert "BTC breaking above" in text


def test_format_without_analysis_shows_upper_only():
    text = format_execution_plan(
        plan=_make_plan(),
        confidence=0.82,
        time_horizon="4h",
    )
    # Should not crash, just no lower section
    assert "開倉建議" in text
    assert "分析摘要" not in text


def test_format_short_side():
    plan = _make_plan()
    short_plan = plan.model_copy(update={"side": "short"})
    text = format_execution_plan(
        plan=short_plan,
        confidence=0.75,
        time_horizon="1d",
    )
    assert "SHORT" in text
    assert "\U0001f534" in text  # red circle emoji
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_formatters_execution_plan.py -v`
Expected: FAIL with `ImportError: cannot import name 'format_execution_plan'`

**Step 3: Write implementation**

Add to `orchestrator/src/orchestrator/telegram/formatters.py` (after the existing `format_pending_approval` function, around line 260):

```python
# ---------------------------------------------------------------------------
# Execution plan (two-section format)
# ---------------------------------------------------------------------------

def format_execution_plan(
    *,
    plan: object,
    confidence: float,
    time_horizon: str,
    analysis_summary: dict[str, str] | None = None,
    rationale: str | None = None,
    model_used: str | None = None,
    expires_minutes: int | None = None,
) -> str:
    """Format an ExecutionPlan as a two-section Telegram message.

    Upper section: concrete trade parameters.
    Lower section: agent analysis summaries + rationale.
    """
    side_str = plan.side.upper()  # type: ignore[attr-defined]
    emoji = _SIDE_EMOJI.get(plan.side, "")  # type: ignore[attr-defined]
    symbol = plan.symbol.replace(":USDT", "")  # type: ignore[attr-defined]

    entry_price: float = plan.entry_price  # type: ignore[attr-defined]
    quantity: float = plan.quantity  # type: ignore[attr-defined]
    notional: float = plan.notional_value  # type: ignore[attr-defined]
    margin: float = plan.margin_required  # type: ignore[attr-defined]
    leverage: int = plan.leverage  # type: ignore[attr-defined]
    margin_mode: str = plan.margin_mode  # type: ignore[attr-defined]
    liq_price: float = plan.liquidation_price  # type: ignore[attr-defined]
    max_loss: float = plan.max_loss  # type: ignore[attr-defined]
    max_loss_pct: float = plan.max_loss_pct  # type: ignore[attr-defined]
    rr: float = plan.risk_reward_ratio  # type: ignore[attr-defined]
    tp_profits: list[float] = plan.tp_profits  # type: ignore[attr-defined]
    order_type: str = plan.entry_order.order_type  # type: ignore[attr-defined]

    # --- Upper section ---
    lines = [
        "\u2501\u2501\u2501 \u958b\u5009\u5efa\u8b70 \u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501",
        "",
        f"{emoji} {symbol} {side_str} \u00b7 Confidence {confidence:.0%}",
        "",
    ]

    # Entry
    entry_str = f"${entry_price:,.1f} ({order_type})"
    if order_type == "limit" and plan.entry_order.price is not None:  # type: ignore[attr-defined]
        entry_str = f"${plan.entry_order.price:,.1f} (limit)"  # type: ignore[attr-defined]
    lines.append(f"\u25b6 Entry:     {entry_str}")
    lines.append(f"  Quantity:  {quantity:.4f} (${notional:,.0f})")
    lines.append(
        f"  Margin:    ${margin:,.0f} \u00b7 {leverage}x {margin_mode}",
    )
    lines.append(f"  Liq:       ${liq_price:,.1f}")

    # SL
    sl_order = plan.sl_order  # type: ignore[attr-defined]
    if sl_order is not None and sl_order.stop_price is not None:
        sl_price = sl_order.stop_price
        sl_pct = (sl_price - entry_price) / entry_price * 100
        lines.append(f"\u26d4 Stop Loss: ${sl_price:,.1f} ({sl_pct:+.1f}%)")

    # TPs
    tp_orders = plan.tp_orders  # type: ignore[attr-defined]
    for i, tp_order in enumerate(tp_orders):
        if tp_order.stop_price is None:
            continue
        tp_pct = (tp_order.stop_price - entry_price) / entry_price * 100
        close_pct_str = f"{tp_order.quantity / quantity * 100:.0f}%"
        profit_str = ""
        if i < len(tp_profits):
            profit_str = f" \u2192 +${tp_profits[i]:,.0f}"
        lines.append(
            f"\u2705 TP{i + 1}:       ${tp_order.stop_price:,.1f}"
            f" ({tp_pct:+.1f}%) \u2192 close {close_pct_str}{profit_str}",
        )

    # Loss / Profit / R:R
    lines.append("")
    lines.append(f"\u26a0\ufe0f \u6700\u5927\u8667\u640d: ${max_loss:,.0f} ({max_loss_pct:.1f}%)")
    if tp_profits:
        tp_parts = " / ".join(
            f"TP{i + 1} +${p:,.0f}" for i, p in enumerate(tp_profits)
        )
        lines.append(f"\U0001f4b0 \u9810\u4f30\u7372\u5229: {tp_parts}")
    lines.append(f"\U0001f4ca Risk/Reward: 1:{rr:.1f}")

    # --- Lower section (optional) ---
    if analysis_summary:
        lines.append("")
        lines.append("\u2501\u2501\u2501 \u5206\u6790\u6458\u8981 \u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501")
        lines.append("")

        label_map = {
            "technical": "\U0001f4c8 Technical",
            "positioning": "\U0001f4b9 Positioning",
            "catalyst": "\U0001f4c5 Catalyst",
            "correlation": "\U0001f310 Correlation",
        }
        for key in ("technical", "positioning", "catalyst", "correlation"):
            if key in analysis_summary:
                label = label_map.get(key, key.title())
                summary_lines = analysis_summary[key].split("\n")
                lines.append(f"{label}: {summary_lines[0]}")
                for extra in summary_lines[1:]:
                    lines.append(f"   {extra}")

    if rationale:
        lines.append("")
        lines.append(f"\U0001f4a1 \u5c40\u52e2\u5206\u6790")
        lines.append(rationale)

    # Footer
    footer_parts: list[str] = []
    if model_used:
        footer_parts.append(f"Model: {model_used.split('/')[-1]}")
    if expires_minutes is not None:
        footer_parts.append(f"Expires in {expires_minutes} min")
    if footer_parts:
        lines.append(f"\n{' \u00b7 '.join(footer_parts)}")

    return "\n".join(lines)
```

Also add the import at the top of formatters.py (if not already present — it likely isn't needed since we use `object` typing).

**Step 4: Run test to verify it passes**

Run: `cd orchestrator && uv run pytest tests/unit/test_formatters_execution_plan.py -v`
Expected: PASS (5 tests)

**Step 5: Run all formatter tests**

Run: `cd orchestrator && uv run pytest tests/unit/test_formatters*.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/formatters.py orchestrator/tests/unit/test_formatters_execution_plan.py
git commit -m "feat: add two-section format_execution_plan formatter"
```

---

### Task 6: Integrate ExecutionPlanner into Pipeline Runner

**Files:**
- Modify: `orchestrator/src/orchestrator/pipeline/runner.py`
- Test: `orchestrator/tests/unit/test_pipeline_execution_plan.py`

**Context:** Currently the pipeline runner goes: Proposer → Validation → Risk Check → Execute/Approve. We insert ExecutionPlanner between Proposer and Validation. The PipelineResult needs to carry the ExecutionPlan so the Telegram bot can format it.

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_pipeline_execution_plan.py
from __future__ import annotations

import pytest

from orchestrator.execution.planner import ExecutionPlanner
from orchestrator.models import EntryOrder, Side, TakeProfit, TradeProposal


class FakeEquityProvider:
    async def get_equity(self) -> float:
        return 10_000.0

    async def get_available_margin(self) -> float:
        return 9_000.0


class FakeSettings:
    trade_margin_amount: float = 500.0
    paper_taker_fee_rate: float = 0.0005


@pytest.mark.asyncio
async def test_planner_integrates_with_proposal():
    """Integration test: real planner with real proposal model."""
    planner = ExecutionPlanner(
        equity_provider=FakeEquityProvider(),
        config=FakeSettings(),  # type: ignore[arg-type]
    )
    proposal = TradeProposal(
        symbol="BTC/USDT:USDT",
        side=Side.LONG,
        entry=EntryOrder(type="market"),
        position_size_risk_pct=1.0,
        stop_loss=93_000.0,
        take_profit=[
            TakeProfit(price=97_000.0, close_pct=50),
            TakeProfit(price=99_000.0, close_pct=100),
        ],
        suggested_leverage=10,
        time_horizon="4h",
        confidence=0.82,
        invalid_if=[],
        rationale="Test rationale.",
    )

    plan = await planner.create_plan(proposal, current_price=95_000.0)

    # Verify plan is complete and consistent
    assert plan.proposal_id == proposal.proposal_id
    assert plan.symbol == "BTC/USDT:USDT"
    assert plan.side == "long"
    assert plan.entry_order.side == "buy"
    assert plan.sl_order is not None
    assert plan.sl_order.side == "sell"
    assert len(plan.tp_orders) == 2
    assert plan.max_loss > 0
    assert plan.risk_reward_ratio > 0
    assert plan.liquidation_price < plan.entry_price
```

**Step 2: Run test to verify it passes** (this is an integration test using already-built components)

Run: `cd orchestrator && uv run pytest tests/unit/test_pipeline_execution_plan.py -v`
Expected: PASS

**Step 3: Modify PipelineRunner**

This is the most involved change. In `runner.py`:

1. Add `execution_planner` to `PipelineRunner.__init__()` (around line 63)
2. After proposer runs and validation passes, call `execution_planner.create_plan()`
3. Store the ExecutionPlan in PipelineResult
4. Pass it to the approval manager / Telegram push

**Specific changes to `runner.py`:**

a. Import ExecutionPlanner at top:
```python
from orchestrator.execution.planner import ExecutionPlanner
```

b. Add to `__init__` params (line ~75):
```python
execution_planner: ExecutionPlanner | None = None,
```

c. Store it:
```python
self._execution_planner = execution_planner
```

d. After validation passes (around line 204), before risk check:
```python
# Compute execution plan
execution_plan = None
if self._execution_planner is not None and proposal.side != Side.FLAT:
    current_price = snapshot_short.current_price
    execution_plan = await self._execution_planner.create_plan(
        proposal, current_price,
    )
```

e. Add `execution_plan` field to `PipelineResult` (look for the dataclass/model definition — it may be in runner.py or a separate file).

**Note to implementer:** The exact line numbers will shift as you add code. Use the field names and method names as anchors. Read the file carefully before editing.

**Step 4: Run full test suite**

Run: `cd orchestrator && uv run pytest tests/ -v --tb=short`
Expected: All PASS (existing tests should not break since `execution_planner=None` is the default)

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/pipeline/runner.py
git commit -m "feat: integrate ExecutionPlanner into pipeline runner"
```

---

### Task 7: Update Telegram Bot — Approval with ExecutionPlan Format

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/bot.py`

**Context:** Currently `push_pending_approval()` (line 657) uses `format_pending_approval()`. We need to update it to use the new `format_execution_plan()` when an ExecutionPlan is available, and change the buttons from [Approve/Reject] to [開倉/調整/跳過].

**Step 1: Update `push_pending_approval`**

In `bot.py`, find `push_pending_approval()` (line 657). Modify it to:

1. Accept an optional `execution_plan` parameter
2. If plan is available, use `format_execution_plan()` instead of `format_pending_approval()`
3. Change inline keyboard buttons:

```python
buttons = [
    [
        InlineKeyboardButton("🚀 開倉", callback_data=f"approve:{approval_id}"),
        InlineKeyboardButton("✏️ 調整", callback_data=f"adjust:{approval_id}"),
        InlineKeyboardButton("❌ 跳過", callback_data=f"reject:{approval_id}"),
    ],
]
```

**Step 2: Add analysis_summary extraction**

Create a helper that extracts one-line summaries from the pipeline result's agent outputs (technical, positioning, catalyst, correlation) so they can be passed to `format_execution_plan()`.

```python
def _extract_analysis_summary(result: PipelineResult) -> dict[str, str]:
    summary: dict[str, str] = {}
    # Extract from result's agent outputs
    # Each agent output has trend/momentum/key fields that can be summarized
    ...
    return summary
```

**Step 3: Test manually**

Since Telegram bot handlers are hard to unit test, verify by:
1. Running `cd orchestrator && uv run ruff check src/orchestrator/telegram/bot.py`
2. Running existing bot tests if any exist
3. Manual testing with Telegram (out of scope for this plan)

**Step 4: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/bot.py
git commit -m "feat: use two-section ExecutionPlan format in Telegram approval"
```

---

### Task 8: Telegram Bot — Parameter Adjustment Flow

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/bot.py`

**Context:** When user taps [✏️ 調整], we need a conversation flow that lets them modify parameters. This task adds the `adjust` callback handler and sub-handlers for each parameter.

**Step 1: Add callback routes**

In `_CALLBACK_DISPATCH` (line 689), add:
```python
"adjust":     (2, "_handle_adjust"),
"adj_lev":    (2, "_handle_adjust_leverage"),
"adj_sl":     (2, "_handle_adjust_sl_prompt"),
"adj_tp":     (2, "_handle_adjust_tp_prompt"),
"adj_qty":    (2, "_handle_adjust_qty"),
"adj_confirm": (2, "_handle_adjust_confirm"),
"adj_cancel": (2, "_handle_adjust_cancel"),
```

**Step 2: Implement `_handle_adjust`**

Shows the adjustment menu:
```python
async def _handle_adjust(self, query, approval_id: str) -> None:
    buttons = [
        [
            InlineKeyboardButton("Leverage", callback_data=f"adj_lev:{approval_id}"),
            InlineKeyboardButton("SL", callback_data=f"adj_sl:{approval_id}"),
        ],
        [
            InlineKeyboardButton("TP", callback_data=f"adj_tp:{approval_id}"),
            InlineKeyboardButton("Quantity", callback_data=f"adj_qty:{approval_id}"),
        ],
        [
            InlineKeyboardButton("🚀 確認開倉", callback_data=f"adj_confirm:{approval_id}"),
            InlineKeyboardButton("❌ 取消", callback_data=f"adj_cancel:{approval_id}"),
        ],
    ]
    await query.edit_message_text(
        "✏️ 要調整哪個參數？",
        reply_markup=InlineKeyboardMarkup(buttons),
    )
```

**Step 3: Implement leverage adjustment (buttons)**

```python
async def _handle_adjust_leverage(self, query, approval_id: str) -> None:
    buttons = [
        [
            InlineKeyboardButton(f"{lev}x", callback_data=f"set_lev:{approval_id}:{lev}")
            for lev in [5, 10, 20, 50]
        ],
        [InlineKeyboardButton("⬅️ 返回", callback_data=f"adjust:{approval_id}")],
    ]
    await query.edit_message_text(
        f"目前 Leverage: {current_lev}x\n選擇新的倍數：",
        reply_markup=InlineKeyboardMarkup(buttons),
    )
```

**Step 4: Implement SL/TP/Qty text input handlers**

For text input params, we need to use Telegram's `ConversationHandler` or store state and listen for the next message. The simplest approach:

1. Store `_pending_adjustments: dict[str, dict]` on the bot — maps `approval_id` to current adjusted values
2. When user taps [SL], send a prompt message and set a flag
3. When next text message arrives, check if there's a pending adjustment

**Note to implementer:** This is the most complex UX piece. Follow the existing patterns in bot.py for handling text input (check if there's a `MessageHandler` pattern). The key data structure is:

```python
# Stored on bot instance
_adjustments: dict[str, ExecutionPlan]  # approval_id → current adjusted plan
```

When any parameter changes, use `ExecutionPlanner.recalculate()` (new method to add) that takes an existing plan + changed params and returns a new plan.

**Step 5: Implement recalculate in ExecutionPlanner**

Add to `orchestrator/src/orchestrator/execution/planner.py`:

```python
async def recalculate(
    self,
    plan: ExecutionPlan,
    *,
    leverage: int | None = None,
    stop_loss: float | None = None,
    take_profit: list[TakeProfit] | None = None,
    quantity: float | None = None,
) -> ExecutionPlan:
    """Recalculate an ExecutionPlan with changed parameters.

    Quantity stays fixed unless explicitly changed.
    """
    ...
```

**Step 6: Test recalculate**

Add tests to `test_execution_planner.py`:
```python
@pytest.mark.asyncio
async def test_recalculate_leverage_keeps_quantity():
    ...

@pytest.mark.asyncio
async def test_recalculate_sl_updates_max_loss():
    ...
```

**Step 7: Commit**

```bash
git add orchestrator/src/orchestrator/execution/planner.py orchestrator/src/orchestrator/telegram/bot.py orchestrator/tests/unit/test_execution_planner.py
git commit -m "feat: add parameter adjustment flow in Telegram bot"
```

---

### Task 9: Telegram Bot — Position Management Menu

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/bot.py`

**Context:** After a trade is executed, users should be able to manage positions via `/status`. Each open position gets an [⚙️ 管理] button. This task adds the position management handlers.

**Step 1: Update `/status` handler to include management buttons**

Find the `/status` handler in bot.py. For each open position, add:
```python
InlineKeyboardButton("⚙️ 管理", callback_data=f"pos_manage:{trade_id}")
```

**Step 2: Add callback routes**

```python
"pos_manage":  (2, "_handle_pos_manage"),
"pos_sl":      (2, "_handle_pos_move_sl"),
"pos_tp":      (2, "_handle_pos_adjust_tp"),
"pos_add":     (2, "_handle_pos_add"),
"pos_reduce":  (2, "_handle_pos_reduce"),
"pos_close":   (2, "_handle_pos_close"),
"pos_confirm_close": (2, "_handle_pos_confirm_close"),
"pos_back":    (2, "_handle_pos_back"),
```

**Step 3: Implement management menu**

```python
async def _handle_pos_manage(self, query, trade_id: str) -> None:
    buttons = [
        [
            InlineKeyboardButton("移 SL", callback_data=f"pos_sl:{trade_id}"),
            InlineKeyboardButton("調 TP", callback_data=f"pos_tp:{trade_id}"),
        ],
        [
            InlineKeyboardButton("加倉", callback_data=f"pos_add:{trade_id}"),
            InlineKeyboardButton("減倉", callback_data=f"pos_reduce:{trade_id}"),
        ],
        [
            InlineKeyboardButton("平倉", callback_data=f"pos_close:{trade_id}"),
        ],
        [
            InlineKeyboardButton("⬅️ 返回", callback_data=f"pos_back:{trade_id}"),
        ],
    ]
    # Show current position info + buttons
    ...
```

**Step 4: Implement each operation handler**

Follow the patterns from Task 8:
- **Move SL**: prompt for text input, call `paper_engine.update_sl(trade_id, new_sl)`
- **Adjust TP**: prompt for text input, call `paper_engine.update_tp(trade_id, new_tps)`
- **Add**: buttons for same/half/custom amount, call `paper_engine.add_to_position()`
- **Reduce**: buttons for 25/50/75%, call `paper_engine.reduce_position()`
- **Close**: confirmation, call `paper_engine.close_position()`

**Note to implementer:** The PaperEngine already has `close_position()`, `add_to_position()`, and `reduce_position()` methods. Check if `update_sl()` and `update_tp()` exist — if not, they need to be added (create a new immutable Position with updated SL/TP).

**Step 5: Add PaperEngine.update_sl / update_tp if missing**

In `paper_engine.py`, add:
```python
async def update_sl(self, trade_id: str, new_sl: float) -> Position:
    pos = self._positions[trade_id]
    updated = pos.model_copy(update={"stop_loss": new_sl})
    self._positions[trade_id] = updated
    return updated

async def update_tp(self, trade_id: str, new_tp: list[TakeProfit]) -> Position:
    pos = self._positions[trade_id]
    updated = pos.model_copy(update={"take_profit": new_tp})
    self._positions[trade_id] = updated
    return updated
```

**Step 6: Test PaperEngine updates**

```python
# In tests/unit/test_paper_engine.py (add to existing)
@pytest.mark.asyncio
async def test_update_sl():
    ...

@pytest.mark.asyncio
async def test_update_tp():
    ...
```

**Step 7: Commit**

```bash
git add orchestrator/src/orchestrator/exchange/paper_engine.py orchestrator/src/orchestrator/telegram/bot.py orchestrator/tests/unit/test_paper_engine.py
git commit -m "feat: add position management menu in Telegram bot"
```

---

### Task 10: Wire Everything Together + Smoke Test

**Files:**
- Modify: `orchestrator/src/orchestrator/__main__.py` or wherever the bot is initialized
- Modify: `orchestrator/src/orchestrator/pipeline/runner.py` (if not done in Task 6)

**Step 1: Initialize ExecutionPlanner in the main entry point**

Find where PipelineRunner is constructed. Add:
```python
from orchestrator.execution.equity import PaperEquityProvider
from orchestrator.execution.planner import ExecutionPlanner

equity_provider = PaperEquityProvider(engine=paper_engine)
execution_planner = ExecutionPlanner(
    equity_provider=equity_provider,
    config=settings,
)
```

Pass `execution_planner` to PipelineRunner.

**Step 2: Run full test suite**

Run: `cd orchestrator && uv run pytest tests/ -v --tb=short --cov=orchestrator`
Expected: All PASS, coverage ≥ 80%

**Step 3: Run linter**

Run: `cd orchestrator && uv run ruff check src/ tests/`
Expected: No errors

**Step 4: Commit**

```bash
git add orchestrator/src/orchestrator/__main__.py
git commit -m "feat: wire ExecutionPlanner into application startup"
```

---

## Task Dependency Graph

```
Task 1 (EquityProvider) ──┐
Task 2 (Config)       ────┤
Task 3 (Models)       ────┼──→ Task 4 (Planner) ──→ Task 6 (Pipeline) ──→ Task 10 (Wire)
                          │                    │
Task 5 (Formatter)   ─────┘                    │
                                               ├──→ Task 7 (TG Approval)
                                               ├──→ Task 8 (TG Adjustment)
                                               └──→ Task 9 (TG Position Mgmt)
```

Tasks 1, 2, 3, 5 can run in parallel.
Task 4 depends on 1, 2, 3.
Task 6 depends on 4.
Tasks 7, 8, 9 depend on 5 and 6.
Task 10 depends on all.

## Summary

| Task | Description | Creates/Modifies | Est. Lines |
|------|-------------|-----------------|-----------|
| 1 | EquityProvider protocol | `execution/equity.py` | ~30 |
| 2 | Config field | `config.py` | ~2 |
| 3 | ExecutionPlan model | `execution/plan.py` | ~30 |
| 4 | ExecutionPlanner | `execution/planner.py` | ~120 |
| 5 | Two-section formatter | `telegram/formatters.py` | ~100 |
| 6 | Pipeline integration | `pipeline/runner.py` | ~20 |
| 7 | TG approval w/ new format | `telegram/bot.py` | ~50 |
| 8 | TG adjustment flow | `telegram/bot.py` + planner | ~200 |
| 9 | TG position management | `telegram/bot.py` + engine | ~250 |
| 10 | Wire everything | `__main__.py` | ~10 |
