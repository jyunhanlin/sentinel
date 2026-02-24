# Agent Leverage & Paper Trading Optimization — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add leverage recommendations to agent output, USDT margin-based position sizing, partial take-profit execution, and detailed Telegram report format.

**Architecture:** Extend existing Pydantic models with new fields (TakeProfit, volatility_pct, suggested_leverage). Update agent prompts. Modify PaperEngine to support partial TP closures with triggered-TP tracking. Update Telegram approval flow for USDT margin input.

**Tech Stack:** Python 3.12, Pydantic v2, structlog, python-telegram-bot, SQLModel, pytest

---

### Task 1: Add TakeProfit model and update TradeProposal & MarketInterpretation

**Files:**
- Modify: `orchestrator/src/orchestrator/models.py`
- Test: `orchestrator/tests/unit/test_models.py`

**Step 1: Write the failing tests**

Add to `orchestrator/tests/unit/test_models.py`:

```python
from orchestrator.models import TakeProfit

class TestTakeProfit:
    def test_valid_take_profit(self):
        tp = TakeProfit(price=65800.0, close_pct=50)
        assert tp.price == 65800.0
        assert tp.close_pct == 50

    def test_close_pct_out_of_range(self):
        with pytest.raises(ValidationError):
            TakeProfit(price=65800.0, close_pct=0)
        with pytest.raises(ValidationError):
            TakeProfit(price=65800.0, close_pct=101)

class TestMarketInterpretationVolatility:
    def test_volatility_pct_field(self):
        interp = MarketInterpretation(
            trend=Trend.UP,
            volatility_regime=VolatilityRegime.MEDIUM,
            volatility_pct=2.3,
            key_levels=[],
            risk_flags=[],
        )
        assert interp.volatility_pct == 2.3

class TestTradeProposalLeverage:
    def test_suggested_leverage_field(self):
        proposal = TradeProposal(
            symbol="BTC/USDT:USDT", side=Side.LONG,
            entry=EntryOrder(type="market"),
            position_size_risk_pct=1.0, stop_loss=64000.0,
            take_profit=[TakeProfit(price=65800.0, close_pct=50), TakeProfit(price=67000.0, close_pct=100)],
            suggested_leverage=10,
            time_horizon="4h", confidence=0.72,
            invalid_if=[], rationale="test",
        )
        assert proposal.suggested_leverage == 10
        assert proposal.take_profit[0].close_pct == 50

    def test_suggested_leverage_default(self):
        proposal = TradeProposal(
            symbol="BTC/USDT:USDT", side=Side.FLAT,
            entry=EntryOrder(type="market"),
            position_size_risk_pct=0.0, stop_loss=None,
            take_profit=[], time_horizon="4h",
            confidence=0.5, invalid_if=[], rationale="no trade",
        )
        assert proposal.suggested_leverage == 10

    def test_suggested_leverage_validation(self):
        with pytest.raises(ValidationError):
            TradeProposal(
                symbol="BTC/USDT:USDT", side=Side.LONG,
                entry=EntryOrder(type="market"),
                position_size_risk_pct=1.0, stop_loss=64000.0,
                take_profit=[], suggested_leverage=100,
                time_horizon="4h", confidence=0.7,
                invalid_if=[], rationale="test",
            )
```

**Step 2: Run tests to verify they fail**

Run: `cd orchestrator && uv run pytest tests/unit/test_models.py -v -x`
Expected: FAIL — `TakeProfit` not found, `volatility_pct` missing, `suggested_leverage` missing

**Step 3: Implement model changes**

In `orchestrator/src/orchestrator/models.py`:

1. Add `TakeProfit` model after `KeyLevel`:
```python
class TakeProfit(BaseModel, frozen=True):
    price: float
    close_pct: int = Field(ge=1, le=100)
```

2. Add `volatility_pct` to `MarketInterpretation`:
```python
class MarketInterpretation(BaseModel, frozen=True):
    trend: Trend
    volatility_regime: VolatilityRegime
    volatility_pct: float = Field(ge=0.0, default=0.0)
    key_levels: list[KeyLevel]
    risk_flags: list[str]
```

3. Update `TradeProposal`:
```python
class TradeProposal(BaseModel, frozen=True):
    proposal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    side: Side
    entry: EntryOrder
    position_size_risk_pct: float = Field(ge=0.0)
    stop_loss: float | None = None
    take_profit: list[TakeProfit]
    suggested_leverage: int = Field(ge=1, le=50, default=10)
    time_horizon: str
    confidence: float = Field(ge=0.0, le=1.0)
    invalid_if: list[str]
    rationale: str
```

**Step 4: Fix all existing tests that construct TradeProposal or MarketInterpretation**

The `take_profit` field now expects `list[TakeProfit]` instead of `list[float]`. Update all test helpers:
- `test_models.py`: Update `TestTradeProposal.test_valid_proposal` — change `take_profit=[95500.0, 97000.0]` to `take_profit=[TakeProfit(price=95500.0, close_pct=50), TakeProfit(price=97000.0, close_pct=100)]`
- `test_models.py`: Update `TestMarketInterpretation.test_valid_interpretation` — add `volatility_pct=2.0`
- `test_paper_engine.py`: Update `_make_proposal()` helper — change `take_profit` default and all call sites
- `test_runner.py`: Update proposal fixtures
- `test_executor.py`: Update proposal fixtures
- `test_aggregator.py`: Update proposal fixtures
- Any other test that constructs `TradeProposal` or `MarketInterpretation`

**Step 5: Run all tests to verify they pass**

Run: `cd orchestrator && uv run pytest tests/ -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/models.py orchestrator/tests/
git commit -m "feat: add TakeProfit model, volatility_pct, suggested_leverage"
```

---

### Task 2: Update MarketAgent prompt for volatility_pct

**Files:**
- Modify: `orchestrator/src/orchestrator/agents/market.py`
- Test: `orchestrator/tests/unit/test_agent_market.py`

**Step 1: Write the failing test**

Add to `test_agent_market.py`:

```python
def test_market_agent_includes_volatility_pct_in_schema(self, snapshot):
    agent = MarketAgent(llm_client=MagicMock())
    messages = agent._build_messages(snapshot=snapshot)
    system_prompt = messages[0]["content"]
    assert "volatility_pct" in system_prompt
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_agent_market.py -v -x`
Expected: FAIL — `volatility_pct` not in system prompt

**Step 3: Update MarketAgent prompts**

In `orchestrator/src/orchestrator/agents/market.py`:

1. Update `system_prompt` to include `volatility_pct` in the JSON schema:
```python
system_prompt = (
    "You are a crypto technical analyst. "
    "Analyze the provided OHLCV data, funding rate, and volume to determine "
    "market structure, trend, volatility regime, key price levels, and risk flags.\n\n"
    "Calculate volatility_pct as the average true range of the last 14 candles "
    "divided by current price, expressed as a percentage (e.g. 2.3 means 2.3%).\n\n"
    "Respond with ONLY a JSON object matching this schema:\n"
    "{\n"
    '  "trend": "up" | "down" | "range",\n'
    '  "volatility_regime": "low" | "medium" | "high",\n'
    '  "volatility_pct": <float>,\n'
    '  "key_levels": [{"type": "support|resistance", "price": <number>}],\n'
    '  "risk_flags": ["<flag_name>"]  '
    "// e.g. funding_elevated, oi_near_ath, volume_declining\n"
    "}"
)
```

2. Update `_get_default_output()` to include `volatility_pct=0.0`

**Step 4: Run tests to verify they pass**

Run: `cd orchestrator && uv run pytest tests/unit/test_agent_market.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/agents/market.py orchestrator/tests/unit/test_agent_market.py
git commit -m "feat: add volatility_pct to MarketAgent prompt"
```

---

### Task 3: Update ProposerAgent prompt for leverage + TakeProfit format

**Files:**
- Modify: `orchestrator/src/orchestrator/agents/proposer.py`
- Test: `orchestrator/tests/unit/test_agent_proposer.py`

**Step 1: Write the failing test**

Add to `test_agent_proposer.py`:

```python
def test_proposer_includes_suggested_leverage_in_schema(self):
    agent = ProposerAgent(llm_client=MagicMock())
    messages = agent._build_messages(
        snapshot=make_snapshot(),
        sentiment=self._make_sentiment(),
        market=self._make_market(),
    )
    system_prompt = messages[0]["content"]
    assert "suggested_leverage" in system_prompt
    assert "close_pct" in system_prompt

def test_proposer_includes_volatility_pct_in_user_prompt(self):
    agent = ProposerAgent(llm_client=MagicMock())
    market = self._make_market()  # needs volatility_pct
    messages = agent._build_messages(
        snapshot=make_snapshot(),
        sentiment=self._make_sentiment(),
        market=market,
    )
    user_prompt = messages[1]["content"]
    assert "Volatility:" in user_prompt or "volatility_pct" in user_prompt.lower()
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_agent_proposer.py -v -x`
Expected: FAIL

**Step 3: Update ProposerAgent prompts**

In `orchestrator/src/orchestrator/agents/proposer.py`:

1. Update `system_prompt` JSON schema to include `suggested_leverage` and `take_profit` with `close_pct`:
```python
system_prompt = (
    "You are a crypto trade proposal generator for futures trading. "
    "Based on sentiment analysis, technical analysis, and current market data, "
    "generate a structured trade proposal with leverage recommendation.\n\n"
    "Rules:\n"
    "- If no clear edge, set side='flat' with position_size_risk_pct=0\n"
    "- stop_loss MUST be below entry for long, above entry for short\n"
    "- position_size_risk_pct: 0.5-2.0% typical range\n"
    "- confidence: be conservative, rarely above 0.8\n"
    "- suggested_leverage: based on volatility_pct:\n"
    "    - volatility_pct < 2% → up to 20x\n"
    "    - volatility_pct 2-4% → 10x\n"
    "    - volatility_pct > 4% → 5x\n"
    "    - if confidence < 0.5 → cap at 5x\n"
    "- take_profit: list of price levels with close_pct (% of remaining position to close).\n"
    "  Last level should have close_pct=100. Example: [{\"price\": 65800, \"close_pct\": 50}, {\"price\": 67000, \"close_pct\": 100}]\n\n"
    "Respond with ONLY a JSON object matching this schema:\n"
    "{\n"
    '  "symbol": "<symbol>",\n'
    '  "side": "long" | "short" | "flat",\n'
    '  "entry": {"type": "market"} or {"type": "limit", "price": <number>},\n'
    '  "position_size_risk_pct": <float 0.0-2.0>,\n'
    '  "stop_loss": <number or null>,\n'
    '  "take_profit": [{"price": <number>, "close_pct": <int 1-100>}],\n'
    '  "suggested_leverage": <int 1-50>,\n'
    '  "time_horizon": "<e.g. 4h, 1d>",\n'
    '  "confidence": <float 0.0-1.0>,\n'
    '  "invalid_if": ["<condition>"],\n'
    '  "rationale": "<1-2 sentence explanation>"\n'
    "}"
)
```

2. Update `user_prompt` to pass `volatility_pct` from market interpretation:
```python
f"Volatility: {market.volatility_regime} ({market.volatility_pct:.1f}%)\n"
```

3. Update `_get_default_output()` to include `suggested_leverage=10, take_profit=[]` with correct type

**Step 4: Run tests to verify they pass**

Run: `cd orchestrator && uv run pytest tests/unit/test_agent_proposer.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/agents/proposer.py orchestrator/tests/unit/test_agent_proposer.py
git commit -m "feat: add leverage and TakeProfit format to ProposerAgent prompt"
```

---

### Task 4: Add MarginSizer for USDT margin-based position sizing

**Files:**
- Modify: `orchestrator/src/orchestrator/risk/position_sizer.py`
- Test: `orchestrator/tests/unit/test_position_sizer.py`

**Step 1: Write the failing test**

Add to `test_position_sizer.py`:

```python
from orchestrator.risk.position_sizer import MarginSizer

class TestMarginSizer:
    def test_basic_calculation(self):
        sizer = MarginSizer()
        qty = sizer.calculate_from_margin(
            margin_usdt=500.0, leverage=10, entry_price=64800.0,
        )
        # qty = 500 * 10 / 64800 ≈ 0.07716
        assert qty == pytest.approx(0.07716, rel=0.01)

    def test_1x_leverage(self):
        sizer = MarginSizer()
        qty = sizer.calculate_from_margin(
            margin_usdt=1000.0, leverage=1, entry_price=64800.0,
        )
        assert qty == pytest.approx(1000.0 / 64800.0, rel=0.01)

    def test_zero_margin_returns_zero(self):
        sizer = MarginSizer()
        qty = sizer.calculate_from_margin(
            margin_usdt=0.0, leverage=10, entry_price=64800.0,
        )
        assert qty == 0.0

    def test_zero_price_raises(self):
        sizer = MarginSizer()
        with pytest.raises(ValueError):
            sizer.calculate_from_margin(
                margin_usdt=500.0, leverage=10, entry_price=0.0,
            )
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_position_sizer.py::TestMarginSizer -v -x`
Expected: FAIL — `MarginSizer` not found

**Step 3: Implement MarginSizer**

Add to `orchestrator/src/orchestrator/risk/position_sizer.py`:

```python
class MarginSizer:
    """quantity = margin_usdt × leverage / entry_price"""

    def calculate_from_margin(
        self, *, margin_usdt: float, leverage: int, entry_price: float,
    ) -> float:
        if entry_price <= 0:
            raise ValueError("entry_price must be positive")
        if margin_usdt == 0.0:
            return 0.0
        return margin_usdt * leverage / entry_price
```

**Step 4: Run tests to verify they pass**

Run: `cd orchestrator && uv run pytest tests/unit/test_position_sizer.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/risk/position_sizer.py orchestrator/tests/unit/test_position_sizer.py
git commit -m "feat: add MarginSizer for USDT-based position sizing"
```

---

### Task 5: Update PaperEngine for partial take-profit

**Files:**
- Modify: `orchestrator/src/orchestrator/exchange/paper_engine.py`
- Modify: `orchestrator/src/orchestrator/storage/models.py`
- Test: `orchestrator/tests/unit/test_paper_engine.py`

This is the most complex task. The key changes:
1. `Position` stores `take_profit_levels: list[TakeProfit]` and `triggered_tp_indices: list[int]`
2. `CloseResult` adds `partial: bool` and `remaining_quantity: float | None`
3. `check_sl_tp` handles partial TP closures instead of full closes
4. `open_position` converts `list[TakeProfit]` from proposal

**Step 1: Write the failing tests**

Add to `test_paper_engine.py`:

```python
from orchestrator.models import TakeProfit

class TestPartialTakeProfit:
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

    def test_tp1_triggers_partial_close(self):
        """When price hits TP1 (close_pct=50), only 50% of position should close."""
        engine = self._make_engine()
        proposal = _make_proposal(
            stop_loss=67000.0,
            take_profit=[
                TakeProfit(price=69000.0, close_pct=50),
                TakeProfit(price=70000.0, close_pct=100),
            ],
        )
        pos = engine.open_position(proposal, current_price=68000.0, leverage=10)
        original_qty = pos.quantity

        # Price hits TP1
        results = engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=69500.0)
        assert len(results) == 1
        assert results[0].reason == "tp"
        assert results[0].partial is True
        assert results[0].quantity == pytest.approx(original_qty * 0.5, rel=0.01)

        # Position still open with 50% remaining
        remaining = engine.get_open_positions()
        assert len(remaining) == 1
        assert remaining[0].quantity == pytest.approx(original_qty * 0.5, rel=0.01)

    def test_tp2_triggers_full_close(self):
        """After TP1 triggered, TP2 (close_pct=100) closes remaining."""
        engine = self._make_engine()
        proposal = _make_proposal(
            stop_loss=67000.0,
            take_profit=[
                TakeProfit(price=69000.0, close_pct=50),
                TakeProfit(price=70000.0, close_pct=100),
            ],
        )
        engine.open_position(proposal, current_price=68000.0, leverage=10)

        # TP1 triggers
        engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=69500.0)
        assert len(engine.get_open_positions()) == 1

        # TP2 triggers
        results = engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=70500.0)
        assert len(results) == 1
        assert results[0].partial is False
        assert len(engine.get_open_positions()) == 0

    def test_sl_still_closes_full_position(self):
        """SL should close entire position regardless of TP levels."""
        engine = self._make_engine()
        proposal = _make_proposal(
            stop_loss=67000.0,
            take_profit=[
                TakeProfit(price=69000.0, close_pct=50),
                TakeProfit(price=70000.0, close_pct=100),
            ],
        )
        engine.open_position(proposal, current_price=68000.0, leverage=10)

        results = engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=66500.0)
        assert len(results) == 1
        assert results[0].reason == "sl"
        assert len(engine.get_open_positions()) == 0

    def test_tp1_moves_sl_to_breakeven(self):
        """After TP1, stop_loss should move to entry price (breakeven)."""
        engine = self._make_engine()
        proposal = _make_proposal(
            stop_loss=67000.0,
            take_profit=[
                TakeProfit(price=69000.0, close_pct=50),
                TakeProfit(price=70000.0, close_pct=100),
            ],
        )
        pos = engine.open_position(proposal, current_price=68000.0, leverage=10)

        # TP1 triggers
        engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=69500.0)
        remaining = engine.get_open_positions()
        assert len(remaining) == 1
        # SL moved to entry price (breakeven)
        assert remaining[0].stop_loss == pytest.approx(pos.entry_price)

    def test_single_tp_100pct_closes_all(self):
        """A single TP with close_pct=100 should close full position."""
        engine = self._make_engine()
        proposal = _make_proposal(
            stop_loss=67000.0,
            take_profit=[TakeProfit(price=70000.0, close_pct=100)],
        )
        engine.open_position(proposal, current_price=68000.0, leverage=10)

        results = engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=70500.0)
        assert len(results) == 1
        assert results[0].partial is False
        assert len(engine.get_open_positions()) == 0

    def test_short_partial_tp(self):
        """Partial TP works for short positions too."""
        engine = self._make_engine()
        proposal = _make_proposal(
            side=Side.SHORT,
            stop_loss=70000.0,
            take_profit=[
                TakeProfit(price=67000.0, close_pct=50),
                TakeProfit(price=65000.0, close_pct=100),
            ],
        )
        pos = engine.open_position(proposal, current_price=68000.0, leverage=10)
        original_qty = pos.quantity

        # TP1 triggers for short (price goes down)
        results = engine.check_sl_tp(symbol="BTC/USDT:USDT", current_price=66500.0)
        assert len(results) == 1
        assert results[0].partial is True
        assert results[0].quantity == pytest.approx(original_qty * 0.5, rel=0.01)
```

**Step 2: Run tests to verify they fail**

Run: `cd orchestrator && uv run pytest tests/unit/test_paper_engine.py::TestPartialTakeProfit -v -x`
Expected: FAIL

**Step 3: Implement partial TP in PaperEngine**

In `orchestrator/src/orchestrator/exchange/paper_engine.py`:

1. Update `Position` model — add `triggered_tp_indices` and change `take_profit` type:
```python
from orchestrator.models import Side, TakeProfit, TradeProposal

class Position(BaseModel, frozen=True):
    trade_id: str
    proposal_id: str
    symbol: str
    side: Side
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: list[TakeProfit]  # changed from list[float]
    opened_at: datetime
    risk_pct: float
    leverage: int = 1
    margin: float = 0.0
    liquidation_price: float = 0.0
    triggered_tp_indices: list[int] = []  # new
```

2. Update `CloseResult` — add `partial` and `remaining_quantity`:
```python
class CloseResult(BaseModel, frozen=True):
    trade_id: str
    symbol: str
    side: Side
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    fees: float
    reason: str
    partial: bool = False
    remaining_quantity: float | None = None
```

3. Rewrite `check_sl_tp` to handle partial TPs. The new logic:
```python
def check_sl_tp(self, *, symbol: str, current_price: float) -> list[CloseResult]:
    closed: list[CloseResult] = []
    remaining: list[Position] = []

    for pos in self._positions:
        if pos.symbol != symbol:
            remaining.append(pos)
            continue

        # Check liquidation (highest priority)
        liq_trigger = self._check_liquidation(pos, current_price)
        if liq_trigger is not None:
            result = self._close(pos, exit_price=current_price, reason="liquidation")
            closed.append(result)
            continue

        # Check SL (full close)
        sl_trigger = self._check_sl(pos, current_price)
        if sl_trigger is not None:
            exit_price, _ = sl_trigger
            result = self._close(pos, exit_price=exit_price, reason="sl")
            closed.append(result)
            continue

        # Check TPs (may be partial)
        tp_result = self._check_tp_levels(pos, current_price)
        if tp_result is not None:
            result, updated_pos = tp_result
            closed.append(result)
            if updated_pos is not None:
                remaining.append(updated_pos)
        else:
            remaining.append(pos)

    self._positions = remaining
    return closed
```

4. Add helper `_check_tp_levels`:
```python
def _check_tp_levels(
    self, pos: Position, current_price: float,
) -> tuple[CloseResult, Position | None] | None:
    """Check TP levels. Returns (CloseResult, remaining_position_or_None) if triggered."""
    for idx, tp in enumerate(pos.take_profit):
        if idx in pos.triggered_tp_indices:
            continue
        # Check if price crossed this TP level
        triggered = (
            (pos.side == Side.LONG and current_price >= tp.price)
            or (pos.side == Side.SHORT and current_price <= tp.price)
        )
        if not triggered:
            continue

        if tp.close_pct >= 100:
            # Full close
            result = self._close(pos, exit_price=tp.price, reason="tp")
            return result, None
        else:
            # Partial close
            close_qty = pos.quantity * tp.close_pct / 100
            remaining_qty = pos.quantity - close_qty

            if pos.side == Side.LONG:
                pnl = (tp.price - pos.entry_price) * close_qty
            else:
                pnl = (pos.entry_price - tp.price) * close_qty

            fee = close_qty * tp.price * self._taker_fee_rate
            self._total_fees += fee
            self._closed_pnl += pnl

            new_triggered = [*pos.triggered_tp_indices, idx]
            remaining_margin = pos.margin * (remaining_qty / pos.quantity)

            # Move SL to entry (breakeven) after first TP
            new_sl = pos.entry_price if not pos.triggered_tp_indices else pos.stop_loss

            updated_pos = Position(
                trade_id=pos.trade_id,
                proposal_id=pos.proposal_id,
                symbol=pos.symbol,
                side=pos.side,
                entry_price=pos.entry_price,
                quantity=remaining_qty,
                stop_loss=new_sl,
                take_profit=pos.take_profit,
                opened_at=pos.opened_at,
                risk_pct=pos.risk_pct * (remaining_qty / pos.quantity),
                leverage=pos.leverage,
                margin=remaining_margin,
                liquidation_price=pos.liquidation_price,
                triggered_tp_indices=new_triggered,
            )

            self._trade_repo.update_trade_partial_close(
                trade_id=pos.trade_id,
                remaining_qty=remaining_qty,
                remaining_margin=remaining_margin,
            )

            self._save_stats_snapshot()

            result = CloseResult(
                trade_id=pos.trade_id,
                symbol=pos.symbol,
                side=pos.side,
                entry_price=pos.entry_price,
                exit_price=tp.price,
                quantity=close_qty,
                pnl=pnl,
                fees=fee,
                reason="tp",
                partial=True,
                remaining_quantity=remaining_qty,
            )
            return result, updated_pos

    return None
```

5. Extract `_check_liquidation` and `_check_sl` from `_check_trigger` (keep backward compat):
```python
def _check_liquidation(self, pos: Position, current_price: float) -> bool:
    if pos.leverage <= 1:
        return False
    if pos.side == Side.LONG and current_price <= pos.liquidation_price:
        return True
    if pos.side == Side.SHORT and current_price >= pos.liquidation_price:
        return True
    return False

def _check_sl(self, pos: Position, current_price: float) -> tuple[float, str] | None:
    if pos.side == Side.LONG and current_price <= pos.stop_loss:
        return pos.stop_loss, "sl"
    if pos.side == Side.SHORT and current_price >= pos.stop_loss:
        return pos.stop_loss, "sl"
    return None
```

6. Update `open_position` — `Position.take_profit` now comes from `proposal.take_profit` which is already `list[TakeProfit]`:
```python
# line 149, was: take_profit=proposal.take_profit,
# still: take_profit=proposal.take_profit,  (type changed from list[float] to list[TakeProfit])
```

7. Update `rebuild_from_db` — deserialize `take_profit_json` into `list[TakeProfit]`:
```python
take_profit_raw = json.loads(t.take_profit_json) if t.take_profit_json else []
take_profit_levels = []
for item in take_profit_raw:
    if isinstance(item, dict):
        take_profit_levels.append(TakeProfit(**item))
    else:
        # Legacy: plain float → treat as 100% close
        take_profit_levels.append(TakeProfit(price=item, close_pct=100))
```

8. Update `save_trade` call in `open_position` — serialize `take_profit` as list of dicts:
```python
self._trade_repo.save_trade(
    ...
    take_profit=[tp.model_dump() for tp in proposal.take_profit],
)
```

**Step 4: Update PaperTradeRecord for triggered_tp tracking**

In `orchestrator/src/orchestrator/storage/models.py`, add to `PaperTradeRecord`:
```python
triggered_tp_json: str = "[]"  # JSON-encoded list[int]
```

**Step 5: Update repository save_trade to accept dict-based take_profit**

In the repository's `save_trade` method, the `take_profit` parameter type changes:
```python
take_profit: list[dict] | list[float] | None = None,
```
(Serialization stays the same: `json.dumps(take_profit or [])`)

**Step 6: Run all tests**

Run: `cd orchestrator && uv run pytest tests/ -v`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add orchestrator/src/orchestrator/exchange/paper_engine.py orchestrator/src/orchestrator/storage/models.py orchestrator/src/orchestrator/storage/repository.py orchestrator/tests/unit/test_paper_engine.py
git commit -m "feat: implement partial take-profit in PaperEngine"
```

---

### Task 6: Update Telegram formatters for detailed report

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/formatters.py`
- Test: `orchestrator/tests/unit/test_formatters.py`

**Step 1: Write the failing test**

Add to `test_formatters.py`:

```python
from orchestrator.models import TakeProfit

def test_format_pending_approval_detailed():
    """Test the new detailed report format with leverage and partial TP."""
    from orchestrator.telegram.formatters import format_pending_approval_detailed

    # Create a mock approval with all required fields
    # ... (construct with TakeProfit levels, suggested_leverage, etc.)
    # Assert the output contains the expected format sections
```

**Step 2: Implement updated formatters**

Update `format_pending_approval` in `formatters.py` to use the new detailed format from the design doc. Key changes:
- Show entry with price and % from current
- Show each TP with close_pct
- Show suggested leverage with volatility info
- Show risk/reward ratio
- Show market analysis section (trend, vol, support/resistance)
- Show sentiment section

Also update:
- `format_proposal` — similar detailed format
- `format_position_card` — update `take_profit` display to show `close_pct`
- `format_status` / `format_status_from_records` — handle new take_profit format

**Step 3: Run tests**

Run: `cd orchestrator && uv run pytest tests/unit/test_formatters.py -v`

**Step 4: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/formatters.py orchestrator/tests/unit/test_formatters.py
git commit -m "feat: detailed Telegram report format with leverage and partial TP"
```

---

### Task 7: Update Telegram bot approval flow for USDT margin input

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/bot.py`
- Test: `orchestrator/tests/unit/test_telegram.py`

**Step 1: Update approval flow**

The current flow: Approve → Select leverage → Preview → Confirm

New flow: Approve → Select leverage (pre-filled with `suggested_leverage`) → Input margin USDT → Preview with computed details → Confirm

Changes to `bot.py`:

1. In `_handle_approve`, pre-select the `suggested_leverage` from the proposal:
```python
# Highlight the suggested leverage button
leverage_buttons = [
    InlineKeyboardButton(
        f"{'✓ ' if lev == approval.proposal.suggested_leverage else ''}{lev}x",
        callback_data=f"leverage:{approval_id}:{lev}",
    )
    for lev in self._leverage_options
]
```

2. In `_handle_leverage_preview`, show predefined margin amounts:
```python
# Show margin selection buttons
margin_buttons = [
    InlineKeyboardButton(f"${m}", callback_data=f"margin:{approval_id}:{leverage}:{m}")
    for m in [100, 250, 500, 1000]
]
```

3. Add new callback handler `_handle_margin_preview`:
```python
async def _handle_margin_preview(
    self, query: CallbackQuery, approval_id: str, leverage_str: str, margin_str: str, *_args: str,
) -> None:
    """Show final confirmation with computed quantity and ROE."""
    leverage = int(leverage_str)
    margin_usdt = float(margin_str)
    # Calculate quantity, liq, ROE for each TP
    # Show confirmation card matching the design doc format
```

4. Update `_handle_confirm_leverage` → rename to `_handle_confirm_trade` and accept margin_usdt

5. Update `_CALLBACK_DISPATCH` with new entries

6. In `_handle_confirm_trade`, use `MarginSizer` instead of `RiskPercentSizer` when margin_usdt is provided

**Step 2: Update stale SL/TP check**

The `_check_stale_sl_tp` function currently takes `take_profit: list[float]`. Update to accept `list[TakeProfit]`:
```python
def _check_stale_sl_tp(
    *,
    side: str,
    current_price: float,
    stop_loss: float | None,
    take_profit: list,  # list[TakeProfit] or list[float]
) -> str | None:
    # Extract first TP price
    first_tp = None
    if take_profit:
        first = take_profit[0]
        first_tp = first.price if hasattr(first, 'price') else first
    ...
```

**Step 3: Run tests**

Run: `cd orchestrator && uv run pytest tests/unit/test_telegram.py -v`

**Step 4: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/bot.py orchestrator/tests/unit/test_telegram.py
git commit -m "feat: USDT margin input and detailed approval flow"
```

---

### Task 8: Update PipelineRunner to pass volatility_pct and store new format

**Files:**
- Modify: `orchestrator/src/orchestrator/pipeline/runner.py`
- Modify: `orchestrator/src/orchestrator/pipeline/aggregator.py`
- Test: `orchestrator/tests/unit/test_runner.py`

**Step 1: Update PipelineResult to carry sentiment and market data**

Add to `PipelineResult`:
```python
sentiment: SentimentReport | None = None
market: MarketInterpretation | None = None
```

**Step 2: Update runner to include sentiment/market in result**

In `PipelineRunner.execute()`, include the analysis results in the PipelineResult so formatters can display them:
```python
return PipelineResult(
    ...
    sentiment=sentiment_result.output,
    market=market_result.output,
)
```

**Step 3: Update aggregator for new take_profit type**

No changes needed to validation logic — `take_profit` is still a list, just with different item type.

**Step 4: Run tests**

Run: `cd orchestrator && uv run pytest tests/unit/test_runner.py -v`

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/pipeline/runner.py orchestrator/src/orchestrator/pipeline/aggregator.py orchestrator/tests/unit/test_runner.py
git commit -m "feat: pass sentiment and market data through PipelineResult"
```

---

### Task 9: Run full test suite, lint, and fix any remaining issues

**Step 1: Run full test suite**

Run: `cd orchestrator && uv run pytest tests/ -v --tb=short`

**Step 2: Run linter**

Run: `cd orchestrator && uv run ruff check src/ tests/`

**Step 3: Run coverage check**

Run: `cd orchestrator && uv run pytest tests/ --cov=orchestrator --cov-report=term-missing`

**Step 4: Fix any failures or lint issues**

**Step 5: Final commit**

```bash
git add -A
git commit -m "chore: fix lint and remaining test issues"
```

---

## Dependency Order

```
Task 1 (models) → Task 2 (market agent) → Task 3 (proposer agent)
Task 1 (models) → Task 4 (margin sizer)
Task 1 (models) → Task 5 (paper engine)
Task 5 (paper engine) → Task 6 (formatters)
Task 5 (paper engine) + Task 4 (margin sizer) → Task 7 (telegram bot)
Task 3 (proposer) → Task 8 (pipeline runner)
All → Task 9 (integration)
```

Tasks 2, 3, 4, 5 can be parallelized after Task 1 is complete.
