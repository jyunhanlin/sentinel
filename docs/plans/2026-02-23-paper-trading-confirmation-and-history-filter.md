# Paper Trading: Confirmation Steps, History Filters & Test Coverage

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add intermediate confirmation cards to approve/add/reduce flows, render symbol filter buttons in /history, and fill test coverage gaps.

**Architecture:** Split each existing two-step flow into three steps (select → confirm card → execute) by introducing new callback actions (`leverage:`, `select_add:`, `select_reduce:`) that show a confirmation card with computed details. The existing `confirm_*` handlers remain as the final execution step. History filtering adds a `get_distinct_closed_symbols` repository method and renders filter buttons below pagination.

**Tech Stack:** Python 3.12+, python-telegram-bot, pytest, structlog

---

### Task 1: Add Leverage Confirmation Card to Approve Flow

Currently `_handle_approve` shows leverage buttons with `confirm_leverage:{id}:{lev}` callback, which directly executes. We need an intermediate step: clicking a leverage button shows a confirmation card with margin/liq details, then the user clicks Confirm to execute.

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/bot.py:573-586` (dispatch table)
- Modify: `orchestrator/src/orchestrator/telegram/bot.py:670-702` (`_handle_approve`)
- Add handler: `_handle_leverage_preview` (new method)
- Test: `orchestrator/tests/unit/test_telegram.py`

**Step 1: Write the failing test**

Add to `TestApproveWithLeverage` in `test_telegram.py`:

```python
@pytest.mark.asyncio
async def test_leverage_selection_shows_confirmation_card(self):
    """After selecting leverage, bot should show confirmation with margin/liq details."""
    bot = self._make_bot()
    approval = MagicMock()
    approval.approval_id = "abc123"
    approval.proposal.symbol = "BTC/USDT:USDT"
    approval.proposal.side = Side.LONG
    approval.proposal.side.value = "long"
    approval.proposal.stop_loss = 67000.0
    approval.proposal.take_profit = [70000.0]
    approval.proposal.position_size_risk_pct = 1.0
    approval.snapshot_price = 68000.0
    bot._approval_manager.get.return_value = approval
    bot._data_fetcher.fetch_current_price = AsyncMock(return_value=68000.0)

    # Paper engine needs to provide calculation methods
    bot._paper_engine.calculate_margin.return_value = 680.0
    bot._paper_engine.calculate_liquidation_price.return_value = 61540.0
    bot._paper_engine._position_sizer.calculate.return_value = 0.1

    query = self._make_callback_query("leverage:abc123:10")
    update = MagicMock()
    update.callback_query = query
    update.effective_chat = MagicMock()
    update.effective_chat.id = 123
    context = MagicMock()

    await bot._callback_router(update, context)

    # Should show confirmation card with details, not execute
    query.edit_message_text.assert_called_once()
    call_kwargs = query.edit_message_text.call_args
    text = call_kwargs.kwargs.get("text", call_kwargs.args[0] if call_kwargs.args else "")
    assert "Margin" in text
    assert "Liq" in text

    markup = call_kwargs.kwargs.get("reply_markup")
    all_data = [
        btn.callback_data for row in markup.inline_keyboard for btn in row
        if btn.callback_data
    ]
    assert any("confirm_leverage:" in d for d in all_data)
    assert any("cancel:" in d for d in all_data)
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_telegram.py::TestApproveWithLeverage::test_leverage_selection_shows_confirmation_card -v`
Expected: FAIL — `"leverage"` action not in dispatch table

**Step 3: Implement**

In `bot.py`:

1. Change `_handle_approve` to use `leverage:{approval_id}:{lev}` instead of `confirm_leverage:{approval_id}:{lev}` for the button callbacks:

```python
# In _handle_approve, change the button callback_data:
# FROM: f"confirm_leverage:{approval_id}:{lev}"
# TO:   f"leverage:{approval_id}:{lev}"
```

2. Add `"leverage"` to the dispatch table:

```python
"leverage":         (3, "_handle_leverage_preview"),
```

3. Add the new handler method after `_handle_approve`:

```python
async def _handle_leverage_preview(
    self, query: CallbackQuery, approval_id: str, leverage_str: str, *_args: str,
) -> None:
    """Show confirmation card with margin/liq details before executing."""
    leverage = int(leverage_str)
    if self._approval_manager is None or self._executor is None:
        await query.answer("Not configured")
        return
    if self._paper_engine is None:
        await query.answer("Paper engine not configured")
        return

    approval = self._approval_manager.get(approval_id)
    if approval is None:
        await query.answer("Expired or not found")
        await query.edit_message_text("Approval expired or already handled.")
        return

    p = approval.proposal
    current_price = approval.snapshot_price
    if self._data_fetcher is not None:
        current_price = await self._data_fetcher.fetch_current_price(p.symbol)

    qty = self._paper_engine._position_sizer.calculate(
        equity=self._paper_engine.equity,
        risk_pct=p.position_size_risk_pct,
        entry_price=current_price,
        stop_loss=p.stop_loss,
    )
    margin = self._paper_engine.calculate_margin(
        quantity=qty, price=current_price, leverage=leverage,
    )
    liq = self._paper_engine.calculate_liquidation_price(
        entry_price=current_price, leverage=leverage, side=p.side,
    )

    side_str = p.side.value.upper()
    tp_str = ", ".join(f"${tp:,.1f}" for tp in p.take_profit) if p.take_profit else "—"
    text = (
        f"━━ CONFIRM ORDER ━━\n"
        f"{p.symbol}  {side_str} · {leverage}x\n\n"
        f"Entry: ~${current_price:,.1f} | Qty: {qty:.4f}\n"
        f"Margin: ${margin:,.2f} | Liq: ~${liq:,.1f}\n"
        f"SL: ${p.stop_loss:,.1f} | TP: {tp_str}\n"
    )
    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                "Confirm", callback_data=f"confirm_leverage:{approval_id}:{leverage}",
            ),
            InlineKeyboardButton("Cancel", callback_data="cancel:0"),
        ],
    ])
    await _safe_callback_reply(query, text=text, reply_markup=keyboard)
```

**Step 4: Run test to verify it passes**

Run: `cd orchestrator && uv run pytest tests/unit/test_telegram.py::TestApproveWithLeverage -v`
Expected: All PASS

**Step 5: Run full tests**

Run: `cd orchestrator && uv run pytest --tb=short`
Expected: All PASS

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/bot.py orchestrator/tests/unit/test_telegram.py
git commit -m "feat: add confirmation card with margin/liq details to approve flow"
```

---

### Task 2: Add Confirmation Card to Add Flow

Currently clicking `[0.5%]` sends `confirm_add:{trade_id}:0.5` and executes immediately. We need an intermediate step that shows how much will be added.

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/bot.py:573-586` (dispatch table)
- Modify: `orchestrator/src/orchestrator/telegram/bot.py:926-953` (`_handle_add`)
- Add handler: `_handle_select_add` (new method)
- Test: `orchestrator/tests/unit/test_telegram.py`

**Step 1: Write the failing tests**

Add `TestAddFlow` class to `test_telegram.py`:

```python
class TestAddFlow:
    def _make_bot(self):
        return SentinelBot(
            token="test-token",
            admin_chat_ids=[123],
            paper_engine=MagicMock(),
            data_fetcher=MagicMock(),
            trade_repo=MagicMock(),
        )

    def _make_update(self, data: str):
        query = MagicMock()
        query.data = data
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        query.message = MagicMock()
        query.message.message_id = 1
        query.message.reply_markup = None
        update = MagicMock()
        update.callback_query = query
        update.effective_chat = MagicMock()
        update.effective_chat.id = 123
        return update, query

    @pytest.mark.asyncio
    async def test_add_shows_risk_options(self):
        """Clicking Add button should show risk % selection."""
        bot = self._make_bot()
        bot._paper_engine.get_position_with_pnl.return_value = {
            "position": MagicMock(
                symbol="BTC/USDT:USDT", side=Side.LONG, leverage=10,
                entry_price=68000.0, quantity=0.1, margin=680.0,
                liquidation_price=61540.0, stop_loss=67000.0,
                take_profit=[70000.0], opened_at=datetime.now(UTC),
                trade_id="t1",
            ),
            "unrealized_pnl": 50.0, "pnl_pct": 0.74, "roe_pct": 7.35,
        }
        bot._data_fetcher.fetch_current_price = AsyncMock(return_value=68500.0)

        update, query = self._make_update("add:t1")
        await bot._callback_router(update, MagicMock())

        markup = query.edit_message_text.call_args.kwargs.get("reply_markup")
        all_data = [
            btn.callback_data for row in markup.inline_keyboard for btn in row
            if btn.callback_data
        ]
        # Should point to select_add (confirmation step), not confirm_add (execution)
        assert any("select_add:" in d for d in all_data)

    @pytest.mark.asyncio
    async def test_select_add_shows_confirmation(self):
        """After selecting risk %, show confirmation card with details."""
        bot = self._make_bot()
        pos = MagicMock(
            symbol="BTC/USDT:USDT", side=Side.LONG, leverage=10,
            entry_price=68000.0, quantity=0.1, margin=680.0,
            stop_loss=67000.0, trade_id="t1",
        )
        bot._paper_engine._find_position.return_value = pos
        bot._paper_engine._position_sizer.calculate.return_value = 0.05
        bot._paper_engine.calculate_margin.return_value = 345.0
        bot._paper_engine.equity = 10000.0
        bot._data_fetcher.fetch_current_price = AsyncMock(return_value=69000.0)

        update, query = self._make_update("select_add:t1:1.0")
        await bot._callback_router(update, MagicMock())

        text = query.edit_message_text.call_args.kwargs.get(
            "text", query.edit_message_text.call_args.args[0]
        )
        assert "Confirm" in text or "CONFIRM" in text
        markup = query.edit_message_text.call_args.kwargs.get("reply_markup")
        all_data = [
            btn.callback_data for row in markup.inline_keyboard for btn in row
            if btn.callback_data
        ]
        assert any("confirm_add:" in d for d in all_data)

    @pytest.mark.asyncio
    async def test_confirm_add_executes(self):
        """Clicking Confirm should execute the add operation."""
        bot = self._make_bot()
        updated_pos = MagicMock(
            symbol="BTC/USDT:USDT", side=MagicMock(value="long"),
            leverage=10, entry_price=68500.0, quantity=0.15, margin=1025.0,
        )
        bot._paper_engine.add_to_position.return_value = updated_pos
        bot._data_fetcher.fetch_current_price = AsyncMock(return_value=69000.0)

        update, query = self._make_update("confirm_add:t1:1.0")
        await bot._callback_router(update, MagicMock())

        bot._paper_engine.add_to_position.assert_called_once()
```

**Step 2: Run tests to verify they fail**

Run: `cd orchestrator && uv run pytest tests/unit/test_telegram.py::TestAddFlow -v`
Expected: FAIL — `select_add` not in dispatch table

**Step 3: Implement**

1. In `_handle_add`, change button callbacks from `confirm_add:{trade_id}:{risk}` to `select_add:{trade_id}:{risk}`:

```python
# In _handle_add, change:
# FROM: f"confirm_add:{trade_id}:0.5"
# TO:   f"select_add:{trade_id}:0.5"
# (same for 1.0 and 2.0)
```

2. Add `"select_add"` to dispatch table:

```python
"select_add":       (3, "_handle_select_add"),
```

3. Add the new handler:

```python
async def _handle_select_add(
    self, query: CallbackQuery, trade_id: str, risk_pct_str: str, *_args: str,
) -> None:
    """Show confirmation card for add operation."""
    risk_pct = float(risk_pct_str)
    if self._paper_engine is None or self._data_fetcher is None:
        await query.answer("Not configured")
        return

    try:
        pos = self._paper_engine._find_position(trade_id)
        current_price = await self._data_fetcher.fetch_current_price(pos.symbol)
    except ValueError:
        await query.answer("Position not found")
        return

    add_qty = self._paper_engine._position_sizer.calculate(
        equity=self._paper_engine.equity,
        risk_pct=risk_pct,
        entry_price=current_price,
        stop_loss=pos.stop_loss,
    )
    add_margin = self._paper_engine.calculate_margin(
        quantity=add_qty, price=current_price, leverage=pos.leverage,
    )

    side_str = pos.side.value.upper() if hasattr(pos.side, "value") else str(pos.side).upper()
    text = (
        f"━━ CONFIRM ADD ━━\n"
        f"{pos.symbol}  {side_str} {pos.leverage}x\n\n"
        f"Current Price: ~${current_price:,.1f}\n"
        f"Add Qty: {add_qty:.4f} | Add Margin: ${add_margin:,.2f}\n"
        f"Risk: {risk_pct}%"
    )
    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                "Confirm Add", callback_data=f"confirm_add:{trade_id}:{risk_pct}",
            ),
            InlineKeyboardButton("Cancel", callback_data="cancel:0"),
        ],
    ])
    await _safe_callback_reply(query, text=text, reply_markup=keyboard)
```

**Step 4: Run tests to verify they pass**

Run: `cd orchestrator && uv run pytest tests/unit/test_telegram.py::TestAddFlow -v`
Expected: All PASS

**Step 5: Run full tests**

Run: `cd orchestrator && uv run pytest --tb=short`
Expected: All PASS

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/bot.py orchestrator/tests/unit/test_telegram.py
git commit -m "feat: add confirmation card to add-to-position flow"
```

---

### Task 3: Add Confirmation Card to Reduce Flow

Currently clicking `[25%]` sends `confirm_reduce:{trade_id}:25` and executes immediately. We need an intermediate step showing estimated PnL.

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/bot.py:573-586` (dispatch table)
- Modify: `orchestrator/src/orchestrator/telegram/bot.py:875-903` (`_handle_reduce`)
- Add handler: `_handle_select_reduce` (new method)
- Test: `orchestrator/tests/unit/test_telegram.py`

**Step 1: Write the failing tests**

Add `TestReduceFlow` class to `test_telegram.py`:

```python
class TestReduceFlow:
    def _make_bot(self):
        return SentinelBot(
            token="test-token",
            admin_chat_ids=[123],
            paper_engine=MagicMock(),
            data_fetcher=MagicMock(),
            trade_repo=MagicMock(),
        )

    def _make_update(self, data: str):
        query = MagicMock()
        query.data = data
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        query.message = MagicMock()
        query.message.message_id = 1
        query.message.reply_markup = None
        update = MagicMock()
        update.callback_query = query
        update.effective_chat = MagicMock()
        update.effective_chat.id = 123
        return update, query

    @pytest.mark.asyncio
    async def test_reduce_shows_pct_options(self):
        """Clicking Reduce should show percentage selection."""
        bot = self._make_bot()
        bot._paper_engine.get_position_with_pnl.return_value = {
            "position": MagicMock(
                symbol="BTC/USDT:USDT", side=Side.LONG, leverage=10,
                entry_price=68000.0, quantity=0.1, margin=680.0,
                liquidation_price=61540.0, stop_loss=67000.0,
                take_profit=[70000.0], opened_at=datetime.now(UTC),
                trade_id="t1",
            ),
            "unrealized_pnl": 100.0, "pnl_pct": 1.47, "roe_pct": 14.7,
        }
        bot._data_fetcher.fetch_current_price = AsyncMock(return_value=69000.0)

        update, query = self._make_update("reduce:t1")
        await bot._callback_router(update, MagicMock())

        markup = query.edit_message_text.call_args.kwargs.get("reply_markup")
        all_data = [
            btn.callback_data for row in markup.inline_keyboard for btn in row
            if btn.callback_data
        ]
        assert any("select_reduce:" in d for d in all_data)

    @pytest.mark.asyncio
    async def test_select_reduce_shows_confirmation(self):
        """After selecting %, show confirmation card with estimated PnL."""
        bot = self._make_bot()
        pos = MagicMock(
            symbol="BTC/USDT:USDT", side=Side.LONG, leverage=10,
            entry_price=68000.0, quantity=0.1, margin=680.0,
            stop_loss=67000.0, trade_id="t1",
        )
        bot._paper_engine._find_position.return_value = pos
        bot._data_fetcher.fetch_current_price = AsyncMock(return_value=69000.0)

        update, query = self._make_update("select_reduce:t1:50")
        await bot._callback_router(update, MagicMock())

        text = query.edit_message_text.call_args.kwargs.get(
            "text", query.edit_message_text.call_args.args[0]
        )
        assert "Confirm" in text or "CONFIRM" in text
        assert "PnL" in text or "pnl" in text
        markup = query.edit_message_text.call_args.kwargs.get("reply_markup")
        all_data = [
            btn.callback_data for row in markup.inline_keyboard for btn in row
            if btn.callback_data
        ]
        assert any("confirm_reduce:" in d for d in all_data)

    @pytest.mark.asyncio
    async def test_confirm_reduce_executes(self):
        """Clicking Confirm should execute the reduce operation."""
        bot = self._make_bot()
        close_result = MagicMock()
        close_result.pnl = 50.0
        close_result.symbol = "BTC/USDT:USDT"
        close_result.side = Side.LONG
        close_result.entry_price = 68000.0
        close_result.exit_price = 69000.0
        close_result.quantity = 0.05
        close_result.fees = 1.7
        close_result.reason = "partial_reduce"
        bot._paper_engine.reduce_position.return_value = close_result
        bot._data_fetcher.fetch_current_price = AsyncMock(return_value=69000.0)

        update, query = self._make_update("confirm_reduce:t1:50")
        await bot._callback_router(update, MagicMock())

        bot._paper_engine.reduce_position.assert_called_once()
```

**Step 2: Run tests to verify they fail**

Run: `cd orchestrator && uv run pytest tests/unit/test_telegram.py::TestReduceFlow -v`
Expected: FAIL — `select_reduce` not in dispatch table

**Step 3: Implement**

1. In `_handle_reduce`, change button callbacks from `confirm_reduce:{trade_id}:{pct}` to `select_reduce:{trade_id}:{pct}`:

```python
# In _handle_reduce, change:
# FROM: f"confirm_reduce:{trade_id}:25"
# TO:   f"select_reduce:{trade_id}:25"
# (same for 50 and 75)
```

2. Add `"select_reduce"` to dispatch table:

```python
"select_reduce":    (3, "_handle_select_reduce"),
```

3. Add the new handler:

```python
async def _handle_select_reduce(
    self, query: CallbackQuery, trade_id: str, pct_str: str, *_args: str,
) -> None:
    """Show confirmation card for reduce operation."""
    pct = float(pct_str)
    if self._paper_engine is None or self._data_fetcher is None:
        await query.answer("Not configured")
        return

    try:
        pos = self._paper_engine._find_position(trade_id)
        current_price = await self._data_fetcher.fetch_current_price(pos.symbol)
    except ValueError:
        await query.answer("Position not found")
        return

    close_qty = pos.quantity * pct / 100
    direction = 1 if pos.side == Side.LONG else -1
    est_pnl = (current_price - pos.entry_price) * close_qty * direction
    pnl_sign = "+" if est_pnl >= 0 else ""

    side_str = pos.side.value.upper() if hasattr(pos.side, "value") else str(pos.side).upper()
    text = (
        f"━━ CONFIRM REDUCE ━━\n"
        f"{pos.symbol}  {side_str} {pos.leverage}x\n\n"
        f"Close {pct:.0f}%: {close_qty:.4f} at ~${current_price:,.1f}\n"
        f"Est. PnL: {pnl_sign}${est_pnl:,.2f}"
    )
    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                "Confirm Reduce", callback_data=f"confirm_reduce:{trade_id}:{pct_str}",
            ),
            InlineKeyboardButton("Cancel", callback_data="cancel:0"),
        ],
    ])
    await _safe_callback_reply(query, text=text, reply_markup=keyboard)
```

**Step 4: Run tests to verify they pass**

Run: `cd orchestrator && uv run pytest tests/unit/test_telegram.py::TestReduceFlow -v`
Expected: All PASS

**Step 5: Run full tests (check existing TestPositionOperationCallbacks still pass)**

Run: `cd orchestrator && uv run pytest --tb=short`
Expected: All PASS. Note: The existing `test_reduce_shows_pct_options` in `TestPositionOperationCallbacks` needs to be updated to expect `select_reduce:` instead of `confirm_reduce:`. Fix this when the test fails.

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/bot.py orchestrator/tests/unit/test_telegram.py
git commit -m "feat: add confirmation card to reduce-position flow"
```

---

### Task 4: Add Symbol Filter Buttons to /history

The `_handle_history_callback` already handles `history:filter:{symbol}`, but filter buttons are never rendered. We need a repository method to get distinct closed symbols, then render them as buttons.

**Files:**
- Modify: `orchestrator/src/orchestrator/storage/repository.py` (add `get_distinct_closed_symbols`)
- Modify: `orchestrator/src/orchestrator/telegram/bot.py:475-503` (`_history_handler`)
- Modify: `orchestrator/src/orchestrator/telegram/bot.py:982-1024` (`_handle_history_callback`)
- Test: `orchestrator/tests/unit/test_repository.py`
- Test: `orchestrator/tests/unit/test_telegram.py`

**Step 1: Write the failing repository test**

Add to `test_repository.py`:

```python
class TestDistinctClosedSymbols:
    def test_returns_unique_symbols(self, repo):
        for i, sym in enumerate(["BTC/USDT:USDT", "ETH/USDT:USDT", "BTC/USDT:USDT"]):
            repo.save_trade(
                trade_id=f"t{i}", proposal_id=f"p{i}", symbol=sym,
                side="long", entry_price=68000.0, quantity=0.1,
            )
            repo.update_trade_closed(f"t{i}", exit_price=69000.0, pnl=100.0, fees=3.4)
        symbols = repo.get_distinct_closed_symbols()
        assert sorted(symbols) == ["BTC/USDT:USDT", "ETH/USDT:USDT"]

    def test_returns_empty_when_no_closed(self, repo):
        repo.save_trade(
            trade_id="t1", proposal_id="p1", symbol="BTC/USDT:USDT",
            side="long", entry_price=68000.0, quantity=0.1,
        )
        # Not closed
        assert repo.get_distinct_closed_symbols() == []
```

**Step 2: Run tests to verify they fail**

Run: `cd orchestrator && uv run pytest tests/unit/test_repository.py::TestDistinctClosedSymbols -v`
Expected: FAIL — method doesn't exist

**Step 3: Implement repository method**

Add to `PaperTradeRepository` in `repository.py` (after `get_closed_paginated`):

```python
def get_distinct_closed_symbols(self) -> list[str]:
    """Return unique symbols from closed trades."""
    from sqlmodel import func
    stmt = (
        select(PaperTradeRecord.symbol)
        .where(PaperTradeRecord.status == "closed")
        .group_by(PaperTradeRecord.symbol)
        .order_by(func.count().desc())
    )
    return list(self._session.exec(stmt).all())
```

**Step 4: Run repository tests to verify they pass**

Run: `cd orchestrator && uv run pytest tests/unit/test_repository.py::TestDistinctClosedSymbols -v`
Expected: PASS

**Step 5: Write the failing history filter test**

Add to `test_telegram.py`:

```python
class TestHistoryFilter:
    @pytest.mark.asyncio
    async def test_history_shows_filter_buttons(self):
        """When there are closed trades, /history should show symbol filter buttons."""
        bot = SentinelBot(
            token="test-token",
            admin_chat_ids=[123],
            trade_repo=MagicMock(),
        )
        bot._trade_repo.get_closed_paginated.return_value = (
            [MagicMock(
                symbol="BTC/USDT:USDT", side="long", leverage=10,
                entry_price=68000.0, exit_price=69000.0,
                pnl=100.0, fees=3.4, close_reason="manual", margin=680.0,
            )],
            1,
        )
        bot._trade_repo.get_distinct_closed_symbols.return_value = [
            "BTC/USDT:USDT", "ETH/USDT:USDT",
        ]

        update = _make_update(123)
        await bot._history_handler(update, _make_context())

        call_kwargs = update.message.reply_text.call_args
        markup = call_kwargs.kwargs.get("reply_markup") or call_kwargs[1].get("reply_markup")
        all_data = [
            btn.callback_data for row in markup.inline_keyboard for btn in row
            if btn.callback_data
        ]
        assert any("history:filter:" in d for d in all_data)

    @pytest.mark.asyncio
    async def test_history_filter_preserves_across_pagination(self):
        """Pagination buttons should preserve the active symbol filter."""
        bot = SentinelBot(
            token="test-token",
            admin_chat_ids=[123],
            trade_repo=MagicMock(),
        )
        bot._trade_repo.get_closed_paginated.return_value = (
            [MagicMock(
                symbol="BTC/USDT:USDT", side="long", leverage=10,
                entry_price=68000.0, exit_price=69000.0,
                pnl=100.0, fees=3.4, close_reason="manual", margin=680.0,
            )] * 5,
            12,  # total > page_size, so next button should appear
        )
        bot._trade_repo.get_distinct_closed_symbols.return_value = ["BTC/USDT:USDT"]

        query = MagicMock()
        query.data = "history:filter:BTC/USDT:USDT"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        query.message = MagicMock()
        query.message.message_id = 1
        query.message.reply_markup = None

        update = MagicMock()
        update.callback_query = query
        update.effective_chat = MagicMock()
        update.effective_chat.id = 123

        await bot._callback_router(update, MagicMock())

        call_kwargs = query.edit_message_text.call_args
        markup = call_kwargs.kwargs.get("reply_markup")
        all_data = [
            btn.callback_data for row in markup.inline_keyboard for btn in row
            if btn.callback_data
        ]
        # Next button should include filter: "history:page:2:BTC/USDT:USDT"
        next_btns = [d for d in all_data if d.startswith("history:page:")]
        assert len(next_btns) > 0
        # The page callback should carry the symbol filter
        assert any("BTC" in d for d in next_btns)
```

**Step 6: Run tests to verify they fail**

Run: `cd orchestrator && uv run pytest tests/unit/test_telegram.py::TestHistoryFilter -v`
Expected: FAIL

**Step 7: Implement history filter buttons and preserved pagination**

Update `_history_handler` in `bot.py` to build filter buttons:

```python
async def _history_handler(
    self, update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    if not await self._check_admin(update):
        return
    if self._trade_repo is None:
        await self._reply(update, "Paper trading not configured.")
        return

    from orchestrator.telegram.formatters import format_history_paginated

    page_size = 5
    trades, total = self._trade_repo.get_closed_paginated(offset=0, limit=page_size)
    total_pages = max(1, (total + page_size - 1) // page_size)
    text = format_history_paginated(trades, page=1, total_pages=total_pages)

    rows: list[list[InlineKeyboardButton]] = []

    # Nav row
    nav_buttons: list[InlineKeyboardButton] = []
    nav_buttons.append(
        InlineKeyboardButton(f"Page 1/{total_pages}", callback_data="cancel:0")
    )
    if total_pages > 1:
        nav_buttons.append(
            InlineKeyboardButton("Next", callback_data="history:page:2")
        )
    rows.append(nav_buttons)

    # Filter row
    symbols = self._trade_repo.get_distinct_closed_symbols()
    if len(symbols) > 1:
        filter_buttons = [
            InlineKeyboardButton(
                sym.split("/")[0],  # "BTC" from "BTC/USDT:USDT"
                callback_data=f"history:filter:{sym}",
            )
            for sym in symbols[:4]  # max 4 filter buttons
        ]
        filter_buttons.append(
            InlineKeyboardButton("All", callback_data="history:filter:all")
        )
        rows.append(filter_buttons)

    if update.message:
        await update.message.reply_text(
            text, reply_markup=InlineKeyboardMarkup(rows),
        )
```

Update `_handle_history_callback` to:
1. Support `history:filter:all` (reset filter)
2. Preserve symbol filter in pagination callbacks by encoding as `history:page:{n}:{symbol}`
3. Parse page callbacks that include a symbol filter

```python
async def _handle_history_callback(
    self, query: CallbackQuery, action: str, value: str, *extra: str,
) -> None:
    # Rejoin extra parts for symbols with colons (e.g. "BTC/USDT:USDT")
    if extra:
        value = ":".join([value, *extra])
    if self._trade_repo is None:
        await query.answer("Not configured")
        return

    from orchestrator.telegram.formatters import format_history_paginated

    page_size = 5
    page = 1
    symbol_filter: str | None = None

    if action == "page":
        # value may be "2" or "2:BTC/USDT:USDT" (page with filter)
        parts = value.split(":", 1)
        page = int(parts[0])
        if len(parts) > 1:
            symbol_filter = parts[1]
    elif action == "filter":
        if value != "all":
            symbol_filter = value

    offset = (page - 1) * page_size
    trades, total = self._trade_repo.get_closed_paginated(
        offset=offset, limit=page_size, symbol=symbol_filter,
    )
    total_pages = max(1, (total + page_size - 1) // page_size)
    text = format_history_paginated(trades, page=page, total_pages=total_pages)

    rows: list[list[InlineKeyboardButton]] = []

    # Nav row — encode filter in page callbacks
    filter_suffix = f":{symbol_filter}" if symbol_filter else ""
    nav_buttons: list[InlineKeyboardButton] = []
    if page > 1:
        nav_buttons.append(
            InlineKeyboardButton("Prev", callback_data=f"history:page:{page - 1}{filter_suffix}")
        )
    nav_buttons.append(
        InlineKeyboardButton(f"Page {page}/{total_pages}", callback_data="cancel:0")
    )
    if page < total_pages:
        nav_buttons.append(
            InlineKeyboardButton("Next", callback_data=f"history:page:{page + 1}{filter_suffix}")
        )
    rows.append(nav_buttons)

    # Filter row
    symbols = self._trade_repo.get_distinct_closed_symbols()
    if len(symbols) > 1:
        filter_buttons = [
            InlineKeyboardButton(
                sym.split("/")[0],
                callback_data=f"history:filter:{sym}",
            )
            for sym in symbols[:4]
        ]
        filter_buttons.append(
            InlineKeyboardButton("All", callback_data="history:filter:all")
        )
        rows.append(filter_buttons)

    keyboard = InlineKeyboardMarkup(rows)
    await _safe_callback_reply(query, text=text, reply_markup=keyboard)
```

**Step 8: Run tests to verify they pass**

Run: `cd orchestrator && uv run pytest tests/unit/test_telegram.py::TestHistoryFilter tests/unit/test_repository.py::TestDistinctClosedSymbols -v`
Expected: All PASS

**Step 9: Run full tests**

Run: `cd orchestrator && uv run pytest --tb=short`
Expected: All PASS. Note: `TestHistoryPagination` tests may need to be updated if they assert on `_handle_history_callback`'s behavior — check and fix if needed.

**Step 10: Commit**

```bash
git add orchestrator/src/orchestrator/storage/repository.py orchestrator/src/orchestrator/telegram/bot.py orchestrator/tests/unit/test_repository.py orchestrator/tests/unit/test_telegram.py
git commit -m "feat: add symbol filter buttons and preserved filter to /history"
```

---

### Task 5: Final Verification

**Step 1: Run full test suite**

Run: `cd orchestrator && uv run pytest -v --tb=short`
Expected: All PASS

**Step 2: Run linter**

Run: `cd orchestrator && uv run ruff check src/ tests/`
Expected: No new errors

**Step 3: Fix any issues**

If ruff reports line-length or other issues in new code, fix them.

**Step 4: Commit fixups if any**

```bash
git commit -m "chore: fix lint issues in confirmation and history filter changes"
```
