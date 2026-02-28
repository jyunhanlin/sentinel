# Approval & Telegram Codemap

**Last Updated:** 2026-02-28

## Overview

Semi-auto trading: proposals are sent to Telegram with inline Approve/Reject buttons. User approval required before execution.

## Architecture

```
PipelineRunner
    │
    ├─ ApprovalManager
    │   ├─ Create approval record
    │   └─ Send Telegram message with buttons
    │
    ├─ Telegram Bot
    │   ├─ /start, /status, /coin, /history, /perf, /eval, /resume, /run
    │   ├─ Button handlers (approve, reject, translate)
    │   └─ Formatters (messages, prices, reports)
    │
    └─ Notification System
        ├─ Trade proposals
        ├─ Approvals/rejections
        ├─ Execution status
        └─ Risk pauses
```

## Approval Manager

**File:** `orchestrator/src/orchestrator/approval/manager.py`

Manages approval state machine and Telegram integration.

### PendingApproval

```python
class PendingApproval(BaseModel, frozen=True):
    approval_id: str  # UUID
    proposal_id: str  # Link to trade proposal
    proposal: TradeProposal
    snapshot_price: float  # Price at approval time
    created_at: datetime
    expires_at: datetime
    message_id: int | None = None  # Telegram message ID
```

### ApprovalManager

```python
class ApprovalManager:
    def __init__(
        self,
        *,
        approval_repo: ApprovalRepository,
        telegram_bot: SentinelBot,
        approval_timeout_minutes: int = 15,
        price_deviation_threshold: float = 0.01,  # 1%
    ) -> None:
        self._approval_repo = approval_repo
        self._telegram_bot = telegram_bot
        self._timeout_minutes = approval_timeout_minutes
        self._price_deviation_threshold = price_deviation_threshold
        self._pending: dict[str, PendingApproval] = {}

    async def create_approval(
        self,
        *,
        proposal: TradeProposal,
        current_price: float,
    ) -> str:
        """
        Create approval request and send Telegram message.
        Returns approval_id.
        """
        approval_id = str(uuid.uuid4())
        expires_at = datetime.now(UTC) + timedelta(
            minutes=self._timeout_minutes
        )

        pending = PendingApproval(
            approval_id=approval_id,
            proposal_id=proposal.proposal_id,
            proposal=proposal,
            snapshot_price=current_price,
            created_at=datetime.now(UTC),
            expires_at=expires_at,
        )

        self._pending[approval_id] = pending

        # Send Telegram message
        message_id = await self._telegram_bot.send_approval_request(
            approval_id=approval_id,
            proposal=proposal,
            snapshot_price=current_price,
        )

        # Store in database
        self._approval_repo.create(
            ApprovalRecord(
                approval_id=approval_id,
                proposal_id=proposal.proposal_id,
                snapshot_price=current_price,
                message_id=message_id,
                expires_at=expires_at,
            )
        )

        return approval_id

    async def handle_approval(self, approval_id: str, approved: bool) -> bool:
        """
        User clicked button. Check if still valid, update state.
        Returns True if approved, False if rejected/expired.
        """
        pending = self._pending.get(approval_id)
        if not pending:
            return False

        # Check expiration
        if datetime.now(UTC) > pending.expires_at:
            await self._telegram_bot.notify_approval_expired(approval_id)
            return False

        # Check price deviation
        if approved:
            # In real usage, would re-fetch current price
            # For now, skip (Telegram bot could re-fetch before calling)
            pass

        # Update database
        self._approval_repo.update(
            approval_id,
            status="approved" if approved else "rejected",
            resolved_at=datetime.now(UTC),
        )

        # Remove from pending
        del self._pending[approval_id]

        return approved

    async def check_and_expire_approvals(self) -> None:
        """Periodically check for expired approvals."""
        now = datetime.now(UTC)
        expired_ids = [
            aid for aid, pending in self._pending.items()
            if now > pending.expires_at
        ]

        for aid in expired_ids:
            pending = self._pending[aid]
            await self._telegram_bot.notify_approval_expired(aid)
            self._approval_repo.update(aid, status="expired")
            del self._pending[aid]
```

## Telegram Bot

**File:** `orchestrator/src/orchestrator/telegram/bot.py`

Telegram command and callback handlers.

### SentinelBot

```python
class SentinelBot:
    def __init__(
        self,
        *,
        token: str,
        admin_chat_ids: list[int],
        approval_manager: ApprovalManager,
        pipeline_scheduler: PipelineScheduler,
        trade_repo: PaperTradeRepository,
        paper_engine: PaperEngine,
        stats_calculator: StatsCalculator,
        eval_runner: EvalRunner,
    ) -> None:
        self._token = token
        self._admin_chat_ids = admin_chat_ids
        self._approval_manager = approval_manager
        self._pipeline_scheduler = pipeline_scheduler
        self._trade_repo = trade_repo
        self._paper_engine = paper_engine
        self._stats_calculator = stats_calculator
        self._eval_runner = eval_runner

        # Build Telegram application
        self._app = Application.builder().token(token).build()
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Register command and callback handlers."""
        # Commands
        self._app.add_handler(CommandHandler("start", self._start))
        self._app.add_handler(CommandHandler("status", self._status))
        self._app.add_handler(CommandHandler("coin", self._coin))
        self._app.add_handler(CommandHandler("history", self._history))
        self._app.add_handler(CommandHandler("perf", self._perf))
        self._app.add_handler(CommandHandler("eval", self._eval))
        self._app.add_handler(CommandHandler("resume", self._resume))
        self._app.add_handler(CommandHandler("run", self._run))

        # Callbacks (inline buttons)
        self._app.add_handler(CallbackQueryHandler(
            self._approve_callback, pattern="^approve:"
        ))
        self._app.add_handler(CallbackQueryHandler(
            self._reject_callback, pattern="^reject:"
        ))
        self._app.add_handler(CallbackQueryHandler(
            self._translate_callback, pattern="^translate:"
        ))

    async def start(self) -> None:
        """Start polling."""
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()

    async def stop(self) -> None:
        """Stop polling."""
        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()
```

### Command Handlers

#### /start
```python
async def _start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Welcome message."""
    chat_id = update.effective_chat.id
    if not self._is_admin(chat_id):
        await update.message.reply_text("Unauthorized")
        return

    text = format_welcome()
    await context.bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
```

#### /status
```python
async def _status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Account overview + latest signals."""
    chat_id = update.effective_chat.id
    if not self._is_admin(chat_id):
        return

    # Get account snapshot
    state = self._stats_calculator.calculate_current_state()

    # Get open positions
    open_positions = self._paper_engine.get_open_positions()

    # Get recent signals
    recent_runs = self._pipeline_repo.get_recent()[:5]

    text = format_status(state, open_positions, recent_runs)
    await context.bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
```

#### /coin
```python
async def _coin(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Detailed analysis for a symbol."""
    symbol = context.args[0] if context.args else "BTC"
    symbol = f"{symbol}/USDT:USDT"

    # Fetch snapshot
    snapshot = await self._data_fetcher.fetch_snapshot(symbol)

    # Run agents
    sentiment = await self._sentiment_agent.analyze(snapshot=snapshot)
    market = await self._market_agent.analyze(snapshot=snapshot)
    proposal = await self._proposer_agent.analyze(
        snapshot=snapshot,
        sentiment=sentiment.output,
        market=market.output,
    )

    # Format
    text = format_coin_analysis(snapshot, sentiment.output, market.output, proposal.output)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=text, parse_mode="HTML")
```

#### /run
```python
async def _run(
    self, update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Manually trigger pipeline."""
    # Parse args: /run [symbol] [sonnet|opus]
    symbol = context.args[0] if context.args else "BTC/USDT:USDT"
    model_override = None
    if len(context.args) > 1:
        model_name = context.args[1]
        if model_name in MODEL_ALIASES:
            model_override = MODEL_ALIASES[model_name]

    # Run pipeline
    result = await self._pipeline_scheduler.run_once(
        symbol=symbol,
        model_override=model_override,
    )

    # Send result
    text = format_pipeline_result(result)
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=text,
        parse_mode="HTML",
    )
```

#### /history
```python
async def _history(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Recent closed trades."""
    closed_trades = self._trade_repo.get_closed_trades(limit=10)
    text = format_trade_history(closed_trades)
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=text,
        parse_mode="HTML",
    )
```

#### /perf
```python
async def _perf(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Performance report."""
    stats = self._stats_calculator.calculate()
    text = format_perf_report(stats)
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=text,
        parse_mode="HTML",
    )
```

#### /eval
```python
async def _eval(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Run LLM evaluation."""
    report = await self._eval_runner.run_default()
    text = format_eval_report(report)
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=text,
        parse_mode="HTML",
    )
```

#### /resume
```python
async def _resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Resume after risk pause."""
    self._pipeline_scheduler.resume()
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Trading resumed ✓",
    )
```

### Callback Handlers (Inline Buttons)

#### Approve Button
```python
async def _approve_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """User clicked Approve button."""
    query = update.callback_query
    approval_id = query.data.split(":")[1]

    # Handle approval
    approved = await self._approval_manager.handle_approval(approval_id, approved=True)

    if approved:
        # Execute trade
        pending = self._approval_manager.get_pending(approval_id)
        result = await self._executor.execute_entry(
            proposal=pending.proposal,
            current_price=pending.snapshot_price,
        )

        # Notify user
        text = format_execution_result(result)
        await _safe_callback_reply(query, text=text)
    else:
        await _safe_callback_reply(query, text="Approval expired or invalid")
```

#### Reject Button
```python
async def _reject_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """User clicked Reject button."""
    query = update.callback_query
    approval_id = query.data.split(":")[1]

    await self._approval_manager.handle_approval(approval_id, approved=False)
    await _safe_callback_reply(query, text="Proposal rejected ✓")
```

#### Translate Button
```python
async def _translate_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Toggle message language (English ↔ 中文)."""
    query = update.callback_query
    lang = query.data.split(":")[1]  # "zh" or "en"

    # Get original message text
    original_text = query.message.text

    # Translate
    translated = to_chinese(original_text) if lang == "zh" else original_text

    # Show keyboard for reverse translation
    keyboard = _english_keyboard() if lang == "zh" else _translate_keyboard()

    await query.edit_message_text(
        text=translated,
        reply_markup=keyboard,
        parse_mode="HTML",
    )
```

## Message Formatting

**File:** `orchestrator/src/orchestrator/telegram/formatters.py`

Functions to format messages for Telegram.

### Proposal Message

```python
def format_proposal(
    approval_id: str,
    proposal: TradeProposal,
    snapshot_price: float,
) -> tuple[str, InlineKeyboardMarkup]:
    """
    Format a trade proposal with inline buttons.
    Returns (message_text, keyboard).
    """
    lines = [
        f"<b>{proposal.symbol} {proposal.side.value.upper()}</b>",
        f"Entry: {proposal.entry.type} @ {proposal.entry.price or snapshot_price:.2f}",
        f"SL: {proposal.stop_loss:.2f}",
        f"TP: {', '.join(str(tp.price) for tp in proposal.take_profit)}",
        f"Risk: {proposal.position_size_risk_pct:.1f}%",
        f"Leverage: {proposal.suggested_leverage}x",
        f"Confidence: {proposal.confidence:.1%}",
        f"",
        f"<i>{proposal.rationale}</i>",
    ]

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✓ Approve", callback_data=f"approve:{approval_id}"),
            InlineKeyboardButton("✗ Reject", callback_data=f"reject:{approval_id}"),
        ],
        [
            InlineKeyboardButton("Translate to zh-TW", callback_data="translate:zh"),
        ],
    ])

    return "\n".join(lines), keyboard
```

### Status Message

```python
def format_status(
    state: AccountState,
    open_positions: list[OpenPosition],
    recent_signals: list[SignalRecord],
) -> str:
    """Format account status."""
    lines = [
        f"<b>Account Overview</b>",
        f"Equity: ${state.equity:,.2f}",
        f"Daily PnL: ${state.daily_pnl:.2f}",
        f"Win Rate: {state.win_rate:.1%}",
        f"Max Drawdown: {state.max_drawdown_pct:.1%}",
        f"",
        f"<b>Open Positions: {len(open_positions)}</b>",
    ]

    for pos in open_positions:
        lines.append(
            f"  {pos.symbol}: {pos.side.value.upper()} {pos.quantity:.4f} "
            f"@ {pos.entry_price:.2f} (LIQ: {pos.liquidation_price:.2f})"
        )

    lines.extend([
        f"",
        f"<b>Recent Signals</b>",
    ])

    for signal in recent_signals:
        lines.append(
            f"  {signal.symbol}: {signal.proposal.side.value.upper()} "
            f"(confidence: {signal.proposal.confidence:.1%})"
        )

    return "\n".join(lines)
```

## Integration with Pipeline

When pipeline produces a proposal that passes risk checks:

```python
# In PipelineRunner.execute()

# Risk check passed
if risk_result.approved:
    # Create approval request
    approval_id = await approval_manager.create_approval(
        proposal=proposal,
        current_price=snapshot.current_price,
    )

    # Return pending approval
    return PipelineResult(
        status="pending_approval",
        approval_id=approval_id,
        proposal=proposal,
    )
else:
    if risk_result.action == "pause":
        # Send risk pause notification
        await telegram_bot.notify_risk_pause(risk_result)
        # Pipeline stops, requires /resume
        return PipelineResult(
            status="risk_paused",
            rejection_reason=risk_result.reason,
        )
    else:
        # Send rejection notification
        await telegram_bot.notify_proposal_rejected(proposal, risk_result)
        return PipelineResult(
            status="risk_rejected",
            rejection_reason=risk_result.reason,
        )
```

## Configuration

Environment variables:

```python
telegram_bot_token: str  # Required, from @BotFather
telegram_admin_chat_ids: list[int]  # Comma-separated list
approval_timeout_minutes: int = 15
price_deviation_threshold: float = 0.01  # 1%
```

Example `.env`:
```
TELEGRAM_BOT_TOKEN=1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh
TELEGRAM_ADMIN_CHAT_IDS=123456789,987654321
APPROVAL_TIMEOUT_MINUTES=15
PRICE_DEVIATION_THRESHOLD=0.01
```

## Workflows

### Approval Flow

```
1. Pipeline generates proposal
2. RiskChecker approves
3. ApprovalManager creates approval
4. Telegram bot sends message with buttons
5. User clicks Approve/Reject within 15 min
6. Callback handler updates approval status
7. If approved: OrderExecutor fills entry
8. User notified of execution
```

### Risk Pause Flow

```
1. Pipeline generates proposal
2. RiskChecker returns action="pause" (too many losses, etc.)
3. Telegram bot sends pause notification
4. Trading stops (scheduler paused)
5. User must run /resume command
6. Scheduler resumes
```

## Testing

**File:** `tests/unit/test_telegram.py`

Mocks Telegram API:

```python
@patch("telegram.ext.Application")
async def test_approve_callback(mock_app):
    bot = SentinelBot(...)
    # Simulate button click
    query = MagicMock(spec=CallbackQuery)
    query.data = "approve:approval-123"

    await bot._approve_callback(query, context)

    # Verify trade was executed
    assert mock_executor.execute_entry.called
```

## Related

- [pipeline.md](pipeline.md) — Where approvals are created
- [execution.md](execution.md) — Trade execution on approval
- [storage.md](storage.md) — Approval record persistence
