# M4 Design: Semi-Auto Trading with Approval Flow

**Date:** 2026-02-19
**Status:** Approved
**Depends on:** M3 (Eval + Performance Stats)

---

## Goal

Transform the pipeline from fully-automatic paper trading to semi-automatic execution with TG inline keyboard approval. Add switchable Paper/Live execution mode with exchange-native SL/TP orders.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Approval UX | TG Inline Keyboard [Approve/Reject] | One-tap interaction, no command typing |
| Approval timeout | 15 minutes | Matches pipeline interval; market conditions change |
| Execution mode | Config switchable paper/live | Test approval flow with paper first, then go live |
| Live SL/TP | Exchange stop orders via CCXT | Works even when bot is offline |
| Safety limits | Existing Risk Checker | No additional absolute $ cap needed |
| Pipeline behavior | All runs wait for approval in semi-auto mode | Fully semi-auto, no auto-execution |
| Rust executor | Deferred | Focus on Python semi-auto MVP first |

## Architecture

### New / Modified Modules

```
execution/
  executor.py            OrderExecutor ABC + PaperExecutor + LiveExecutor
  order_manager.py       SL/TP order placement and tracking

approval/
  manager.py             PendingApproval state machine + expiry logic
  models.py              PendingApproval, ApprovalRecord models

pipeline/
  runner.py              + pending_approval status path

telegram/
  bot.py                 + CallbackQueryHandler, InlineKeyboardMarkup
  formatters.py          + format_pending_approval(), format_execution_result()

storage/
  models.py              + extend PaperTradeRecord with live fields
  repository.py          + ApprovalRepository

config.py                + trading_mode, approval_timeout_minutes, price_deviation_threshold
__main__.py              + wiring new components
```

### Pipeline Flow (Semi-Auto Mode)

```
Scheduler / TG /run
    │
    ▼
Runner.execute(symbol)
    │
    ① DataFetcher.fetch_snapshot()
    │
    ② PaperEngine.check_sl_tp(snapshot)  ← still runs for paper positions
    │
    ③ [asyncio.gather] Sentiment + Market agents
    │
    ④ ProposerAgent.propose()
    │
    ⑤ Aggregator.validate()
    │   └─ ❌ → rejected, skip to ⑧
    │
    ⑥ RiskChecker.check()
    │   ├─ ❌ → rejected/paused, skip to ⑧
    │   └─ ✅ → continue
    │
    ⑦ ApprovalManager.create(proposal)  ← NEW: don't execute yet
    │   └─ returns PendingApproval with approval_id
    │
    ⑧ Storage.save(run, llm_calls, proposal, risk_result)
    │
    ⑨ TG Push
        ├─ approved by risk → InlineKeyboard [Approve] [Reject]
        ├─ rejected         → rejection notification
        └─ SL/TP (②)       → close report

--- async callback ---

User clicks [Approve]:
    │
    ▼
CallbackHandler
    │
    ① ApprovalManager.approve(approval_id)
    │   └─ checks not expired
    │
    ② Re-fetch current price
    │   └─ check deviation from proposal time
    │   └─ > 1% threshold → notify user, don't execute
    │
    ③ OrderExecutor.execute_entry(proposal, current_price)
    │   ├─ PaperExecutor: paper_engine.open_position()
    │   └─ LiveExecutor: CCXT market order + SL/TP stop orders
    │
    ④ TG Push: execution confirmation

User clicks [Reject]:
    │
    ▼
CallbackHandler
    │
    ① ApprovalManager.reject(approval_id)
    │
    ② TG Update: mark message as rejected

15min timeout:
    │
    ▼
Expiry scheduler job (every 1 min)
    │
    ① ApprovalManager.expire_stale()
    │
    ② TG Update: mark message as expired
```

## Component Details

### ApprovalManager (`approval/manager.py`)

In-memory state + DB persistence. Manages PendingApproval lifecycle.

```python
class PendingApproval(BaseModel, frozen=True):
    approval_id: str
    proposal: TradeProposal
    run_id: str
    snapshot_price: float        # price at proposal time
    created_at: datetime
    expires_at: datetime         # created_at + timeout
    status: str = "pending"      # pending | approved | rejected | expired
    message_id: int | None = None  # TG message ID for editing

class ApprovalManager:
    _pending: dict[str, PendingApproval]

    def create(self, *, proposal, run_id, snapshot_price, timeout_minutes) -> PendingApproval
    def approve(self, approval_id) -> PendingApproval | None
    def reject(self, approval_id) -> PendingApproval | None
    def get(self, approval_id) -> PendingApproval | None
    def expire_stale(self) -> list[PendingApproval]
    def get_pending_count(self) -> int
```

### OrderExecutor (`execution/executor.py`)

Strategy pattern. Two implementations switchable via config.

```python
class ExecutionResult(BaseModel, frozen=True):
    trade_id: str
    symbol: str
    side: str
    entry_price: float
    quantity: float
    fees: float
    mode: str                    # "paper" | "live"
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
        stop_loss: float, take_profit: list[float]
    ) -> list[str]: ...  # returns order IDs

    @abstractmethod
    async def cancel_orders(self, order_ids: list[str]) -> None: ...

class PaperExecutor(OrderExecutor):
    """Wraps existing PaperEngine. Delegates to paper_engine.open_position()."""

class LiveExecutor(OrderExecutor):
    """Uses CCXT async client for real exchange orders."""
    # execute_entry: market order
    # place_sl_tp: stop_market + take_profit_market orders
    # cancel_orders: cancel by order ID
```

### LiveExecutor Safety

Before executing:
1. **Re-fetch price** via `exchange_client.fetch_ticker()`
2. **Deviation check**: `abs(current - snapshot_price) / snapshot_price`
   - `<= price_deviation_threshold` (default 1%) → execute
   - `> threshold` → reject with message: "Price deviated {X}% since proposal"
3. **Order execution**: Market order → get actual fill price
4. **SL/TP placement**: Exchange stop orders immediately after fill
5. **Record keeping**: Store exchange order IDs in DB

### Storage Extensions

**PaperTradeRecord** new fields:
```python
mode: str = "paper"              # "paper" | "live"
exchange_order_id: str = ""      # main order ID (live)
sl_order_id: str = ""            # SL stop order ID (live)
tp_order_id: str = ""            # TP stop order ID (live)
```

**New table: `approval_records`**
```python
class ApprovalRecord(SQLModel, table=True):
    __tablename__ = "approval_records"

    id: int | None = Field(default=None, primary_key=True)
    approval_id: str = Field(unique=True, index=True)
    proposal_id: str = Field(index=True)
    run_id: str
    snapshot_price: float
    status: str = "pending"       # pending | approved | rejected | expired
    message_id: int | None = None
    created_at: datetime
    expires_at: datetime
    resolved_at: datetime | None = None
```

### TG Enhancements

**InlineKeyboardMarkup for proposals:**

```python
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

keyboard = InlineKeyboardMarkup([
    [
        InlineKeyboardButton("✅ Approve", callback_data=f"approve:{approval_id}"),
        InlineKeyboardButton("❌ Reject", callback_data=f"reject:{approval_id}"),
    ]
])
```

**CallbackQueryHandler:**
- Parse `callback_data` → extract action + approval_id
- `approve:` → execute order flow
- `reject:` → mark rejected, edit message

**Message editing on resolve:**
- On approve: edit original message to append "✅ APPROVED — executed at $X"
- On reject: edit to append "❌ REJECTED by user"
- On expire: edit to append "⏰ EXPIRED"

**New formatters:**

`format_pending_approval(approval: PendingApproval) -> str`:
```
[PENDING APPROVAL] BTC/USDT:USDT
Side: LONG
Entry: market @ ~$95,200
Risk: 1.5%
SL: $93,000 | TP: $97,000
Confidence: 75%
Rationale: Strong breakout above resistance...

⏱ Expires in 15 minutes
```

`format_execution_result(result: ExecutionResult) -> str`:
```
✅ [EXECUTED] BTC/USDT:USDT LONG
Mode: live
Entry: $95,350 | Qty: 0.075 BTC
SL order: $93,000
TP order: $97,000
```

### Config Additions

```python
# Semi-auto trading
trading_mode: str = "paper"                    # "paper" | "live"
approval_timeout_minutes: int = 15
price_deviation_threshold: float = 0.01        # 1%
```

### ExchangeClient Extensions

New methods on existing CCXT wrapper for live trading:

```python
class ExchangeClient:
    # Existing:
    async def fetch_ohlcv(...)
    async def fetch_ticker(...)

    # New for M4:
    async def create_market_order(self, symbol, side, amount) -> dict
    async def create_stop_order(self, symbol, side, amount, stop_price) -> dict
    async def cancel_order(self, order_id, symbol) -> dict
    async def fetch_order(self, order_id, symbol) -> dict
```

## Out of Scope

- Rust executor (future milestone)
- Limit orders (market only for MVP)
- Partial close / trailing stop (future)
- Multi-exchange live trading (future)
- Automated approval rules (e.g., auto-approve if confidence > 90%)
- WebSocket real-time price monitoring
