# M2 Design: Risk Management + Paper Trading

**Date:** 2026-02-18
**Status:** Approved
**Depends on:** M1 (3-model pipeline)

---

## Goal

Add risk gate and paper trading engine after the M1 pipeline, completing the proposal → risk check → simulated execution → TG report loop.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Price check frequency | Pipeline trigger only (every 15min) | MVP simplest; no extra polling loop |
| Initial equity | Config fixed value (default $10,000) | Resettable, no exchange API dependency |
| Risk rejection | Reject + TG notification | Full transparency |
| Same-symbol positions | Allow multiple | More flexible position management |
| Total exposure calc | Sum all open positions' risk% | Simple and explicit |
| Performance stats | Deferred to M3 | M2 stores raw data only |
| Engine architecture | In-memory ledger + DB persistence | Simple, fast; rebuild from DB on restart |

## Architecture

### New / Modified Modules

```
risk/
  checker.py              Rule-based risk engine (non-LLM, pure logic)
  position_sizer.py       Risk % position size calculator (strategy pattern)

exchange/
  paper_engine.py         In-memory ledger + DB persistence

agents/
  utils.py                Shared utilities (summarize_ohlcv)

pipeline/
  runner.py               + risk check + paper execution integration

telegram/
  bot.py                  + /history, /resume, trade report push
  formatters.py           + format_trade_report(), format_risk_rejection()

storage/
  repository.py           + PaperTradeRepository, AccountSnapshotRepository

config.py                 + paper trading config fields
__main__.py               + wiring new components
```

### Pipeline Data Flow (after M2)

```
Scheduler / TG /run
    │
    ▼
Runner.execute(symbol)
    │
    ① DataFetcher.fetch_snapshot()
    │
    ② PaperEngine.check_sl_tp(snapshot)
    │   └─ current_price crosses SL/TP → close position → TG push
    │   └─ engine_state is now up-to-date
    │
    ③ [asyncio.gather]
    │   ├── SentimentAgent.analyze(snapshot)
    │   └── MarketAgent.analyze(snapshot)
    │
    ④ ProposerAgent.propose(sentiment, market, snapshot)
    │
    ⑤ Aggregator.validate(proposal)
    │   └─ proposal-level sanity (SL direction, field completeness)
    │   └─ ❌ → rejected, skip to ⑧
    │
    ⑥ RiskChecker.check(proposal, engine_state)
    │   └─ account-level risk rules (exposure, daily loss, etc.)
    │   ├─ ✅ ALL PASS → ⑦
    │   └─ ❌ ANY FAIL → rejected/paused, skip to ⑧
    │
    ⑦ PaperEngine.open_position(proposal, snapshot)
    │   └─ PositionSizer calculates quantity
    │   └─ Simulated fill at current_price, deduct taker fee
    │
    ⑧ Storage.save(run, llm_calls, proposal, risk_result, trade)
    │
    ⑨ TG Push
        ├─ approved  → proposal + fill report
        ├─ rejected  → proposal + rejection reason
        ├─ paused    → proposal + pause notification
        └─ SL/TP (②) → close report
```

## Component Details

### Risk Checker (`risk/checker.py`)

Pure code logic, no LLM. Checks rules sequentially; any failure stops evaluation.

```python
class RiskResult(BaseModel, frozen=True):
    approved: bool
    rule_violated: str = ""     # e.g. "max_total_exposure"
    reason: str = ""            # e.g. "Total exposure 22% exceeds 20% limit"
    action: str = "reject"      # "reject" | "pause"
```

| Rule | Input | Condition | Action |
|------|-------|-----------|--------|
| Max single risk | proposal | `risk_pct > max_single_risk_pct` (2%) | Reject |
| Max total exposure | proposal + all open positions | `sum(open_risk%) + new_risk% > max_total_exposure_pct` (20%) | Reject |
| Max consecutive losses | recent closed trades | `consecutive_losses >= max_consecutive_losses` (5) | Pause |
| Max daily loss | today's closed trades | `daily_loss_pct > max_daily_loss_pct` (5%) | Pause |
| invalid_if check | proposal.invalid_if + snapshot | proposal self-declared cancel conditions | Cancel |

Note: Direction sanity is already handled by M1 Aggregator. RiskChecker focuses on account-level constraints.

**Pause vs Reject:**
- **Reject**: blocks this proposal only
- **Pause**: sets `pipeline_paused = True`, blocks all subsequent proposals until manual `/resume` or next-day auto-reset

### Position Sizer (`risk/position_sizer.py`)

Strategy pattern. MVP implements Risk % mode only.

```python
class PositionSizer(ABC):
    @abstractmethod
    def calculate(self, *, equity: float, risk_pct: float,
                  entry_price: float, stop_loss: float) -> float:
        """Return quantity in base currency units."""

class RiskPercentSizer(PositionSizer):
    """quantity = (equity * risk_pct / 100) / abs(entry - stop_loss)"""
```

Example: equity=$10,000, risk=1.5%, entry=$95,000, SL=$93,000
→ risk_amount=$150, distance=$2,000, quantity=0.075 BTC

### Paper Trading Engine (`exchange/paper_engine.py`)

In-memory ledger with DB persistence. Rebuild from DB on startup.

```python
class PaperEngine:
    _initial_equity: float
    _positions: dict[str, list[Position]]  # symbol → [Position, ...]
    _closed_pnl: float
    _total_fees: float
    _paused: bool
```

**Position model:**

```python
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
```

**Core methods:**

| Method | Trigger | Logic |
|--------|---------|-------|
| `open_position(proposal, snapshot)` | After risk check pass (⑦) | PositionSizer → quantity, fill at current_price, deduct taker fee, write DB |
| `check_sl_tp(snapshot)` | Pipeline start (②) | Iterate positions, check current_price vs SL/TP, close if triggered |
| `close_position(trade_id, exit_price)` | SL/TP trigger or manual | Calculate PnL + fee, remove from _positions, write DB |
| `rebuild_from_db()` | App startup | Read all open PaperTradeRecords, reconstruct _positions |

**Fill model (MVP simplified):**

| Order type | Logic |
|-----------|-------|
| Market (open) | Fill at `current_price`, no slippage |
| SL trigger | `current_price` crosses `stop_loss` → fill at SL price |
| TP trigger | `current_price` crosses `take_profit[0]` → fill at TP price, close full position |

**Fee calculation:**
- Open: `quantity * entry_price * taker_fee_rate` (default 0.05%)
- Close: `quantity * exit_price * taker_fee_rate`

**Equity:**
```
equity = initial_equity + closed_pnl - total_fees
```
Excludes unrealized PnL (MVP simplification).

### Telegram Enhancements

**New commands:**

| Command | Function |
|---------|----------|
| `/history` | Recent N closed trades (symbol, side, PnL, duration) |
| `/resume` | Manually un-pause pipeline after risk pause |

**New push notifications:**
- SL/TP triggered → trade close report to admins
- Risk rejection → rejection reason to admins

**New formatters:**

`format_trade_report()`:
```
[CLOSED] BTC/USDT:USDT
Side: LONG
Entry: 95,000.0 → Exit: 93,000.0 (SL)
Quantity: 0.075 BTC
PnL: -$150.00 (fees: $7.13)
Duration: 2h 15m
```

`format_risk_rejection()`:
```
[RISK REJECTED] BTC/USDT:USDT
Proposed: LONG @ 95,000
Rule: max_total_exposure
Reason: Total exposure 22% exceeds 20% limit
```

### Config Additions

```python
# Paper Trading
paper_initial_equity: float = 10000.0
paper_taker_fee_rate: float = 0.0005   # 0.05%
paper_maker_fee_rate: float = 0.0002   # 0.02%
```

Risk fields already exist from M0: `max_single_risk_pct`, `max_total_exposure_pct`, `max_daily_loss_pct`, `max_consecutive_losses`.

### Storage Additions

DB tables `paper_trades` and `account_snapshots` already defined in M0. M2 adds repositories:

```python
class PaperTradeRepository:
    save_trade(...)
    update_trade_closed(trade_id, exit_price, pnl, fees)
    get_open_positions() → list[PaperTradeRecord]
    get_recent_closed(limit=10) → list[PaperTradeRecord]
    count_consecutive_losses() → int
    get_daily_pnl(date) → float

class AccountSnapshotRepository:
    save_snapshot(equity, open_count, daily_pnl)
    get_latest() → AccountSnapshotRecord | None
```

## M1 Review Issues (fixed in M2)

| # | Issue | Fix |
|---|-------|-----|
| 1 | `_summarize_ohlcv` duplicated in sentiment.py and market.py | Extract to `agents/utils.py` with `max_candles` parameter |
| 4 | `_latest_results` in-memory only | `/status` `/coin` read from DB via `TradeProposalRepository` |
| 5 | Prompt field hardcoded `"(see messages)"` | Store full messages JSON in `_save_llm_calls` |

## Out of Scope

- Performance statistics (M3)
- Evaluation dataset / regression tests (M3)
- Approval flow for proposals (M4)
- Live trading (M4)
- WebSocket real-time price monitoring (future)
- Slippage simulation (future)
