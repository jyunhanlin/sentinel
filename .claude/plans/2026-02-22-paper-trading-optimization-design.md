# Paper Trading Optimization Design

**Date:** 2026-02-22
**Status:** Approved

## Goal

Optimize the paper trading experience to simulate real perpetual futures trading on exchanges (Binance-style), with manual position management via Telegram inline buttons.

## Current State

- PaperEngine: in-memory + DB, positions only close via SL/TP
- No manual operations (add/reduce/close)
- No leverage/margin simulation
- `/status` shows basic info, no unrealized PnL
- `/history` shows last 10 closed trades, no filtering/pagination

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Position model | Single position + avg price update | Simplest, minimal changes, matches user mental model |
| Immutability | Create new Position to replace old | Maintains frozen=True convention |
| Leverage | Per-trade selection on approve | Matches real exchange UX |
| Margin mode | Cross margin (MVP) | Simpler; isolated margin can be added later |
| Risk % | Kept internally for LLM proposals | LLM proposes risk %, engine converts to leverage/margin for display |
| Interaction | Telegram inline buttons | One-tap operations, no command typing |

## Section 1: Position Model + PaperEngine

### Position Model Changes

Add fields to existing `Position` (frozen Pydantic model):

```python
class Position(BaseModel, frozen=True):
    # existing fields...
    leverage: int               # leverage multiplier (e.g. 10)
    margin: float               # used margin = notional / leverage
    liquidation_price: float    # forced liquidation price
```

### PaperEngine New Methods

```python
def add_to_position(self, trade_id: str, risk_pct: float, current_price: float) -> Position:
    """Add to position: calculate new qty, update avg price, replace Position."""
    # 1. Find existing position
    # 2. PositionSizer calculates additional qty (risk_pct + current_price + existing SL)
    # 3. New avg_entry = (old_qty * old_entry + new_qty * current_price) / total_qty
    # 4. New margin = old_margin + (new_qty * current_price / leverage)
    # 5. Recalculate liquidation price
    # 6. Create new Position, replace in _positions list
    # 7. Save to DB, deduct fee

def reduce_position(self, trade_id: str, pct: float, current_price: float) -> CloseResult:
    """Reduce position: close partial qty, record partial PnL."""
    # 1. Find existing position
    # 2. close_qty = quantity * pct / 100
    # 3. Calculate partial PnL
    # 4. Release proportional margin
    # 5. Create new Position with remaining qty (avg price unchanged)
    # 6. Replace in _positions list
    # 7. Record partial close to DB, deduct fee

def close_position(self, trade_id: str, current_price: float) -> CloseResult:
    """Close entire position. Equivalent to reduce_position(100%)."""

def get_position_with_pnl(self, trade_id: str, current_price: float) -> dict:
    """Get position with unrealized PnL and ROE%."""
    # unrealized_pnl = (current_price - entry_price) * quantity * direction
    # roe_pct = unrealized_pnl / margin * 100
```

### Account State

```
Total Equity = initial_equity + closed_pnl - total_fees + unrealized_pnl
Available Balance = equity - total_used_margin - unrealized_losses
Used Margin = sum of all open positions' margin
```

### DB Changes

`PaperTradeRecord` new fields:

```python
leverage: int | None          # leverage multiplier
margin: float | None          # used margin
liquidation_price: float | None
close_reason: str | None      # "sl" | "tp" | "liquidation" | "manual" | "partial_reduce"
original_entry: float | None  # original entry before avg price updates
```

Add/reduce operations also create DB records (as operation logs) for audit trail.

## Section 2: Telegram UI Interaction

### Position Cards (/status)

```
📊 Account Overview
Equity: $10,150.00 (+1.5%)
Available: $7,320.00 | Used Margin: $2,680.00
Open Positions: 2

────────────────────
BTC/USDT:USDT · LONG · 10x
Entry: $68,333 | Qty: 0.15 BTC
Margin: $1,025 | Liq: $61,880
SL: $67,000 | TP: $70,000
PnL: +$125.00 (+1.22% / ROE +12.2%)
Duration: 2h 35m
[➕ Add] [➖ Reduce] [❌ Close]
```

### Interaction Flows

**Add (加倉):**
```
[➕ Add]
→ "Add to BTC LONG — Select risk %"
→ [0.5%] [1%] [2%]
→ "Confirm: Add to BTC LONG with 1% risk at ~$69,500?"
→ [✅ Confirm] [❌ Cancel]
→ Execute, update position card
```

**Reduce (減倉):**
```
[➖ Reduce]
→ "Reduce BTC LONG — Select amount"
→ [25%] [50%] [75%]
→ "Confirm: Reduce 50% of BTC LONG (0.075 BTC) at ~$69,500?"
→ [✅ Confirm] [❌ Cancel]
→ Execute, show partial PnL, update position card
```

**Close (平倉):**
```
[❌ Close]
→ "Confirm: Close entire BTC LONG (0.15 BTC) at ~$69,500?"
→ [✅ Confirm] [❌ Cancel]
→ Execute, show CloseResult
```

### Approve Flow (with leverage selection)

```
LLM proposal message (unchanged):
  BTC/USDT LONG | Risk: 1% | SL: $67,000 | TP: $70,000
  [Approve] [Reject]

User clicks [Approve]:
→ "Select leverage for BTC LONG"
→ [5x] [10x] [20x] [50x]

User selects [10x]:
→ "BTC LONG · 10x Leverage
   Entry: ~$68,000 | Qty: 0.1 BTC
   Margin: $680 | Liq. Price: ~$61,880
   SL: $67,000 | TP: $70,000"
→ [✅ Confirm] [❌ Cancel]
```

### Callback Data Format

```
add:{trade_id}                    → show risk % options
add:{trade_id}:{risk_pct}        → show confirmation
confirm_add:{trade_id}:{risk_pct} → execute
reduce:{trade_id}                 → show % options
reduce:{trade_id}:{pct}          → show confirmation
confirm_reduce:{trade_id}:{pct}  → execute
close:{trade_id}                  → show confirmation
confirm_close:{trade_id}         → execute
cancel:{message_context}         → cancel operation
leverage:{approval_id}:{value}   → select leverage during approve
confirm_leverage:{approval_id}:{value} → confirm and execute
```

## Section 3: History Optimization

### /history Improvements

```
📜 Trade History (Page 1/3)

1. BTC/USDT LONG · Closed (SL) · 10x
   Entry: $68,000 → Exit: $67,000
   PnL: -$100.00 (-1.47% / ROE -14.7%) | Duration: 3h 20m
   Feb 22, 14:30

2. ETH/USDT SHORT · Closed (TP) · 20x
   Entry: $2,450 → Exit: $2,300
   PnL: +$300.00 (+6.12% / ROE +122.4%) | Duration: 1d 2h
   Feb 21, 09:15

[⬅️ Prev] [Page 1/3] [➡️ Next]
[🔍 BTC] [🔍 ETH] [All]
```

### Filtering & Pagination

- Filter by symbol: inline buttons per active symbol
- Pagination: 5 trades per page, prev/next buttons
- Callback: `history:page:{n}`, `history:filter:{symbol}`, `history:filter:all`

### Repository Changes

```python
def get_closed_paginated(
    self, offset: int = 0, limit: int = 5, symbol: str | None = None
) -> tuple[list[PaperTradeRecord], int]:
    """Return (trades, total_count) with pagination and filtering."""

def get_trade_by_id(self, trade_id: str) -> PaperTradeRecord | None:
    """Single trade lookup for operation confirmations."""
```

## Section 4: Leverage & Margin Simulation

### Liquidation Price Calculation

**Cross margin, simplified Binance formula:**

```
LONG:  liq_price = entry_price * (1 - 1/leverage + maintenance_margin_rate/100)
SHORT: liq_price = entry_price * (1 + 1/leverage - maintenance_margin_rate/100)
```

### Liquidation Check

In `check_sl_tp()`, add liquidation check with highest priority:

```
Priority: Liquidation > SL > TP
LONG:  if current_price <= liquidation_price → liquidate (loss ≈ margin)
SHORT: if current_price >= liquidation_price → liquidate (loss ≈ margin)
```

### Margin Validation

Before opening/adding to position:
- Check `available_balance >= required_margin`
- If insufficient → reject with "Insufficient margin" message

### Config

```python
paper_default_leverage: int = 10
paper_maintenance_margin_rate: float = 0.5  # Binance default for BTC
paper_leverage_options: list[int] = [5, 10, 20, 50]
```

## Scope Boundaries

**In scope:**
- Manual add/reduce/close via TG inline buttons
- Leverage selection on approve
- Margin/liquidation simulation (cross margin)
- Unrealized PnL + ROE% display
- History filtering and pagination
- Operation audit trail in DB

**Out of scope (future):**
- Isolated margin mode
- Funding rate simulation
- Partial TP (multi-target)
- Trailing stop
- Market/limit order types (all fills at current price)
