# Agent Leverage Recommendation & Paper Trading Optimization

**Date:** 2026-02-24
**Status:** Approved

## Goal

1. Agent produces concrete futures trading recommendations (entry, SL/TP, leverage)
2. Paper trading accepts USDT margin input instead of risk %
3. Support partial take-profit (split TP with different close percentages)
4. Detailed report format on Telegram with full analysis

## Design Decisions

- **Leverage strategy**: LLM-recommended based on volatility (Approach A — simplest)
- **Position sizing**: User inputs USDT margin amount, system calculates quantity
- **Leverage does NOT affect SL/TP prices** — SL/TP are price levels from technical analysis; leverage affects margin, liquidation price, and ROE
- **Partial TP**: Multiple take-profit levels with close percentages (e.g., TP1 close 50%, TP2 close 100%)

---

## 1. Model Changes

### 1.1 MarketInterpretation — add volatility_pct

```python
class MarketInterpretation(BaseModel, frozen=True):
    trend: Trend
    volatility_regime: VolatilityRegime
    volatility_pct: float  # ATR-based volatility as % of price, e.g. 2.3 = 2.3%
    key_levels: list[KeyLevel]
    risk_flags: list[str]
```

`volatility_pct` is the average true range (ATR) of recent candles as a percentage of current price. MarketAgent estimates this from OHLCV data.

### 1.2 TradeProposal — add suggested_leverage, change take_profit

```python
class TakeProfit(BaseModel, frozen=True):
    price: float
    close_pct: int = Field(ge=1, le=100)  # % of remaining position to close

class TradeProposal(BaseModel, frozen=True):
    # ... existing fields ...
    suggested_leverage: int = Field(ge=1, le=50, default=10)
    take_profit: list[TakeProfit]  # was list[float]
```

Leverage guidelines for ProposerAgent:
- `volatility_pct < 2%` → suggest up to 20x
- `volatility_pct 2-4%` → suggest 10x
- `volatility_pct > 4%` → suggest 5x
- `confidence < 0.5` → cap at 5x

---

## 2. Agent Prompt Changes

### 2.1 MarketAgent

Add `volatility_pct` to output schema:

```json
{
  "trend": "up",
  "volatility_regime": "low",
  "volatility_pct": 1.8,
  "key_levels": [...],
  "risk_flags": [...]
}
```

Prompt addition: "Calculate volatility_pct as the average true range of the last 14 candles divided by current price, expressed as a percentage."

### 2.2 ProposerAgent

Update output schema:

```json
{
  "symbol": "BTC/USDT:USDT",
  "side": "long",
  "entry": {"type": "limit", "price": 64800},
  "position_size_risk_pct": 1.0,
  "stop_loss": 64000,
  "take_profit": [
    {"price": 65800, "close_pct": 50},
    {"price": 67000, "close_pct": 100}
  ],
  "suggested_leverage": 10,
  "time_horizon": "4h",
  "confidence": 0.72,
  "invalid_if": ["price drops below 63500"],
  "rationale": "Strong support at 64k..."
}
```

Prompt additions:
- Pass `volatility_pct` from MarketAgent into context
- Add leverage recommendation rules (vol-based guidelines above)
- `take_profit` now requires `close_pct` per level; last TP should be 100%

---

## 3. PaperEngine — Partial Take-Profit

### 3.1 Position tracking

Add to Position model:
```python
triggered_tp_indices: list[int] = []  # indices of TPs already triggered
take_profit_levels: list[TakeProfit] = []  # the full TP plan
```

### 3.2 check_sl_tp logic change

Current: triggers first TP → full close.

New:
1. Check liquidation (unchanged — highest priority)
2. Check SL (unchanged — full close)
3. Check TPs in order:
   - For each untriggered TP where price crosses the level:
     - If `close_pct == 100` → `close_position()` (full close)
     - Else → `reduce_position(trade_id, pct=close_pct)` (partial close)
     - Mark TP index as triggered
     - Optional: after TP1, move SL to entry price (breakeven)

### 3.3 CloseResult extension

```python
class CloseResult:
    # ... existing fields ...
    partial: bool = False  # True for partial TP closes
    remaining_quantity: float | None = None
```

---

## 4. Position Sizing — USDT Margin Input

### Current flow
```
risk_pct → quantity = (equity × risk_pct / 100) / |entry - SL|
```

### New flow
```
margin_usdt (user input) → quantity = margin_usdt × leverage / entry_price
```

The `RiskPercentSizer` remains available for risk checks, but actual position size comes from user's USDT input.

### Validation
- `margin_usdt <= available_balance` (existing check)
- Warn if implied risk % > 5% of equity

---

## 5. Telegram Message Format

### 5.1 Proposal notification (detailed report)

```
🟢 LONG BTC/USDT:USDT
──────────────────────
▶ Entry:  $64,800 (limit)
⛔ SL:     $64,000 (-1.2%)
✅ TP1:    $65,800 (+1.5%) → close 50%
✅ TP2:    $67,000 (+3.4%) → close 100%
──────────────────────
Leverage: 10x (vol: 1.8%)
Risk/Reward: 1:2.7
Confidence: 72%
──────────────────────

📊 Market Analysis:
Trend: UP | Vol: LOW (1.8%)
Support: 64,000 / 63,200
Resist:  65,800 / 67,000
Funding: 0.005% (neutral)

🗣 Sentiment: 65/100 (bullish)
Key: ETF inflows +$200M

⚠️ Risk: No flags

Rationale: Strong support at
64k, funding neutral, low vol
favors breakout.

[Approve] [Reject]
```

### 5.2 Confirmation flow (after Approve)

Step 1 — Input margin & adjust leverage:
```
📋 Trade Confirmation
──────────────────────
🟢 LONG BTC/USDT:USDT @ $64,800

Margin (USDT): [user types amount]
Leverage: [5x] [10x✓] [20x] [50x]
```

Step 2 — Show computed details:
```
📋 Trade Confirmation
──────────────────────
🟢 LONG BTC/USDT:USDT @ $64,800

Margin:     500 USDT
Leverage:   10x
Quantity:   0.0772 BTC
Notional:   $5,000
──────────────────────
⛔ Liq:     $58,320
⛔ SL ROE:  -12.3%
✅ TP1 ROE: +15.4% → close 50%
✅ TP2 ROE: +33.9% → close 100%

[Confirm] [Cancel]
```

---

## 6. Database Changes

### PaperTradeRecord

- Add `take_profit_levels_json: str` — serialized `list[TakeProfit]`
- Add `triggered_tp_json: str` — serialized `list[int]`
- Existing `close_reason` already supports "partial_reduce"

---

## 7. Files to Modify

| File | Change |
|------|--------|
| `models.py` | Add TakeProfit, volatility_pct, suggested_leverage |
| `agents/market.py` | Update prompt for volatility_pct |
| `agents/proposer.py` | Update prompt for leverage + TakeProfit format |
| `exchange/paper_engine.py` | Partial TP logic in check_sl_tp |
| `risk/position_sizer.py` | Add USDT margin → quantity calculation |
| `storage/models.py` | Add new fields to PaperTradeRecord |
| `storage/repository.py` | Handle new fields in save/load |
| `telegram/` handlers | Updated message formatting + USDT input flow |
| `pipeline/runner.py` | Pass volatility_pct to proposer |
| `execution/executor.py` | Accept USDT margin instead of risk_pct |

---

## 8. Out of Scope (Future)

- Live executor leverage wiring (Binance API)
- Trailing stop-loss
- Dynamic leverage adjustment during position lifetime
- Multiple concurrent positions same symbol
