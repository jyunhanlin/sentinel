# Proposal → Trade → Position Management Design

**Date:** 2026-03-03
**Status:** Approved

## Problem

The current flow from proposal to trade execution has gaps:
1. TradeProposal outputs abstract `risk_pct` — users can't see concrete numbers (quantity, margin, fees)
2. Telegram approval is binary (approve/reject) — no way to adjust parameters before execution
3. Position management lacks a Telegram interface for post-trade operations

## Design Overview

```
Proposer → TradeProposal (risk %)
  → ExecutionPlanner (Python, not LLM)
    → ExecutionPlan (quantity, margin, liq price, fees)
  → Telegram notification (two-section format)
  → User: [開倉] / [調整] / [跳過]
  → Execute trade
  → Position management via Telegram
```

## 1. Architecture: ExecutionPlan Middle Layer

### Why a middle layer?

- **LLM is bad at math** — Python computes quantity/margin/liq price reliably
- **Proposer stays simple** — it decides *what* to trade (direction, risk level), not *how much*
- **Fresh data** — ExecutionPlan uses latest equity and price at computation time
- **Clean separation** — Proposer = trading intent, ExecutionPlanner = concrete execution

### EquityProvider Interface

Unified interface for equity source:
- `PaperEquityProvider` — reads from PaperEngine simulated account
- `LiveEquityProvider` (future) — reads from exchange API

```python
class EquityProvider(Protocol):
    async def get_equity(self) -> float: ...
    async def get_available_margin(self) -> float: ...
```

### Position Sizing: Fixed Amount Margin

Each trade uses a fixed dollar amount as margin (configured in settings).

```python
# Config
trade_margin_amount: float = 500.0  # $500 per trade
```

No internal safety valve in v1. See "Future Considerations" for risk guardrails.

### ExecutionPlan Model

```python
class OrderInstruction(BaseModel, frozen=True):
    symbol: str
    side: str                    # "buy" / "sell"
    order_type: str              # "market" / "limit"
    quantity: float
    price: float | None          # for limit orders
    reduce_only: bool = False    # for SL/TP orders
    stop_price: float | None     # for stop orders

class ExecutionPlan(BaseModel, frozen=True):
    proposal: TradeProposal       # original proposal reference
    entry_order: OrderInstruction
    sl_order: OrderInstruction | None
    tp_orders: list[OrderInstruction]
    margin_mode: str = "isolated"
    leverage: int
    quantity: float
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

### ExecutionPlanner

Pure Python computation (no LLM):

```python
class ExecutionPlanner:
    def __init__(self, equity_provider: EquityProvider, config: Settings): ...

    async def create_plan(
        self, proposal: TradeProposal, current_price: float,
    ) -> ExecutionPlan:
        equity = await self.equity_provider.get_equity()
        margin = self.config.trade_margin_amount
        quantity = margin * proposal.suggested_leverage / current_price
        # ... compute all derived values
```

## 2. Telegram Proposal Message (Two-Section Format)

### Upper Section — Trade Parameters

All numbers needed to open the position:

```
━━━ 開倉建議 ━━━━━━━━━━━━━━━

🟢 BTC/USDT LONG · Confidence 82%

▶ Entry:     $95,000 (market)
  Quantity:  0.05 BTC ($4,750)
  Margin:    $475 · 10x isolated
  Liq:       $85,950
⛔ Stop Loss: $93,000 (-2.1%)
✅ TP1:       $97,000 (+2.1%) → close 50%
✅ TP2:       $99,000 (+4.2%) → close 100%

⚠️ 最大虧損: $100 (1.0%)
💰 預估獲利: TP1 +$50 / TP2 +$100
📊 Risk/Reward: 1:2.0
```

### Lower Section — Analysis Summary

Agent analysis summaries + situation rationale:

```
━━━ 分析摘要 ━━━━━━━━━━━━━━━

📈 Technical: UP, BULLISH momentum
   Support: 93,200 / Resist: 97,500
💹 Positioning: Funding -0.01%
   OI +3.2%, 多空比偏空 → 軋空潛力
📅 Catalyst: 無近期高影響事件
🌐 Correlation: DXY↓ S&P risk-on

💡 局勢分析
BTC 在 4h 級別突破前高 $94,500 後回踩確認支撐，
RSI 55 尚未過熱。Funding -0.01% 顯示市場偏空，
若價格持續站穩 $95,000 上方，空頭回補可能推動
進一步上漲。近期無重大宏觀事件，DXY 走弱提供
順風環境。

Model: sonnet-4-6 · Expires in 15 min

[🚀 開倉] [✏️ 調整] [❌ 跳過]
```

### Rationale Field

The `rationale` field in TradeProposal is no longer a 1-2 sentence summary.
It should be a **situation analysis paragraph** covering current market conditions,
key factors supporting the trade, and what to watch for.

## 3. Parameter Adjustment Flow (Hybrid Mode)

When user taps [✏️ 調整]:

```
✏️ 要調整哪個參數？

[Leverage] [SL] [TP] [Quantity]
[🚀 確認開倉] [❌ 取消]
```

### Interaction by parameter type:

| Parameter | Input Method | Notes |
|-----------|-------------|-------|
| Leverage | Inline buttons (5x/10x/20x/50x) | Quick selection |
| SL | Text input (price) | Precise value |
| TP | Text input (price close_pct) | Supports modify/add/delete |
| Quantity | Buttons (50%/75%/100%/150%) + custom text | Relative to original or exact |

### Recalculation Rules

When a parameter changes:

| Changed | Recalculated | NOT changed |
|---------|-------------|-------------|
| Leverage | margin, liq price | quantity |
| SL | max loss ($, %) | quantity |
| TP | estimated profit | quantity |
| Quantity | margin, max loss, profit, liq price | — |

**Key rule:** Changing SL does NOT recalculate quantity. Quantity stays fixed; only the displayed max loss amount changes.

After each adjustment, display the updated upper section with changed values highlighted.

## 4. Position Management (Post-Trade Operations)

### Entry Point

Via `/status` command — each open position shows an [⚙️ 管理] button.

### Management Menu

```
⚙️ BTC/USDT LONG — 管理

[移 SL] [調 TP] [加倉] [減倉] [平倉]
[⬅️ 返回]
```

### Operation Details

#### Move SL (text input)
- Show current SL and current price
- User inputs new SL price
- Display updated max loss after change

#### Adjust TP (text input)
- Show current TP levels
- Support: modify (`98000 50`), add (`add 101000 30`), delete (`del 1`)
- Display updated estimated profits

#### Add to Position (hybrid)
- Buttons: [同量] [一半] [自訂]
- Custom: text input for exact quantity
- After adding: show new avg entry, new margin, new liq price

#### Reduce Position (buttons)
- Buttons: [25%] [50%] [75%] [自訂]
- Custom: text input for exact percentage
- After reducing: show remaining quantity and realized PnL

#### Close Position (confirmation)
- Show current PnL preview
- [✅ 確認平倉] [❌ 取消]
- After close: show final PnL report

### Post-Operation Display

Every operation shows an updated position card with all current values.

## 5. Data Flow Summary

```
┌─────────────────────────────────────────────────────┐
│ Pipeline                                             │
│                                                      │
│ DataFetch + EquityProvider.get_equity()               │
│   → 5 Analysis Agents (parallel)                     │
│   → ProposerAgent → TradeProposal (risk %, rationale)│
│   → ExecutionPlanner → ExecutionPlan (concrete nums)  │
│   → Validation + Risk Check                          │
│   → Telegram: two-section message                    │
│       → [開倉] → Execute → Position created          │
│       → [調整] → Modify params → [確認] → Execute    │
│       → [跳過] → Log & skip                          │
│                                                      │
│ Position Management (via /status)                    │
│   → [管理] → Move SL / Adjust TP / Add / Reduce     │
│   → [平倉] → Close with PnL report                  │
└─────────────────────────────────────────────────────┘
```

## Future Considerations

### Internal Safety Valve (Not in v1)

Risk guardrails that can be added later when needed:

| Check | Rule | Purpose |
|-------|------|---------|
| Max single loss | SL loss ≤ equity × 5% | Prevent outsized single-trade loss |
| Max total exposure | All open margin ≤ equity × 60% | Ensure capital for new trades |
| Min liquidation distance | Liq price ≥ X% from entry | Prevent flash-crash liquidation |
| Max daily loss | Daily cumulative loss ≤ equity × 10% | Prevent tilt/streak losses |

These would be configurable values with the safety valve intercepting between
ExecutionPlan creation and trade execution.

### Additional Order Types

- Post-only limit orders (maker fee savings)
- Scaled/DCA entry (multiple limit orders at different prices)

### Multi-Exchange Support

EquityProvider and ExecutionPlanner are exchange-agnostic by design.
Adding a new exchange only requires implementing the provider interface.
