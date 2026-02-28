# Risk Management Codemap

**Last Updated:** 2026-02-28

## Overview

Risk management enforces position sizing, daily loss limits, and prevents excessive leverage. It operates at two levels:

1. **Position Sizer** — Calculates quantity based on risk % and account equity
2. **Risk Checker** — Validates proposals against hard limits before execution

## Architecture

```
TradeProposal (position_size_risk_pct, side, SL, TP)
    │
    ├─ RiskChecker
    │   └─ Validates against:
    │       ├─ Max single risk %
    │       ├─ Max total open exposure %
    │       ├─ Max consecutive losses
    │       └─ Max daily loss %
    │
    └─ PositionSizer
        ├─ RiskPercentSizer (most common)
        │   └─ Calculate quantity from risk_pct + account equity
        │
        └─ MarginSizer (for leverage)
            └─ Calculate quantity from margin amount + leverage
```

## Risk Checker

**File:** `orchestrator/src/orchestrator/risk/checker.py`

The gatekeeper before trade execution.

### Interface

```python
class RiskChecker:
    def __init__(
        self,
        *,
        max_single_risk_pct: float,
        max_total_exposure_pct: float,
        max_consecutive_losses: int,
        max_daily_loss_pct: float,
    ) -> None: ...

    def check(
        self,
        *,
        proposal: TradeProposal,
        open_positions_risk_pct: float,
        consecutive_losses: int,
        daily_loss_pct: float,
    ) -> RiskResult: ...
```

### RiskResult

```python
class RiskResult(BaseModel, frozen=True):
    approved: bool
    rule_violated: str = ""  # e.g., "max_single_risk", "max_daily_loss"
    reason: str = ""  # Human-readable explanation
    action: str = "reject"  # "reject" (normal) | "pause" (stop trading)
```

### Validation Rules

| Rule | Condition | Action | Example |
|------|-----------|--------|---------|
| **Max Single Risk** | `proposal.position_size_risk_pct > max_single_risk_pct` | REJECT | Proposal: 2.5%, Max: 2% → REJECT |
| **Max Total Exposure** | `open_risk_pct + proposal_risk_pct > max_total_exposure_pct` | REJECT | Open: 15%, New: 8%, Max: 20% → REJECT |
| **Max Consecutive Losses** | `consecutive_losses >= max_consecutive_losses` | PAUSE | 5 losses in a row → PAUSE trading |
| **Max Daily Loss** | `daily_loss_pct < -max_daily_loss_pct` | PAUSE | Lost 5.5% today, Max: 5% → PAUSE |

### Decision Tree

```
check(proposal, open_risk_pct, consecutive_losses, daily_loss_pct)
    │
    ├─ Is side == FLAT?
    │   └─ Return approved=True (no position to check)
    │
    ├─ Rule 1: Max single risk
    │   └─ proposal.risk_pct > max_single_risk_pct?
    │       └─ REJECT
    │
    ├─ Rule 2: Max total exposure
    │   └─ (open_risk_pct + proposal.risk_pct) > max_total_exposure_pct?
    │       └─ REJECT
    │
    ├─ Rule 3: Consecutive losses
    │   └─ consecutive_losses >= max_consecutive_losses?
    │       └─ PAUSE (prevents revenge trading)
    │
    └─ Rule 4: Daily loss
        └─ daily_loss_pct < -max_daily_loss_pct?
            └─ PAUSE (circuit breaker for the day)
```

### Example

```python
checker = RiskChecker(
    max_single_risk_pct=2.0,
    max_total_exposure_pct=20.0,
    max_consecutive_losses=5,
    max_daily_loss_pct=5.0,
)

proposal = TradeProposal(
    symbol="BTC/USDT:USDT",
    side=Side.LONG,
    position_size_risk_pct=1.5,
    ...
)

# Check against current account state
result = checker.check(
    proposal=proposal,
    open_positions_risk_pct=12.0,  # Currently have 12% exposure
    consecutive_losses=0,
    daily_loss_pct=-1.5,
)

if result.approved:
    # OK to execute
    print(f"Approved. Total exposure after: {12.0 + 1.5}%")
else:
    if result.action == "reject":
        print(f"Rejected: {result.reason}")
    else:  # pause
        print(f"Trading paused: {result.reason}")
        # Notify user, require /resume command to resume
```

## Position Sizers

**File:** `orchestrator/src/orchestrator/risk/position_sizer.py`

Converts risk % into actual position size (quantity).

### RiskPercentSizer (Main)

```python
class RiskPercentSizer(PositionSizer):
    """Position size based on risk percentage of account equity."""

    def calculate(
        self,
        *,
        side: Side,
        entry_price: float,
        stop_loss: float,
        account_equity: float,
        risk_pct: float,
    ) -> float:
        """
        Calculate position quantity to risk exactly risk_pct of equity.

        Formula:
            risk_amount = account_equity * risk_pct / 100
            distance = abs(entry_price - stop_loss)
            quantity = risk_amount / distance

        Args:
            side: "long" or "short"
            entry_price: Entry price
            stop_loss: Stop loss price
            account_equity: Current account balance
            risk_pct: Risk as % of equity (e.g., 1.0 = 1%)

        Returns:
            Quantity to buy/sell
        """
```

**Example:**
```
Account: $10,000
Risk: 1.0% = $100
Entry: $45,000
Stop: $44,500 (distance: $500)
Quantity = $100 / $500 = 0.2 BTC

If trade goes SL:
  Loss = 0.2 * $500 = $100 = 1% of $10,000
```

### MarginSizer

```python
class MarginSizer(PositionSizer):
    """Position size based on margin amount + leverage."""

    def calculate_from_margin(
        self,
        *,
        margin_usdt: float,
        leverage: int,
        entry_price: float,
    ) -> float:
        """
        Calculate quantity from margin.

        Formula:
            notional = margin_usdt * leverage
            quantity = notional / entry_price

        Args:
            margin_usdt: Margin to use (e.g., 100 USDT)
            leverage: Leverage multiplier (e.g., 5x, 10x)
            entry_price: Entry price

        Returns:
            Quantity to open
        """
```

**Example:**
```
Margin: 100 USDT
Leverage: 5x
Entry: $45,000

Notional = 100 * 5 = $500
Quantity = $500 / $45,000 = 0.0111 BTC
```

### Interface

```python
class PositionSizer(ABC):
    @abstractmethod
    def calculate(self, *, **kwargs) -> float:
        """Calculate position quantity."""
```

## Integration Points

### In Pipeline

After approval, executor uses sizer:

```python
# In OrderExecutor.execute_entry()
if margin_usdt is not None:
    sizer = MarginSizer()
    qty = sizer.calculate_from_margin(
        margin_usdt=margin_usdt,
        leverage=leverage,
        entry_price=current_price,
    )
else:
    sizer = RiskPercentSizer()
    qty = sizer.calculate(
        side=proposal.side,
        entry_price=current_price,
        stop_loss=proposal.stop_loss,
        account_equity=current_equity,
        risk_pct=proposal.position_size_risk_pct,
    )
```

### In Paper Engine

```python
# In PaperEngine.open_position()
sizer = RiskPercentSizer()
quantity = sizer.calculate(
    side=proposal.side,
    entry_price=current_price,
    stop_loss=proposal.stop_loss,
    account_equity=self._equity,
    risk_pct=proposal.position_size_risk_pct,
)

position = OpenPosition(
    symbol=proposal.symbol,
    side=proposal.side,
    entry_price=current_price,
    quantity=quantity,
    leverage=leverage,
    # Calculate margin based on quantity and leverage
    margin=self._calculate_margin(quantity, current_price, leverage),
)
```

## Configuration

Environment variables in `config.py`:

```python
# Risk Limits
max_single_risk_pct: float = 2.0          # Max risk per trade
max_total_exposure_pct: float = 20.0      # Max total open exposure
max_daily_loss_pct: float = 5.0           # Pause trading if hit
max_consecutive_losses: int = 5           # Pause if hit

# Paper Trading
paper_initial_equity: float = 10000.0     # Starting balance
paper_taker_fee_rate: float = 0.0005      # 0.05%
paper_maker_fee_rate: float = 0.0002      # 0.02%
paper_default_leverage: int = 10
paper_maintenance_margin_rate: float = 0.5  # %
paper_leverage_options: list[int] = [5, 10, 20, 50]
```

## Paper Engine

**File:** `orchestrator/src/orchestrator/exchange/paper_engine.py`

Simulates trading with realistic:
- Margin requirements
- Liquidation prices
- Fee calculations
- Leverage constraints

### Key Classes

```python
class PaperEngine:
    """Simulates paper trading with margin, leverage, fees."""

    def open_position(
        self,
        proposal: TradeProposal,
        current_price: float,
        leverage: int,
    ) -> OpenPosition: ...

    def close_position(
        self,
        trade_id: str,
        close_price: float,
        close_reason: str,
    ) -> CloseResult: ...
```

### Liquidation Calculation

```python
liquidation_price = {
    "long": entry_price * (1 - 1 / leverage + maintenance_margin_rate / leverage),
    "short": entry_price * (1 + 1 / leverage - maintenance_margin_rate / leverage),
}
```

### Fee Simulation

```python
# Entry fee
entry_fee = quantity * entry_price * taker_fee_rate

# Exit fee
exit_fee = quantity * exit_price * taker_fee_rate

# Total PnL = (close_price - entry_price) * quantity - entry_fee - exit_fee
```

## Statistics & Account State

**File:** `orchestrator/src/orchestrator/stats/calculator.py`

Tracks:
- `daily_loss_pct` — Total PnL since midnight
- `consecutive_losses` — Number of consecutive losing trades
- `open_positions_risk_pct` — Sum of risk % for all open positions

Used by RiskChecker:

```python
stats = StatsCalculator(paper_engine=paper_engine)
state = stats.calculate_current_state()

risk_result = risk_checker.check(
    proposal=proposal,
    open_positions_risk_pct=state.open_positions_risk_pct,
    consecutive_losses=state.consecutive_losses,
    daily_loss_pct=state.daily_loss_pct,
)
```

## Testing

**File:** `tests/unit/test_risk_checker.py`
- Tests each risk rule
- Tests edge cases (position_size_risk_pct = exactly max)
- Tests action (reject vs pause)

**File:** `tests/unit/test_position_sizer.py`
- Tests quantity calculation
- Tests with different account sizes and risk %

**File:** `tests/unit/test_paper_trading_flow.py`
- End-to-end paper trading with sizer + paper engine
- Tests margin, liquidation, fees

## Workflow

### Before Trade Execution

```
1. Calculate current account state (StatsCalculator)
   → daily_loss_pct, consecutive_losses, open_positions_risk_pct

2. Check risk (RiskChecker)
   → approved=True → continue
   → approved=False, action="reject" → send rejection notification
   → approved=False, action="pause" → send pause notification + require /resume

3. Size position (PositionSizer)
   → quantity = calculate(risk_pct, account_equity, SL distance)

4. Execute trade (OrderExecutor)
   → PaperExecutor calls PaperEngine
   → Entry order, SL/TP orders created

5. Monitor (PriceMonitor + PaperEngine)
   → Check SL/TP fills every N seconds
   → Update daily_loss_pct after each close
   → Check consecutive_losses for next pipeline run
```

### After Trade Close

```
1. PaperEngine closes position
   → Calculates PnL = (close_price - entry_price) * quantity - fees

2. StatsCalculator updates:
   → daily_loss_pct += pnl / account_equity * 100
   → If pnl < 0: consecutive_losses += 1
   → If pnl > 0: consecutive_losses = 0

3. Next pipeline run checks RiskResult:
   → If daily_loss_pct < -max_daily_loss_pct
   → Pause trading until /resume command
```

## Related

- [pipeline.md](pipeline.md) — Where risk checking happens
- [execution.md](execution.md) — Trade execution with sizing
- [exchange.md](exchange.md) — Paper engine details
- [configuration.md](configuration.md) — Risk configuration
