# Trade Execution Codemap

**Last Updated:** 2026-02-28

## Overview

Trade execution converts a validated proposal into actual orders on exchange or paper engine. It handles:

1. **Position sizing** — Calculate quantity from risk %
2. **Entry order** — Market or limit order at snapshot price
3. **Stop loss & take profit** — OCO (one-cancels-other) orders
4. **Mode switching** — Paper vs live trading

## Architecture

```
OrderExecutor (Abstract)
    │
    ├─ PaperExecutor
    │   └─ Uses PaperEngine for simulation
    │
    └─ LiveExecutor
        └─ Uses ExchangeClient (CCXT) for real orders
```

## OrderExecutor Interface

**File:** `orchestrator/src/orchestrator/execution/executor.py`

```python
class OrderExecutor(ABC):
    @abstractmethod
    async def execute_entry(
        self,
        proposal: TradeProposal,
        current_price: float,
        leverage: int = 1,
        margin_usdt: float | None = None,
    ) -> ExecutionResult: ...

    @abstractmethod
    async def place_sl_tp(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        stop_loss: float,
        take_profit: list[float],
    ) -> list[str]: ...

    @abstractmethod
    async def cancel_orders(self, order_ids: list[str]) -> None: ...
```

### ExecutionResult

```python
class ExecutionResult(BaseModel, frozen=True):
    trade_id: str
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    quantity: float
    fees: float  # Entry order fee
    mode: str  # "paper" or "live"
    exchange_order_id: str = ""  # Only for live
    sl_order_id: str = ""  # Order IDs for SL/TP
    tp_order_id: str = ""
```

## PaperExecutor

**File:** `orchestrator/src/orchestrator/execution/executor.py`

Simulates trading on the PaperEngine.

### Entry Execution

```python
async def execute_entry(
    self,
    proposal: TradeProposal,
    current_price: float,
    leverage: int = 1,
    margin_usdt: float | None = None,
) -> ExecutionResult:
    """
    1. Size position (risk % or margin-based)
    2. Open position on PaperEngine
    3. Return ExecutionResult with quantity + fees
    """
    if margin_usdt is not None:
        # Use MarginSizer for specific margin amount
        sizer = MarginSizer()
        qty = sizer.calculate_from_margin(
            margin_usdt=margin_usdt,
            leverage=leverage,
            entry_price=current_price,
        )
        position = self._paper_engine.open_position_with_quantity(
            proposal, current_price, leverage=leverage,
            quantity=qty, margin=margin_usdt,
        )
    else:
        # Use RiskPercentSizer for risk % of account
        position = self._paper_engine.open_position(
            proposal, current_price, leverage=leverage
        )

    return ExecutionResult(
        trade_id=position.trade_id,
        symbol=position.symbol,
        side=position.side.value,
        entry_price=position.entry_price,
        quantity=position.quantity,
        fees=position.quantity * position.entry_price * fee_rate,
        mode="paper",
    )
```

### SL/TP Placement

```python
async def place_sl_tp(
    self,
    *,
    symbol: str,
    side: str,
    quantity: float,
    stop_loss: float,
    take_profit: list[float],
) -> list[str]:
    """
    On PaperEngine, SL/TP are internally tracked.
    This method returns dummy order IDs.
    """
    order_ids = []
    if stop_loss:
        order_ids.append(f"SL_{uuid.uuid4()}")
    for tp in take_profit:
        order_ids.append(f"TP_{uuid.uuid4()}")
    return order_ids
```

### Order Cancellation

```python
async def cancel_orders(self, order_ids: list[str]) -> None:
    """On PaperEngine, just acknowledge cancellation."""
    logger.info("paper_orders_cancelled", count=len(order_ids))
```

## LiveExecutor

**File:** `orchestrator/src/orchestrator/execution/executor.py`

Uses real exchange API (CCXT) for live trading.

### Entry Execution

```python
async def execute_entry(
    self,
    proposal: TradeProposal,
    current_price: float,
    leverage: int = 1,
    margin_usdt: float | None = None,
) -> ExecutionResult:
    """
    1. Set leverage on exchange
    2. Size position
    3. Place market order via CCXT
    4. Return result with exchange order IDs
    """
    # Set leverage (Binance, etc.)
    await self._exchange_client.set_leverage(
        symbol=proposal.symbol,
        leverage=leverage,
    )

    # Calculate quantity
    if margin_usdt is not None:
        sizer = MarginSizer()
        qty = sizer.calculate_from_margin(...)
    else:
        sizer = RiskPercentSizer()
        qty = sizer.calculate(...)

    # Place order
    order = await self._exchange_client.create_order(
        symbol=proposal.symbol,
        order_type=proposal.entry.type,  # "market" or "limit"
        side="buy" if proposal.side == Side.LONG else "sell",
        amount=qty,
        price=proposal.entry.price if proposal.entry.type == "limit" else None,
    )

    return ExecutionResult(
        trade_id=str(uuid.uuid4()),
        symbol=proposal.symbol,
        side=proposal.side.value,
        entry_price=order["average"],
        quantity=order["amount"],
        fees=order["fee"]["cost"],
        mode="live",
        exchange_order_id=order["id"],
    )
```

### SL/TP Placement

```python
async def place_sl_tp(
    self,
    *,
    symbol: str,
    side: str,
    quantity: float,
    stop_loss: float,
    take_profit: list[float],
) -> list[str]:
    """
    Place stop loss and take profit orders on exchange.
    Returns list of order IDs.
    """
    order_ids = []

    # Place stop loss
    if stop_loss:
        sl_order = await self._exchange_client.create_order(
            symbol=symbol,
            order_type="stop_market",
            side="sell" if side == "long" else "buy",
            amount=quantity,
            stopPrice=stop_loss,
        )
        order_ids.append(sl_order["id"])

    # Place take profits
    remaining_qty = quantity
    for tp in take_profit:
        qty_to_close = quantity * tp.close_pct / 100
        remaining_qty -= qty_to_close
        tp_order = await self._exchange_client.create_order(
            symbol=symbol,
            order_type="take_profit_market",
            side="sell" if side == "long" else "buy",
            amount=qty_to_close,
            stopPrice=tp.price,
        )
        order_ids.append(tp_order["id"])

    return order_ids
```

### Order Cancellation

```python
async def cancel_orders(self, order_ids: list[str]) -> None:
    """Cancel orders on exchange."""
    for order_id in order_ids:
        await self._exchange_client.cancel_order(order_id)
```

## Integration with Pipeline

**In `PipelineRunner.execute()` after approval:**

```python
# 1. Get current price snapshot
current_price = snapshot.current_price

# 2. Execute entry
execution_result = await executor.execute_entry(
    proposal=proposal,
    current_price=current_price,
    leverage=proposal.suggested_leverage,
)

# 3. Place SL/TP
order_ids = await executor.place_sl_tp(
    symbol=proposal.symbol,
    side=proposal.side.value,
    quantity=execution_result.quantity,
    stop_loss=proposal.stop_loss,
    take_profit=[tp.price for tp in proposal.take_profit],
)

# 4. Store execution result
trade_repo.create(
    proposal_id=proposal.proposal_id,
    symbol=proposal.symbol,
    side=proposal.side.value,
    entry_price=execution_result.entry_price,
    quantity=execution_result.quantity,
    leverage=proposal.suggested_leverage,
    sl_order_id=order_ids[0] if proposal.stop_loss else "",
    tp_order_ids=order_ids[1:],
)

# 5. Monitor SL/TP in PriceMonitor
```

## Configuration

Environment variables:

```python
trading_mode: str = "paper"  # "paper" | "live"

# Paper Trading
paper_initial_equity: float = 10000.0
paper_taker_fee_rate: float = 0.0005
paper_maker_fee_rate: float = 0.0002
paper_default_leverage: int = 10
paper_leverage_options: list[int] = [5, 10, 20, 50]

# Exchange (live mode only)
exchange_id: str = "binance"
exchange_api_key: str = ""  # Required for live
exchange_api_secret: str = ""  # Required for live
```

## Leverage

Leverage is set at:

1. **Proposal level** — `proposal.suggested_leverage` (1-50)
2. **Execution level** — Can override via `execute_entry(leverage=N)`

For Binance:
- Max leverage: 50x (up to 125x with USDe collateral)
- Margin requirement: 1 / leverage
- Liquidation: Entry price ± (entry / leverage * maintenance_margin)

## Testing

**File:** `tests/unit/test_paper_trading_flow.py`

Tests:
- Entry order execution
- Position sizing with different risk %
- Fee calculations
- SL/TP order placement (paper)
- Position closing

**File:** `tests/integration/` (if enabled)

Tests:
- Live API calls (requires credentials)
- Order placement on testnet

## Example Workflow

```python
from orchestrator.execution.executor import PaperExecutor
from orchestrator.exchange.paper_engine import PaperEngine
from orchestrator.risk.position_sizer import RiskPercentSizer

# Setup
paper_engine = PaperEngine(initial_equity=10000.0)
executor = PaperExecutor(paper_engine=paper_engine)

# Proposal
proposal = TradeProposal(
    symbol="BTC/USDT:USDT",
    side=Side.LONG,
    entry=EntryOrder(type="market"),
    position_size_risk_pct=1.0,
    stop_loss=44000.0,
    take_profit=[TakeProfit(price=46000.0, close_pct=50)],
    suggested_leverage=10,
)

# Execute entry at market price
current_price = 45000.0
result = await executor.execute_entry(
    proposal=proposal,
    current_price=current_price,
    leverage=proposal.suggested_leverage,
)

# Result
print(f"Trade {result.trade_id}:")
print(f"  Entry: {result.quantity:.4f} BTC @ {result.entry_price}")
print(f"  Fees: ${result.fees:.2f}")

# Place SL/TP
order_ids = await executor.place_sl_tp(
    symbol=proposal.symbol,
    side=proposal.side.value,
    quantity=result.quantity,
    stop_loss=proposal.stop_loss,
    take_profit=[tp.price for tp in proposal.take_profit],
)

print(f"  Orders placed: {len(order_ids)}")

# Monitor with PriceMonitor
# (See exchange.md for details)
```

## Related

- [pipeline.md](pipeline.md) — Pipeline integration
- [risk.md](risk.md) — Position sizing
- [exchange.md](exchange.md) — Paper engine, price monitoring
- [storage.md](storage.md) — Trade record persistence
