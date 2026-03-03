from __future__ import annotations

from pydantic import BaseModel


class OrderInstruction(BaseModel, frozen=True):
    symbol: str
    side: str                    # "buy" / "sell"
    order_type: str              # "market" / "limit"
    quantity: float
    price: float | None = None
    reduce_only: bool = False
    stop_price: float | None = None


class ExecutionPlan(BaseModel, frozen=True):
    proposal_id: str
    symbol: str
    side: str                     # "long" / "short"
    entry_order: OrderInstruction
    sl_order: OrderInstruction | None
    tp_orders: list[OrderInstruction]
    margin_mode: str = "isolated"
    leverage: int
    quantity: float
    entry_price: float
    notional_value: float         # quantity × entry_price
    margin_required: float
    liquidation_price: float
    estimated_fees: float
    max_loss: float               # dollar amount if SL hit
    max_loss_pct: float           # as % of equity
    tp_profits: list[float]       # estimated profit per TP level
    risk_reward_ratio: float
    equity_snapshot: float        # equity at computation time
