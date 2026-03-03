from __future__ import annotations

from typing import TYPE_CHECKING

from orchestrator.execution.plan import ExecutionPlan, OrderInstruction
from orchestrator.models import Side

if TYPE_CHECKING:
    from orchestrator.execution.equity import EquityProvider
    from orchestrator.models import TradeProposal


class ExecutionPlanner:
    """Computes concrete trade numbers from a TradeProposal.

    Pure Python — no LLM calls. Reads equity from an EquityProvider
    and uses fixed-amount margin sizing from config.
    """

    def __init__(self, equity_provider: EquityProvider, config: object) -> None:
        self._equity = equity_provider
        self._config = config

    async def create_plan(
        self, proposal: TradeProposal, current_price: float,
    ) -> ExecutionPlan:
        equity = await self._equity.get_equity()
        margin = self._config.trade_margin_amount  # type: ignore[attr-defined]
        fee_rate: float = getattr(self._config, "paper_taker_fee_rate", 0.0005)
        leverage = proposal.suggested_leverage

        # Entry price: use limit price if provided, else current market price
        entry_price = (
            proposal.entry.price
            if proposal.entry.type == "limit" and proposal.entry.price is not None
            else current_price
        )

        # Quantity from fixed margin
        quantity = margin * leverage / entry_price

        # Notional value
        notional = quantity * entry_price

        # Order sides
        entry_side = "buy" if proposal.side == Side.LONG else "sell"
        exit_side = "sell" if proposal.side == Side.LONG else "buy"

        # Entry order
        entry_order = OrderInstruction(
            symbol=proposal.symbol,
            side=entry_side,
            order_type=proposal.entry.type,
            quantity=quantity,
            price=proposal.entry.price if proposal.entry.type == "limit" else None,
        )

        # SL order
        sl_order = None
        if proposal.stop_loss is not None:
            sl_order = OrderInstruction(
                symbol=proposal.symbol,
                side=exit_side,
                order_type="market",
                quantity=quantity,
                stop_price=proposal.stop_loss,
                reduce_only=True,
            )

        # TP orders
        tp_orders: list[OrderInstruction] = []
        remaining_qty = quantity
        for tp in proposal.take_profit:
            close_qty = quantity * tp.close_pct / 100
            close_qty = min(close_qty, remaining_qty)
            tp_orders.append(
                OrderInstruction(
                    symbol=proposal.symbol,
                    side=exit_side,
                    order_type="market",
                    quantity=close_qty,
                    stop_price=tp.price,
                    reduce_only=True,
                ),
            )
            remaining_qty -= close_qty

        # Max loss
        max_loss = 0.0
        if proposal.stop_loss is not None:
            max_loss = quantity * abs(entry_price - proposal.stop_loss)
        max_loss_pct = (max_loss / equity * 100) if equity > 0 else 0.0

        # TP profits
        tp_profits: list[float] = []
        remaining_for_profit = quantity
        for tp in proposal.take_profit:
            portion = quantity * tp.close_pct / 100
            portion = min(portion, remaining_for_profit)
            profit = portion * abs(tp.price - entry_price)
            tp_profits.append(profit)
            remaining_for_profit -= portion

        # Risk/Reward ratio
        risk_reward = 0.0
        if proposal.stop_loss is not None and proposal.take_profit:
            risk_dist = abs(entry_price - proposal.stop_loss)
            reward_dist = abs(proposal.take_profit[-1].price - entry_price)
            if risk_dist > 0:
                risk_reward = reward_dist / risk_dist

        # Liquidation price (simplified)
        if proposal.side == Side.LONG:
            liq_price = entry_price * (1 - 1 / leverage)
        else:
            liq_price = entry_price * (1 + 1 / leverage)

        # Estimated fees (entry only)
        estimated_fees = notional * fee_rate

        return ExecutionPlan(
            proposal_id=proposal.proposal_id,
            symbol=proposal.symbol,
            side=proposal.side.value,
            entry_order=entry_order,
            sl_order=sl_order,
            tp_orders=tp_orders,
            margin_mode="isolated",
            leverage=leverage,
            quantity=quantity,
            entry_price=entry_price,
            notional_value=notional,
            margin_required=margin,
            liquidation_price=liq_price,
            estimated_fees=estimated_fees,
            max_loss=max_loss,
            max_loss_pct=max_loss_pct,
            tp_profits=tp_profits,
            risk_reward_ratio=risk_reward,
            equity_snapshot=equity,
        )
