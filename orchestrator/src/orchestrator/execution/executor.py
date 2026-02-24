from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel

from orchestrator.models import TradeProposal

if TYPE_CHECKING:
    from orchestrator.exchange.client import ExchangeClient
    from orchestrator.exchange.paper_engine import PaperEngine
    from orchestrator.risk.position_sizer import PositionSizer

logger = structlog.get_logger(__name__)


class ExecutionResult(BaseModel, frozen=True):
    trade_id: str
    symbol: str
    side: str
    entry_price: float
    quantity: float
    fees: float
    mode: str  # "paper" | "live"
    exchange_order_id: str = ""
    sl_order_id: str = ""
    tp_order_id: str = ""


class OrderExecutor(ABC):
    @abstractmethod
    async def execute_entry(
        self, proposal: TradeProposal, current_price: float, leverage: int = 1,
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


class PaperExecutor(OrderExecutor):
    def __init__(self, *, paper_engine: PaperEngine) -> None:
        self._paper_engine = paper_engine

    async def execute_entry(
        self, proposal: TradeProposal, current_price: float, leverage: int = 1,
        margin_usdt: float | None = None,
    ) -> ExecutionResult:
        if margin_usdt is not None:
            from orchestrator.risk.position_sizer import MarginSizer

            sizer = MarginSizer()
            qty = sizer.calculate_from_margin(
                margin_usdt=margin_usdt, leverage=leverage, entry_price=current_price,
            )
            # Override position sizing by temporarily setting quantity directly
            position = self._paper_engine.open_position_with_quantity(
                proposal, current_price, leverage=leverage,
                quantity=qty, margin=margin_usdt,
            )
        else:
            position = self._paper_engine.open_position(proposal, current_price, leverage=leverage)
        logger.info(
            "paper_execution",
            trade_id=position.trade_id,
            symbol=position.symbol,
            entry_price=position.entry_price,
        )
        return ExecutionResult(
            trade_id=position.trade_id,
            symbol=position.symbol,
            side=position.side.value,
            entry_price=position.entry_price,
            quantity=position.quantity,
            fees=position.quantity
            * position.entry_price
            * self._paper_engine._taker_fee_rate,
            mode="paper",
        )

    async def place_sl_tp(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        stop_loss: float,
        take_profit: list[float],
    ) -> list[str]:
        # Paper mode: SL/TP handled by PaperEngine.check_sl_tp()
        return []

    async def cancel_orders(self, order_ids: list[str]) -> None:
        # Paper mode: nothing to cancel
        pass


class LiveExecutor(OrderExecutor):
    def __init__(
        self,
        *,
        exchange_client: ExchangeClient,
        position_sizer: PositionSizer,
        paper_engine: PaperEngine,
        price_deviation_threshold: float = 0.01,
    ) -> None:
        self._exchange = exchange_client
        self._position_sizer = position_sizer
        self._paper_engine = paper_engine
        self._threshold = price_deviation_threshold

    def check_price_deviation(
        self, snapshot_price: float, current_price: float
    ) -> float:
        """Return deviation as a ratio. Raises ValueError if above threshold."""
        if snapshot_price <= 0:
            return 0.0
        deviation = abs(current_price - snapshot_price) / snapshot_price
        if deviation > self._threshold:
            raise ValueError(
                f"Price deviated {deviation:.1%} from proposal time "
                f"(was ${snapshot_price:,.1f}, now ${current_price:,.1f}). "
                f"Threshold: {self._threshold:.1%}"
            )
        return deviation

    async def execute_entry(
        self, proposal: TradeProposal, current_price: float, leverage: int = 1,
        margin_usdt: float | None = None,
    ) -> ExecutionResult:
        if proposal.stop_loss is None:
            raise ValueError(
                f"Cannot execute {proposal.symbol}: stop_loss is required"
            )

        quantity = self._position_sizer.calculate(
            equity=self._paper_engine.equity,
            risk_pct=proposal.position_size_risk_pct,
            entry_price=current_price,
            stop_loss=proposal.stop_loss,
        )

        ccxt_side = "buy" if proposal.side.value == "long" else "sell"
        order = await self._exchange.create_market_order(
            proposal.symbol, ccxt_side, quantity
        )

        fill_price = order.get("price", current_price)
        fill_qty = order.get("filled", quantity)
        fee_info = order.get("fee", {})
        fees = fee_info.get("cost", fill_qty * fill_price * 0.0005)

        trade_id = str(uuid.uuid4())

        logger.info(
            "live_execution",
            trade_id=trade_id,
            order_id=order.get("id", ""),
            symbol=proposal.symbol,
            fill_price=fill_price,
            quantity=fill_qty,
        )

        return ExecutionResult(
            trade_id=trade_id,
            symbol=proposal.symbol,
            side=proposal.side.value,
            entry_price=fill_price,
            quantity=fill_qty,
            fees=fees,
            mode="live",
            exchange_order_id=order.get("id", ""),
        )

    async def place_sl_tp(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        stop_loss: float,
        take_profit: list[float],
    ) -> list[str]:
        order_ids: list[str] = []
        close_side = "sell" if side == "long" else "buy"

        sl_order = await self._exchange.create_stop_order(
            symbol, close_side, quantity, stop_price=stop_loss
        )
        order_ids.append(sl_order.get("id", ""))

        # TP: first target only for MVP
        if take_profit:
            tp_order = await self._exchange.create_stop_order(
                symbol, close_side, quantity, stop_price=take_profit[0]
            )
            order_ids.append(tp_order.get("id", ""))

        logger.info("sl_tp_placed", symbol=symbol, order_ids=order_ids)
        return order_ids

    async def cancel_orders(self, order_ids: list[str]) -> None:
        for oid in order_ids:
            if oid:
                try:
                    await self._exchange.cancel_order(oid, "")
                except Exception as e:
                    logger.warning(
                        "cancel_order_failed", order_id=oid, error=str(e)
                    )
