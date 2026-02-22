from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel

from orchestrator.models import Side, TradeProposal
from orchestrator.risk.position_sizer import PositionSizer
from orchestrator.stats.calculator import StatsCalculator

if TYPE_CHECKING:
    from orchestrator.storage.repository import AccountSnapshotRepository, PaperTradeRepository

logger = structlog.get_logger(__name__)


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
    leverage: int = 1
    margin: float = 0.0
    liquidation_price: float = 0.0


class CloseResult(BaseModel, frozen=True):
    trade_id: str
    symbol: str
    side: Side
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    fees: float
    reason: str  # "sl" | "tp"


class PaperEngine:
    def __init__(
        self,
        *,
        initial_equity: float,
        taker_fee_rate: float,
        position_sizer: PositionSizer,
        trade_repo: PaperTradeRepository,
        snapshot_repo: AccountSnapshotRepository,
        stats_calculator: StatsCalculator | None = None,
        maintenance_margin_rate: float = 0.5,
    ) -> None:
        self._initial_equity = initial_equity
        self._taker_fee_rate = taker_fee_rate
        self._position_sizer = position_sizer
        self._trade_repo = trade_repo
        self._snapshot_repo = snapshot_repo
        self._stats_calculator = stats_calculator
        self._maintenance_margin_rate = maintenance_margin_rate
        self._positions: list[Position] = []
        self._closed_pnl: float = 0.0
        self._total_fees: float = 0.0
        self._paused: bool = False

    @property
    def equity(self) -> float:
        return self._initial_equity + self._closed_pnl - self._total_fees

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def open_positions_risk_pct(self) -> float:
        return sum(p.risk_pct for p in self._positions)

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        logger.info("engine_pause_state", paused=paused)

    def get_open_positions(self) -> list[Position]:
        return list(self._positions)

    def calculate_margin(self, *, quantity: float, price: float, leverage: int) -> float:
        return quantity * price / leverage

    def calculate_liquidation_price(
        self, *, entry_price: float, leverage: int, side: Side,
    ) -> float:
        mmr = self._maintenance_margin_rate / 100
        if side == Side.LONG:
            return entry_price * (1 - 1 / leverage + mmr)
        return entry_price * (1 + 1 / leverage - mmr)

    @property
    def used_margin(self) -> float:
        return sum(p.margin for p in self._positions)

    @property
    def available_balance(self) -> float:
        return self.equity - self.used_margin

    def open_position(
        self, proposal: TradeProposal, current_price: float, leverage: int = 1,
    ) -> Position:
        if proposal.stop_loss is None:
            raise ValueError(
                f"Cannot open position for {proposal.symbol}: stop_loss is required"
            )

        stop_loss = proposal.stop_loss

        quantity = self._position_sizer.calculate(
            equity=self.equity,
            risk_pct=proposal.position_size_risk_pct,
            entry_price=current_price,
            stop_loss=stop_loss,
        )

        margin = self.calculate_margin(quantity=quantity, price=current_price, leverage=leverage)
        if margin > self.available_balance:
            raise ValueError(
                f"Insufficient margin: need ${margin:,.2f}, available ${self.available_balance:,.2f}"
            )

        liquidation_price = self.calculate_liquidation_price(
            entry_price=current_price, leverage=leverage, side=proposal.side,
        )

        open_fee = quantity * current_price * self._taker_fee_rate
        self._total_fees += open_fee

        position = Position(
            trade_id=str(uuid.uuid4()),
            proposal_id=proposal.proposal_id,
            symbol=proposal.symbol,
            side=proposal.side,
            entry_price=current_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=proposal.take_profit,
            opened_at=datetime.now(UTC),
            risk_pct=proposal.position_size_risk_pct,
            leverage=leverage,
            margin=margin,
            liquidation_price=liquidation_price,
        )
        self._positions.append(position)

        self._trade_repo.save_trade(
            trade_id=position.trade_id,
            proposal_id=position.proposal_id,
            symbol=position.symbol,
            side=position.side.value,
            entry_price=position.entry_price,
            quantity=position.quantity,
            risk_pct=position.risk_pct,
            leverage=leverage,
            margin=margin,
            liquidation_price=liquidation_price,
            stop_loss=stop_loss,
            take_profit=proposal.take_profit,
        )

        logger.info(
            "position_opened",
            trade_id=position.trade_id,
            symbol=position.symbol,
            side=position.side,
            quantity=position.quantity,
            entry_price=current_price,
            leverage=leverage,
            margin=margin,
            fee=open_fee,
        )

        return position

    def add_to_position(
        self, *, trade_id: str, risk_pct: float, current_price: float,
    ) -> Position:
        pos = self._find_position(trade_id)
        add_qty = self._position_sizer.calculate(
            equity=self.equity, risk_pct=risk_pct,
            entry_price=current_price, stop_loss=pos.stop_loss,
        )
        add_margin = self.calculate_margin(
            quantity=add_qty, price=current_price, leverage=pos.leverage,
        )
        if add_margin > self.available_balance:
            raise ValueError(
                f"Insufficient margin: need ${add_margin:,.2f}, "
                f"available ${self.available_balance:,.2f}"
            )

        fee = add_qty * current_price * self._taker_fee_rate
        self._total_fees += fee

        total_qty = pos.quantity + add_qty
        new_entry = (pos.quantity * pos.entry_price + add_qty * current_price) / total_qty
        new_margin = pos.margin + add_margin
        new_liq = self.calculate_liquidation_price(
            entry_price=new_entry, leverage=pos.leverage, side=pos.side,
        )

        new_pos = Position(
            trade_id=pos.trade_id,
            proposal_id=pos.proposal_id,
            symbol=pos.symbol,
            side=pos.side,
            entry_price=new_entry,
            quantity=total_qty,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            opened_at=pos.opened_at,
            risk_pct=pos.risk_pct + risk_pct,
            leverage=pos.leverage,
            margin=new_margin,
            liquidation_price=new_liq,
        )
        self._replace_position(pos.trade_id, new_pos)

        self._trade_repo.update_trade_position(
            trade_id=pos.trade_id, entry_price=new_entry,
            quantity=total_qty, margin=new_margin, liquidation_price=new_liq,
        )
        logger.info(
            "position_added", trade_id=pos.trade_id, add_qty=add_qty,
            new_avg_entry=new_entry, new_total_qty=total_qty, fee=fee,
        )
        return new_pos

    def reduce_position(
        self, *, trade_id: str, pct: float, current_price: float,
    ) -> CloseResult:
        if pct >= 100.0:
            return self.close_position(trade_id=trade_id, current_price=current_price)

        pos = self._find_position(trade_id)
        close_qty = pos.quantity * pct / 100
        remaining_qty = pos.quantity - close_qty

        if pos.side == Side.LONG:
            pnl = (current_price - pos.entry_price) * close_qty
        else:
            pnl = (pos.entry_price - current_price) * close_qty

        fee = close_qty * current_price * self._taker_fee_rate
        self._total_fees += fee
        self._closed_pnl += pnl

        remaining_margin = pos.margin * (remaining_qty / pos.quantity)

        new_pos = Position(
            trade_id=pos.trade_id,
            proposal_id=pos.proposal_id,
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            quantity=remaining_qty,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            opened_at=pos.opened_at,
            risk_pct=pos.risk_pct * (remaining_qty / pos.quantity),
            leverage=pos.leverage,
            margin=remaining_margin,
            liquidation_price=pos.liquidation_price,
        )
        self._replace_position(pos.trade_id, new_pos)

        self._trade_repo.update_trade_partial_close(
            trade_id=pos.trade_id,
            remaining_qty=remaining_qty,
            remaining_margin=remaining_margin,
        )

        logger.info(
            "position_reduced", trade_id=pos.trade_id,
            close_qty=close_qty, remaining_qty=remaining_qty, pnl=pnl,
        )

        self._save_stats_snapshot()

        return CloseResult(
            trade_id=pos.trade_id,
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=current_price,
            quantity=close_qty,
            pnl=pnl,
            fees=fee,
            reason="partial_reduce",
        )

    def close_position(self, *, trade_id: str, current_price: float) -> CloseResult:
        pos = self._find_position(trade_id)
        self._positions = [p for p in self._positions if p.trade_id != trade_id]
        return self._close(pos, exit_price=current_price, reason="manual")

    def get_position_with_pnl(
        self, *, trade_id: str, current_price: float,
    ) -> dict:
        pos = self._find_position(trade_id)
        direction = 1 if pos.side == Side.LONG else -1
        unrealized_pnl = (current_price - pos.entry_price) * pos.quantity * direction
        notional = pos.entry_price * pos.quantity
        pnl_pct = (unrealized_pnl / notional * 100) if notional else 0
        roe_pct = (unrealized_pnl / pos.margin * 100) if pos.margin else pnl_pct
        return {
            "position": pos,
            "unrealized_pnl": unrealized_pnl,
            "pnl_pct": pnl_pct,
            "roe_pct": roe_pct,
        }

    def _find_position(self, trade_id: str) -> Position:
        for p in self._positions:
            if p.trade_id == trade_id:
                return p
        raise ValueError(f"Position {trade_id} not found")

    def _replace_position(self, trade_id: str, new_pos: Position) -> None:
        self._positions = [
            new_pos if p.trade_id == trade_id else p for p in self._positions
        ]

    def check_sl_tp(self, *, symbol: str, current_price: float) -> list[CloseResult]:
        closed: list[CloseResult] = []
        remaining: list[Position] = []

        for pos in self._positions:
            if pos.symbol != symbol:
                remaining.append(pos)
                continue

            trigger = self._check_trigger(pos, current_price)
            if trigger is not None:
                exit_price, reason = trigger
                result = self._close(pos, exit_price=exit_price, reason=reason)
                closed.append(result)
            else:
                remaining.append(pos)

        self._positions = remaining
        return closed

    def rebuild_from_db(self) -> None:
        import json

        open_trades = self._trade_repo.get_open_positions()
        self._positions = [
            Position(
                trade_id=t.trade_id,
                proposal_id=t.proposal_id,
                symbol=t.symbol,
                side=Side(t.side),
                entry_price=t.entry_price,
                quantity=t.quantity,
                stop_loss=t.stop_loss,
                take_profit=json.loads(t.take_profit_json) if t.take_profit_json else [],
                opened_at=t.opened_at,
                risk_pct=t.risk_pct,
                leverage=t.leverage,
                margin=t.margin,
                liquidation_price=t.liquidation_price,
            )
            for t in open_trades
        ]
        # Rebuild closed PnL and fees
        closed_trades = self._trade_repo.get_recent_closed(limit=1000)
        self._closed_pnl = sum(t.pnl for t in closed_trades)
        self._total_fees = sum(t.fees for t in closed_trades)
        logger.info(
            "engine_rebuilt",
            open_positions=len(self._positions),
            closed_pnl=self._closed_pnl,
            total_fees=self._total_fees,
        )

    def _check_trigger(
        self, pos: Position, current_price: float
    ) -> tuple[float, str] | None:
        # Liquidation check (highest priority)
        if pos.leverage > 1:
            if pos.side == Side.LONG and current_price <= pos.liquidation_price:
                return current_price, "liquidation"
            if pos.side == Side.SHORT and current_price >= pos.liquidation_price:
                return current_price, "liquidation"

        # SL/TP checks
        if pos.side == Side.LONG:
            if current_price <= pos.stop_loss:
                return pos.stop_loss, "sl"
            if pos.take_profit and current_price >= pos.take_profit[0]:
                return pos.take_profit[0], "tp"
        elif pos.side == Side.SHORT:
            if current_price >= pos.stop_loss:
                return pos.stop_loss, "sl"
            if pos.take_profit and current_price <= pos.take_profit[0]:
                return pos.take_profit[0], "tp"
        return None

    def _close(self, pos: Position, *, exit_price: float, reason: str) -> CloseResult:
        close_fee = pos.quantity * exit_price * self._taker_fee_rate
        self._total_fees += close_fee

        if reason == "liquidation":
            # Total loss of margin on liquidation
            pnl = -pos.margin
        elif pos.side == Side.LONG:
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        self._closed_pnl += pnl

        self._trade_repo.update_trade_closed(
            trade_id=pos.trade_id,
            exit_price=exit_price,
            pnl=pnl,
            fees=close_fee,
        )
        self._trade_repo.update_trade_close_reason(
            trade_id=pos.trade_id, reason=reason,
        )

        logger.info(
            "position_closed",
            trade_id=pos.trade_id,
            symbol=pos.symbol,
            reason=reason,
            pnl=pnl,
            fee=close_fee,
        )

        self._save_stats_snapshot()

        return CloseResult(
            trade_id=pos.trade_id,
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            pnl=pnl,
            fees=close_fee,
            reason=reason,
        )

    def _save_stats_snapshot(self) -> None:
        """Calculate performance stats and save snapshot."""
        if self._stats_calculator is None:
            return

        closed_trades = self._trade_repo.get_all_closed()
        stats = self._stats_calculator.calculate(
            closed_trades=closed_trades, initial_equity=self._initial_equity
        )
        today = datetime.now(UTC).date()
        daily_pnl = self._trade_repo.get_daily_pnl(today)
        self._snapshot_repo.save_snapshot(
            equity=self.equity,
            open_count=len(self._positions),
            daily_pnl=daily_pnl,
            total_pnl=stats.total_pnl,
            win_rate=stats.win_rate,
            profit_factor=stats.profit_factor,
            max_drawdown_pct=stats.max_drawdown_pct,
            sharpe_ratio=stats.sharpe_ratio,
            total_trades=stats.total_trades,
        )
        logger.info("stats_snapshot_saved", total_pnl=stats.total_pnl, win_rate=stats.win_rate)
