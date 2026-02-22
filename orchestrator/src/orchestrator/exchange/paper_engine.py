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

    def open_position(self, proposal: TradeProposal, current_price: float) -> Position:
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
        )

        logger.info(
            "position_opened",
            trade_id=position.trade_id,
            symbol=position.symbol,
            side=position.side,
            quantity=position.quantity,
            entry_price=current_price,
            fee=open_fee,
        )

        return position

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
        open_trades = self._trade_repo.get_open_positions()
        self._positions = [
            Position(
                trade_id=t.trade_id,
                proposal_id=t.proposal_id,
                symbol=t.symbol,
                side=Side(t.side),
                entry_price=t.entry_price,
                quantity=t.quantity,
                stop_loss=0.0,  # not stored in DB, positions will rely on next check
                take_profit=[],
                opened_at=t.opened_at,
                risk_pct=t.risk_pct,
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

        if pos.side == Side.LONG:
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
