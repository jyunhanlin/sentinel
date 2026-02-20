from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, datetime
from typing import Protocol

from pydantic import BaseModel


class ClosedTrade(Protocol):
    pnl: float
    closed_at: datetime | None


class PerformanceStats(BaseModel, frozen=True):
    total_pnl: float
    total_pnl_pct: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float


class StatsCalculator:
    def calculate(
        self, *, closed_trades: list[ClosedTrade], initial_equity: float
    ) -> PerformanceStats:
        if not closed_trades:
            return PerformanceStats(
                total_pnl=0.0, total_pnl_pct=0.0, win_rate=0.0,
                total_trades=0, winning_trades=0, losing_trades=0,
                profit_factor=0.0, max_drawdown_pct=0.0, sharpe_ratio=0.0,
            )

        total_pnl = sum(t.pnl for t in closed_trades)
        total_trades = len(closed_trades)
        winning_trades = sum(1 for t in closed_trades if t.pnl > 0)
        losing_trades = sum(1 for t in closed_trades if t.pnl < 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        gross_profit = sum(t.pnl for t in closed_trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in closed_trades if t.pnl < 0))
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        max_drawdown_pct = self._calc_max_drawdown(closed_trades, initial_equity)
        sharpe_ratio = self._calc_sharpe(closed_trades, initial_equity)

        return PerformanceStats(
            total_pnl=total_pnl,
            total_pnl_pct=(total_pnl / initial_equity) * 100 if initial_equity > 0 else 0.0,
            win_rate=win_rate,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            profit_factor=profit_factor,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
        )

    def _calc_max_drawdown(self, trades: list[ClosedTrade], initial_equity: float) -> float:
        equity = initial_equity
        peak = equity
        max_dd = 0.0
        for t in trades:
            equity += t.pnl
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd * 100  # as percentage

    def _calc_sharpe(self, trades: list[ClosedTrade], initial_equity: float) -> float:
        # Group PnL by date
        daily_pnl: dict[date, float] = defaultdict(float)
        for t in trades:
            if t.closed_at is not None:
                day = t.closed_at.date()
                daily_pnl[day] += t.pnl

        if len(daily_pnl) < 2:
            return 0.0

        daily_returns = [pnl / initial_equity for pnl in daily_pnl.values()]
        mean_return = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean_return) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
        std_return = math.sqrt(variance) if variance > 0 else 0.0

        if std_return == 0.0:
            return 0.0

        return (mean_return / std_return) * math.sqrt(365)
