import pytest

from orchestrator.stats.calculator import PerformanceStats, StatsCalculator


class TestStatsCalculator:
    def _make_trade(self, *, pnl: float, closed_at_str: str = "2026-02-19"):
        """Helper to create a minimal trade-like object."""
        from datetime import datetime
        from unittest.mock import MagicMock

        trade = MagicMock()
        trade.pnl = pnl
        trade.fees = abs(pnl) * 0.001  # tiny fee
        trade.closed_at = datetime.fromisoformat(f"{closed_at_str}T12:00:00+00:00")
        return trade

    def test_no_trades_returns_zeros(self):
        calc = StatsCalculator()
        stats = calc.calculate(closed_trades=[], initial_equity=10000.0)
        assert stats.total_pnl == 0.0
        assert stats.win_rate == 0.0
        assert stats.profit_factor == 0.0
        assert stats.max_drawdown_pct == 0.0
        assert stats.sharpe_ratio == 0.0
        assert stats.total_trades == 0

    def test_all_winning_trades(self):
        trades = [self._make_trade(pnl=100.0), self._make_trade(pnl=200.0)]
        calc = StatsCalculator()
        stats = calc.calculate(closed_trades=trades, initial_equity=10000.0)
        assert stats.total_pnl == 300.0
        assert stats.total_pnl_pct == pytest.approx(3.0, rel=0.01)
        assert stats.win_rate == 1.0
        assert stats.winning_trades == 2
        assert stats.losing_trades == 0
        assert stats.profit_factor == float("inf")

    def test_mixed_trades(self):
        trades = [
            self._make_trade(pnl=200.0),
            self._make_trade(pnl=-100.0),
            self._make_trade(pnl=150.0),
            self._make_trade(pnl=-50.0),
        ]
        calc = StatsCalculator()
        stats = calc.calculate(closed_trades=trades, initial_equity=10000.0)
        assert stats.total_pnl == 200.0
        assert stats.total_trades == 4
        assert stats.winning_trades == 2
        assert stats.losing_trades == 2
        assert stats.win_rate == 0.5
        # profit_factor = 350 / 150 = 2.333...
        assert stats.profit_factor == pytest.approx(2.333, rel=0.01)

    def test_all_losing_trades(self):
        trades = [self._make_trade(pnl=-100.0), self._make_trade(pnl=-50.0)]
        calc = StatsCalculator()
        stats = calc.calculate(closed_trades=trades, initial_equity=10000.0)
        assert stats.win_rate == 0.0
        assert stats.profit_factor == 0.0

    def test_max_drawdown(self):
        """Drawdown should capture peak-to-trough decline."""
        trades = [
            self._make_trade(pnl=500.0, closed_at_str="2026-02-01"),
            self._make_trade(pnl=-800.0, closed_at_str="2026-02-02"),
            self._make_trade(pnl=200.0, closed_at_str="2026-02-03"),
        ]
        calc = StatsCalculator()
        stats = calc.calculate(closed_trades=trades, initial_equity=10000.0)
        # Peak = 10000 + 500 = 10500, trough = 10500 - 800 = 9700
        # Drawdown = (10500 - 9700) / 10500 = 7.619%
        assert stats.max_drawdown_pct == pytest.approx(7.619, rel=0.01)

    def test_sharpe_ratio_insufficient_data(self):
        """With < 2 unique days, Sharpe should be 0."""
        trades = [self._make_trade(pnl=100.0)]
        calc = StatsCalculator()
        stats = calc.calculate(closed_trades=trades, initial_equity=10000.0)
        assert stats.sharpe_ratio == 0.0

    def test_performance_stats_is_frozen(self):
        stats = PerformanceStats(
            total_pnl=0.0, total_pnl_pct=0.0, win_rate=0.0,
            total_trades=0, winning_trades=0, losing_trades=0,
            profit_factor=0.0, max_drawdown_pct=0.0, sharpe_ratio=0.0,
        )
        with pytest.raises(Exception):
            stats.total_pnl = 999.0
