import json
from unittest.mock import MagicMock

import pytest

from orchestrator.stats.evaluator import (
    ConfidenceBucket,
    EvaluationReport,
    PeriodStats,
    PipelineEvaluator,
    SymbolStats,
    TradeEvaluation,
)


class TestEvaluationModels:
    def test_trade_evaluation_frozen(self):
        ev = TradeEvaluation(
            trade_id="t-1",
            proposal_id="p-1",
            symbol="BTC/USDT:USDT",
            direction_correct=True,
            entry_deviation_pct=0.3,
            close_reason="tp",
            confidence=75,
            pnl=100.0,
        )
        assert ev.direction_correct is True
        with pytest.raises(Exception):
            ev.pnl = 999.0

    def test_evaluation_report_frozen(self):
        report = EvaluationReport(
            total_evaluated=1,
            total_unmatched=0,
            direction_accuracy=1.0,
            avg_entry_deviation_pct=0.3,
            sl_hit_rate=0.0,
            tp_hit_rate=1.0,
            liquidation_rate=0.0,
            by_symbol=[],
            by_confidence=[],
            by_period=[],
            evaluations=[],
        )
        assert report.total_evaluated == 1
        with pytest.raises(Exception):
            report.total_evaluated = 99


def _make_trade_record(
    *,
    trade_id: str = "t-1",
    proposal_id: str = "p-1",
    symbol: str = "BTC/USDT:USDT",
    side: str = "long",
    entry_price: float = 69000.0,
    pnl: float = 100.0,
    close_reason: str = "tp",
) -> MagicMock:
    rec = MagicMock()
    rec.trade_id = trade_id
    rec.proposal_id = proposal_id
    rec.symbol = symbol
    rec.side = side
    rec.entry_price = entry_price
    rec.pnl = pnl
    rec.close_reason = close_reason
    return rec


def _make_proposal_record(
    *,
    proposal_id: str = "p-1",
    side: str = "long",
    entry_price: float | None = 68800.0,
    confidence: float = 0.75,
) -> MagicMock:
    proposal = {
        "side": side,
        "entry": {"type": "limit", "price": entry_price},
        "confidence": confidence,
    }
    rec = MagicMock()
    rec.proposal_id = proposal_id
    rec.proposal_json = json.dumps(proposal)
    return rec


class TestEvaluateSingle:
    def test_direction_correct_long_profit(self):
        trade_repo = MagicMock()
        proposal_repo = MagicMock()
        trade_repo.get_by_trade_id.return_value = _make_trade_record(pnl=100.0, side="long")
        proposal_repo.get_by_proposal_id.return_value = _make_proposal_record(side="long")

        evaluator = PipelineEvaluator(trade_repo=trade_repo, proposal_repo=proposal_repo)
        result = evaluator.evaluate_single("t-1")

        assert result is not None
        assert result.direction_correct is True
        assert result.confidence == 75

    def test_direction_wrong_long_loss(self):
        trade_repo = MagicMock()
        proposal_repo = MagicMock()
        trade_repo.get_by_trade_id.return_value = _make_trade_record(pnl=-50.0, side="long")
        proposal_repo.get_by_proposal_id.return_value = _make_proposal_record(side="long")

        evaluator = PipelineEvaluator(trade_repo=trade_repo, proposal_repo=proposal_repo)
        result = evaluator.evaluate_single("t-1")

        assert result is not None
        assert result.direction_correct is False

    def test_entry_deviation_calculated(self):
        trade_repo = MagicMock()
        proposal_repo = MagicMock()
        trade_repo.get_by_trade_id.return_value = _make_trade_record(entry_price=69000.0)
        proposal_repo.get_by_proposal_id.return_value = _make_proposal_record(entry_price=68800.0)

        evaluator = PipelineEvaluator(trade_repo=trade_repo, proposal_repo=proposal_repo)
        result = evaluator.evaluate_single("t-1")

        assert result is not None
        # (69000 - 68800) / 68800 * 100 ≈ 0.291%
        assert result.entry_deviation_pct == pytest.approx(0.291, rel=0.01)

    def test_market_order_no_entry_deviation(self):
        trade_repo = MagicMock()
        proposal_repo = MagicMock()
        trade_repo.get_by_trade_id.return_value = _make_trade_record()
        proposal_repo.get_by_proposal_id.return_value = _make_proposal_record(entry_price=None)

        evaluator = PipelineEvaluator(trade_repo=trade_repo, proposal_repo=proposal_repo)
        result = evaluator.evaluate_single("t-1")

        assert result is not None
        assert result.entry_deviation_pct is None

    def test_no_proposal_returns_none(self):
        trade_repo = MagicMock()
        proposal_repo = MagicMock()
        trade_repo.get_by_trade_id.return_value = _make_trade_record()
        proposal_repo.get_by_proposal_id.return_value = None

        evaluator = PipelineEvaluator(trade_repo=trade_repo, proposal_repo=proposal_repo)
        result = evaluator.evaluate_single("t-1")

        assert result is None

    def test_no_trade_returns_none(self):
        trade_repo = MagicMock()
        proposal_repo = MagicMock()
        trade_repo.get_by_trade_id.return_value = None

        evaluator = PipelineEvaluator(trade_repo=trade_repo, proposal_repo=proposal_repo)
        result = evaluator.evaluate_single("t-1")

        assert result is None


class TestEvaluate:
    def _setup_repos(self, trades, proposals):
        """Setup mock repos with given trades and proposals."""
        trade_repo = MagicMock()
        proposal_repo = MagicMock()
        trade_repo.get_all_closed.return_value = trades
        trade_repo.get_by_trade_id.side_effect = lambda tid: next(
            (t for t in trades if t.trade_id == tid), None
        )
        proposal_repo.get_by_proposal_id.side_effect = lambda pid: next(
            (p for p in proposals if p.proposal_id == pid), None
        )
        return trade_repo, proposal_repo

    def test_evaluate_overall_accuracy(self):
        trades = [
            _make_trade_record(trade_id="t-1", proposal_id="p-1", pnl=100.0, close_reason="tp"),
            _make_trade_record(trade_id="t-2", proposal_id="p-2", pnl=-50.0, close_reason="sl"),
        ]
        proposals = [
            _make_proposal_record(proposal_id="p-1"),
            _make_proposal_record(proposal_id="p-2"),
        ]
        trade_repo, proposal_repo = self._setup_repos(trades, proposals)
        evaluator = PipelineEvaluator(trade_repo=trade_repo, proposal_repo=proposal_repo)

        report = evaluator.evaluate()

        assert report.total_evaluated == 2
        assert report.direction_accuracy == 0.5
        assert report.tp_hit_rate == 0.5
        assert report.sl_hit_rate == 0.5

    def test_evaluate_by_symbol(self):
        trades = [
            _make_trade_record(trade_id="t-1", proposal_id="p-1", symbol="BTC/USDT:USDT", pnl=100.0),
            _make_trade_record(trade_id="t-2", proposal_id="p-2", symbol="ETH/USDT:USDT", pnl=-50.0),
        ]
        proposals = [
            _make_proposal_record(proposal_id="p-1"),
            _make_proposal_record(proposal_id="p-2"),
        ]
        trade_repo, proposal_repo = self._setup_repos(trades, proposals)
        evaluator = PipelineEvaluator(trade_repo=trade_repo, proposal_repo=proposal_repo)

        report = evaluator.evaluate()

        assert len(report.by_symbol) == 2
        btc = next(s for s in report.by_symbol if s.symbol == "BTC/USDT:USDT")
        assert btc.direction_accuracy == 1.0

    def test_evaluate_by_confidence(self):
        trades = [
            _make_trade_record(trade_id="t-1", proposal_id="p-1", pnl=100.0),
            _make_trade_record(trade_id="t-2", proposal_id="p-2", pnl=-50.0),
        ]
        proposals = [
            _make_proposal_record(proposal_id="p-1", confidence=0.80),
            _make_proposal_record(proposal_id="p-2", confidence=0.30),
        ]
        trade_repo, proposal_repo = self._setup_repos(trades, proposals)
        evaluator = PipelineEvaluator(trade_repo=trade_repo, proposal_repo=proposal_repo)

        report = evaluator.evaluate()

        assert len(report.by_confidence) >= 2
        high = next(b for b in report.by_confidence if "high" in b.bucket)
        low = next(b for b in report.by_confidence if "low" in b.bucket)
        assert high.direction_accuracy == 1.0
        assert low.direction_accuracy == 0.0

    def test_evaluate_unmatched_trades(self):
        trades = [
            _make_trade_record(trade_id="t-1", proposal_id="p-1", pnl=100.0),
            _make_trade_record(trade_id="t-2", proposal_id="p-missing", pnl=-50.0),
        ]
        proposals = [
            _make_proposal_record(proposal_id="p-1"),
        ]
        trade_repo, proposal_repo = self._setup_repos(trades, proposals)
        evaluator = PipelineEvaluator(trade_repo=trade_repo, proposal_repo=proposal_repo)

        report = evaluator.evaluate()

        assert report.total_evaluated == 1
        assert report.total_unmatched == 1

    def test_evaluate_empty_trades(self):
        trade_repo, proposal_repo = self._setup_repos([], [])
        evaluator = PipelineEvaluator(trade_repo=trade_repo, proposal_repo=proposal_repo)

        report = evaluator.evaluate()

        assert report.total_evaluated == 0
        assert report.direction_accuracy == 0.0

    def test_evaluate_symbol_filter(self):
        trades = [
            _make_trade_record(trade_id="t-1", proposal_id="p-1", symbol="BTC/USDT:USDT", pnl=100.0),
            _make_trade_record(trade_id="t-2", proposal_id="p-2", symbol="ETH/USDT:USDT", pnl=-50.0),
        ]
        proposals = [
            _make_proposal_record(proposal_id="p-1"),
            _make_proposal_record(proposal_id="p-2"),
        ]
        trade_repo, proposal_repo = self._setup_repos(trades, proposals)
        evaluator = PipelineEvaluator(trade_repo=trade_repo, proposal_repo=proposal_repo)

        report = evaluator.evaluate(symbol="BTC/USDT:USDT")

        assert report.total_evaluated == 1
        assert report.direction_accuracy == 1.0
