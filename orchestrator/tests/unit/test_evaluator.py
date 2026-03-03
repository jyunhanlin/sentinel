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
