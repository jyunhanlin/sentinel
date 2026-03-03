import pytest

from orchestrator.stats.evaluator import (
    ConfidenceBucket,
    EvaluationReport,
    PeriodStats,
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
