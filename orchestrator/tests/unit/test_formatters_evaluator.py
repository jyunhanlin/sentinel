from orchestrator.stats.evaluator import (
    ConfidenceBucket,
    EvaluationReport,
    SymbolStats,
    TradeEvaluation,
)
from orchestrator.telegram.formatters import (
    format_evaluation_report,
    format_trade_evaluation,
)


class TestFormatTradeEvaluation:
    def test_direction_correct(self):
        ev = TradeEvaluation(
            trade_id="t-1", proposal_id="p-1", symbol="BTC/USDT:USDT",
            direction_correct=True, entry_deviation_pct=0.3,
            close_reason="tp", confidence=75, pnl=100.0,
        )
        result = format_trade_evaluation(ev)
        assert "✅" in result or "Direction correct" in result.lower() or "correct" in result.lower()
        assert "0.3" in result
        assert "75" in result

    def test_direction_wrong(self):
        ev = TradeEvaluation(
            trade_id="t-1", proposal_id="p-1", symbol="BTC/USDT:USDT",
            direction_correct=False, entry_deviation_pct=-1.2,
            close_reason="sl", confidence=40, pnl=-50.0,
        )
        result = format_trade_evaluation(ev)
        assert "❌" in result or "Direction wrong" in result.lower() or "wrong" in result.lower()

    def test_no_entry_deviation(self):
        ev = TradeEvaluation(
            trade_id="t-1", proposal_id="p-1", symbol="BTC/USDT:USDT",
            direction_correct=True, entry_deviation_pct=None,
            close_reason="tp", confidence=75, pnl=100.0,
        )
        result = format_trade_evaluation(ev)
        assert "market" in result.lower() or "N/A" in result or "n/a" in result.lower()


class TestFormatEvaluationReport:
    def test_empty_report(self):
        report = EvaluationReport(
            total_evaluated=0, total_unmatched=0,
            direction_accuracy=0.0, avg_entry_deviation_pct=0.0,
            sl_hit_rate=0.0, tp_hit_rate=0.0, liquidation_rate=0.0,
            by_symbol=[], by_confidence=[], by_period=[], evaluations=[],
        )
        result = format_evaluation_report(report)
        assert "no" in result.lower() or "0" in result

    def test_full_report_contains_sections(self):
        report = EvaluationReport(
            total_evaluated=10, total_unmatched=2,
            direction_accuracy=0.7, avg_entry_deviation_pct=0.5,
            sl_hit_rate=0.3, tp_hit_rate=0.5, liquidation_rate=0.2,
            by_symbol=[
                SymbolStats(
                    symbol="BTC/USDT:USDT", total_trades=10,
                    direction_accuracy=0.7, avg_entry_deviation_pct=0.5,
                    sl_hit_rate=0.3, tp_hit_rate=0.5, total_pnl=500.0,
                ),
            ],
            by_confidence=[
                ConfidenceBucket(
                    bucket="high (67-100)", total_trades=5,
                    direction_accuracy=0.8, avg_pnl=100.0,
                ),
            ],
            by_period=[],
            evaluations=[],
        )
        result = format_evaluation_report(report)
        assert "70" in result  # direction accuracy
        assert "BTC" in result
        assert "high" in result.lower()
