from datetime import UTC, datetime

from orchestrator.exchange.paper_engine import CloseResult
from orchestrator.models import EntryOrder, Side, TradeProposal
from orchestrator.pipeline.runner import PipelineResult
from orchestrator.risk.checker import RiskResult
from orchestrator.storage.models import PaperTradeRecord, TradeProposalRecord
from orchestrator.telegram.bot import SentinelBot, is_admin
from orchestrator.telegram.formatters import (
    format_help,
    format_history,
    format_proposal,
    format_risk_rejection,
    format_status,
    format_trade_report,
    format_welcome,
)


class TestIsAdmin:
    def test_admin_allowed(self):
        assert is_admin(123, admin_ids=[123, 456]) is True

    def test_non_admin_rejected(self):
        assert is_admin(789, admin_ids=[123, 456]) is False

    def test_empty_admin_list_rejects_all(self):
        assert is_admin(123, admin_ids=[]) is False


class TestFormatters:
    def test_format_welcome(self):
        msg = format_welcome()
        assert "Sentinel" in msg
        assert "/help" in msg

    def test_format_help(self):
        msg = format_help()
        assert "/status" in msg
        assert "/coin" in msg
        assert "/run" in msg


class TestSentinelBot:
    def test_create_bot(self):
        bot = SentinelBot(token="test-token", admin_chat_ids=[123])
        assert bot.admin_chat_ids == [123]


class TestFormatProposal:
    def test_format_long_proposal(self):
        result = PipelineResult(
            run_id="test-run",
            symbol="BTC/USDT:USDT",
            status="completed",
            proposal=TradeProposal(
                symbol="BTC/USDT:USDT",
                side=Side.LONG,
                entry=EntryOrder(type="market"),
                position_size_risk_pct=1.5,
                stop_loss=93000.0,
                take_profit=[97000.0],
                time_horizon="4h",
                confidence=0.75,
                invalid_if=[],
                rationale="Bullish momentum with strong support",
            ),
        )
        msg = format_proposal(result)
        assert "LONG" in msg
        assert "BTC/USDT:USDT" in msg
        assert "93000" in msg or "93,000" in msg
        assert "97000" in msg or "97,000" in msg
        assert "Bullish" in msg

    def test_format_flat_proposal(self):
        result = PipelineResult(
            run_id="test-run",
            symbol="BTC/USDT:USDT",
            status="completed",
            proposal=TradeProposal(
                symbol="BTC/USDT:USDT",
                side=Side.FLAT,
                entry=EntryOrder(type="market"),
                position_size_risk_pct=0.0,
                stop_loss=None,
                take_profit=[],
                time_horizon="4h",
                confidence=0.5,
                invalid_if=[],
                rationale="No clear signal",
            ),
        )
        msg = format_proposal(result)
        assert "FLAT" in msg

    def test_format_rejected_proposal(self):
        result = PipelineResult(
            run_id="test-run",
            symbol="BTC/USDT:USDT",
            status="rejected",
            proposal=TradeProposal(
                symbol="BTC/USDT:USDT",
                side=Side.LONG,
                entry=EntryOrder(type="market"),
                position_size_risk_pct=1.5,
                stop_loss=97000.0,
                take_profit=[99000.0],
                time_horizon="4h",
                confidence=0.75,
                invalid_if=[],
                rationale="Bad",
            ),
            rejection_reason="SL on wrong side",
        )
        msg = format_proposal(result)
        assert "REJECTED" in msg or "rejected" in msg


class TestFormatStatus:
    def test_format_status_with_results(self):
        results = [
            PipelineResult(
                run_id="r1",
                symbol="BTC/USDT:USDT",
                status="completed",
                proposal=TradeProposal(
                    symbol="BTC/USDT:USDT",
                    side=Side.LONG,
                    entry=EntryOrder(type="market"),
                    position_size_risk_pct=1.0,
                    stop_loss=93000.0,
                    take_profit=[97000.0],
                    time_horizon="4h",
                    confidence=0.7,
                    invalid_if=[],
                    rationale="test",
                ),
            ),
        ]
        msg = format_status(results)
        assert "BTC" in msg

    def test_format_status_empty(self):
        msg = format_status([])
        assert "No" in msg or "no" in msg


class TestFormatTradeReport:
    def test_format_long_close_sl(self):
        result = CloseResult(
            trade_id="t-001",
            symbol="BTC/USDT:USDT",
            side=Side.LONG,
            entry_price=95000.0,
            exit_price=93000.0,
            quantity=0.075,
            pnl=-150.0,
            fees=7.13,
            reason="sl",
        )
        text = format_trade_report(result)
        assert "[CLOSED]" in text
        assert "BTC/USDT:USDT" in text
        assert "LONG" in text
        assert "93,000.0" in text
        assert "-$150.00" in text
        assert "SL" in text

    def test_format_short_close_tp(self):
        result = CloseResult(
            trade_id="t-002",
            symbol="ETH/USDT:USDT",
            side=Side.SHORT,
            entry_price=3000.0,
            exit_price=2800.0,
            quantity=1.0,
            pnl=200.0,
            fees=2.90,
            reason="tp",
        )
        text = format_trade_report(result)
        assert "SHORT" in text
        assert "TP" in text
        assert "$200.00" in text


class TestFormatRiskRejection:
    def test_format_rejection(self):
        risk_result = RiskResult(
            approved=False,
            rule_violated="max_total_exposure",
            reason="Total exposure 22% exceeds 20% limit",
            action="reject",
        )
        text = format_risk_rejection(
            symbol="BTC/USDT:USDT",
            side="LONG",
            entry_price=95000.0,
            risk_result=risk_result,
        )
        assert "[RISK REJECTED]" in text
        assert "BTC/USDT:USDT" in text
        assert "max_total_exposure" in text

    def test_format_pause(self):
        risk_result = RiskResult(
            approved=False,
            rule_violated="max_consecutive_losses",
            reason="5 consecutive losses reached 5 limit",
            action="pause",
        )
        text = format_risk_rejection(
            symbol="BTC/USDT:USDT",
            side="LONG",
            entry_price=95000.0,
            risk_result=risk_result,
        )
        assert "[RISK PAUSED]" in text


class TestFormatHistory:
    def test_format_empty_history(self):
        text = format_history([])
        assert "No closed trades" in text

    def test_format_history_with_trades(self):
        trade = PaperTradeRecord(
            trade_id="t-001", proposal_id="p-001",
            symbol="BTC/USDT:USDT", side="long",
            entry_price=95000.0, exit_price=93000.0,
            quantity=0.075, pnl=-150.0, fees=7.13,
            status="closed", risk_pct=1.5,
            opened_at=datetime.now(UTC), closed_at=datetime.now(UTC),
        )
        text = format_history([trade])
        assert "BTC/USDT:USDT" in text
        assert "-$150.00" in text


class TestFormatHelpUpdated:
    def test_help_includes_history(self):
        text = format_help()
        assert "/history" in text

    def test_help_includes_resume(self):
        text = format_help()
        assert "/resume" in text


class TestFormatPerfReport:
    def test_format_perf_report_positive(self):
        from orchestrator.stats.calculator import PerformanceStats
        from orchestrator.telegram.formatters import format_perf_report

        stats = PerformanceStats(
            total_pnl=1250.0, total_pnl_pct=12.5, win_rate=0.625,
            total_trades=16, winning_trades=10, losing_trades=6,
            profit_factor=1.85, max_drawdown_pct=4.2, sharpe_ratio=1.32,
        )
        text = format_perf_report(stats)
        assert "+$1,250.00" in text
        assert "12.5%" in text
        assert "62.5%" in text
        assert "10/16" in text
        assert "1.85" in text
        assert "4.2%" in text
        assert "1.32" in text

    def test_format_perf_report_no_trades(self):
        from orchestrator.stats.calculator import PerformanceStats
        from orchestrator.telegram.formatters import format_perf_report

        stats = PerformanceStats(
            total_pnl=0.0, total_pnl_pct=0.0, win_rate=0.0,
            total_trades=0, winning_trades=0, losing_trades=0,
            profit_factor=0.0, max_drawdown_pct=0.0, sharpe_ratio=0.0,
        )
        text = format_perf_report(stats)
        assert "No trades" in text or "0" in text


class TestFormatEvalReport:
    def test_format_eval_report_with_failures(self):
        from orchestrator.telegram.formatters import format_eval_report

        report = {
            "dataset_name": "golden_v1",
            "total_cases": 5,
            "passed_cases": 4,
            "failed_cases": 1,
            "accuracy": 0.8,
            "consistency_score": 0.933,
            "failures": [{"case_id": "bear_divergence", "reason": "expected SHORT, got LONG"}],
        }
        text = format_eval_report(report)
        assert "golden_v1" in text
        assert "80" in text
        assert "bear_divergence" in text
        assert "93" in text

    def test_format_eval_report_all_passed(self):
        from orchestrator.telegram.formatters import format_eval_report

        report = {
            "dataset_name": "golden_v1",
            "total_cases": 5,
            "passed_cases": 5,
            "failed_cases": 0,
            "accuracy": 1.0,
            "consistency_score": 1.0,
            "failures": [],
        }
        text = format_eval_report(report)
        assert "100" in text


class TestBotStatusFromDB:
    def test_bot_has_proposal_repo_setter(self):
        """Verify the bot accepts a proposal_repo for DB-backed status."""
        bot = SentinelBot(token="test-token", admin_chat_ids=[123])
        assert hasattr(bot, "set_proposal_repo")

    def test_format_status_from_records(self):
        """format_status_from_records should render DB proposal records."""
        from orchestrator.telegram.formatters import format_status_from_records

        record = TradeProposalRecord(
            proposal_id="p-001",
            run_id="run-1",
            proposal_json='{"symbol":"BTC/USDT:USDT","side":"long","entry":{"type":"market"},'
            '"position_size_risk_pct":1.5,"stop_loss":93000.0,"take_profit":[97000.0],'
            '"time_horizon":"4h","confidence":0.75,"invalid_if":[],"rationale":"test"}',
            risk_check_result="approved",
        )
        text = format_status_from_records([record])
        assert "BTC/USDT:USDT" in text
        assert "approved" in text.lower()
