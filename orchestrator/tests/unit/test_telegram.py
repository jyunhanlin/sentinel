from orchestrator.telegram.bot import SentinelBot, is_admin
from orchestrator.telegram.formatters import (
    format_help,
    format_proposal,
    format_status,
    format_welcome,
)
from orchestrator.models import EntryOrder, Side, TradeProposal
from orchestrator.pipeline.runner import PipelineResult


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
