from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

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


class TestFormatPendingApproval:
    def test_format_pending_long(self):
        from datetime import UTC, datetime, timedelta

        from orchestrator.approval.manager import PendingApproval
        from orchestrator.telegram.formatters import format_pending_approval

        now = datetime.now(UTC)
        approval = PendingApproval(
            approval_id="a-001",
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
                rationale="Strong breakout",
            ),
            run_id="run-1",
            snapshot_price=95200.0,
            created_at=now,
            expires_at=now + timedelta(minutes=15),
        )
        text = format_pending_approval(approval)
        assert "PENDING" in text
        assert "BTC/USDT:USDT" in text
        assert "LONG" in text
        assert "93,000" in text or "93000" in text
        assert "15" in text  # timeout mention


class TestFormatExecutionResult:
    def test_format_live_execution(self):
        from orchestrator.execution.executor import ExecutionResult
        from orchestrator.telegram.formatters import format_execution_result

        result = ExecutionResult(
            trade_id="t-001",
            symbol="BTC/USDT:USDT",
            side="long",
            entry_price=95350.0,
            quantity=0.075,
            fees=3.57,
            mode="live",
            exchange_order_id="binance-001",
            sl_order_id="sl-001",
            tp_order_id="tp-001",
        )
        text = format_execution_result(result)
        assert "EXECUTED" in text
        assert "live" in text.lower()
        assert "95,350" in text or "95350" in text

    def test_format_paper_execution(self):
        from orchestrator.execution.executor import ExecutionResult
        from orchestrator.telegram.formatters import format_execution_result

        result = ExecutionResult(
            trade_id="t-002",
            symbol="BTC/USDT:USDT",
            side="long",
            entry_price=95200.0,
            quantity=0.075,
            fees=3.57,
            mode="paper",
        )
        text = format_execution_result(result)
        assert "EXECUTED" in text
        assert "paper" in text.lower()


class TestApprovalCallback:
    @pytest.mark.asyncio
    async def test_approve_callback_executes(self):
        """Clicking Approve should execute the order and send confirmation."""
        from datetime import UTC, datetime, timedelta

        from orchestrator.approval.manager import PendingApproval
        from orchestrator.execution.executor import ExecutionResult

        now = datetime.now(UTC)
        approval = PendingApproval(
            approval_id="a-001",
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
                rationale="test",
            ),
            run_id="run-1",
            snapshot_price=95200.0,
            created_at=now,
            expires_at=now + timedelta(minutes=15),
        )
        approval_mgr = MagicMock()
        approval_mgr.approve.return_value = approval

        executor = AsyncMock()
        executor.execute_entry.return_value = ExecutionResult(
            trade_id="t-001",
            symbol="BTC/USDT:USDT",
            side="long",
            entry_price=95250.0,
            quantity=0.075,
            fees=3.57,
            mode="paper",
        )
        executor.place_sl_tp.return_value = []

        data_fetcher = AsyncMock()
        data_fetcher.fetch_current_price.return_value = 95250.0

        bot = SentinelBot(
            token="test-token", admin_chat_ids=[123],
            approval_manager=approval_mgr,
            executor=executor,
            data_fetcher=data_fetcher,
        )

        # Simulate callback
        query = MagicMock()
        query.data = "approve:a-001"
        query.from_user.id = 123
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        update.effective_chat = MagicMock()
        update.effective_chat.id = 123
        context = MagicMock()

        await bot._callback_router(update, context)
        executor.execute_entry.assert_called_once()
        query.answer.assert_called()

    @pytest.mark.asyncio
    async def test_reject_callback(self):
        from datetime import UTC, datetime, timedelta

        from orchestrator.approval.manager import PendingApproval

        now = datetime.now(UTC)
        approval = PendingApproval(
            approval_id="a-002",
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
                rationale="test",
            ),
            run_id="run-1",
            snapshot_price=95200.0,
            created_at=now,
            expires_at=now + timedelta(minutes=15),
        )
        approval_mgr = MagicMock()
        approval_mgr.reject.return_value = approval
        bot = SentinelBot(
            token="test-token", admin_chat_ids=[123],
            approval_manager=approval_mgr,
        )

        query = MagicMock()
        query.data = "reject:a-002"
        query.from_user.id = 123
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        update.effective_chat = MagicMock()
        update.effective_chat.id = 123
        context = MagicMock()

        await bot._callback_router(update, context)
        approval_mgr.reject.assert_called_once_with("a-002")
        query.edit_message_text.assert_called()


class TestBotStatusFromDB:
    def test_bot_accepts_proposal_repo(self):
        """Verify the bot accepts a proposal_repo for DB-backed status."""
        mock_repo = MagicMock()
        bot = SentinelBot(token="test-token", admin_chat_ids=[123], proposal_repo=mock_repo)
        assert bot._proposal_repo is mock_repo

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


class TestEvalHandler:
    @pytest.mark.asyncio
    async def test_eval_handler_no_runner(self):
        """Without eval runner, /eval should say not configured."""
        bot = SentinelBot(token="test-token", admin_chat_ids=[123])
        update = MagicMock()
        update.effective_chat.id = 123
        update.message.reply_text = AsyncMock()
        context = MagicMock()

        await bot._eval_handler(update, context)
        text = update.message.reply_text.call_args[0][0]
        assert "not configured" in text.lower()


def _make_update(chat_id: int = 123):
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.message.reply_text = AsyncMock()
    return update


def _make_context(args=None):
    ctx = MagicMock()
    ctx.args = args or []
    return ctx


class TestStartHandler:
    @pytest.mark.asyncio
    async def test_start_replies_welcome(self):
        bot = SentinelBot(token="t", admin_chat_ids=[123])
        update = _make_update(123)
        await bot._start_handler(update, _make_context())
        text = update.message.reply_text.call_args[0][0]
        assert "Sentinel" in text

    @pytest.mark.asyncio
    async def test_start_rejects_non_admin(self):
        bot = SentinelBot(token="t", admin_chat_ids=[123])
        update = _make_update(999)
        await bot._start_handler(update, _make_context())
        update.message.reply_text.assert_not_called()


class TestHelpHandler:
    @pytest.mark.asyncio
    async def test_help_replies(self):
        bot = SentinelBot(token="t", admin_chat_ids=[123])
        update = _make_update(123)
        await bot._help_handler(update, _make_context())
        text = update.message.reply_text.call_args[0][0]
        assert "/status" in text


def _make_pipeline_result():
    return PipelineResult(
        run_id="r1", symbol="BTC/USDT:USDT", status="completed",
        proposal=TradeProposal(
            symbol="BTC/USDT:USDT", side=Side.LONG,
            entry=EntryOrder(type="market"),
            position_size_risk_pct=1.0, stop_loss=93000.0,
            take_profit=[97000.0], time_horizon="4h",
            confidence=0.7, invalid_if=[], rationale="test",
        ),
    )


class TestStatusHandler:
    @pytest.mark.asyncio
    async def test_status_with_results(self):
        bot = SentinelBot(token="t", admin_chat_ids=[123])
        bot._latest_results = {"BTC/USDT:USDT": _make_pipeline_result()}
        update = _make_update(123)
        await bot._status_handler(update, _make_context())
        update.message.reply_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_status_empty_with_proposal_repo(self):
        mock_repo = MagicMock()
        mock_repo.get_recent.return_value = []
        bot = SentinelBot(token="t", admin_chat_ids=[123], proposal_repo=mock_repo)
        update = _make_update(123)
        await bot._status_handler(update, _make_context())
        update.message.reply_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_status_empty_no_repo(self):
        bot = SentinelBot(token="t", admin_chat_ids=[123])
        update = _make_update(123)
        await bot._status_handler(update, _make_context())
        text = update.message.reply_text.call_args[0][0]
        assert "No" in text or "no" in text


class TestCoinHandler:
    @pytest.mark.asyncio
    async def test_coin_no_args(self):
        bot = SentinelBot(token="t", admin_chat_ids=[123])
        update = _make_update(123)
        await bot._coin_handler(update, _make_context())
        text = update.message.reply_text.call_args[0][0]
        assert "Usage" in text

    @pytest.mark.asyncio
    async def test_coin_matching_result(self):
        bot = SentinelBot(token="t", admin_chat_ids=[123])
        bot._latest_results = {"BTC/USDT:USDT": _make_pipeline_result()}
        update = _make_update(123)
        await bot._coin_handler(update, _make_context(args=["BTC"]))
        update.message.reply_text.assert_called()

    @pytest.mark.asyncio
    async def test_coin_no_match(self):
        bot = SentinelBot(token="t", admin_chat_ids=[123])
        update = _make_update(123)
        await bot._coin_handler(update, _make_context(args=["XYZ"]))
        text = update.message.reply_text.call_args[0][0]
        assert "No recent" in text


class TestRunHandler:
    @pytest.mark.asyncio
    async def test_run_no_scheduler(self):
        bot = SentinelBot(token="t", admin_chat_ids=[123])
        update = _make_update(123)
        await bot._run_handler(update, _make_context())
        text = update.message.reply_text.call_args[0][0]
        assert "not configured" in text.lower()

    @pytest.mark.asyncio
    async def test_run_all_symbols(self):
        mock_scheduler = MagicMock()
        mock_scheduler.symbols = ["BTC/USDT:USDT"]
        mock_scheduler.run_once = AsyncMock(return_value=[])
        bot = SentinelBot(token="t", admin_chat_ids=[123], scheduler=mock_scheduler)
        update = _make_update(123)
        await bot._run_handler(update, _make_context())
        mock_scheduler.run_once.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_model_alias(self):
        mock_scheduler = MagicMock()
        mock_scheduler.symbols = ["BTC/USDT:USDT"]
        mock_scheduler.run_once = AsyncMock(return_value=[])
        bot = SentinelBot(token="t", admin_chat_ids=[123], scheduler=mock_scheduler)
        update = _make_update(123)
        await bot._run_handler(update, _make_context(args=["opus"]))
        call_kwargs = mock_scheduler.run_once.call_args[1]
        assert call_kwargs.get("model_override") == "anthropic/claude-opus-4-6"


class TestHistoryHandler:
    @pytest.mark.asyncio
    async def test_history_no_repo(self):
        bot = SentinelBot(token="t", admin_chat_ids=[123])
        update = _make_update(123)
        await bot._history_handler(update, _make_context())
        text = update.message.reply_text.call_args[0][0]
        assert "not configured" in text.lower()

    @pytest.mark.asyncio
    async def test_history_with_trades(self):
        mock_repo = MagicMock()
        mock_repo.get_recent_closed.return_value = []
        bot = SentinelBot(token="t", admin_chat_ids=[123], trade_repo=mock_repo)
        update = _make_update(123)
        await bot._history_handler(update, _make_context())
        update.message.reply_text.assert_called_once()


class TestResumeHandler:
    @pytest.mark.asyncio
    async def test_resume_no_engine(self):
        bot = SentinelBot(token="t", admin_chat_ids=[123])
        update = _make_update(123)
        await bot._resume_handler(update, _make_context())
        text = update.message.reply_text.call_args[0][0]
        assert "not configured" in text.lower()

    @pytest.mark.asyncio
    async def test_resume_unpauses(self):
        mock_engine = MagicMock()
        bot = SentinelBot(token="t", admin_chat_ids=[123], paper_engine=mock_engine)
        update = _make_update(123)
        await bot._resume_handler(update, _make_context())
        mock_engine.set_paused.assert_called_once_with(False)
        text = update.message.reply_text.call_args[0][0]
        assert "resumed" in text.lower()


class TestPushMethods:
    @pytest.mark.asyncio
    async def test_push_proposal_no_app(self):
        bot = SentinelBot(token="t", admin_chat_ids=[123])
        await bot.push_proposal(123, MagicMock())
        # Should return silently when _app is None

    @pytest.mark.asyncio
    async def test_push_close_report_no_app(self):
        bot = SentinelBot(token="t", admin_chat_ids=[123])
        await bot.push_close_report(MagicMock())

    @pytest.mark.asyncio
    async def test_push_risk_rejection_no_app(self):
        bot = SentinelBot(token="t", admin_chat_ids=[123])
        await bot.push_risk_rejection(
            symbol="BTC", side="LONG", entry_price=95000.0,
            risk_result=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_push_to_admins(self):
        bot = SentinelBot(token="t", admin_chat_ids=[123, 456])
        bot._app = MagicMock()
        bot._app.bot.send_message = AsyncMock()
        await bot.push_to_admins(_make_pipeline_result())
        assert bot._app.bot.send_message.call_count == 2


class TestBuildBot:
    def test_build_registers_handlers(self):
        bot = SentinelBot(token="test-token", admin_chat_ids=[123])
        app = bot.build()
        # Should have registered handlers without error
        assert app is not None
        assert bot._app is not None


class TestPushWithApp:
    @pytest.mark.asyncio
    async def test_push_proposal_with_app(self):
        bot = SentinelBot(token="t", admin_chat_ids=[123])
        bot._app = MagicMock()
        bot._app.bot.send_message = AsyncMock()
        await bot.push_proposal(123, _make_pipeline_result())
        bot._app.bot.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_push_close_report_with_app(self):
        bot = SentinelBot(token="t", admin_chat_ids=[123, 456])
        bot._app = MagicMock()
        bot._app.bot.send_message = AsyncMock()
        result = CloseResult(
            trade_id="t-1", symbol="BTC/USDT:USDT", side=Side.LONG,
            entry_price=95000.0, exit_price=93000.0, quantity=0.01,
            pnl=-20.0, fees=1.0, reason="sl",
        )
        await bot.push_close_report(result)
        assert bot._app.bot.send_message.call_count == 2

    @pytest.mark.asyncio
    async def test_push_risk_rejection_with_app(self):
        bot = SentinelBot(token="t", admin_chat_ids=[123])
        bot._app = MagicMock()
        bot._app.bot.send_message = AsyncMock()
        rr = RiskResult(
            approved=False, rule_violated="max_single_risk",
            reason="Too risky", action="reject",
        )
        await bot.push_risk_rejection(
            symbol="BTC/USDT:USDT", side="LONG",
            entry_price=95000.0, risk_result=rr,
        )
        bot._app.bot.send_message.assert_called_once()


class TestRunHandlerAdvanced:
    @pytest.mark.asyncio
    async def test_run_with_symbol_arg(self):
        mock_scheduler = MagicMock()
        mock_scheduler.symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        result = _make_pipeline_result()
        mock_scheduler.run_once = AsyncMock(return_value=[result])
        bot = SentinelBot(token="t", admin_chat_ids=[123], scheduler=mock_scheduler)
        update = _make_update(123)
        await bot._run_handler(update, _make_context(args=["BTC"]))
        call_kwargs = mock_scheduler.run_once.call_args[1]
        assert call_kwargs["symbols"] == ["BTC/USDT:USDT"]

    @pytest.mark.asyncio
    async def test_run_unknown_symbol(self):
        mock_scheduler = MagicMock()
        mock_scheduler.symbols = ["BTC/USDT:USDT"]
        bot = SentinelBot(token="t", admin_chat_ids=[123], scheduler=mock_scheduler)
        update = _make_update(123)
        await bot._run_handler(update, _make_context(args=["XYZ"]))
        text = update.message.reply_text.call_args[0][0]
        assert "Unknown" in text

    @pytest.mark.asyncio
    async def test_run_with_premium_model_setting(self):
        mock_scheduler = MagicMock()
        mock_scheduler.symbols = ["BTC/USDT:USDT"]
        mock_scheduler.run_once = AsyncMock(return_value=[])
        bot = SentinelBot(
            token="t", admin_chat_ids=[123],
            premium_model="anthropic/claude-opus-4-6",
            scheduler=mock_scheduler,
        )
        update = _make_update(123)
        await bot._run_handler(update, _make_context())
        call_kwargs = mock_scheduler.run_once.call_args[1]
        assert call_kwargs.get("model_override") == "anthropic/claude-opus-4-6"


class TestCoinHandlerDB:
    @pytest.mark.asyncio
    async def test_coin_db_fallback(self):
        mock_repo = MagicMock()
        record = TradeProposalRecord(
            proposal_id="p-1", run_id="r-1",
            proposal_json='{"symbol":"BTC/USDT:USDT","side":"long"}',
            risk_check_result="approved",
        )
        mock_repo.get_recent.return_value = [record]
        bot = SentinelBot(token="t", admin_chat_ids=[123], proposal_repo=mock_repo)
        update = _make_update(123)
        await bot._coin_handler(update, _make_context(args=["BTC"]))
        update.message.reply_text.assert_called()


class TestEvalHandlerWithRunner:
    @pytest.mark.asyncio
    async def test_eval_handler_runs(self):
        from orchestrator.eval.runner import CaseResult, EvalReport

        mock_runner = AsyncMock()
        mock_runner.run_default.return_value = EvalReport(
            dataset_name="golden_v1", total_cases=1,
            passed_cases=1, failed_cases=0, accuracy=1.0,
            case_results=[
                CaseResult(case_id="test", passed=True, scores=[]),
            ],
        )
        bot = SentinelBot(token="t", admin_chat_ids=[123], eval_runner=mock_runner)
        update = _make_update(123)
        await bot._eval_handler(update, _make_context())
        # Should have 2 calls: "Running evaluation..." and the report
        assert update.message.reply_text.call_count == 2


class TestPerfHandler:
    @pytest.mark.asyncio
    async def test_perf_handler_returns_stats(self):
        """The /perf command should show performance stats."""
        mock_snapshot = MagicMock()
        mock_snapshot.total_pnl = 500.0
        mock_snapshot.win_rate = 0.6
        mock_snapshot.profit_factor = 1.5
        mock_snapshot.max_drawdown_pct = 3.0
        mock_snapshot.sharpe_ratio = 1.1
        mock_snapshot.total_trades = 10
        mock_snapshot.equity = 10500.0

        mock_snapshot_repo = MagicMock()
        mock_snapshot_repo.get_latest.return_value = mock_snapshot
        bot = SentinelBot(token="test-token", admin_chat_ids=[123], snapshot_repo=mock_snapshot_repo)

        update = MagicMock()
        update.effective_chat.id = 123
        update.message.reply_text = AsyncMock()
        context = MagicMock()

        await bot._perf_handler(update, context)
        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "Win Rate" in text or "win" in text.lower()

    @pytest.mark.asyncio
    async def test_perf_handler_no_data(self):
        """The /perf command should say no data when no snapshots exist."""
        mock_snapshot_repo = MagicMock()
        mock_snapshot_repo.get_latest.return_value = None
        bot = SentinelBot(token="test-token", admin_chat_ids=[123], snapshot_repo=mock_snapshot_repo)

        update = MagicMock()
        update.effective_chat.id = 123
        update.message.reply_text = AsyncMock()
        context = MagicMock()

        await bot._perf_handler(update, context)
        text = update.message.reply_text.call_args[0][0]
        assert "No" in text or "no" in text


class TestTranslation:
    def test_to_chinese_labels(self):
        from orchestrator.telegram.translations import to_chinese

        text = "[NEW] BTC/USDT:USDT  (02/22 11:00)\nSide: LONG\nEntry: market\nRisk: 1.5%"
        zh = to_chinese(text)
        assert "新提案" in zh
        assert "方向：" in zh
        assert "進場：" in zh
        assert "風險：" in zh
        # Original values preserved
        assert "BTC/USDT:USDT" in zh
        assert "LONG" in zh

    def test_to_chinese_status(self):
        from orchestrator.telegram.translations import to_chinese

        text = "Latest pipeline results:\n─────────────────────\nBTC — LONG [completed]"
        zh = to_chinese(text)
        assert "最新 Pipeline 結果" in zh

    def test_to_chinese_perf(self):
        from orchestrator.telegram.translations import to_chinese

        text = "Performance Report\n─────\nWin Rate: 60%\nSharpe Ratio: 1.2"
        zh = to_chinese(text)
        assert "績效報告" in zh
        assert "勝率" in zh

    def test_to_chinese_passthrough(self):
        from orchestrator.telegram.translations import to_chinese

        text = "BTC/USDT:USDT 95000.0"
        assert to_chinese(text) == text  # No labels to translate


class TestTranslateCallback:
    @pytest.mark.asyncio
    async def test_translate_to_chinese(self):
        bot = SentinelBot(token="t", admin_chat_ids=[123])
        bot._msg_cache.store(42, "[NEW] BTC\nSide: LONG")

        query = MagicMock()
        query.data = "translate:zh"
        query.message.message_id = 42
        query.message.reply_markup = None
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        update.effective_chat = MagicMock()
        update.effective_chat.id = 123
        context = MagicMock()

        await bot._callback_router(update, context)
        edited_text = query.edit_message_text.call_args[1]["text"]
        assert "新提案" in edited_text
        assert "方向：" in edited_text

    @pytest.mark.asyncio
    async def test_translate_back_to_english(self):
        bot = SentinelBot(token="t", admin_chat_ids=[123])
        original = "[NEW] BTC\nSide: LONG"
        bot._msg_cache.store(42, original)

        query = MagicMock()
        query.data = "translate:en"
        query.message.message_id = 42
        query.message.reply_markup = None
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        update.effective_chat = MagicMock()
        update.effective_chat.id = 123
        context = MagicMock()

        await bot._callback_router(update, context)
        edited_text = query.edit_message_text.call_args[1]["text"]
        assert edited_text == original
