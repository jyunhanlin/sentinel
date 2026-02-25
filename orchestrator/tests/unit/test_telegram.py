from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.exchange.paper_engine import CloseResult
from orchestrator.models import EntryOrder, Side, TakeProfit, TradeProposal
from orchestrator.pipeline.runner import PipelineResult
from orchestrator.risk.checker import RiskResult
from orchestrator.storage.models import PaperTradeRecord, TradeProposalRecord
from orchestrator.telegram.bot import SentinelBot, is_admin
from orchestrator.telegram.formatters import (
    format_account_overview,
    format_help,
    format_history,
    format_history_paginated,
    format_position_card,
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
                take_profit=[TakeProfit(price=97000.0, close_pct=100)],
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
        assert "Stop Loss" in msg
        assert "Take Profit" in msg
        assert "Confidence" in msg

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
                take_profit=[TakeProfit(price=99000.0, close_pct=100)],
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
                    take_profit=[TakeProfit(price=97000.0, close_pct=100)],
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
        assert "SL" in text
        assert "BTC/USDT:USDT" in text
        assert "LONG" in text
        assert "93,000.0" in text
        assert "-$150.00" in text

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
        assert "RISK REJECTED" in text
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
        assert "RISK PAUSED" in text


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
                take_profit=[TakeProfit(price=97000.0, close_pct=100)],
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
        assert "BTC/USDT:USDT" in text
        assert "LONG" in text
        assert "93,000" in text or "93000" in text
        assert "Stop Loss" in text
        assert "Take Profit" in text
        assert "Leverage: 10x" in text
        assert "Confidence: 75%" in text
        assert "Risk: 1.5%" in text
        assert "Time Horizon: 4h" in text
        assert "Expires in 15 min" in text


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
        assert "LONG" in text
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
        assert "LONG" in text
        assert "paper" in text.lower()


class TestApprovalCallback:
    @pytest.mark.asyncio
    async def test_approve_callback_shows_leverage(self):
        """Clicking Approve should show leverage selection (not execute immediately)."""
        from datetime import UTC, datetime, timedelta

        from orchestrator.approval.manager import PendingApproval

        now = datetime.now(UTC)
        approval = PendingApproval(
            approval_id="a-001",
            proposal=TradeProposal(
                symbol="BTC/USDT:USDT",
                side=Side.LONG,
                entry=EntryOrder(type="market"),
                position_size_risk_pct=1.5,
                stop_loss=93000.0,
                take_profit=[TakeProfit(price=97000.0, close_pct=100)],
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
        approval_mgr.get.return_value = approval

        executor = AsyncMock()

        bot = SentinelBot(
            token="test-token", admin_chat_ids=[123],
            approval_manager=approval_mgr,
            executor=executor,
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
        # Should show leverage selection, NOT execute
        executor.execute_entry.assert_not_called()
        query.edit_message_text.assert_called_once()
        # Verify leverage buttons in markup
        markup = query.edit_message_text.call_args.kwargs.get("reply_markup")
        assert markup is not None
        all_data = [
            btn.callback_data for row in markup.inline_keyboard for btn in row
            if btn.callback_data
        ]
        assert any("leverage:" in d for d in all_data)

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
                take_profit=[TakeProfit(price=97000.0, close_pct=100)],
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
            take_profit=[TakeProfit(price=97000.0, close_pct=100)], time_horizon="4h",
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


class TestStatusWithPositions:
    @pytest.mark.asyncio
    async def test_status_shows_position_cards(self):
        """Status should show open positions with PnL when engine is available."""
        mock_engine = MagicMock()
        pos = MagicMock(
            trade_id="t1", symbol="BTC/USDT:USDT", side=Side.LONG,
            leverage=10, entry_price=68000.0, quantity=0.1,
            margin=680.0, liquidation_price=61540.0,
            stop_loss=67000.0, take_profit=[TakeProfit(price=70000.0, close_pct=100)],
            opened_at=datetime.now(UTC),
        )
        mock_engine.get_open_positions.return_value = [pos]
        mock_engine.get_position_with_pnl.return_value = {
            "position": pos,
            "unrealized_pnl": 100.0,
            "pnl_pct": 1.47,
            "roe_pct": 14.7,
        }
        mock_engine.equity = 10100.0
        mock_engine.available_balance = 9420.0
        mock_engine.used_margin = 680.0
        mock_engine._initial_equity = 10000.0

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_current_price = AsyncMock(return_value=69000.0)

        bot = SentinelBot(
            token="t", admin_chat_ids=[123],
            paper_engine=mock_engine, data_fetcher=mock_fetcher,
        )
        update = _make_update(123)
        await bot._status_handler(update, _make_context())

        # 1 overview + 1 position card = 2 messages
        assert update.message.reply_text.call_count == 2
        calls = update.message.reply_text.call_args_list
        # First call: overview (no reply_markup)
        overview_text = calls[0][0][0]
        assert "Account Overview" in overview_text
        assert "Open Positions: 1" in overview_text
        # Second call: position card with action buttons
        card_text = calls[1][0][0]
        assert "BTC" in card_text
        assert calls[1][1]["reply_markup"] is not None

    @pytest.mark.asyncio
    async def test_status_no_positions_shows_overview(self):
        """When no positions, status should show account overview with 'No open positions'."""
        mock_engine = MagicMock()
        mock_engine.get_open_positions.return_value = []
        mock_engine.equity = 10000.0
        mock_engine.available_balance = 10000.0
        mock_engine.used_margin = 0.0
        mock_engine._initial_equity = 10000.0

        mock_fetcher = MagicMock()

        bot = SentinelBot(
            token="t", admin_chat_ids=[123],
            paper_engine=mock_engine, data_fetcher=mock_fetcher,
        )
        update = _make_update(123)
        await bot._status_handler(update, _make_context())
        text = update.message.reply_text.call_args[0][0]
        assert "Account Overview" in text
        assert "No open positions" in text

    @pytest.mark.asyncio
    async def test_status_no_positions_includes_recent_signals(self):
        """When no positions but has pipeline results, show overview + signals."""
        mock_engine = MagicMock()
        mock_engine.get_open_positions.return_value = []
        mock_engine.equity = 10000.0
        mock_engine.available_balance = 10000.0
        mock_engine.used_margin = 0.0
        mock_engine._initial_equity = 10000.0

        mock_fetcher = MagicMock()

        bot = SentinelBot(
            token="t", admin_chat_ids=[123],
            paper_engine=mock_engine, data_fetcher=mock_fetcher,
        )
        bot._latest_results = {"BTC/USDT:USDT": _make_pipeline_result()}
        update = _make_update(123)
        await bot._status_handler(update, _make_context())
        text = update.message.reply_text.call_args[0][0]
        assert "Account Overview" in text
        assert "Pipeline Status" in text
        assert "BTC" in text


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
        mock_repo.get_closed_paginated.return_value = ([], 0)
        bot = SentinelBot(token="t", admin_chat_ids=[123], trade_repo=mock_repo)
        update = _make_update(123)
        await bot._history_handler(update, _make_context())
        update.message.reply_text.assert_called_once()


class TestHistoryPagination:
    @pytest.mark.asyncio
    async def test_history_shows_paginated_with_buttons(self):
        """The /history command should show paginated trades with nav buttons."""
        trade = MagicMock(
            symbol="BTC/USDT:USDT", side="long", leverage=10,
            entry_price=68000.0, exit_price=67000.0,
            pnl=-100.0, fees=3.4, opened_at=datetime.now(UTC),
            closed_at=datetime.now(UTC), close_reason="sl", margin=680.0,
        )
        mock_repo = MagicMock()
        mock_repo.get_closed_paginated.return_value = ([trade], 8)  # 8 total = 2 pages

        bot = SentinelBot(token="t", admin_chat_ids=[123], trade_repo=mock_repo)
        update = _make_update(123)
        await bot._history_handler(update, _make_context())

        # Should call get_closed_paginated instead of get_recent_closed
        mock_repo.get_closed_paginated.assert_called_once()
        # Should have reply with pagination buttons
        call_kwargs = update.message.reply_text.call_args
        markup = call_kwargs.kwargs.get("reply_markup") or call_kwargs[1].get("reply_markup")
        assert markup is not None
        all_data = [
            btn.callback_data for row in markup.inline_keyboard for btn in row
            if btn.callback_data
        ]
        assert any("history:page:" in d for d in all_data)

    @pytest.mark.asyncio
    async def test_history_single_page_no_nav(self):
        """Single page of results should have no Prev/Next buttons."""
        mock_repo = MagicMock()
        mock_repo.get_closed_paginated.return_value = ([], 0)

        bot = SentinelBot(token="t", admin_chat_ids=[123], trade_repo=mock_repo)
        update = _make_update(123)
        await bot._history_handler(update, _make_context())

        call_kwargs = update.message.reply_text.call_args
        markup = call_kwargs.kwargs.get("reply_markup") or call_kwargs[1].get("reply_markup")
        if markup:
            all_data = [
                btn.callback_data for row in markup.inline_keyboard for btn in row
                if btn.callback_data and btn.callback_data.startswith("history:page:")
            ]
            # Page 1/1, no Prev or Next
            assert len(all_data) == 0


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
        bot = SentinelBot(
            token="test-token", admin_chat_ids=[123],
            snapshot_repo=mock_snapshot_repo,
        )

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
        bot = SentinelBot(
            token="test-token", admin_chat_ids=[123],
            snapshot_repo=mock_snapshot_repo,
        )

        update = MagicMock()
        update.effective_chat.id = 123
        update.message.reply_text = AsyncMock()
        context = MagicMock()

        await bot._perf_handler(update, context)
        text = update.message.reply_text.call_args[0][0]
        assert "No" in text or "no" in text


class TestTranslation:
    @pytest.mark.asyncio
    async def test_to_chinese_calls_llm(self):
        from orchestrator.telegram.translations import to_chinese

        mock_client = MagicMock()
        mock_client.call = AsyncMock(return_value=MagicMock(content="[新提案] BTC\n方向：LONG"))

        result = await to_chinese("[NEW] BTC\nSide: LONG", mock_client)
        assert result == "[新提案] BTC\n方向：LONG"
        mock_client.call.assert_awaited_once()
        call_args = mock_client.call.call_args
        messages = call_args.kwargs.get("messages") or call_args[0][0]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "[NEW] BTC" in messages[1]["content"]


class TestTranslateCallback:
    @pytest.mark.asyncio
    async def test_translate_to_chinese(self):
        mock_llm = MagicMock()
        mock_llm.call = AsyncMock(return_value=MagicMock(content="[新提案] BTC\n方向：LONG"))
        bot = SentinelBot(token="t", admin_chat_ids=[123], llm_client=mock_llm)
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
        mock_llm.call.assert_awaited_once()

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

    @pytest.mark.asyncio
    async def test_translate_no_llm_client(self):
        bot = SentinelBot(token="t", admin_chat_ids=[123])  # no llm_client
        bot._msg_cache.store(42, "[NEW] BTC")

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
        query.answer.assert_awaited_with("Translation not available")
        query.edit_message_text.assert_not_called()


class TestLeverageFormatting:
    def test_format_position_card(self):
        info = {
            "position": MagicMock(
                symbol="BTC/USDT:USDT",
                side=Side.LONG,
                leverage=10,
                entry_price=68000.0,
                quantity=0.1,
                margin=680.0,
                liquidation_price=61540.0,
                stop_loss=67000.0,
                take_profit=[TakeProfit(price=70000.0, close_pct=100)],
                opened_at=datetime.now(UTC),
                trade_id="t1",
            ),
            "unrealized_pnl": 125.0,
            "pnl_pct": 1.84,
            "roe_pct": 18.4,
        }
        text = format_position_card(info)
        assert "10x" in text
        assert "Margin" in text
        assert "Liq" in text
        assert "ROE" in text

    def test_format_account_overview(self):
        text = format_account_overview(
            equity=10150.0,
            available=7320.0,
            used_margin=2680.0,
            initial_equity=10000.0,
            position_count=1,
        )
        assert "Available" in text
        assert "Used Margin" in text
        assert "Open Positions: 1" in text

    def test_format_history_paginated(self):
        trade = MagicMock(
            symbol="BTC/USDT:USDT",
            side="long",
            leverage=10,
            entry_price=68000.0,
            exit_price=67000.0,
            pnl=-100.0,
            fees=3.4,
            opened_at=datetime.now(UTC),
            closed_at=datetime.now(UTC),
            close_reason="sl",
            margin=680.0,
        )
        text = format_history_paginated([trade], page=1, total_pages=3)
        assert "1/3" in text
        assert "10x" in text
        assert "ROE" in text


class TestApproveWithLeverage:
    def _make_bot(self, **kwargs):
        defaults = dict(
            token="test-token",
            admin_chat_ids=[123],
            approval_manager=MagicMock(),
            executor=MagicMock(),
            data_fetcher=MagicMock(),
            paper_engine=MagicMock(),
        )
        defaults.update(kwargs)
        return SentinelBot(**defaults)

    def _make_callback_query(self, data: str):
        query = MagicMock()
        query.data = data
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        query.message = MagicMock()
        query.message.message_id = 1
        query.message.reply_markup = None
        return query

    @pytest.mark.asyncio
    async def test_approve_shows_leverage_selection(self):
        """When user clicks Approve, bot should show leverage options."""
        bot = self._make_bot()
        approval = MagicMock()
        approval.proposal.symbol = "BTC/USDT:USDT"
        approval.proposal.side.value = "long"
        approval.proposal.stop_loss = 67000.0
        approval.proposal.take_profit = [TakeProfit(price=70000.0, close_pct=100)]
        approval.proposal.position_size_risk_pct = 1.0
        approval.snapshot_price = 68000.0
        bot._approval_manager.get.return_value = approval

        query = self._make_callback_query("approve:abc123")
        update = MagicMock()
        update.callback_query = query
        update.effective_chat = MagicMock()
        update.effective_chat.id = 123
        context = MagicMock()

        await bot._callback_router(update, context)

        # Should show leverage buttons, not execute immediately
        query.edit_message_text.assert_called_once()
        call_kwargs = query.edit_message_text.call_args
        markup = call_kwargs.kwargs.get("reply_markup") or call_kwargs[1].get("reply_markup")
        assert markup is not None
        all_data = [
            btn.callback_data
            for row in markup.inline_keyboard
            for btn in row
            if btn.callback_data
        ]
        assert any("leverage:" in d for d in all_data)

    @pytest.mark.asyncio
    async def test_leverage_selection_shows_margin_buttons(self):
        """After selecting leverage, bot should show margin amount selection."""
        bot = self._make_bot()
        approval = MagicMock()
        approval.approval_id = "abc123"
        approval.proposal.symbol = "BTC/USDT:USDT"
        approval.proposal.side = Side.LONG
        approval.proposal.stop_loss = 67000.0
        approval.proposal.take_profit = [TakeProfit(price=70000.0, close_pct=100)]
        approval.proposal.position_size_risk_pct = 1.0
        approval.snapshot_price = 68000.0
        bot._approval_manager.get.return_value = approval

        query = self._make_callback_query("leverage:abc123:10")
        update = MagicMock()
        update.callback_query = query
        update.effective_chat = MagicMock()
        update.effective_chat.id = 123

        await bot._callback_router(update, MagicMock())

        query.edit_message_text.assert_called_once()
        call_kwargs = query.edit_message_text.call_args
        text = call_kwargs.kwargs.get("text", call_kwargs.args[0] if call_kwargs.args else "")
        assert "SELECT MARGIN" in text
        assert "10x" in text
        markup = call_kwargs.kwargs.get("reply_markup")
        all_data = [
            btn.callback_data for row in markup.inline_keyboard for btn in row
            if btn.callback_data
        ]
        assert any("margin:" in d for d in all_data)
        assert any("cancel:" in d for d in all_data)

    @pytest.mark.asyncio
    async def test_confirm_leverage_executes_trade(self):
        """Confirming leverage executes the trade with selected leverage."""
        bot = self._make_bot()
        approval = MagicMock()
        approval.proposal.symbol = "BTC/USDT:USDT"
        approval.proposal.side.value = "long"
        approval.proposal.stop_loss = 67000.0
        approval.proposal.take_profit = [TakeProfit(price=70000.0, close_pct=100)]
        approval.proposal.position_size_risk_pct = 1.0
        approval.snapshot_price = 68000.0
        bot._approval_manager.get.return_value = approval
        bot._data_fetcher.fetch_current_price = AsyncMock(return_value=68000.0)

        exec_result = MagicMock()
        exec_result.trade_id = "t1"
        exec_result.symbol = "BTC/USDT:USDT"
        exec_result.side = "long"
        exec_result.entry_price = 68000.0
        exec_result.quantity = 0.1
        exec_result.fees = 3.4
        exec_result.mode = "paper"
        exec_result.sl_order_id = ""
        exec_result.tp_order_id = ""
        bot._executor.execute_entry = AsyncMock(return_value=exec_result)
        bot._executor.place_sl_tp = AsyncMock(return_value=[])

        query = self._make_callback_query("confirm_leverage:abc123:10")
        update = MagicMock()
        update.callback_query = query
        update.effective_chat = MagicMock()
        update.effective_chat.id = 123
        context = MagicMock()

        await bot._callback_router(update, context)

        bot._executor.execute_entry.assert_called_once()
        call_args = bot._executor.execute_entry.call_args
        assert call_args.kwargs.get("leverage") == 10 or (
            len(call_args.args) > 2 and call_args.args[2] == 10
        )


class TestPositionOperationCallbacks:
    def _make_bot(self):
        return SentinelBot(
            token="test-token",
            admin_chat_ids=[123],
            paper_engine=MagicMock(),
            data_fetcher=MagicMock(),
            trade_repo=MagicMock(),
        )

    def _make_update(self, data: str):
        query = MagicMock()
        query.data = data
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        query.message = MagicMock()
        query.message.message_id = 1
        query.message.reply_markup = None
        update = MagicMock()
        update.callback_query = query
        update.effective_chat = MagicMock()
        update.effective_chat.id = 123
        return update, query

    def _make_position_info(self):
        return {
            "position": MagicMock(
                symbol="BTC/USDT:USDT", side=Side.LONG, leverage=10,
                entry_price=68000.0, quantity=0.1, margin=680.0,
                liquidation_price=61540.0, stop_loss=67000.0,
                take_profit=[TakeProfit(price=70000.0, close_pct=100)], opened_at=datetime.now(UTC),
                trade_id="t1",
            ),
            "unrealized_pnl": 50.0,
            "pnl_pct": 0.74,
            "roe_pct": 7.35,
        }

    @pytest.mark.asyncio
    async def test_close_shows_confirmation(self):
        bot = self._make_bot()
        bot._paper_engine.get_position_with_pnl.return_value = self._make_position_info()
        bot._data_fetcher.fetch_current_price = AsyncMock(return_value=68500.0)

        update, query = self._make_update("close:t1")
        context = MagicMock()
        await bot._callback_router(update, context)

        query.edit_message_text.assert_called_once()
        markup = query.edit_message_text.call_args.kwargs.get("reply_markup")
        assert markup is not None
        all_data = [
            btn.callback_data for row in markup.inline_keyboard for btn in row
            if btn.callback_data
        ]
        assert any("confirm_close:" in d for d in all_data)

    @pytest.mark.asyncio
    async def test_confirm_close_executes(self):
        bot = self._make_bot()
        close_result = MagicMock()
        close_result.pnl = 100.0
        close_result.symbol = "BTC/USDT:USDT"
        close_result.side = Side.LONG
        close_result.entry_price = 68000.0
        close_result.exit_price = 69000.0
        close_result.quantity = 0.1
        close_result.fees = 3.4
        close_result.reason = "manual"
        bot._paper_engine.close_position.return_value = close_result
        bot._data_fetcher.fetch_current_price = AsyncMock(return_value=69000.0)

        update, query = self._make_update("confirm_close:t1")
        context = MagicMock()
        await bot._callback_router(update, context)

        bot._paper_engine.close_position.assert_called_once()

    @pytest.mark.asyncio
    async def test_reduce_shows_pct_options(self):
        bot = self._make_bot()
        bot._paper_engine.get_position_with_pnl.return_value = self._make_position_info()
        bot._data_fetcher.fetch_current_price = AsyncMock(return_value=68500.0)

        update, query = self._make_update("reduce:t1")
        context = MagicMock()
        await bot._callback_router(update, context)

        markup = query.edit_message_text.call_args.kwargs.get("reply_markup")
        all_data = [
            btn.callback_data for row in markup.inline_keyboard for btn in row
            if btn.callback_data
        ]
        assert any("select_reduce:" in d for d in all_data)


class TestReduceFlow:
    def _make_bot(self):
        return SentinelBot(
            token="test-token",
            admin_chat_ids=[123],
            paper_engine=MagicMock(),
            data_fetcher=MagicMock(),
            trade_repo=MagicMock(),
        )

    def _make_update(self, data: str):
        query = MagicMock()
        query.data = data
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        query.message = MagicMock()
        query.message.message_id = 1
        query.message.reply_markup = None
        update = MagicMock()
        update.callback_query = query
        update.effective_chat = MagicMock()
        update.effective_chat.id = 123
        return update, query

    @pytest.mark.asyncio
    async def test_select_reduce_shows_confirmation(self):
        """After selecting %, show confirmation card with estimated PnL."""
        bot = self._make_bot()
        pos = MagicMock(
            symbol="BTC/USDT:USDT", side=Side.LONG, leverage=10,
            entry_price=68000.0, quantity=0.1, margin=680.0,
            stop_loss=67000.0, trade_id="t1",
        )
        bot._paper_engine._find_position.return_value = pos
        bot._data_fetcher.fetch_current_price = AsyncMock(return_value=69000.0)

        update, query = self._make_update("select_reduce:t1:50")
        await bot._callback_router(update, MagicMock())

        call_kwargs = query.edit_message_text.call_args.kwargs
        text = call_kwargs.get("text", "")
        assert "CONFIRM REDUCE" in text
        assert "PnL" in text
        markup = call_kwargs.get("reply_markup")
        all_data = [
            btn.callback_data for row in markup.inline_keyboard for btn in row
            if btn.callback_data
        ]
        assert any("confirm_reduce:" in d for d in all_data)

    @pytest.mark.asyncio
    async def test_confirm_reduce_executes(self):
        """Clicking Confirm should execute the reduce operation."""
        bot = self._make_bot()
        close_result = MagicMock()
        close_result.pnl = 50.0
        close_result.symbol = "BTC/USDT:USDT"
        close_result.side = Side.LONG
        close_result.entry_price = 68000.0
        close_result.exit_price = 69000.0
        close_result.quantity = 0.05
        close_result.fees = 1.7
        close_result.reason = "partial_reduce"
        bot._paper_engine.reduce_position.return_value = close_result
        bot._data_fetcher.fetch_current_price = AsyncMock(return_value=69000.0)

        update, query = self._make_update("confirm_reduce:t1:50")
        await bot._callback_router(update, MagicMock())

        bot._paper_engine.reduce_position.assert_called_once()


class TestAddFlow:
    def _make_bot(self):
        return SentinelBot(
            token="test-token",
            admin_chat_ids=[123],
            paper_engine=MagicMock(),
            data_fetcher=MagicMock(),
            trade_repo=MagicMock(),
        )

    def _make_update(self, data: str):
        query = MagicMock()
        query.data = data
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        query.message = MagicMock()
        query.message.message_id = 1
        query.message.reply_markup = None
        update = MagicMock()
        update.callback_query = query
        update.effective_chat = MagicMock()
        update.effective_chat.id = 123
        return update, query

    @pytest.mark.asyncio
    async def test_add_shows_risk_options_with_select_add(self):
        """Clicking Add button should show risk % selection pointing to select_add."""
        bot = self._make_bot()
        bot._paper_engine.get_position_with_pnl.return_value = {
            "position": MagicMock(
                symbol="BTC/USDT:USDT", side=Side.LONG, leverage=10,
                entry_price=68000.0, quantity=0.1, margin=680.0,
                liquidation_price=61540.0, stop_loss=67000.0,
                take_profit=[TakeProfit(price=70000.0, close_pct=100)], opened_at=datetime.now(UTC),
                trade_id="t1",
            ),
            "unrealized_pnl": 50.0, "pnl_pct": 0.74, "roe_pct": 7.35,
        }
        bot._data_fetcher.fetch_current_price = AsyncMock(return_value=68500.0)

        update, query = self._make_update("add:t1")
        await bot._callback_router(update, MagicMock())

        markup = query.edit_message_text.call_args.kwargs.get("reply_markup")
        all_data = [
            btn.callback_data for row in markup.inline_keyboard for btn in row
            if btn.callback_data
        ]
        assert any("select_add:" in d for d in all_data)

    @pytest.mark.asyncio
    async def test_select_add_shows_confirmation(self):
        """After selecting risk %, show confirmation card with details."""
        bot = self._make_bot()
        pos = MagicMock(
            symbol="BTC/USDT:USDT", side=Side.LONG, leverage=10,
            entry_price=68000.0, quantity=0.1, margin=680.0,
            stop_loss=67000.0, trade_id="t1",
        )
        bot._paper_engine._find_position.return_value = pos
        bot._paper_engine._position_sizer.calculate.return_value = 0.05
        bot._paper_engine.calculate_margin.return_value = 345.0
        bot._paper_engine.equity = 10000.0
        bot._data_fetcher.fetch_current_price = AsyncMock(return_value=69000.0)

        update, query = self._make_update("select_add:t1:1.0")
        await bot._callback_router(update, MagicMock())

        call_kwargs = query.edit_message_text.call_args.kwargs
        text = call_kwargs.get("text", "")
        assert "CONFIRM ADD" in text
        markup = query.edit_message_text.call_args.kwargs.get("reply_markup")
        all_data = [
            btn.callback_data for row in markup.inline_keyboard for btn in row
            if btn.callback_data
        ]
        assert any("confirm_add:" in d for d in all_data)

    @pytest.mark.asyncio
    async def test_confirm_add_executes(self):
        """Clicking Confirm should execute the add operation."""
        bot = self._make_bot()
        updated_pos = MagicMock(
            symbol="BTC/USDT:USDT", side=MagicMock(value="long"),
            leverage=10, entry_price=68500.0, quantity=0.15, margin=1025.0,
        )
        bot._paper_engine.add_to_position.return_value = updated_pos
        bot._data_fetcher.fetch_current_price = AsyncMock(return_value=69000.0)

        update, query = self._make_update("confirm_add:t1:1.0")
        await bot._callback_router(update, MagicMock())

        bot._paper_engine.add_to_position.assert_called_once()


class TestHistoryFilter:
    @pytest.mark.asyncio
    async def test_history_shows_filter_buttons(self):
        """When multiple symbols have closed trades, /history should show filter buttons."""
        bot = SentinelBot(
            token="test-token",
            admin_chat_ids=[123],
            trade_repo=MagicMock(),
        )
        bot._trade_repo.get_closed_paginated.return_value = (
            [MagicMock(
                symbol="BTC/USDT:USDT", side="long", leverage=10,
                entry_price=68000.0, exit_price=69000.0,
                pnl=100.0, fees=3.4, close_reason="manual", margin=680.0,
            )],
            1,
        )
        bot._trade_repo.get_distinct_closed_symbols.return_value = [
            "BTC/USDT:USDT", "ETH/USDT:USDT",
        ]

        update = _make_update(123)
        await bot._history_handler(update, _make_context())

        call_kwargs = update.message.reply_text.call_args
        markup = call_kwargs.kwargs.get("reply_markup") or call_kwargs[1].get("reply_markup")
        all_data = [
            btn.callback_data for row in markup.inline_keyboard for btn in row
            if btn.callback_data
        ]
        assert any("history:filter:" in d for d in all_data)
        assert any("history:filter:all" == d for d in all_data)

    @pytest.mark.asyncio
    async def test_history_no_filter_when_single_symbol(self):
        """When only one symbol exists, no filter row should appear."""
        bot = SentinelBot(
            token="test-token",
            admin_chat_ids=[123],
            trade_repo=MagicMock(),
        )
        bot._trade_repo.get_closed_paginated.return_value = (
            [MagicMock(
                symbol="BTC/USDT:USDT", side="long", leverage=10,
                entry_price=68000.0, exit_price=69000.0,
                pnl=100.0, fees=3.4, close_reason="manual", margin=680.0,
            )],
            1,
        )
        bot._trade_repo.get_distinct_closed_symbols.return_value = ["BTC/USDT:USDT"]

        update = _make_update(123)
        await bot._history_handler(update, _make_context())

        call_kwargs = update.message.reply_text.call_args
        markup = call_kwargs.kwargs.get("reply_markup") or call_kwargs[1].get("reply_markup")
        all_data = [
            btn.callback_data for row in markup.inline_keyboard for btn in row
            if btn.callback_data
        ]
        assert not any("history:filter:" in d for d in all_data)

    @pytest.mark.asyncio
    async def test_history_filter_preserves_across_pagination(self):
        """Pagination buttons should preserve the active symbol filter."""
        bot = SentinelBot(
            token="test-token",
            admin_chat_ids=[123],
            trade_repo=MagicMock(),
        )
        bot._trade_repo.get_closed_paginated.return_value = (
            [MagicMock(
                symbol="BTC/USDT:USDT", side="long", leverage=10,
                entry_price=68000.0, exit_price=69000.0,
                pnl=100.0, fees=3.4, close_reason="manual", margin=680.0,
            )] * 5,
            12,  # total > page_size → next button should appear
        )
        bot._trade_repo.get_distinct_closed_symbols.return_value = [
            "BTC/USDT:USDT", "ETH/USDT:USDT",
        ]

        query = MagicMock()
        query.data = "history:filter:BTC/USDT:USDT"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        query.message = MagicMock()
        query.message.message_id = 1
        query.message.reply_markup = None

        update = MagicMock()
        update.callback_query = query
        update.effective_chat = MagicMock()
        update.effective_chat.id = 123

        await bot._callback_router(update, MagicMock())

        call_kwargs = query.edit_message_text.call_args.kwargs
        markup = call_kwargs.get("reply_markup")
        all_data = [
            btn.callback_data for row in markup.inline_keyboard for btn in row
            if btn.callback_data
        ]
        # Next button should include filter
        next_btns = [d for d in all_data if d.startswith("history:page:")]
        assert len(next_btns) > 0
        assert any("BTC" in d for d in next_btns)
