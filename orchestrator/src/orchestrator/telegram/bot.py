from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from telegram import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

from orchestrator.telegram.formatters import (
    format_eval_report,
    format_execution_result,
    format_help,
    format_history,
    format_pending_approval,
    format_perf_report,
    format_proposal,
    format_risk_rejection,
    format_status,
    format_status_from_records,
    format_trade_report,
    format_welcome,
)

if TYPE_CHECKING:
    from orchestrator.approval.manager import ApprovalManager, PendingApproval
    from orchestrator.eval.runner import EvalRunner
    from orchestrator.exchange.data_fetcher import DataFetcher
    from orchestrator.exchange.paper_engine import CloseResult, PaperEngine
    from orchestrator.execution.executor import OrderExecutor
    from orchestrator.pipeline.runner import PipelineResult
    from orchestrator.pipeline.scheduler import PipelineScheduler
    from orchestrator.risk.checker import RiskResult
    from orchestrator.storage.repository import (
        AccountSnapshotRepository,
        PaperTradeRepository,
        TradeProposalRepository,
    )

logger = structlog.get_logger(__name__)

MODEL_ALIASES: dict[str, str] = {
    "sonnet": "anthropic/claude-sonnet-4-6",
    "opus": "anthropic/claude-opus-4-6",
}


def is_admin(chat_id: int, *, admin_ids: list[int]) -> bool:
    return chat_id in admin_ids


class SentinelBot:
    def __init__(self, token: str, admin_chat_ids: list[int], *, premium_model: str = "") -> None:
        self.token = token
        self.admin_chat_ids = admin_chat_ids
        self._premium_model = premium_model
        self._app: Application | None = None
        self._scheduler: PipelineScheduler | None = None
        self._latest_results: dict[str, object] = {}  # symbol → PipelineResult
        self._paper_engine: PaperEngine | None = None
        self._trade_repo: PaperTradeRepository | None = None
        self._proposal_repo: TradeProposalRepository | None = None
        self._snapshot_repo: AccountSnapshotRepository | None = None
        self._eval_runner: EvalRunner | None = None
        self._approval_manager: ApprovalManager | None = None
        self._executor: OrderExecutor | None = None
        self._data_fetcher: DataFetcher | None = None

    def set_approval_manager(self, mgr: ApprovalManager) -> None:
        self._approval_manager = mgr

    def set_executor(self, executor: OrderExecutor) -> None:
        self._executor = executor

    def set_data_fetcher(self, fetcher: DataFetcher) -> None:
        self._data_fetcher = fetcher

    def set_scheduler(self, scheduler: PipelineScheduler) -> None:
        self._scheduler = scheduler

    def set_paper_engine(self, engine: PaperEngine) -> None:
        self._paper_engine = engine

    def set_trade_repo(self, repo: PaperTradeRepository) -> None:
        self._trade_repo = repo

    def set_proposal_repo(self, repo: TradeProposalRepository) -> None:
        self._proposal_repo = repo

    def set_snapshot_repo(self, repo: AccountSnapshotRepository) -> None:
        self._snapshot_repo = repo

    def set_eval_runner(self, runner: EvalRunner) -> None:
        self._eval_runner = runner

    def build(self) -> Application:
        self._app = Application.builder().token(self.token).build()
        self._app.add_handler(CommandHandler("start", self._start_handler))
        self._app.add_handler(CommandHandler("help", self._help_handler))
        self._app.add_handler(CommandHandler("status", self._status_handler))
        self._app.add_handler(CommandHandler("coin", self._coin_handler))
        self._app.add_handler(CommandHandler("run", self._run_handler))
        self._app.add_handler(CommandHandler("history", self._history_handler))
        self._app.add_handler(CommandHandler("resume", self._resume_handler))
        self._app.add_handler(CommandHandler("perf", self._perf_handler))
        self._app.add_handler(CommandHandler("eval", self._eval_handler))
        self._app.add_handler(CallbackQueryHandler(self._approval_callback))
        return self._app

    async def push_proposal(self, chat_id: int, result: PipelineResult) -> None:
        """Push a pipeline result to a specific chat."""
        if self._app is None:
            return
        msg = format_proposal(result)
        await self._app.bot.send_message(chat_id=chat_id, text=msg)

    async def push_to_admins(self, result: PipelineResult) -> None:
        """Push a pipeline result to all admin chats."""
        for chat_id in self.admin_chat_ids:
            await self.push_proposal(chat_id, result)

    async def push_close_report(self, result: CloseResult) -> None:
        """Push a trade close report to all admin chats."""
        if self._app is None:
            return
        msg = format_trade_report(result)
        for chat_id in self.admin_chat_ids:
            await self._app.bot.send_message(chat_id=chat_id, text=msg)

    async def push_risk_rejection(
        self, *, symbol: str, side: str, entry_price: float, risk_result: RiskResult
    ) -> None:
        """Push a risk rejection notification to all admin chats."""
        if self._app is None:
            return
        msg = format_risk_rejection(
            symbol=symbol, side=side, entry_price=entry_price, risk_result=risk_result
        )
        for chat_id in self.admin_chat_ids:
            await self._app.bot.send_message(chat_id=chat_id, text=msg)

    @staticmethod
    async def _reply(update: Update, text: str) -> None:
        """Send a reply, handling the case where update.message may be None."""
        if update.message is not None:
            await update.message.reply_text(text)

    async def _check_admin(self, update: Update) -> bool:
        chat_id = update.effective_chat.id if update.effective_chat else 0
        if not is_admin(chat_id, admin_ids=self.admin_chat_ids):
            logger.warning("unauthorized_access", chat_id=chat_id)
            return False
        return True

    async def _start_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        await self._reply(update,format_welcome())

    async def _help_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        await self._reply(update,format_help())

    async def _status_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        results = list(self._latest_results.values())
        if results:
            await self._reply(update,format_status(results))
        elif self._proposal_repo is not None:
            records = self._proposal_repo.get_recent(limit=10)
            await self._reply(update,format_status_from_records(records))
        else:
            await self._reply(update,format_status([]))

    async def _coin_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        args = context.args
        if not args:
            await self._reply(update,"Usage: /coin <symbol> (e.g. /coin BTC)")
            return

        query = args[0].upper()
        # Find matching symbol from in-memory cache
        matching = [
            r for sym, r in self._latest_results.items() if query in sym.upper()
        ]

        if matching:
            for result in matching:
                await self._reply(update,format_proposal(result))
            return

        # Fallback: search DB records
        if self._proposal_repo is not None:
            import json

            records = self._proposal_repo.get_recent(limit=20)
            matching_records = []
            for r in records:
                try:
                    proposal = json.loads(r.proposal_json)
                    if query in proposal.get("symbol", "").upper():
                        matching_records.append(r)
                except (json.JSONDecodeError, AttributeError):
                    pass
            if matching_records:
                await self._reply(update,format_status_from_records(matching_records))
                return

        await self._reply(update,
            f"No recent analysis for {query}. Use /run to trigger analysis."
        )

    async def _run_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        if self._scheduler is None:
            await self._reply(update,"Pipeline not configured.")
            return

        args = list(context.args or [])

        # Parse model from args: /run [symbol] [model]
        # model can be "sonnet", "opus", or a full model ID
        model_override: str | None = self._premium_model or None
        symbol_args: list[str] = []

        for arg in args:
            lower = arg.lower()
            if lower in MODEL_ALIASES:
                model_override = MODEL_ALIASES[lower]
            elif lower.startswith("anthropic/"):
                model_override = lower
            else:
                symbol_args.append(arg)

        model_label = (model_override or "default").split("/")[-1]
        await self._reply(update,f"Running pipeline (model: {model_label})...")

        if symbol_args:
            query = symbol_args[0].upper()
            symbols = [
                s for s in self._scheduler.symbols if query in s.upper()
            ]
            if not symbols:
                await self._reply(update,f"Unknown symbol: {query}")
                return
            results = await self._scheduler.run_once(symbols=symbols, model_override=model_override)
        else:
            results = await self._scheduler.run_once(model_override=model_override)

        for result in results:
            self._latest_results[result.symbol] = result
            await self._reply(update,format_proposal(result))

    async def _history_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        if self._trade_repo is None:
            await self._reply(update,"Paper trading not configured.")
            return
        trades = self._trade_repo.get_recent_closed(limit=10)
        await self._reply(update,format_history(trades))

    async def _resume_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        if self._paper_engine is None:
            await self._reply(update,"Paper trading not configured.")
            return
        self._paper_engine.set_paused(False)
        await self._reply(update,"Pipeline resumed. Trading un-paused.")

    async def _perf_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        if self._snapshot_repo is None:
            await self._reply(update,"Stats not configured.")
            return
        snapshot = self._snapshot_repo.get_latest()
        if snapshot is None or snapshot.total_trades == 0:
            await self._reply(update,
                "No performance data yet. Close some positions first."
            )
            return
        from orchestrator.stats.calculator import PerformanceStats

        stats = PerformanceStats(
            total_pnl=snapshot.total_pnl,
            total_pnl_pct=(
                (snapshot.total_pnl / snapshot.equity * 100) if snapshot.equity > 0 else 0.0
            ),
            win_rate=snapshot.win_rate,
            total_trades=snapshot.total_trades,
            winning_trades=int(snapshot.win_rate * snapshot.total_trades),
            losing_trades=snapshot.total_trades - int(snapshot.win_rate * snapshot.total_trades),
            profit_factor=snapshot.profit_factor,
            max_drawdown_pct=snapshot.max_drawdown_pct,
            sharpe_ratio=snapshot.sharpe_ratio,
        )
        await self._reply(update,format_perf_report(stats))

    async def push_pending_approval(self, chat_id: int, approval: PendingApproval) -> int | None:
        """Push proposal with Approve/Reject keyboard. Returns message_id."""
        if self._app is None:
            return None
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton(
                    "Approve", callback_data=f"approve:{approval.approval_id}"
                ),
                InlineKeyboardButton(
                    "Reject", callback_data=f"reject:{approval.approval_id}"
                ),
            ]
        ])
        msg = await self._app.bot.send_message(
            chat_id=chat_id,
            text=format_pending_approval(approval),
            reply_markup=keyboard,
        )
        return msg.message_id

    async def _approval_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        if query is None:
            return
        chat_id = update.effective_chat.id if update.effective_chat else 0
        if not is_admin(chat_id, admin_ids=self.admin_chat_ids):
            await query.answer("Unauthorized")
            return

        data = query.data or ""
        parts = data.split(":", 1)
        if len(parts) != 2:
            await query.answer("Invalid action")
            return

        action, approval_id = parts

        if action == "approve":
            await self._handle_approve(query, approval_id)
        elif action == "reject":
            await self._handle_reject(query, approval_id)
        else:
            await query.answer("Unknown action")

    async def _handle_approve(self, query: CallbackQuery, approval_id: str) -> None:
        if self._approval_manager is None or self._executor is None:
            await query.answer("Not configured")
            return

        approval = self._approval_manager.approve(approval_id)
        if approval is None:
            await query.answer("Expired or not found")
            await query.edit_message_text("Approval expired or already handled.")
            return

        try:
            # Re-fetch price for deviation check
            current_price = approval.snapshot_price
            if self._data_fetcher is not None:
                current_price = await self._data_fetcher.fetch_current_price(
                    approval.proposal.symbol
                )

            result = await self._executor.execute_entry(
                approval.proposal, current_price=current_price
            )
            sl_tp_ids = await self._executor.place_sl_tp(
                symbol=approval.proposal.symbol,
                side=approval.proposal.side.value,
                quantity=result.quantity,
                stop_loss=approval.proposal.stop_loss or 0.0,
                take_profit=approval.proposal.take_profit,
            )

            await query.answer("Approved!")
            await query.edit_message_text(format_execution_result(result))
            logger.info(
                "approval_executed",
                approval_id=approval_id,
                trade_id=result.trade_id,
                sl_tp_ids=sl_tp_ids,
            )
        except Exception as e:
            logger.error("approval_execution_failed", error=str(e))
            await query.answer("Execution failed")
            await query.edit_message_text(f"Execution failed: {e}")

    async def _handle_reject(self, query: CallbackQuery, approval_id: str) -> None:
        if self._approval_manager is None:
            await query.answer("Not configured")
            return

        approval = self._approval_manager.reject(approval_id)
        if approval is None:
            await query.answer("Not found or already handled")
            await query.edit_message_text("Approval not found or already handled.")
            return

        await query.answer("Rejected")
        await query.edit_message_text(
            f"[REJECTED] {approval.proposal.symbol} {approval.proposal.side.value.upper()}\n"
            "Trade proposal rejected by admin."
        )

    async def _eval_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        if self._eval_runner is None:
            await self._reply(update,"Eval not configured.")
            return
        await self._reply(update,"Running evaluation...")
        report = await self._eval_runner.run_default()
        report_dict = report.model_dump()
        # Add failure details for the formatter
        report_dict["failures"] = [
            {"case_id": cr["case_id"], "reason": "; ".join(
                s["reason"] for s in cr["scores"] if not s["passed"]
            )}
            for cr in report_dict["case_results"]
            if not cr["passed"]
        ]
        await self._reply(update,format_eval_report(report_dict))
