from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import structlog
from telegram import BotCommand, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest
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
from orchestrator.telegram.translations import to_chinese

if TYPE_CHECKING:
    from orchestrator.approval.manager import ApprovalManager, PendingApproval
    from orchestrator.eval.runner import EvalRunner
    from orchestrator.exchange.data_fetcher import DataFetcher
    from orchestrator.exchange.paper_engine import CloseResult, PaperEngine
    from orchestrator.execution.executor import OrderExecutor
    from orchestrator.llm.client import LLMClient
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

_TRANSLATE_CACHE_MAX = 200


def _translate_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Translate to zh-TW", callback_data="translate:zh")]
    ])


def _english_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Translate to en", callback_data="translate:en")]
    ])


def _check_stale_sl_tp(
    *,
    side: str,
    current_price: float,
    stop_loss: float | None,
    take_profit: list[float],
) -> str | None:
    """Return a human-readable reason if SL/TP would immediately trigger, else None."""
    if side == "long":
        if stop_loss is not None and current_price <= stop_loss:
            return f"Price ${current_price:,.1f} already at/below SL ${stop_loss:,.1f}"
        if take_profit and current_price >= take_profit[0]:
            return f"Price ${current_price:,.1f} already at/above TP ${take_profit[0]:,.1f}"
    elif side == "short":
        if stop_loss is not None and current_price >= stop_loss:
            return f"Price ${current_price:,.1f} already at/above SL ${stop_loss:,.1f}"
        if take_profit and current_price <= take_profit[0]:
            return f"Price ${current_price:,.1f} already at/below TP ${take_profit[0]:,.1f}"
    return None


async def _safe_callback_reply(
    query: CallbackQuery,
    *,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> None:
    """Answer callback + edit message, tolerating expired/duplicate queries."""
    try:
        await query.answer()
    except BadRequest:
        pass  # callback query expired (e.g. duplicate click)
    try:
        await query.edit_message_text(text=text, reply_markup=reply_markup)
    except BadRequest as e:
        if "Message is not modified" not in str(e):
            raise


def is_admin(chat_id: int, *, admin_ids: list[int]) -> bool:
    return chat_id in admin_ids


class _MessageCache:
    """Bounded LRU cache mapping message_id → original English text."""

    def __init__(self, maxsize: int = _TRANSLATE_CACHE_MAX) -> None:
        self._data: OrderedDict[int, str] = OrderedDict()
        self._maxsize = maxsize

    def store(self, message_id: int, text: str) -> None:
        self._data[message_id] = text
        self._data.move_to_end(message_id)
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def get(self, message_id: int) -> str | None:
        return self._data.get(message_id)


class SentinelBot:
    def __init__(
        self,
        token: str,
        admin_chat_ids: list[int],
        *,
        premium_model: str = "",
        scheduler: PipelineScheduler | None = None,
        paper_engine: PaperEngine | None = None,
        trade_repo: PaperTradeRepository | None = None,
        proposal_repo: TradeProposalRepository | None = None,
        snapshot_repo: AccountSnapshotRepository | None = None,
        eval_runner: EvalRunner | None = None,
        approval_manager: ApprovalManager | None = None,
        executor: OrderExecutor | None = None,
        data_fetcher: DataFetcher | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.token = token
        self.admin_chat_ids = admin_chat_ids
        self._premium_model = premium_model
        self._llm_client = llm_client
        self._app: Application | None = None
        self._scheduler = scheduler
        self._latest_results: dict[str, object] = {}  # symbol → PipelineResult
        self._running_symbols: set[str] = set()
        self._msg_cache = _MessageCache()
        self._paper_engine = paper_engine
        self._trade_repo = trade_repo
        self._proposal_repo = proposal_repo
        self._snapshot_repo = snapshot_repo
        self._eval_runner = eval_runner
        self._approval_manager = approval_manager
        self._executor = executor
        self._data_fetcher = data_fetcher
        self._leverage_options: list[int] = [5, 10, 20, 50]

    _BOT_COMMANDS = [
        BotCommand("status", "Account overview & latest proposals"),
        BotCommand("run", "Trigger pipeline for all symbols"),
        BotCommand("coin", "Detailed analysis for a symbol"),
        BotCommand("history", "Recent trade records"),
        BotCommand("perf", "Performance report"),
        BotCommand("eval", "Run LLM evaluation"),
        BotCommand("resume", "Un-pause after risk pause"),
        BotCommand("help", "Show available commands"),
    ]

    async def register_commands(self) -> None:
        """Register bot commands with Telegram (call after app.initialize)."""
        if self._app is not None:
            await self._app.bot.set_my_commands(self._BOT_COMMANDS)

    def build(self, *, post_init: Any = None) -> Application:
        builder = Application.builder().token(self.token)
        if post_init is not None:
            builder = builder.post_init(post_init)
        self._app = builder.build()
        self._app.add_handler(CommandHandler("start", self._start_handler))
        self._app.add_handler(CommandHandler("help", self._help_handler))
        self._app.add_handler(CommandHandler("status", self._status_handler))
        self._app.add_handler(CommandHandler("coin", self._coin_handler))
        self._app.add_handler(CommandHandler("run", self._run_handler))
        self._app.add_handler(CommandHandler("history", self._history_handler))
        self._app.add_handler(CommandHandler("resume", self._resume_handler))
        self._app.add_handler(CommandHandler("perf", self._perf_handler))
        self._app.add_handler(CommandHandler("eval", self._eval_handler))
        self._app.add_handler(CallbackQueryHandler(self._callback_router))
        return self._app

    # --- Push methods (for scheduled / background notifications) ---

    async def push_proposal(self, chat_id: int, result: PipelineResult) -> None:
        """Push a pipeline result to a specific chat."""
        if self._app is None:
            return
        msg = format_proposal(result)
        sent = await self._app.bot.send_message(
            chat_id=chat_id, text=msg, reply_markup=_translate_keyboard(),
        )
        self._msg_cache.store(sent.message_id, msg)

    async def push_to_admins(self, result: PipelineResult) -> None:
        """Push a pipeline result to all admin chats."""
        for chat_id in self.admin_chat_ids:
            await self.push_proposal(chat_id, result)

    async def push_to_admins_with_approval(self, result: PipelineResult) -> None:
        """Push a pipeline result to all admins, with approval buttons if applicable."""
        if self._app is None:
            return
        for chat_id in self.admin_chat_ids:
            if result.status == "pending_approval" and result.approval_id and self._approval_manager:
                approval = self._approval_manager.get(result.approval_id)
                if approval:
                    await self.push_pending_approval(chat_id, approval)
                    continue
            await self.push_proposal(chat_id, result)

    async def push_close_report(self, result: CloseResult) -> None:
        """Push a trade close report to all admin chats."""
        if self._app is None:
            return
        msg = format_trade_report(result)
        for chat_id in self.admin_chat_ids:
            sent = await self._app.bot.send_message(
                chat_id=chat_id, text=msg, reply_markup=_translate_keyboard(),
            )
            self._msg_cache.store(sent.message_id, msg)

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
            sent = await self._app.bot.send_message(
                chat_id=chat_id, text=msg, reply_markup=_translate_keyboard(),
            )
            self._msg_cache.store(sent.message_id, msg)

    # --- Reply helper with translate button ---

    async def _reply(self, update: Update, text: str) -> None:
        """Send a reply with a translate button, caching the English text."""
        if update.message is None:
            return
        sent = await update.message.reply_text(
            text, reply_markup=_translate_keyboard(),
        )
        self._msg_cache.store(sent.message_id, text)

    async def _check_admin(self, update: Update) -> bool:
        chat_id = update.effective_chat.id if update.effective_chat else 0
        if not is_admin(chat_id, admin_ids=self.admin_chat_ids):
            logger.warning("unauthorized_access", chat_id=chat_id)
            return False
        return True

    # --- Command handlers ---

    async def _start_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        await self._reply(update, format_welcome())

    async def _help_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        await self._reply(update, format_help())

    async def _status_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return

        # Show position cards if paper engine is available
        if self._paper_engine is not None and self._data_fetcher is not None:
            positions = self._paper_engine.get_open_positions()
            if positions:
                from orchestrator.telegram.formatters import (
                    format_account_overview,
                    format_position_card,
                )

                position_cards: list[str] = []
                for pos in positions:
                    try:
                        current_price = await self._data_fetcher.fetch_current_price(
                            pos.symbol
                        )
                        info = self._paper_engine.get_position_with_pnl(
                            trade_id=pos.trade_id, current_price=current_price,
                        )
                        position_cards.append(format_position_card(info))
                    except Exception as e:
                        logger.warning("position_card_error", trade_id=pos.trade_id, error=str(e))

                overview = format_account_overview(
                    equity=self._paper_engine.equity,
                    available=self._paper_engine.available_balance,
                    used_margin=self._paper_engine.used_margin,
                    initial_equity=self._paper_engine._initial_equity,
                    position_cards=position_cards,
                )
                # Send overview, then per-position messages with action buttons
                await self._reply(update, overview)
                for pos in positions:
                    keyboard = InlineKeyboardMarkup([
                        [
                            InlineKeyboardButton(
                                "Close", callback_data=f"close:{pos.trade_id}",
                            ),
                            InlineKeyboardButton(
                                "Reduce", callback_data=f"reduce:{pos.trade_id}",
                            ),
                            InlineKeyboardButton(
                                "Add", callback_data=f"add:{pos.trade_id}",
                            ),
                        ],
                    ])
                    if update.message:
                        sent = await update.message.reply_text(
                            f"{pos.symbol} actions:", reply_markup=keyboard,
                        )
                return

        parts: list[str] = []
        if self._running_symbols:
            syms = ", ".join(sorted(self._running_symbols))
            parts.append(f"Running pipeline: {syms}\n")

        results = list(self._latest_results.values())
        if results:
            parts.append(format_status(results))
        elif self._proposal_repo is not None:
            records = self._proposal_repo.get_recent(limit=10)
            parts.append(format_status_from_records(records))
        else:
            parts.append(format_status([]))

        await self._reply(update, "\n".join(parts))

    async def _coin_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        args = context.args
        if not args:
            await self._reply(update, "Usage: /coin <symbol> (e.g. /coin BTC)")
            return

        query = args[0].upper()
        matching = [
            r for sym, r in self._latest_results.items() if query in sym.upper()
        ]

        if matching:
            for result in matching:
                await self._reply(update, format_proposal(result))
            return

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
                await self._reply(update, format_status_from_records(matching_records))
                return

        await self._reply(update, f"No recent analysis for {query}. Use /run to trigger analysis.")

    async def _run_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        if self._scheduler is None:
            await self._reply(update, "Pipeline not configured.")
            return

        args = list(context.args or [])

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
        status_msg = await update.message.reply_text(
            f"Running pipeline (model: {model_label})..."
        ) if update.message else None

        if symbol_args:
            query = symbol_args[0].upper()
            symbols = [
                s for s in self._scheduler.symbols if query in s.upper()
            ]
            if not symbols:
                await self._reply(update, f"Unknown symbol: {query}")
                return
        else:
            symbols = list(self._scheduler.symbols)

        self._running_symbols.update(symbols)
        try:
            results = await self._scheduler.run_once(
                symbols=symbols, model_override=model_override,
                source="command", notify=False,
            )
        finally:
            self._running_symbols.difference_update(symbols)

        if status_msg:
            try:
                await status_msg.delete()
            except Exception:
                pass  # message may already be gone

        for result in results:
            self._latest_results[result.symbol] = result
            if result.status == "pending_approval" and result.approval_id:
                approval = (
                    self._approval_manager.get(result.approval_id)
                    if self._approval_manager
                    else None
                )
                if approval and update.effective_chat:
                    await self.push_pending_approval(
                        update.effective_chat.id, approval
                    )
                else:
                    await self._reply(update, format_proposal(result))
            else:
                await self._reply(update, format_proposal(result))

    async def _history_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        if self._trade_repo is None:
            await self._reply(update, "Paper trading not configured.")
            return

        from orchestrator.telegram.formatters import format_history_paginated

        page_size = 5
        trades, total = self._trade_repo.get_closed_paginated(offset=0, limit=page_size)
        total_pages = max(1, (total + page_size - 1) // page_size)
        text = format_history_paginated(trades, page=1, total_pages=total_pages)

        nav_buttons: list[InlineKeyboardButton] = []
        nav_buttons.append(
            InlineKeyboardButton(f"Page 1/{total_pages}", callback_data="cancel:0")
        )
        if total_pages > 1:
            nav_buttons.append(
                InlineKeyboardButton("Next", callback_data="history:page:2")
            )

        if update.message:
            await update.message.reply_text(
                text, reply_markup=InlineKeyboardMarkup([nav_buttons]),
            )

    async def _resume_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        if self._paper_engine is None:
            await self._reply(update, "Paper trading not configured.")
            return
        self._paper_engine.set_paused(False)
        await self._reply(update, "Pipeline resumed. Trading un-paused.")

    async def _perf_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        if self._snapshot_repo is None:
            await self._reply(update, "Stats not configured.")
            return
        snapshot = self._snapshot_repo.get_latest()
        if snapshot is None or snapshot.total_trades == 0:
            await self._reply(update, "No performance data yet. Close some positions first.")
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
        await self._reply(update, format_perf_report(stats))

    # --- Approval flow ---

    async def push_pending_approval(self, chat_id: int, approval: PendingApproval) -> int | None:
        """Push proposal with Approve/Reject + Translate keyboard. Returns message_id."""
        if self._app is None:
            return None
        text = format_pending_approval(approval)
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton(
                    "Approve", callback_data=f"approve:{approval.approval_id}"
                ),
                InlineKeyboardButton(
                    "Reject", callback_data=f"reject:{approval.approval_id}"
                ),
            ],
            [InlineKeyboardButton("Translate to zh-TW", callback_data="translate:zh")],
        ])
        msg = await self._app.bot.send_message(
            chat_id=chat_id, text=text, reply_markup=keyboard,
        )
        self._msg_cache.store(msg.message_id, text)
        return msg.message_id

    # --- Callback router ---

    # Dispatch table: action → (min_parts, handler)
    # Each handler receives (query, parts[1:]) for uniform argument passing.
    _CALLBACK_DISPATCH: dict[str, tuple[int, str]] = {
        "approve":          (2, "_handle_approve"),
        "reject":           (2, "_handle_reject"),
        "translate":        (2, "_handle_translate"),
        "confirm_leverage": (3, "_handle_confirm_leverage"),
        "close":            (2, "_handle_close"),
        "confirm_close":    (2, "_handle_confirm_close"),
        "reduce":           (2, "_handle_reduce"),
        "confirm_reduce":   (3, "_handle_confirm_reduce"),
        "add":              (2, "_handle_add"),
        "confirm_add":      (3, "_handle_confirm_add"),
        "cancel":           (1, "_handle_cancel"),
        "history":          (3, "_handle_history_callback"),
    }

    async def _callback_router(
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
        parts = data.split(":")
        action = parts[0] if parts else ""

        entry = self._CALLBACK_DISPATCH.get(action)
        if entry is None or len(parts) < entry[0]:
            await query.answer("Unknown action")
            return

        handler = getattr(self, entry[1])
        await handler(query, *parts[1:])

    async def _handle_cancel(self, query: CallbackQuery, *_args: str) -> None:
        await _safe_callback_reply(query, text="Cancelled.")

    async def _handle_translate(self, query: CallbackQuery, lang: str) -> None:
        """Toggle message between English and Chinese."""
        msg_id = query.message.message_id if query.message else None
        if msg_id is None:
            await query.answer()
            return

        original = self._msg_cache.get(msg_id)
        if original is None:
            await query.answer("Message too old to translate")
            return

        # Preserve approval buttons if present
        existing_markup = query.message.reply_markup if query.message else None
        has_approval_buttons = False
        approval_row: list[InlineKeyboardButton] = []
        if existing_markup:
            for row in existing_markup.inline_keyboard:
                for btn in row:
                    if btn.callback_data and (
                        btn.callback_data.startswith("approve:")
                        or btn.callback_data.startswith("reject:")
                    ):
                        has_approval_buttons = True
                        approval_row = list(row)
                        break

        if lang == "zh":
            if self._llm_client is None:
                await query.answer("Translation not available")
                return
            structlog.contextvars.bind_contextvars(source="translate")
            try:
                translated = await to_chinese(original, self._llm_client)
            finally:
                structlog.contextvars.unbind_contextvars("source")
            rows = []
            if has_approval_buttons:
                rows.append(approval_row)
            rows.append([InlineKeyboardButton("Translate to en", callback_data="translate:en")])
            await _safe_callback_reply(
                query,
                text=translated,
                reply_markup=InlineKeyboardMarkup(rows),
            )
        else:
            rows = []
            if has_approval_buttons:
                rows.append(approval_row)
            rows.append([InlineKeyboardButton("Translate to zh-TW", callback_data="translate:zh")])
            await _safe_callback_reply(
                query,
                text=original,
                reply_markup=InlineKeyboardMarkup(rows),
            )

    async def _handle_approve(self, query: CallbackQuery, approval_id: str) -> None:
        """Show leverage selection when user clicks Approve."""
        if self._approval_manager is None or self._executor is None:
            await query.answer("Not configured")
            return

        approval = self._approval_manager.get(approval_id)
        if approval is None:
            await query.answer("Expired or not found")
            await query.edit_message_text("Approval expired or already handled.")
            return

        # Show leverage selection buttons
        leverage_buttons = [
            InlineKeyboardButton(
                f"{lev}x", callback_data=f"confirm_leverage:{approval_id}:{lev}"
            )
            for lev in self._leverage_options
        ]
        keyboard = InlineKeyboardMarkup([
            leverage_buttons,
            [InlineKeyboardButton("Cancel", callback_data="cancel:0")],
        ])

        p = approval.proposal
        text = (
            f"━━ SELECT LEVERAGE ━━\n"
            f"{p.symbol}  {p.side.value.upper()}\n\n"
            f"Risk: {p.position_size_risk_pct}%\n"
            f"SL: ${p.stop_loss:,.1f}\n"
            f"Choose leverage:"
        )
        await _safe_callback_reply(query, text=text, reply_markup=keyboard)

    async def _handle_confirm_leverage(
        self, query: CallbackQuery, approval_id: str, leverage_str: str, *_args: str,
    ) -> None:
        leverage = int(leverage_str)
        """Execute trade with selected leverage."""
        if self._approval_manager is None or self._executor is None:
            await query.answer("Not configured")
            return

        approval = self._approval_manager.get(approval_id)
        if approval is None:
            await query.answer("Expired or not found")
            await query.edit_message_text("Approval expired or already handled.")
            return

        try:
            current_price = approval.snapshot_price
            if self._data_fetcher is not None:
                current_price = await self._data_fetcher.fetch_current_price(
                    approval.proposal.symbol
                )

            stale_reason = _check_stale_sl_tp(
                side=approval.proposal.side.value,
                current_price=current_price,
                stop_loss=approval.proposal.stop_loss,
                take_profit=approval.proposal.take_profit,
            )
            if stale_reason:
                self._approval_manager.reject(approval_id)
                text = (
                    f"━━ STALE PROPOSAL ━━\n"
                    f"{approval.proposal.symbol}  {approval.proposal.side.value.upper()}\n\n"
                    f"{stale_reason}\n"
                    f"Use /run to get a fresh analysis."
                )
                await _safe_callback_reply(query, text=text, reply_markup=_translate_keyboard())
                if query.message:
                    self._msg_cache.store(query.message.message_id, text)
                return

            self._approval_manager.approve(approval_id)

            result = await self._executor.execute_entry(
                approval.proposal, current_price=current_price, leverage=leverage,
            )
            sl_tp_ids = await self._executor.place_sl_tp(
                symbol=approval.proposal.symbol,
                side=approval.proposal.side.value,
                quantity=result.quantity,
                stop_loss=approval.proposal.stop_loss or 0.0,
                take_profit=approval.proposal.take_profit,
            )

            text = format_execution_result(result)
            await _safe_callback_reply(query, text=text, reply_markup=_translate_keyboard())
            if query.message:
                self._msg_cache.store(query.message.message_id, text)

            symbol = approval.proposal.symbol
            if symbol in self._latest_results:
                self._latest_results[symbol] = self._latest_results[symbol].model_copy(
                    update={"status": "executed"}
                )
            logger.info(
                "approval_executed",
                approval_id=approval_id,
                symbol=approval.proposal.symbol,
                trade_id=result.trade_id,
                leverage=leverage,
                sl_tp_ids=sl_tp_ids,
            )
        except Exception as e:
            logger.error("approval_execution_failed", symbol=approval.proposal.symbol, error=str(e))
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
        text = (
            f"━━ REJECTED ━━\n"
            f"{approval.proposal.symbol}  {approval.proposal.side.value.upper()}\n\n"
            "Trade proposal rejected by admin."
        )
        await query.edit_message_text(text=text, reply_markup=_translate_keyboard())
        if query.message:
            self._msg_cache.store(query.message.message_id, text)
        symbol = approval.proposal.symbol
        if symbol in self._latest_results:
            self._latest_results[symbol] = self._latest_results[symbol].model_copy(
                update={"status": "rejected"}
            )

    async def _eval_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        if self._eval_runner is None:
            await self._reply(update, "Eval not configured.")
            return
        await self._reply(update, "Running evaluation...")
        report = await self._eval_runner.run_default()
        report_dict = report.model_dump()
        report_dict["failures"] = [
            {"case_id": cr["case_id"], "reason": "; ".join(
                s["reason"] for s in cr["scores"] if not s["passed"]
            )}
            for cr in report_dict["case_results"]
            if not cr["passed"]
        ]
        await self._reply(update, format_eval_report(report_dict))

    # --- Position operation handlers ---

    async def _handle_close(self, query: CallbackQuery, trade_id: str) -> None:
        """Show close confirmation with current PnL."""
        if self._paper_engine is None or self._data_fetcher is None:
            await query.answer("Not configured")
            return

        try:
            # Need to find position first to get symbol for price fetch
            pos = self._paper_engine._find_position(trade_id)
            current_price = await self._data_fetcher.fetch_current_price(pos.symbol)
            info = self._paper_engine.get_position_with_pnl(
                trade_id=trade_id, current_price=current_price,
            )
        except ValueError:
            await query.answer("Position not found")
            return

        from orchestrator.telegram.formatters import format_position_card

        text = f"━━ CLOSE POSITION? ━━\n\n{format_position_card(info)}"
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("Confirm Close", callback_data=f"confirm_close:{trade_id}"),
                InlineKeyboardButton("Cancel", callback_data="cancel:0"),
            ],
        ])
        await _safe_callback_reply(query, text=text, reply_markup=keyboard)

    async def _handle_confirm_close(self, query: CallbackQuery, trade_id: str) -> None:
        if self._paper_engine is None or self._data_fetcher is None:
            await query.answer("Not configured")
            return

        try:
            pos = self._paper_engine._find_position(trade_id)
            current_price = await self._data_fetcher.fetch_current_price(pos.symbol)
            result = self._paper_engine.close_position(
                trade_id=trade_id, current_price=current_price,
            )
            text = format_trade_report(result)
            await _safe_callback_reply(query, text=text, reply_markup=_translate_keyboard())
            if query.message:
                self._msg_cache.store(query.message.message_id, text)
        except Exception as e:
            await query.answer(f"Error: {e}")

    async def _handle_reduce(self, query: CallbackQuery, trade_id: str) -> None:
        """Show reduce percentage options."""
        if self._paper_engine is None or self._data_fetcher is None:
            await query.answer("Not configured")
            return

        try:
            pos = self._paper_engine._find_position(trade_id)
            current_price = await self._data_fetcher.fetch_current_price(pos.symbol)
            info = self._paper_engine.get_position_with_pnl(
                trade_id=trade_id, current_price=current_price,
            )
        except ValueError:
            await query.answer("Position not found")
            return

        from orchestrator.telegram.formatters import format_position_card

        text = f"━━ REDUCE POSITION ━━\n\n{format_position_card(info)}\n\nSelect percentage to close:"
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("25%", callback_data=f"confirm_reduce:{trade_id}:25"),
                InlineKeyboardButton("50%", callback_data=f"confirm_reduce:{trade_id}:50"),
                InlineKeyboardButton("75%", callback_data=f"confirm_reduce:{trade_id}:75"),
            ],
            [InlineKeyboardButton("Cancel", callback_data="cancel:0")],
        ])
        await _safe_callback_reply(query, text=text, reply_markup=keyboard)

    async def _handle_confirm_reduce(
        self, query: CallbackQuery, trade_id: str, pct_str: str, *_args: str,
    ) -> None:
        pct = float(pct_str)
        if self._paper_engine is None or self._data_fetcher is None:
            await query.answer("Not configured")
            return

        try:
            pos = self._paper_engine._find_position(trade_id)
            current_price = await self._data_fetcher.fetch_current_price(pos.symbol)
            result = self._paper_engine.reduce_position(
                trade_id=trade_id, pct=pct, current_price=current_price,
            )
            text = format_trade_report(result)
            await _safe_callback_reply(query, text=text, reply_markup=_translate_keyboard())
            if query.message:
                self._msg_cache.store(query.message.message_id, text)
        except Exception as e:
            await query.answer(f"Error: {e}")

    async def _handle_add(self, query: CallbackQuery, trade_id: str) -> None:
        """Show risk % options for adding to position."""
        if self._paper_engine is None or self._data_fetcher is None:
            await query.answer("Not configured")
            return

        try:
            pos = self._paper_engine._find_position(trade_id)
            current_price = await self._data_fetcher.fetch_current_price(pos.symbol)
            info = self._paper_engine.get_position_with_pnl(
                trade_id=trade_id, current_price=current_price,
            )
        except ValueError:
            await query.answer("Position not found")
            return

        from orchestrator.telegram.formatters import format_position_card

        text = f"━━ ADD TO POSITION ━━\n\n{format_position_card(info)}\n\nSelect risk % to add:"
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("0.5%", callback_data=f"confirm_add:{trade_id}:0.5"),
                InlineKeyboardButton("1%", callback_data=f"confirm_add:{trade_id}:1.0"),
                InlineKeyboardButton("2%", callback_data=f"confirm_add:{trade_id}:2.0"),
            ],
            [InlineKeyboardButton("Cancel", callback_data="cancel:0")],
        ])
        await _safe_callback_reply(query, text=text, reply_markup=keyboard)

    async def _handle_confirm_add(
        self, query: CallbackQuery, trade_id: str, risk_pct_str: str, *_args: str,
    ) -> None:
        risk_pct = float(risk_pct_str)
        if self._paper_engine is None or self._data_fetcher is None:
            await query.answer("Not configured")
            return

        try:
            pos = self._paper_engine._find_position(trade_id)
            current_price = await self._data_fetcher.fetch_current_price(pos.symbol)
            updated = self._paper_engine.add_to_position(
                trade_id=trade_id, risk_pct=risk_pct, current_price=current_price,
            )
            text = (
                f"━━ POSITION UPDATED ━━\n"
                f"{updated.symbol}  {updated.side.value.upper()} {updated.leverage}x\n\n"
                f"New Avg Entry: ${updated.entry_price:,.1f}\n"
                f"New Qty: {updated.quantity:.4f}\n"
                f"Margin: ${updated.margin:,.2f}"
            )
            await _safe_callback_reply(query, text=text, reply_markup=_translate_keyboard())
            if query.message:
                self._msg_cache.store(query.message.message_id, text)
        except Exception as e:
            await query.answer(f"Error: {e}")

    async def _handle_history_callback(
        self, query: CallbackQuery, action: str, value: str, *extra: str,
    ) -> None:
        # Rejoin extra parts for symbols with colons (e.g. "BTC/USDT:USDT")
        if extra:
            value = ":".join([value, *extra])
        """Handle history pagination callbacks (history:page:N or history:filter:SYMBOL)."""
        if self._trade_repo is None:
            await query.answer("Not configured")
            return

        from orchestrator.telegram.formatters import format_history_paginated

        page_size = 5
        page = 1
        symbol_filter: str | None = None

        if action == "page":
            page = int(value)
        elif action == "filter":
            symbol_filter = value

        offset = (page - 1) * page_size
        trades, total = self._trade_repo.get_closed_paginated(
            offset=offset, limit=page_size, symbol=symbol_filter,
        )
        total_pages = max(1, (total + page_size - 1) // page_size)

        text = format_history_paginated(trades, page=page, total_pages=total_pages)

        nav_buttons: list[InlineKeyboardButton] = []
        filter_prefix = f"history:filter:{symbol_filter}" if symbol_filter else ""
        if page > 1:
            prev_data = f"history:page:{page - 1}"
            nav_buttons.append(InlineKeyboardButton("Prev", callback_data=prev_data))
        nav_buttons.append(
            InlineKeyboardButton(f"Page {page}/{total_pages}", callback_data="cancel:0")
        )
        if page < total_pages:
            next_data = f"history:page:{page + 1}"
            nav_buttons.append(InlineKeyboardButton("Next", callback_data=next_data))

        keyboard = InlineKeyboardMarkup([nav_buttons])
        await _safe_callback_reply(query, text=text, reply_markup=keyboard)
