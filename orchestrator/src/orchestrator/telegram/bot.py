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
    format_execution_plan,
    format_execution_result,
    format_help,
    format_pending_approval,
    format_perf_report,
    format_proposal,
    format_risk_pause,
    format_risk_rejection,
    format_status,
    format_status_from_records,
    format_trade_report,
    format_welcome,
)
from orchestrator.telegram.translations import to_chinese

if TYPE_CHECKING:
    from orchestrator.approval.manager import ApprovalManager, PendingApproval
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


def _extract_analysis_summary(result: Any) -> dict[str, str]:
    """Extract one-line summaries from pipeline result's agent outputs."""
    if result is None:
        return {}
    summary: dict[str, str] = {}

    tech = getattr(result, "technical_short", None)
    if tech is not None:
        trend = (
            tech.trend.value.upper()
            if hasattr(tech.trend, "value")
            else str(tech.trend).upper()
        )
        momentum = getattr(tech, "momentum", "")
        if hasattr(momentum, "value"):
            momentum = momentum.value.upper()
        parts = [trend]
        if momentum:
            parts.append(f"{momentum} momentum")
        supports = [
            kl for kl in getattr(tech, "key_levels", [])
            if kl.type == "support"
        ]
        resists = [
            kl for kl in getattr(tech, "key_levels", [])
            if kl.type == "resistance"
        ]
        extra_lines: list[str] = []
        if supports:
            s_str = " / ".join(f"{kl.price:,.0f}" for kl in supports)
            extra_lines.append(f"Support: {s_str}")
        if resists:
            r_str = " / ".join(f"{kl.price:,.0f}" for kl in resists)
            extra_lines.append(f"Resist: {r_str}")
        line = ", ".join(parts)
        if extra_lines:
            line += "\n" + " / ".join(extra_lines)
        summary["technical"] = line

    pos = getattr(result, "positioning", None)
    if pos is not None:
        funding_trend = getattr(pos, "funding_trend", None)
        oi_chg = getattr(pos, "oi_change_pct", None)
        parts_p: list[str] = []
        if funding_trend:
            parts_p.append(f"Funding {funding_trend}")
        if oi_chg is not None:
            parts_p.append(f"OI {oi_chg:+.1f}%")
        if parts_p:
            summary["positioning"] = ", ".join(parts_p)

    cat = getattr(result, "catalyst", None)
    if cat is not None:
        recommendation = getattr(cat, "recommendation", None)
        risk_level = getattr(cat, "risk_level", None)
        upcoming = getattr(cat, "upcoming_events", [])
        parts_c: list[str] = []
        if recommendation:
            parts_c.append(recommendation)
        if risk_level:
            parts_c.append(f"{risk_level} risk")
        if upcoming:
            parts_c.append(f"{len(upcoming)} upcoming")
        if parts_c:
            summary["catalyst"] = ", ".join(parts_c)

    corr = getattr(result, "correlation", None)
    if corr is not None:
        dxy = getattr(corr, "dxy_trend", None)
        sp500 = getattr(corr, "sp500_regime", None)
        alignment = getattr(corr, "cross_market_alignment", None)
        parts_r: list[str] = []
        if dxy:
            parts_r.append(f"DXY {dxy}")
        if sp500:
            parts_r.append(f"S&P {sp500.replace('_', '-')}")
        if alignment:
            parts_r.append(alignment)
        if parts_r:
            summary["correlation"] = ", ".join(parts_r)

    return summary


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
        self._approval_manager = approval_manager
        self._executor = executor
        self._data_fetcher = data_fetcher
        self._leverage_options: list[int] = [5, 10, 20, 50]
        self._price_board_msg_ids: dict[int, int] = {}  # chat_id → message_id
        self._adjustments: dict[str, dict[str, Any]] = {}  # approval_id → adjusted params

    _BOT_COMMANDS = [
        BotCommand("status", "Account overview & latest proposals"),
        BotCommand("run", "Trigger pipeline for all symbols"),
        BotCommand("coin", "Detailed analysis for a symbol"),
        BotCommand("history", "Recent trade records"),
        BotCommand("perf", "Performance report"),
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
        self._app.add_handler(CommandHandler("preview", self._preview_handler))
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

        # Risk pause: send dedicated notification with resume button
        if (
            result.status == "risk_paused"
            and result.risk_result
            and result.proposal
        ):
            await self.push_risk_pause(
                symbol=result.symbol,
                side=result.proposal.side.value.upper(),
                entry_price=result.proposal.stop_loss or 0.0,
                risk_result=result.risk_result,
            )
            return

        for chat_id in self.admin_chat_ids:
            is_pending = (
                result.status == "pending_approval"
                and result.approval_id
                and self._approval_manager
            )
            if is_pending:
                approval = self._approval_manager.get(result.approval_id)
                if approval:
                    await self.push_pending_approval(
                        chat_id, approval,
                        technical_short=result.technical_short,
                        execution_plan=getattr(result, "execution_plan", None),
                        pipeline_result=result,
                    )
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

    async def push_risk_pause(
        self, *, symbol: str, side: str, entry_price: float, risk_result: RiskResult
    ) -> None:
        """Push a risk pause notification with resume button to all admin chats."""
        if self._app is None:
            return
        msg = format_risk_pause(
            symbol=symbol, side=side, entry_price=entry_price, risk_result=risk_result
        )
        rows = [
            [InlineKeyboardButton("\u25b6\ufe0f Resume Trading", callback_data="resume:confirm")],
            [InlineKeyboardButton("Translate to zh-TW", callback_data="translate:zh")],
        ]
        for chat_id in self.admin_chat_ids:
            sent = await self._app.bot.send_message(
                chat_id=chat_id, text=msg, reply_markup=InlineKeyboardMarkup(rows),
            )
            self._msg_cache.store(sent.message_id, msg)

    async def update_price_board(self, summaries: list) -> None:
        """Update or create the pinned price board message in all admin chats."""
        if self._app is None:
            return

        from orchestrator.telegram.formatters import format_price_board

        text = format_price_board(summaries)

        for chat_id in self.admin_chat_ids:
            msg_id = self._price_board_msg_ids.get(chat_id)

            if msg_id is not None:
                # Try to edit existing message
                try:
                    await self._app.bot.edit_message_text(
                        text=text, chat_id=chat_id, message_id=msg_id,
                    )
                    continue
                except BadRequest:
                    # Message deleted or not found — fall through to send new
                    pass
                except Exception:
                    logger.warning("price_board_edit_failed", chat_id=chat_id)
                    continue

            # Send new message and pin it
            try:
                sent = await self._app.bot.send_message(chat_id=chat_id, text=text)
                self._price_board_msg_ids[chat_id] = sent.message_id
                await self._app.bot.pin_chat_message(
                    chat_id=chat_id, message_id=sent.message_id, disable_notification=True,
                )
            except Exception:
                logger.exception("price_board_send_failed", chat_id=chat_id)

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
            from orchestrator.telegram.formatters import (
                format_account_overview,
                format_position_card,
            )

            positions = self._paper_engine.get_open_positions()

            overview = format_account_overview(
                equity=self._paper_engine.equity,
                available=self._paper_engine.available_balance,
                used_margin=self._paper_engine.used_margin,
                initial_equity=self._paper_engine._initial_equity,
                position_count=len(positions),
            )

            if positions:
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

                # Send overview header, then each position card with action buttons
                await self._reply(update, overview)
                for pos, card in zip(positions, position_cards):
                    keyboard = InlineKeyboardMarkup([
                        [
                            InlineKeyboardButton(
                                "\u2699\ufe0f Manage",
                                callback_data=f"pos_manage:{pos.trade_id}",
                            ),
                            InlineKeyboardButton(
                                "Close",
                                callback_data=f"close:{pos.trade_id}",
                            ),
                        ],
                    ])
                    if update.message:
                        await update.message.reply_text(
                            card, reply_markup=keyboard,
                        )
                return

            # No open positions — show account overview + recent signals
            parts: list[str] = [overview]

            results = list(self._latest_results.values())
            if results:
                parts.append(format_status(results))
            elif self._proposal_repo is not None:
                records = self._proposal_repo.get_recent(limit=10)
                parts.append(format_status_from_records(records))

            await self._reply(update, "\n\n".join(parts))
            return

        # Fallback: no paper engine
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
                        update.effective_chat.id, approval,
                        execution_plan=getattr(result, "execution_plan", None),
                        pipeline_result=result,
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

        rows: list[list[InlineKeyboardButton]] = []

        # Navigation row
        nav_buttons: list[InlineKeyboardButton] = []
        nav_buttons.append(
            InlineKeyboardButton(f"Page 1/{total_pages}", callback_data="cancel:0")
        )
        if total_pages > 1:
            nav_buttons.append(
                InlineKeyboardButton("Next", callback_data="history:page:2")
            )
        rows.append(nav_buttons)

        # Filter row (only when multiple symbols exist)
        symbols = self._trade_repo.get_distinct_closed_symbols()
        if len(symbols) > 1:
            filter_buttons = [
                InlineKeyboardButton(
                    sym.split("/")[0],
                    callback_data=f"history:filter:{sym}",
                )
                for sym in symbols[:4]
            ]
            filter_buttons.append(
                InlineKeyboardButton("All", callback_data="history:filter:all")
            )
            rows.append(filter_buttons)

        if update.message:
            await update.message.reply_text(
                text, reply_markup=InlineKeyboardMarkup(rows),
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

    async def push_pending_approval(
        self,
        chat_id: int,
        approval: PendingApproval,
        *,
        technical_short: Any | None = None,
        execution_plan: Any | None = None,
        pipeline_result: Any | None = None,
    ) -> int | None:
        """Push proposal with action keyboard. Returns message_id."""
        if self._app is None:
            return None

        if execution_plan is not None:
            analysis_summary = _extract_analysis_summary(pipeline_result)
            text = format_execution_plan(
                plan=execution_plan,
                confidence=approval.proposal.confidence,
                time_horizon=approval.proposal.time_horizon,
                analysis_summary=analysis_summary or None,
                rationale=approval.proposal.rationale,
                model_used=approval.model_used,
                expires_minutes=int(
                    (approval.expires_at - approval.created_at).total_seconds() / 60
                ),
            )
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton(
                        "\U0001f680 Execute",
                        callback_data=f"approve:{approval.approval_id}",
                    ),
                    InlineKeyboardButton(
                        "\u270f\ufe0f Adjust",
                        callback_data=f"adjust:{approval.approval_id}",
                    ),
                    InlineKeyboardButton(
                        "\u274c Skip",
                        callback_data=f"reject:{approval.approval_id}",
                    ),
                ],
                [InlineKeyboardButton(
                    "Translate to zh-TW", callback_data="translate:zh",
                )],
            ])
        else:
            text = format_pending_approval(
                approval, technical_short=technical_short,
            )
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton(
                        "Approve",
                        callback_data=f"approve:{approval.approval_id}",
                    ),
                    InlineKeyboardButton(
                        "Reject",
                        callback_data=f"reject:{approval.approval_id}",
                    ),
                ],
                [InlineKeyboardButton(
                    "Translate to zh-TW", callback_data="translate:zh",
                )],
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
        "leverage":         (3, "_handle_leverage_preview"),
        "margin":           (4, "_handle_margin_preview"),
        "confirm_margin":   (4, "_handle_confirm_margin"),
        "confirm_leverage": (3, "_handle_confirm_leverage"),
        "close":            (2, "_handle_close"),
        "confirm_close":    (2, "_handle_confirm_close"),
        "reduce":           (2, "_handle_reduce"),
        "select_reduce":    (3, "_handle_select_reduce"),
        "confirm_reduce":   (3, "_handle_confirm_reduce"),
        "add":              (2, "_handle_add"),
        "select_add":       (3, "_handle_select_add"),
        "confirm_add":      (3, "_handle_confirm_add"),
        "resume":           (2, "_handle_resume"),
        "cancel":           (1, "_handle_cancel"),
        "history":          (3, "_handle_history_callback"),
        "adjust":           (2, "_handle_adjust"),
        "adj_lev":          (2, "_handle_adjust_leverage"),
        "set_lev":          (3, "_handle_set_leverage"),
        "adj_sl":           (2, "_handle_adjust_sl_prompt"),
        "adj_tp":           (2, "_handle_adjust_tp_prompt"),
        "adj_qty":          (2, "_handle_adjust_qty"),
        "adj_confirm":      (2, "_handle_adjust_confirm"),
        "adj_cancel":       (2, "_handle_adjust_cancel"),
        "pos_manage":       (2, "_handle_pos_manage"),
        "pos_sl":           (2, "_handle_pos_move_sl"),
        "pos_tp":           (2, "_handle_pos_adjust_tp"),
        "pos_add":          (2, "_handle_pos_add"),
        "pos_reduce":       (2, "_handle_pos_reduce"),
        "pos_close":        (2, "_handle_pos_close"),
        "pos_confirm_close": (2, "_handle_pos_confirm_close"),
        "pos_back":         (2, "_handle_pos_back"),
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

    async def _handle_resume(self, query: CallbackQuery, *_args: str) -> None:
        if self._paper_engine is None:
            await _safe_callback_reply(query, text="Paper trading not configured.")
            return
        if not self._paper_engine.paused:
            await _safe_callback_reply(query, text="Trading is already active.")
            return
        self._paper_engine.set_paused(False)
        await _safe_callback_reply(
            query, text="\u25b6\ufe0f Trading resumed. New proposals will be executed."
        )

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

        # Show leverage selection buttons with suggested leverage highlighted
        p = approval.proposal
        leverage_buttons = [
            InlineKeyboardButton(
                f"{'✓ ' if lev == p.suggested_leverage else ''}{lev}x",
                callback_data=f"leverage:{approval_id}:{lev}",
            )
            for lev in self._leverage_options
        ]
        keyboard = InlineKeyboardMarkup([
            leverage_buttons,
            [InlineKeyboardButton("Cancel", callback_data="cancel:0")],
        ])

        text = (
            f"SELECT LEVERAGE\n"
            f"{p.symbol}  {p.side.value.upper()}\n\n"
            f"Suggested: {p.suggested_leverage}x\n"
            f"Stop Loss: ${p.stop_loss:,.1f}\n"
            f"Choose leverage:"
        )
        await _safe_callback_reply(query, text=text, reply_markup=keyboard)

    async def _handle_leverage_preview(
        self, query: CallbackQuery, approval_id: str, leverage_str: str, *_args: str,
    ) -> None:
        """Show margin amount selection after leverage is chosen."""
        leverage = int(leverage_str)
        if self._approval_manager is None or self._executor is None:
            await query.answer("Not configured")
            return

        approval = self._approval_manager.get(approval_id)
        if approval is None:
            await query.answer("Expired or not found")
            await query.edit_message_text("Approval expired or already handled.")
            return

        p = approval.proposal
        margin_buttons = [
            InlineKeyboardButton(
                f"${m}", callback_data=f"margin:{approval_id}:{leverage}:{m}"
            )
            for m in [100, 250, 500, 1000]
        ]
        keyboard = InlineKeyboardMarkup([
            margin_buttons,
            [InlineKeyboardButton("Cancel", callback_data="cancel:0")],
        ])

        text = (
            f"SELECT MARGIN\n"
            f"{p.symbol}  {p.side.value.upper()} · {leverage}x\n\n"
            f"Select margin amount (USDT):"
        )
        await _safe_callback_reply(query, text=text, reply_markup=keyboard)

    async def _handle_margin_preview(
        self, query: CallbackQuery, approval_id: str,
        leverage_str: str, margin_str: str, *_args: str,
    ) -> None:
        """Show confirmation card with computed quantity, liq, and ROE details."""
        leverage = int(leverage_str)
        margin_usdt = float(margin_str)
        if self._approval_manager is None or self._executor is None:
            await query.answer("Not configured")
            return
        if self._paper_engine is None:
            await query.answer("Paper engine not configured")
            return

        approval = self._approval_manager.get(approval_id)
        if approval is None:
            await query.answer("Expired or not found")
            await query.edit_message_text("Approval expired or already handled.")
            return

        p = approval.proposal
        current_price = approval.snapshot_price
        if self._data_fetcher is not None:
            current_price = await self._data_fetcher.fetch_current_price(p.symbol)

        from orchestrator.risk.position_sizer import MarginSizer

        sizer = MarginSizer()
        qty = sizer.calculate_from_margin(
            margin_usdt=margin_usdt, leverage=leverage, entry_price=current_price,
        )
        notional = qty * current_price
        liq = self._paper_engine.calculate_liquidation_price(
            entry_price=current_price, leverage=leverage, side=p.side,
        )

        side_str = p.side.value.upper()
        lines = [
            "CONFIRM ORDER",
            f"{p.symbol}  {side_str} @ ${current_price:,.1f}",
            "",
            f"Margin:     {margin_usdt:,.0f} USDT",
            f"Leverage:   {leverage}x",
            f"Quantity:   {qty:.4f}",
            f"Notional:   ${notional:,.0f}",
            "",
            f"\u26d4 Liq:     ${liq:,.1f}",
        ]

        # SL ROE
        if p.stop_loss is not None:
            direction = 1 if p.side.value == "long" else -1
            sl_pnl = (p.stop_loss - current_price) * qty * direction
            sl_roe = (sl_pnl / margin_usdt * 100) if margin_usdt > 0 else 0
            lines.append(f"\u26d4 SL ROE:  {sl_roe:+.1f}%")

        # TP ROE for each level
        for i, tp in enumerate(p.take_profit, 1):
            direction = 1 if p.side.value == "long" else -1
            tp_pnl = (tp.price - current_price) * qty * direction
            tp_roe = (tp_pnl / margin_usdt * 100) if margin_usdt > 0 else 0
            lines.append(f"\u2705 TP{i} ROE: {tp_roe:+.1f}% \u2192 close {tp.close_pct}%")

        text = "\n".join(lines)
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton(
                    "Confirm",
                    callback_data=f"confirm_margin:{approval_id}:{leverage}:{int(margin_usdt)}",
                ),
                InlineKeyboardButton("Cancel", callback_data="cancel:0"),
            ],
        ])
        await _safe_callback_reply(query, text=text, reply_markup=keyboard)

    async def _handle_confirm_leverage(
        self, query: CallbackQuery, approval_id: str, leverage_str: str, *_args: str,
    ) -> None:
        """Execute trade with selected leverage."""
        leverage = int(leverage_str)
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

            tp_prices = [tp.price for tp in approval.proposal.take_profit]
            stale_reason = _check_stale_sl_tp(
                side=approval.proposal.side.value,
                current_price=current_price,
                stop_loss=approval.proposal.stop_loss,
                take_profit=tp_prices,
            )
            if stale_reason:
                self._approval_manager.reject(approval_id)
                text = (
                    f"STALE PROPOSAL\n"
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
                take_profit=tp_prices,
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

    async def _handle_confirm_margin(
        self, query: CallbackQuery, approval_id: str,
        leverage_str: str, margin_str: str, *_args: str,
    ) -> None:
        """Execute trade with selected leverage and USDT margin amount."""
        leverage = int(leverage_str)
        margin_usdt = float(margin_str)
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

            tp_prices = [tp.price for tp in approval.proposal.take_profit]
            stale_reason = _check_stale_sl_tp(
                side=approval.proposal.side.value,
                current_price=current_price,
                stop_loss=approval.proposal.stop_loss,
                take_profit=tp_prices,
            )
            if stale_reason:
                self._approval_manager.reject(approval_id)
                text = (
                    f"STALE PROPOSAL\n"
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
                approval.proposal, current_price=current_price,
                leverage=leverage, margin_usdt=margin_usdt,
            )
            sl_tp_ids = await self._executor.place_sl_tp(
                symbol=approval.proposal.symbol,
                side=approval.proposal.side.value,
                quantity=result.quantity,
                stop_loss=approval.proposal.stop_loss or 0.0,
                take_profit=tp_prices,
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
                margin_usdt=margin_usdt,
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
            f"REJECTED\n"
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

    # --- Adjustment flow handlers ---

    async def _handle_adjust(
        self, query: CallbackQuery, approval_id: str, *_args: str,
    ) -> None:
        """Show parameter adjustment menu."""
        if self._approval_manager is None:
            await query.answer("Not configured")
            return
        approval = self._approval_manager.get(approval_id)
        if approval is None:
            await query.answer("Expired or not found")
            return

        buttons = [
            [
                InlineKeyboardButton(
                    "Leverage",
                    callback_data=f"adj_lev:{approval_id}",
                ),
                InlineKeyboardButton(
                    "Stop Loss",
                    callback_data=f"adj_sl:{approval_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    "Take Profit",
                    callback_data=f"adj_tp:{approval_id}",
                ),
                InlineKeyboardButton(
                    "Quantity",
                    callback_data=f"adj_qty:{approval_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    "\U0001f680 Confirm",
                    callback_data=f"adj_confirm:{approval_id}",
                ),
                InlineKeyboardButton(
                    "\u274c Cancel",
                    callback_data=f"adj_cancel:{approval_id}",
                ),
            ],
        ]
        await _safe_callback_reply(
            query,
            text="\u270f\ufe0f Which parameter to adjust?",
            reply_markup=InlineKeyboardMarkup(buttons),
        )

    async def _handle_adjust_leverage(
        self, query: CallbackQuery, approval_id: str, *_args: str,
    ) -> None:
        """Show leverage selection buttons for adjustment."""
        if self._approval_manager is None:
            await query.answer("Not configured")
            return
        approval = self._approval_manager.get(approval_id)
        if approval is None:
            await query.answer("Expired or not found")
            return

        current_lev = self._adjustments.get(
            approval_id, {},
        ).get("leverage", approval.proposal.suggested_leverage)

        buttons = [
            [
                InlineKeyboardButton(
                    f"{'> ' if lev == current_lev else ''}{lev}x",
                    callback_data=f"set_lev:{approval_id}:{lev}",
                )
                for lev in self._leverage_options
            ],
            [InlineKeyboardButton(
                "\u2b05\ufe0f Back",
                callback_data=f"adjust:{approval_id}",
            )],
        ]
        await _safe_callback_reply(
            query,
            text=(
                f"Current Leverage: {current_lev}x\n"
                "Select new leverage:"
            ),
            reply_markup=InlineKeyboardMarkup(buttons),
        )

    async def _handle_set_leverage(
        self, query: CallbackQuery, approval_id: str, leverage_str: str, *_args: str,
    ) -> None:
        """Store adjusted leverage and return to adjustment menu."""
        leverage = int(leverage_str)
        adj = self._adjustments.setdefault(approval_id, {})
        adj["leverage"] = leverage
        await query.answer(f"Leverage set to {leverage}x")
        # Return to adjustment menu
        await self._handle_adjust(query, approval_id)

    async def _handle_adjust_sl_prompt(
        self, query: CallbackQuery, approval_id: str, *_args: str,
    ) -> None:
        """Prompt user to type new SL price."""
        await _safe_callback_reply(
            query,
            text=(
                "\u26d4 Enter new Stop Loss price:\n"
                "e.g. 92500"
            ),
        )
        adj = self._adjustments.setdefault(approval_id, {})
        adj["_awaiting"] = "sl"
        adj["_approval_id"] = approval_id

    async def _handle_adjust_tp_prompt(
        self, query: CallbackQuery, approval_id: str, *_args: str,
    ) -> None:
        """Prompt user to type new TP prices."""
        await _safe_callback_reply(
            query,
            text=(
                "\u2705 Enter new Take Profit levels:\n"
                "e.g. 97000 50%, 99000 100%\n\n"
                "Format: price close%, "
                "price close%"
            ),
        )
        adj = self._adjustments.setdefault(approval_id, {})
        adj["_awaiting"] = "tp"
        adj["_approval_id"] = approval_id

    async def _handle_adjust_qty(
        self, query: CallbackQuery, approval_id: str, *_args: str,
    ) -> None:
        """Prompt user to type new margin amount."""
        await _safe_callback_reply(
            query,
            text=(
                "\U0001f4b0 Enter new margin amount (USDT):\n"
                "e.g. 300"
            ),
        )
        adj = self._adjustments.setdefault(approval_id, {})
        adj["_awaiting"] = "margin"
        adj["_approval_id"] = approval_id

    async def _handle_adjust_confirm(
        self, query: CallbackQuery, approval_id: str, *_args: str,
    ) -> None:
        """Confirm adjustment and proceed to execute with adjusted params."""
        adj = self._adjustments.pop(approval_id, {})
        leverage = adj.get("leverage")
        margin = adj.get("margin")

        if leverage is not None or margin is not None:
            # Route to existing execution flow with adjusted params
            lev = leverage or (
                self._approval_manager.get(approval_id).proposal.suggested_leverage
                if self._approval_manager and self._approval_manager.get(approval_id)
                else 10
            )
            if margin is not None:
                await self._handle_confirm_margin(
                    query, approval_id, str(lev), str(int(margin)),
                )
            else:
                await self._handle_confirm_leverage(
                    query, approval_id, str(lev),
                )
        else:
            # No adjustments made, route to standard approve
            await self._handle_approve(query, approval_id)

    async def _handle_adjust_cancel(
        self, query: CallbackQuery, approval_id: str, *_args: str,
    ) -> None:
        """Cancel adjustment and return to original message."""
        self._adjustments.pop(approval_id, None)
        await _safe_callback_reply(
            query, text="\u274c Adjustment cancelled.",
        )

    # --- Preview handler (hidden debug command) ---

    async def _preview_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Hidden command to preview message formats with mock data."""
        if not await self._check_admin(update):
            return

        args = context.args
        preview_type = args[0].lower() if args else "plan"

        from datetime import UTC, datetime, timedelta

        from orchestrator.execution.plan import ExecutionPlan, OrderInstruction
        from orchestrator.models import (
            CatalystEvent,
            CatalystReport,
            CorrelationAnalysis,
            EntryOrder,
            KeyLevel,
            Momentum,
            PositioningAnalysis,
            Side,
            TakeProfit,
            TechnicalAnalysis,
            TradeProposal,
            Trend,
            VolatilityRegime,
        )
        from orchestrator.pipeline.runner import PipelineResult

        _proposal = TradeProposal(
            proposal_id="preview-001",
            symbol="ETH/USDT:USDT",
            side=Side.LONG,
            entry=EntryOrder(type="limit", price=2020.0),
            position_size_risk_pct=0.5,
            stop_loss=1975.0,
            take_profit=[
                TakeProfit(price=2055.0, close_pct=50),
                TakeProfit(price=2089.0, close_pct=50),
            ],
            suggested_leverage=5,
            time_horizon="4h",
            confidence=0.40,
            invalid_if=["price drops below 1950"],
            rationale=(
                "Short-term uptrend (ADX 32) with bullish momentum. "
                "Funding stable and OI expanding — room to run. "
                "DXY weakening provides macro tailwind."
            ),
        )

        _tech = TechnicalAnalysis(
            label="short_term",
            trend=Trend.UP,
            trend_strength=32.0,
            volatility_regime=VolatilityRegime.MEDIUM,
            volatility_pct=2.8,
            momentum=Momentum.BULLISH,
            rsi=58.0,
            key_levels=[
                KeyLevel(type="support", price=2020.0),
                KeyLevel(type="support", price=2000.0),
                KeyLevel(type="resistance", price=2055.0),
                KeyLevel(type="resistance", price=2089.0),
            ],
            risk_flags=[],
        )

        _positioning = PositioningAnalysis(
            funding_trend="stable",
            funding_extreme=False,
            oi_change_pct=3.2,
            retail_bias="long",
            smart_money_bias="long",
            squeeze_risk="none",
            liquidity_assessment="normal",
            risk_flags=[],
            confidence=0.65,
        )

        _catalyst = CatalystReport(
            upcoming_events=[
                CatalystEvent(
                    event="FOMC Minutes",
                    time="2026-03-05T18:00:00Z",
                    impact="high",
                    direction_bias="uncertain",
                ),
                CatalystEvent(
                    event="ETH Dencun Upgrade",
                    time="2026-03-08T00:00:00Z",
                    impact="medium",
                    direction_bias="bullish",
                ),
            ],
            active_events=[],
            risk_level="low",
            recommendation="proceed",
            confidence=0.70,
        )

        _correlation = CorrelationAnalysis(
            dxy_trend="weakening",
            dxy_impact="tailwind",
            sp500_regime="risk_on",
            btc_dominance_trend="falling",
            cross_market_alignment="favorable",
            risk_flags=[],
            confidence=0.72,
        )

        _exec_plan = ExecutionPlan(
            proposal_id="preview-001",
            symbol="ETH/USDT:USDT",
            side="long",
            entry_order=OrderInstruction(
                symbol="ETH/USDT:USDT",
                side="buy",
                order_type="limit",
                quantity=1.2376,
                price=2020.0,
            ),
            sl_order=OrderInstruction(
                symbol="ETH/USDT:USDT",
                side="sell",
                order_type="market",
                quantity=1.2376,
                stop_price=1975.0,
                reduce_only=True,
            ),
            tp_orders=[
                OrderInstruction(
                    symbol="ETH/USDT:USDT",
                    side="sell",
                    order_type="market",
                    quantity=0.6188,
                    stop_price=2055.0,
                    reduce_only=True,
                ),
                OrderInstruction(
                    symbol="ETH/USDT:USDT",
                    side="sell",
                    order_type="market",
                    quantity=0.6188,
                    stop_price=2089.0,
                    reduce_only=True,
                ),
            ],
            margin_mode="isolated",
            leverage=5,
            quantity=1.2376,
            entry_price=2020.0,
            notional_value=2500.0,
            margin_required=500.0,
            liquidation_price=1616.0,
            estimated_fees=1.25,
            max_loss=56.0,
            max_loss_pct=0.5,
            tp_profits=[22.0, 43.0],
            risk_reward_ratio=1.5,
            equity_snapshot=10_000.0,
        )

        now = datetime.now(UTC)

        _pipeline_result = PipelineResult(
            run_id="preview-run",
            symbol="ETH/USDT:USDT",
            status="pending_approval",
            model_used="anthropic/claude-sonnet-4-6",
            created_at=now,
            proposal=_proposal,
            technical_short=_tech,
            positioning=_positioning,
            catalyst=_catalyst,
            correlation=_correlation,
            execution_plan=_exec_plan,
        )

        if preview_type == "plan":
            analysis_summary = _extract_analysis_summary(_pipeline_result)
            text = format_execution_plan(
                plan=_exec_plan,
                confidence=_proposal.confidence,
                time_horizon=_proposal.time_horizon,
                analysis_summary=analysis_summary or None,
                rationale=_proposal.rationale,
                model_used="anthropic/claude-sonnet-4-6",
                expires_minutes=30,
            )
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton(
                        "\U0001f680 Execute",
                        callback_data="cancel:0",
                    ),
                    InlineKeyboardButton(
                        "\u270f\ufe0f Adjust",
                        callback_data="cancel:0",
                    ),
                    InlineKeyboardButton(
                        "\u274c Skip",
                        callback_data="cancel:0",
                    ),
                ],
                [InlineKeyboardButton(
                    "Translate to zh-TW", callback_data="cancel:0",
                )],
            ])
            if update.message:
                await update.message.reply_text(text, reply_markup=keyboard)

        elif preview_type == "flat":
            flat_proposal = _proposal.model_copy(update={
                "side": Side.FLAT,
                "stop_loss": None,
                "take_profit": [],
                "rationale": "Market ranging with no clear directional bias.",
            })
            flat_result = _pipeline_result.model_copy(update={
                "status": "completed",
                "proposal": flat_proposal,
            })
            await self._reply(update, format_proposal(flat_result))

        elif preview_type == "rejected":
            rejected_result = _pipeline_result.model_copy(update={
                "status": "rejected",
                "rejection_reason": "Risk/reward below threshold (0.8 < 1.0).",
            })
            await self._reply(update, format_proposal(rejected_result))

        elif preview_type == "approval":
            from orchestrator.approval.manager import PendingApproval

            approval = PendingApproval(
                approval_id="preview-appr",
                proposal=_proposal,
                run_id="preview-run",
                snapshot_price=2020.0,
                created_at=now,
                expires_at=now + timedelta(minutes=30),
                model_used="anthropic/claude-sonnet-4-6",
            )
            text = format_pending_approval(
                approval, technical_short=_tech,
            )
            if update.message:
                await update.message.reply_text(text)

        elif preview_type == "status":
            completed = _pipeline_result.model_copy(update={
                "status": "completed",
            })
            rejected = _pipeline_result.model_copy(update={
                "status": "rejected",
                "symbol": "BTC/USDT:USDT",
                "rejection_reason": "Confidence below threshold.",
                "proposal": _proposal.model_copy(update={
                    "symbol": "BTC/USDT:USDT",
                }),
            })
            await self._reply(
                update, format_status([completed, rejected]),
            )

        else:
            await self._reply(
                update,
                "Usage: /preview [plan|flat|rejected|approval|status]",
            )

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

        text = (
            f"CLOSE POSITION?\n"
            f"Now: ${current_price:,.1f}\n\n"
            f"{format_position_card(info)}"
        )
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

        card = format_position_card(info)
        text = f"REDUCE POSITION\n\n{card}\n\nSelect percentage to close:"
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("25%", callback_data=f"select_reduce:{trade_id}:25"),
                InlineKeyboardButton("50%", callback_data=f"select_reduce:{trade_id}:50"),
                InlineKeyboardButton("75%", callback_data=f"select_reduce:{trade_id}:75"),
            ],
            [InlineKeyboardButton("Cancel", callback_data="cancel:0")],
        ])
        await _safe_callback_reply(query, text=text, reply_markup=keyboard)

    async def _handle_select_reduce(
        self, query: CallbackQuery, trade_id: str, pct_str: str, *_args: str,
    ) -> None:
        """Show confirmation card for reduce operation with estimated PnL."""
        from orchestrator.models import Side

        pct = float(pct_str)
        if self._paper_engine is None or self._data_fetcher is None:
            await query.answer("Not configured")
            return

        try:
            pos = self._paper_engine._find_position(trade_id)
            current_price = await self._data_fetcher.fetch_current_price(pos.symbol)
        except ValueError:
            await query.answer("Position not found")
            return

        close_qty = pos.quantity * pct / 100
        direction = 1 if pos.side == Side.LONG else -1
        est_pnl = (current_price - pos.entry_price) * close_qty * direction
        pnl_sign = "+" if est_pnl >= 0 else ""

        side_str = (
            pos.side.value.upper()
            if hasattr(pos.side, "value")
            else str(pos.side).upper()
        )
        text = (
            f"CONFIRM REDUCE\n"
            f"{pos.symbol}  {side_str} {pos.leverage}x\n\n"
            f"Now: ${current_price:,.1f}\n"
            f"Close {pct:.0f}%: {close_qty:.4f}\n"
            f"Est. PnL: {pnl_sign}${est_pnl:,.2f}"
        )
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton(
                    "Confirm Reduce",
                    callback_data=f"confirm_reduce:{trade_id}:{pct_str}",
                ),
                InlineKeyboardButton("Cancel", callback_data="cancel:0"),
            ],
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

        text = f"ADD TO POSITION\n\n{format_position_card(info)}\n\nSelect risk % to add:"
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("0.5%", callback_data=f"select_add:{trade_id}:0.5"),
                InlineKeyboardButton("1%", callback_data=f"select_add:{trade_id}:1.0"),
                InlineKeyboardButton("2%", callback_data=f"select_add:{trade_id}:2.0"),
            ],
            [InlineKeyboardButton("Cancel", callback_data="cancel:0")],
        ])
        await _safe_callback_reply(query, text=text, reply_markup=keyboard)

    async def _handle_select_add(
        self, query: CallbackQuery, trade_id: str, risk_pct_str: str, *_args: str,
    ) -> None:
        """Show confirmation card for add operation."""
        risk_pct = float(risk_pct_str)
        if self._paper_engine is None or self._data_fetcher is None:
            await query.answer("Not configured")
            return

        try:
            pos = self._paper_engine._find_position(trade_id)
            current_price = await self._data_fetcher.fetch_current_price(pos.symbol)
        except ValueError:
            await query.answer("Position not found")
            return

        add_qty = self._paper_engine._position_sizer.calculate(
            equity=self._paper_engine.equity,
            risk_pct=risk_pct,
            entry_price=current_price,
            stop_loss=pos.stop_loss,
        )
        add_margin = self._paper_engine.calculate_margin(
            quantity=add_qty, price=current_price, leverage=pos.leverage,
        )

        side_str = (
            pos.side.value.upper()
            if hasattr(pos.side, "value")
            else str(pos.side).upper()
        )
        text = (
            f"CONFIRM ADD\n"
            f"{pos.symbol}  {side_str} {pos.leverage}x\n\n"
            f"Now: ${current_price:,.1f}\n"
            f"Add Qty: {add_qty:.4f} | Add Margin: ${add_margin:,.2f}\n"
            f"Risk: {risk_pct}%"
        )
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton(
                    "Confirm Add",
                    callback_data=f"confirm_add:{trade_id}:{risk_pct}",
                ),
                InlineKeyboardButton("Cancel", callback_data="cancel:0"),
            ],
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
                f"POSITION UPDATED\n"
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

    # --- Position management handlers ---

    async def _handle_pos_manage(
        self, query: CallbackQuery, trade_id: str, *_args: str,
    ) -> None:
        """Show position management menu."""
        if self._paper_engine is None:
            await query.answer("Not configured")
            return
        try:
            pos = self._paper_engine._find_position(trade_id)
        except ValueError:
            await query.answer("Position not found")
            return

        buttons = [
            [
                InlineKeyboardButton(
                    "Move SL",
                    callback_data=f"pos_sl:{trade_id}",
                ),
                InlineKeyboardButton(
                    "Adjust TP",
                    callback_data=f"pos_tp:{trade_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    "Add",
                    callback_data=f"pos_add:{trade_id}",
                ),
                InlineKeyboardButton(
                    "Reduce",
                    callback_data=f"pos_reduce:{trade_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    "Close",
                    callback_data=f"pos_close:{trade_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    "\u2b05\ufe0f Back",
                    callback_data=f"pos_back:{trade_id}",
                ),
            ],
        ]
        side_str = (
            pos.side.value.upper()
            if hasattr(pos.side, "value")
            else str(pos.side).upper()
        )
        text = (
            f"\u2699\ufe0f {pos.symbol} {side_str}"
            f" {pos.leverage}x\n"
            f"Entry: ${pos.entry_price:,.1f}"
            f" | Qty: {pos.quantity:.4f}\n"
            f"SL: ${pos.stop_loss:,.1f}"
        )
        await _safe_callback_reply(
            query, text=text,
            reply_markup=InlineKeyboardMarkup(buttons),
        )

    async def _handle_pos_move_sl(
        self, query: CallbackQuery, trade_id: str, *_args: str,
    ) -> None:
        """Prompt user to enter new SL price."""
        await _safe_callback_reply(
            query,
            text=(
                "\u26d4 Enter new Stop Loss price:\n"
                "e.g. 92500"
            ),
        )
        adj = self._adjustments.setdefault(
            f"pos_{trade_id}", {},
        )
        adj["_awaiting"] = "pos_sl"
        adj["_trade_id"] = trade_id

    async def _handle_pos_adjust_tp(
        self, query: CallbackQuery, trade_id: str, *_args: str,
    ) -> None:
        """Prompt user to enter new TP levels."""
        await _safe_callback_reply(
            query,
            text=(
                "\u2705 Enter new Take Profit levels:\n"
                "e.g. 97000 50%, 99000 100%\n\n"
                "Format: price close%, "
                "price close%"
            ),
        )
        adj = self._adjustments.setdefault(
            f"pos_{trade_id}", {},
        )
        adj["_awaiting"] = "pos_tp"
        adj["_trade_id"] = trade_id

    async def _handle_pos_add(
        self, query: CallbackQuery, trade_id: str, *_args: str,
    ) -> None:
        """Route to existing add handler."""
        await self._handle_add(query, trade_id)

    async def _handle_pos_reduce(
        self, query: CallbackQuery, trade_id: str, *_args: str,
    ) -> None:
        """Route to existing reduce handler."""
        await self._handle_reduce(query, trade_id)

    async def _handle_pos_close(
        self, query: CallbackQuery, trade_id: str, *_args: str,
    ) -> None:
        """Show close confirmation."""
        await self._handle_close(query, trade_id)

    async def _handle_pos_confirm_close(
        self, query: CallbackQuery, trade_id: str, *_args: str,
    ) -> None:
        """Route to existing confirm close handler."""
        await self._handle_confirm_close(query, trade_id)

    async def _handle_pos_back(
        self, query: CallbackQuery, trade_id: str, *_args: str,
    ) -> None:
        """Return to position card view."""
        if self._paper_engine is None or self._data_fetcher is None:
            await query.answer("Not configured")
            return
        try:
            pos = self._paper_engine._find_position(trade_id)
            current_price = (
                await self._data_fetcher.fetch_current_price(
                    pos.symbol,
                )
            )
            info = self._paper_engine.get_position_with_pnl(
                trade_id=trade_id, current_price=current_price,
            )
        except (ValueError, Exception):
            await query.answer("Position not found")
            return

        from orchestrator.telegram.formatters import (
            format_position_card,
        )

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton(
                    "\u2699\ufe0f Manage",
                    callback_data=f"pos_manage:{trade_id}",
                ),
                InlineKeyboardButton(
                    "Close",
                    callback_data=f"close:{trade_id}",
                ),
            ],
        ])
        await _safe_callback_reply(
            query,
            text=format_position_card(info),
            reply_markup=keyboard,
        )

    async def _handle_history_callback(
        self, query: CallbackQuery, action: str, value: str, *extra: str,
    ) -> None:
        """Handle history pagination/filter callbacks.

        Callback formats:
          history:page:N              — plain pagination
          history:page:N:SYMBOL       — pagination with filter preserved
          history:filter:SYMBOL       — apply symbol filter (page 1)
          history:filter:all          — reset filter
        """
        # Rejoin extra parts for symbols with colons (e.g. "BTC/USDT:USDT")
        if extra:
            value = ":".join([value, *extra])
        if self._trade_repo is None:
            await query.answer("Not configured")
            return

        from orchestrator.telegram.formatters import format_history_paginated

        page_size = 5
        page = 1
        symbol_filter: str | None = None

        if action == "page":
            # value may be "2" or "2:BTC/USDT:USDT" (page with filter)
            parts = value.split(":", 1)
            page = int(parts[0])
            if len(parts) > 1:
                symbol_filter = parts[1]
        elif action == "filter":
            if value != "all":
                symbol_filter = value

        offset = (page - 1) * page_size
        trades, total = self._trade_repo.get_closed_paginated(
            offset=offset, limit=page_size, symbol=symbol_filter,
        )
        total_pages = max(1, (total + page_size - 1) // page_size)
        text = format_history_paginated(trades, page=page, total_pages=total_pages)

        rows: list[list[InlineKeyboardButton]] = []

        # Navigation row — encode filter in page callbacks
        filter_suffix = f":{symbol_filter}" if symbol_filter else ""
        nav_buttons: list[InlineKeyboardButton] = []
        if page > 1:
            nav_buttons.append(InlineKeyboardButton(
                "Prev", callback_data=f"history:page:{page - 1}{filter_suffix}",
            ))
        nav_buttons.append(
            InlineKeyboardButton(f"Page {page}/{total_pages}", callback_data="cancel:0")
        )
        if page < total_pages:
            nav_buttons.append(InlineKeyboardButton(
                "Next", callback_data=f"history:page:{page + 1}{filter_suffix}",
            ))
        rows.append(nav_buttons)

        # Filter row
        symbols = self._trade_repo.get_distinct_closed_symbols()
        if len(symbols) > 1:
            filter_buttons = [
                InlineKeyboardButton(
                    sym.split("/")[0],
                    callback_data=f"history:filter:{sym}",
                )
                for sym in symbols[:4]
            ]
            filter_buttons.append(
                InlineKeyboardButton("All", callback_data="history:filter:all")
            )
            rows.append(filter_buttons)

        keyboard = InlineKeyboardMarkup(rows)
        await _safe_callback_reply(query, text=text, reply_markup=keyboard)
