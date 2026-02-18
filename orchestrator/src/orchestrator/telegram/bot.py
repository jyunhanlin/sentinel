from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from orchestrator.telegram.formatters import (
    format_help,
    format_proposal,
    format_status,
    format_welcome,
)

if TYPE_CHECKING:
    from orchestrator.pipeline.scheduler import PipelineScheduler

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

    def set_scheduler(self, scheduler: PipelineScheduler) -> None:
        self._scheduler = scheduler

    def build(self) -> Application:
        self._app = Application.builder().token(self.token).build()
        self._app.add_handler(CommandHandler("start", self._start_handler))
        self._app.add_handler(CommandHandler("help", self._help_handler))
        self._app.add_handler(CommandHandler("status", self._status_handler))
        self._app.add_handler(CommandHandler("coin", self._coin_handler))
        self._app.add_handler(CommandHandler("run", self._run_handler))
        return self._app

    async def push_proposal(self, chat_id: int, result) -> None:
        """Push a pipeline result to a specific chat."""
        if self._app is None:
            return
        msg = format_proposal(result)
        await self._app.bot.send_message(chat_id=chat_id, text=msg)

    async def push_to_admins(self, result) -> None:
        """Push a pipeline result to all admin chats."""
        for chat_id in self.admin_chat_ids:
            await self.push_proposal(chat_id, result)

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
        await update.message.reply_text(format_welcome())

    async def _help_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        await update.message.reply_text(format_help())

    async def _status_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        results = list(self._latest_results.values())
        await update.message.reply_text(format_status(results))

    async def _coin_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        args = context.args
        if not args:
            await update.message.reply_text("Usage: /coin <symbol> (e.g. /coin BTC)")
            return

        query = args[0].upper()
        # Find matching symbol
        matching = [
            r for sym, r in self._latest_results.items() if query in sym.upper()
        ]

        if not matching:
            await update.message.reply_text(
                f"No recent analysis for {query}. Use /run to trigger analysis."
            )
            return

        for result in matching:
            await update.message.reply_text(format_proposal(result))

    async def _run_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        if self._scheduler is None:
            await update.message.reply_text("Pipeline not configured.")
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
        await update.message.reply_text(f"Running pipeline (model: {model_label})...")

        if symbol_args:
            query = symbol_args[0].upper()
            symbols = [
                s for s in self._scheduler.symbols if query in s.upper()
            ]
            if not symbols:
                await update.message.reply_text(f"Unknown symbol: {query}")
                return
            results = await self._scheduler.run_once(symbols=symbols, model_override=model_override)
        else:
            results = await self._scheduler.run_once(model_override=model_override)

        for result in results:
            self._latest_results[result.symbol] = result
            await update.message.reply_text(format_proposal(result))
