from __future__ import annotations

import structlog
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from orchestrator.telegram.formatters import format_help, format_welcome

logger = structlog.get_logger(__name__)


def is_admin(chat_id: int, *, admin_ids: list[int]) -> bool:
    return chat_id in admin_ids


class SentinelBot:
    def __init__(self, token: str, admin_chat_ids: list[int]) -> None:
        self.token = token
        self.admin_chat_ids = admin_chat_ids
        self._app: Application | None = None

    def build(self) -> Application:
        self._app = Application.builder().token(self.token).build()
        self._app.add_handler(CommandHandler("start", self._start_handler))
        self._app.add_handler(CommandHandler("help", self._help_handler))
        self._app.add_handler(CommandHandler("status", self._status_handler))
        self._app.add_handler(CommandHandler("coin", self._coin_handler))
        return self._app

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
        # Placeholder — will be wired to pipeline in M1
        await update.message.reply_text("Status: Paper trading mode. No active proposals yet.")

    async def _coin_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        args = context.args
        if not args:
            await update.message.reply_text("Usage: /coin <symbol> (e.g. /coin BTC)")
            return
        symbol = args[0].upper()
        # Placeholder — will be wired to pipeline in M1
        await update.message.reply_text(f"Analysis for {symbol}: Coming soon in M1.")
