from __future__ import annotations

import structlog

from orchestrator.config import Settings
from orchestrator.exchange.client import ExchangeClient
from orchestrator.logging import setup_logging
from orchestrator.storage.database import create_db_engine, init_db
from orchestrator.telegram.bot import SentinelBot

logger = structlog.get_logger(__name__)


def create_app_components(
    *,
    telegram_bot_token: str,
    telegram_admin_chat_ids: list[int],
    exchange_id: str,
    database_url: str,
    anthropic_api_key: str,
) -> dict:
    db_engine = create_db_engine(database_url)
    init_db(db_engine)

    exchange_client = ExchangeClient(exchange_id=exchange_id)
    bot = SentinelBot(token=telegram_bot_token, admin_chat_ids=telegram_admin_chat_ids)

    return {
        "bot": bot,
        "exchange_client": exchange_client,
        "db_engine": db_engine,
    }


def main() -> None:
    setup_logging(json_output=True)

    settings = Settings()  # type: ignore[call-arg]
    logger.info("starting_sentinel", exchange=settings.exchange_id)

    components = create_app_components(
        telegram_bot_token=settings.telegram_bot_token,
        telegram_admin_chat_ids=settings.telegram_admin_chat_ids,
        exchange_id=settings.exchange_id,
        database_url=settings.database_url,
        anthropic_api_key=settings.anthropic_api_key,
    )

    app = components["bot"].build()
    logger.info("bot_ready", admin_ids=settings.telegram_admin_chat_ids)
    app.run_polling()


if __name__ == "__main__":
    main()
