from __future__ import annotations

import structlog
from sqlmodel import Session

from orchestrator.agents.market import MarketAgent
from orchestrator.agents.proposer import ProposerAgent
from orchestrator.agents.sentiment import SentimentAgent
from orchestrator.config import Settings
from orchestrator.exchange.client import ExchangeClient
from orchestrator.exchange.data_fetcher import DataFetcher
from orchestrator.llm.client import LLMClient
from orchestrator.logging import setup_logging
from orchestrator.pipeline.runner import PipelineRunner
from orchestrator.pipeline.scheduler import PipelineScheduler
from orchestrator.storage.database import create_db_engine, init_db
from orchestrator.storage.repository import (
    LLMCallRepository,
    PipelineRepository,
    TradeProposalRepository,
)
from orchestrator.telegram.bot import SentinelBot

logger = structlog.get_logger(__name__)


def create_app_components(
    *,
    telegram_bot_token: str,
    telegram_admin_chat_ids: list[int],
    exchange_id: str,
    database_url: str,
    anthropic_api_key: str,
    llm_model: str = "anthropic/claude-sonnet-4-6",
    llm_model_premium: str = "anthropic/claude-opus-4-6",
    llm_temperature: float = 0.2,
    llm_max_tokens: int = 2000,
    llm_max_retries: int = 1,
    pipeline_symbols: list[str] | None = None,
    pipeline_interval_minutes: int = 15,
) -> dict:
    # Database
    db_engine = create_db_engine(database_url)
    init_db(db_engine)
    session = Session(db_engine)

    # LLM
    llm_client = LLMClient(
        model=llm_model,
        api_key=anthropic_api_key,
        temperature=llm_temperature,
        max_tokens=llm_max_tokens,
    )

    # Agents
    sentiment_agent = SentimentAgent(client=llm_client, max_retries=llm_max_retries)
    market_agent = MarketAgent(client=llm_client, max_retries=llm_max_retries)
    proposer_agent = ProposerAgent(client=llm_client, max_retries=llm_max_retries)

    # Exchange
    exchange_client = ExchangeClient(exchange_id=exchange_id)
    data_fetcher = DataFetcher(exchange_client)

    # Repositories
    pipeline_repo = PipelineRepository(session)
    llm_call_repo = LLMCallRepository(session)
    proposal_repo = TradeProposalRepository(session)

    # Pipeline
    runner = PipelineRunner(
        data_fetcher=data_fetcher,
        sentiment_agent=sentiment_agent,
        market_agent=market_agent,
        proposer_agent=proposer_agent,
        pipeline_repo=pipeline_repo,
        llm_call_repo=llm_call_repo,
        proposal_repo=proposal_repo,
    )

    symbols = pipeline_symbols or ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    scheduler = PipelineScheduler(
        runner=runner,
        symbols=symbols,
        interval_minutes=pipeline_interval_minutes,
        premium_model=llm_model_premium,
    )

    # Telegram
    bot = SentinelBot(
        token=telegram_bot_token,
        admin_chat_ids=telegram_admin_chat_ids,
        premium_model=llm_model_premium,
    )
    bot.set_scheduler(scheduler)

    return {
        "bot": bot,
        "exchange_client": exchange_client,
        "db_engine": db_engine,
        "scheduler": scheduler,
        "runner": runner,
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
        llm_model=settings.llm_model,
        llm_model_premium=settings.llm_model_premium,
        llm_temperature=settings.llm_temperature,
        llm_max_tokens=settings.llm_max_tokens,
        llm_max_retries=settings.llm_max_retries,
        pipeline_symbols=settings.pipeline_symbols,
        pipeline_interval_minutes=settings.pipeline_interval_minutes,
    )

    # Start scheduler
    components["scheduler"].start()

    # Start bot (blocking)
    app = components["bot"].build()
    logger.info("bot_ready", admin_ids=settings.telegram_admin_chat_ids)
    app.run_polling()


if __name__ == "__main__":
    main()
