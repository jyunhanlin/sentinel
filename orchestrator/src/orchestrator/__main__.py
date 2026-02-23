from __future__ import annotations

import argparse
import asyncio
import signal
from typing import Any

import structlog
from sqlmodel import Session

from orchestrator.agents.market import MarketAgent
from orchestrator.agents.proposer import ProposerAgent
from orchestrator.agents.sentiment import SentimentAgent
from orchestrator.approval.manager import ApprovalManager
from orchestrator.config import Settings
from orchestrator.eval.runner import EvalRunner
from orchestrator.exchange.client import ExchangeClient
from orchestrator.exchange.data_fetcher import DataFetcher
from orchestrator.exchange.paper_engine import PaperEngine
from orchestrator.execution.executor import LiveExecutor, PaperExecutor
from orchestrator.llm.backend import ClaudeCLIBackend, LiteLLMBackend
from orchestrator.llm.client import LLMClient
from orchestrator.logging import setup_logging
from orchestrator.pipeline.runner import PipelineRunner
from orchestrator.pipeline.scheduler import PipelineScheduler
from orchestrator.risk.checker import RiskChecker
from orchestrator.risk.position_sizer import RiskPercentSizer
from orchestrator.stats.calculator import StatsCalculator
from orchestrator.storage.database import create_db_engine, init_db
from orchestrator.storage.repository import (
    AccountSnapshotRepository,
    ApprovalRepository,
    LLMCallRepository,
    PaperTradeRepository,
    PipelineRepository,
    TradeProposalRepository,
)
from orchestrator.telegram.bot import SentinelBot

logger = structlog.get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="orchestrator", description="Sentinel Orchestrator")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("eval", help="Run LLM evaluation against golden dataset")
    subparsers.add_parser("perf", help="Print performance report")
    return parser.parse_args(argv)


def create_app_components(
    *,
    telegram_bot_token: str,
    telegram_admin_chat_ids: list[int],
    exchange_id: str,
    database_url: str,
    anthropic_api_key: str = "",
    llm_model: str = "anthropic/claude-opus-4-6",
    llm_model_premium: str = "anthropic/claude-opus-4-6",
    llm_temperature: float = 0.2,
    llm_max_tokens: int = 2000,
    llm_max_retries: int = 1,
    pipeline_symbols: list[str] | None = None,
    pipeline_interval_minutes: int = 15,
    # Risk
    max_single_risk_pct: float = 2.0,
    max_total_exposure_pct: float = 20.0,
    max_daily_loss_pct: float = 5.0,
    max_consecutive_losses: int = 5,
    # Paper Trading
    paper_initial_equity: float = 10000.0,
    paper_taker_fee_rate: float = 0.0005,
    paper_maker_fee_rate: float = 0.0002,
    # LLM Backend
    llm_backend: str = "api",
    claude_cli_path: str = "claude",
    claude_cli_timeout: int = 120,
    # Semi-auto Trading
    trading_mode: str = "paper",
    approval_timeout_minutes: int = 15,
    price_deviation_threshold: float = 0.01,
    # Price Monitor
    price_monitor_interval_seconds: int = 60,
    price_monitor_enabled: bool = True,
) -> dict[str, Any]:
    # Database
    db_engine = create_db_engine(database_url)
    init_db(db_engine)
    session = Session(db_engine)

    # LLM
    if llm_backend == "cli":
        backend = ClaudeCLIBackend(cli_path=claude_cli_path, timeout=claude_cli_timeout)
    else:
        backend = LiteLLMBackend(api_key=anthropic_api_key)

    llm_client = LLMClient(
        backend=backend,
        model=llm_model,
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
    paper_trade_repo = PaperTradeRepository(session)
    account_snapshot_repo = AccountSnapshotRepository(session)

    # Risk
    risk_checker = RiskChecker(
        max_single_risk_pct=max_single_risk_pct,
        max_total_exposure_pct=max_total_exposure_pct,
        max_consecutive_losses=max_consecutive_losses,
        max_daily_loss_pct=max_daily_loss_pct,
    )

    # Stats
    stats_calculator = StatsCalculator()

    # Paper Engine
    paper_engine = PaperEngine(
        initial_equity=paper_initial_equity,
        taker_fee_rate=paper_taker_fee_rate,
        position_sizer=RiskPercentSizer(),
        trade_repo=paper_trade_repo,
        snapshot_repo=account_snapshot_repo,
        stats_calculator=stats_calculator,
    )
    paper_engine.rebuild_from_db()

    # Price Monitor
    from orchestrator.exchange.price_monitor import PriceMonitor

    price_monitor: PriceMonitor | None = None
    if price_monitor_enabled:
        price_monitor = PriceMonitor(
            paper_engine=paper_engine,
            data_fetcher=data_fetcher,
        )

    # Approval & Execution (M4)
    approval_repo = ApprovalRepository(session)
    approval_manager = ApprovalManager(
        repo=approval_repo, timeout_minutes=approval_timeout_minutes
    )

    position_sizer = RiskPercentSizer()
    if trading_mode == "live":
        executor = LiveExecutor(
            exchange_client=exchange_client,
            position_sizer=position_sizer,
            paper_engine=paper_engine,
            price_deviation_threshold=price_deviation_threshold,
        )
    else:
        executor = PaperExecutor(paper_engine=paper_engine)

    # Pipeline
    runner = PipelineRunner(
        data_fetcher=data_fetcher,
        sentiment_agent=sentiment_agent,
        market_agent=market_agent,
        proposer_agent=proposer_agent,
        pipeline_repo=pipeline_repo,
        llm_call_repo=llm_call_repo,
        proposal_repo=proposal_repo,
        risk_checker=risk_checker,
        paper_engine=paper_engine,
        approval_manager=approval_manager,
    )

    symbols = pipeline_symbols or ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    scheduler = PipelineScheduler(
        runner=runner,
        symbols=symbols,
        interval_minutes=pipeline_interval_minutes,
        default_model=llm_model,
        premium_model=llm_model_premium,
        approval_manager=approval_manager,
        price_monitor=price_monitor,
        price_monitor_interval_seconds=price_monitor_interval_seconds,
    )

    # Eval
    eval_runner = EvalRunner(
        sentiment_agent=sentiment_agent,
        market_agent=market_agent,
        proposer_agent=proposer_agent,
    )

    # Telegram
    bot = SentinelBot(
        token=telegram_bot_token,
        admin_chat_ids=telegram_admin_chat_ids,
        premium_model=llm_model_premium,
        scheduler=scheduler,
        paper_engine=paper_engine,
        trade_repo=paper_trade_repo,
        proposal_repo=proposal_repo,
        snapshot_repo=account_snapshot_repo,
        approval_manager=approval_manager,
        executor=executor,
        data_fetcher=data_fetcher,
        eval_runner=eval_runner,
        llm_client=llm_client,
    )

    return {
        "bot": bot,
        "exchange_client": exchange_client,
        "db_engine": db_engine,
        "scheduler": scheduler,
        "runner": runner,
        "risk_checker": risk_checker,
        "paper_engine": paper_engine,
        "stats_calculator": stats_calculator,
        "eval_runner": eval_runner,
        "snapshot_repo": account_snapshot_repo,
        "approval_manager": approval_manager,
        "executor": executor,
        "price_monitor": price_monitor,
    }


def _build_components(settings: Settings) -> dict[str, Any]:
    return create_app_components(
        telegram_bot_token=settings.telegram_bot_token,
        telegram_admin_chat_ids=settings.telegram_admin_chat_ids,
        exchange_id=settings.exchange_id,
        database_url=settings.database_url,
        anthropic_api_key=settings.anthropic_api_key,
        llm_backend=settings.llm_backend,
        claude_cli_path=settings.claude_cli_path,
        claude_cli_timeout=settings.claude_cli_timeout,
        llm_model=settings.llm_model,
        llm_model_premium=settings.llm_model_premium,
        llm_temperature=settings.llm_temperature,
        llm_max_tokens=settings.llm_max_tokens,
        llm_max_retries=settings.llm_max_retries,
        pipeline_symbols=settings.pipeline_symbols,
        pipeline_interval_minutes=settings.pipeline_interval_minutes,
        max_single_risk_pct=settings.max_single_risk_pct,
        max_total_exposure_pct=settings.max_total_exposure_pct,
        max_daily_loss_pct=settings.max_daily_loss_pct,
        max_consecutive_losses=settings.max_consecutive_losses,
        paper_initial_equity=settings.paper_initial_equity,
        paper_taker_fee_rate=settings.paper_taker_fee_rate,
        paper_maker_fee_rate=settings.paper_maker_fee_rate,
        trading_mode=settings.trading_mode,
        approval_timeout_minutes=settings.approval_timeout_minutes,
        price_deviation_threshold=settings.price_deviation_threshold,
        price_monitor_interval_seconds=settings.price_monitor_interval_seconds,
        price_monitor_enabled=settings.price_monitor_enabled,
    )


async def _run_eval(components: dict[str, Any]) -> None:
    eval_runner: EvalRunner = components["eval_runner"]
    report = await eval_runner.run_default()
    print(f"Dataset: {report.dataset_name}")
    print(
        f"Cases: {report.total_cases} | Passed: {report.passed_cases}"
        f" | Failed: {report.failed_cases}"
    )
    print(f"Accuracy: {report.accuracy:.0%}")
    for cr in report.case_results:
        status = "PASS" if cr.passed else "FAIL"
        print(f"  [{status}] {cr.case_id}")
        for s in cr.scores:
            if not s.passed:
                print(f"    {s.field}: {s.reason}")


def _run_perf(components: dict[str, Any]) -> None:
    from orchestrator.telegram.formatters import format_perf_report

    snapshot_repo = components.get("snapshot_repo")
    if snapshot_repo is None:
        print("Snapshot repo not available.")
        return
    snapshot = snapshot_repo.get_latest()
    if snapshot is None or snapshot.total_trades == 0:
        print("No performance data yet.")
        return
    from orchestrator.stats.calculator import PerformanceStats

    stats = PerformanceStats(
        total_pnl=snapshot.total_pnl,
        total_pnl_pct=(snapshot.total_pnl / snapshot.equity * 100) if snapshot.equity > 0 else 0.0,
        win_rate=snapshot.win_rate,
        total_trades=snapshot.total_trades,
        winning_trades=int(snapshot.win_rate * snapshot.total_trades),
        losing_trades=snapshot.total_trades - int(snapshot.win_rate * snapshot.total_trades),
        profit_factor=snapshot.profit_factor,
        max_drawdown_pct=snapshot.max_drawdown_pct,
        sharpe_ratio=snapshot.sharpe_ratio,
    )
    print(format_perf_report(stats))


async def _run_bot(components: dict[str, Any], settings: Settings) -> None:
    scheduler = components["scheduler"]
    bot = components["bot"]

    # Wire scheduler -> bot notification
    scheduler._on_result = bot.push_to_admins_with_approval

    # Wire price monitor -> bot notification
    price_monitor = components.get("price_monitor")
    if price_monitor is not None:
        price_monitor._on_close = bot.push_close_report

    app = bot.build()
    stop = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    async with app:
        await bot.register_commands()
        await app.start()
        scheduler.start()
        await app.updater.start_polling()
        logger.info("bot_ready", admin_ids=settings.telegram_admin_chat_ids)

        await stop.wait()

        logger.info("shutting_down")
        scheduler.stop()
        await app.updater.stop()
        await app.stop()


def main() -> None:
    args = parse_args()
    settings = Settings()  # type: ignore[call-arg]
    setup_logging(json_output=settings.log_json)
    logger.info("starting_sentinel", exchange=settings.exchange_id)

    components = _build_components(settings)

    if args.command == "eval":
        asyncio.run(_run_eval(components))
        return

    if args.command == "perf":
        _run_perf(components)
        return

    # Default: run bot + scheduler
    asyncio.run(_run_bot(components, settings))


if __name__ == "__main__":
    main()
