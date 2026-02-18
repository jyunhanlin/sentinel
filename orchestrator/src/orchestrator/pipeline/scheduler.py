from __future__ import annotations

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from orchestrator.pipeline.runner import PipelineResult, PipelineRunner

logger = structlog.get_logger(__name__)


class PipelineScheduler:
    def __init__(
        self,
        *,
        runner: PipelineRunner,
        symbols: list[str],
        interval_minutes: int = 15,
        premium_model: str = "",
    ) -> None:
        self.symbols = symbols
        self.interval_minutes = interval_minutes
        self.premium_model = premium_model
        self._runner = runner
        self._scheduler: AsyncIOScheduler | None = None

    async def run_once(
        self,
        *,
        symbols: list[str] | None = None,
        model_override: str | None = None,
    ) -> list[PipelineResult]:
        target_symbols = symbols or self.symbols
        results = []
        for symbol in target_symbols:
            logger.info("scheduler_running_symbol", symbol=symbol, model_override=model_override)
            result = await self._runner.execute(symbol, model_override=model_override)
            results.append(result)
            logger.info(
                "scheduler_symbol_done",
                symbol=symbol,
                status=result.status,
            )
        return results

    async def _run_daily_premium(self) -> None:
        """Daily deep analysis using premium model (Opus)."""
        logger.info("daily_premium_start", model=self.premium_model)
        await self.run_once(model_override=self.premium_model)

    def start(self) -> None:
        self._scheduler = AsyncIOScheduler()

        # Regular interval — Sonnet (default model)
        self._scheduler.add_job(
            self.run_once,
            trigger=IntervalTrigger(minutes=self.interval_minutes),
            id="pipeline_interval",
            name="Pipeline Interval (Sonnet)",
            replace_existing=True,
        )

        # Daily deep analysis — Opus (premium model)
        if self.premium_model:
            self._scheduler.add_job(
                self._run_daily_premium,
                trigger=CronTrigger(hour=0, minute=0),  # 00:00 UTC
                id="pipeline_daily_premium",
                name="Pipeline Daily (Opus)",
                replace_existing=True,
            )

        self._scheduler.start()
        logger.info(
            "scheduler_started",
            interval_minutes=self.interval_minutes,
            premium_model=self.premium_model or "(none)",
            symbols=self.symbols,
        )

    def stop(self) -> None:
        if self._scheduler:
            self._scheduler.shutdown(wait=False)
            logger.info("scheduler_stopped")
