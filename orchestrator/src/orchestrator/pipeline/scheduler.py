from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from orchestrator.pipeline.runner import PipelineResult, PipelineRunner

if TYPE_CHECKING:
    from orchestrator.approval.manager import ApprovalManager
    from orchestrator.exchange.price_monitor import PriceMonitor

logger = structlog.get_logger(__name__)

# Callback type: receives a PipelineResult, pushes it somewhere
ResultCallback = Callable[[PipelineResult], Awaitable[None]]


class PipelineScheduler:
    def __init__(
        self,
        *,
        runner: PipelineRunner,
        symbols: list[str],
        interval_minutes: int = 15,
        default_model: str = "",
        premium_model: str = "",
        approval_manager: ApprovalManager | None = None,
        on_result: ResultCallback | None = None,
        price_monitor: PriceMonitor | None = None,
        price_monitor_interval_seconds: int = 60,
    ) -> None:
        self.symbols = symbols
        self.interval_minutes = interval_minutes
        self._runner_default_model = default_model
        self.premium_model = premium_model
        self._runner = runner
        self._approval_manager = approval_manager
        self._on_result = on_result
        self._price_monitor = price_monitor
        self._price_monitor_interval = price_monitor_interval_seconds
        self._scheduler: AsyncIOScheduler | None = None

    async def run_once(
        self,
        *,
        symbols: list[str] | None = None,
        model_override: str | None = None,
        source: str = "scheduler",
        notify: bool = True,
    ) -> list[PipelineResult]:
        target_symbols = symbols or self.symbols
        effective_model = model_override or self._runner_default_model or "default"

        async def _run_symbol(symbol: str) -> PipelineResult:
            logger.info("pipeline_running", symbol=symbol, model=effective_model, source=source)
            result = await self._runner.execute(symbol, model_override=model_override)
            logger.info(
                "pipeline_done", symbol=symbol,
                model=effective_model, status=result.status, source=source,
            )
            if notify and self._on_result is not None:
                try:
                    await self._on_result(result)
                except Exception:
                    logger.exception("on_result_callback_failed", symbol=symbol)
            return result

        results = await asyncio.gather(*[_run_symbol(s) for s in target_symbols])
        return list(results)

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

        # Approval expiry check
        if self._approval_manager is not None:
            self._scheduler.add_job(
                self._expire_stale_approvals,
                trigger=IntervalTrigger(minutes=1),
                id="approval_expiry",
                name="Approval Expiry Check",
                replace_existing=True,
            )

        # Price monitor — lightweight SL/TP check
        if self._price_monitor is not None:
            self._scheduler.add_job(
                self._price_monitor.check,
                trigger=IntervalTrigger(seconds=self._price_monitor_interval),
                id="price_monitor",
                name="Price Monitor (SL/TP/Liq)",
                replace_existing=True,
            )

        self._scheduler.start()
        logger.info(
            "scheduler_started",
            interval_minutes=self.interval_minutes,
            premium_model=self.premium_model or "(none)",
            symbols=self.symbols,
        )

    async def _expire_stale_approvals(self) -> None:
        """Expire stale pending approvals."""
        if self._approval_manager is None:
            return
        expired = self._approval_manager.expire_stale()
        if expired:
            logger.info("approvals_expired", count=len(expired))

    def stop(self) -> None:
        if self._scheduler:
            self._scheduler.shutdown(wait=False)
            logger.info("scheduler_stopped")
