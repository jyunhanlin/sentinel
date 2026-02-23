from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.pipeline.scheduler import PipelineScheduler


class TestPipelineScheduler:
    def test_create_scheduler(self):
        mock_runner = AsyncMock()
        scheduler = PipelineScheduler(
            runner=mock_runner,
            symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
            interval_minutes=15,
        )
        assert scheduler.symbols == ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        assert scheduler.interval_minutes == 15

    @pytest.mark.asyncio
    async def test_run_once_all_symbols(self):
        mock_runner = AsyncMock()
        mock_runner.execute.return_value = MagicMock(status="completed")

        scheduler = PipelineScheduler(
            runner=mock_runner,
            symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
            interval_minutes=15,
        )

        results = await scheduler.run_once()

        assert len(results) == 2
        assert mock_runner.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_run_once_single_symbol(self):
        mock_runner = AsyncMock()
        mock_runner.execute.return_value = MagicMock(status="completed")

        scheduler = PipelineScheduler(
            runner=mock_runner,
            symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
            interval_minutes=15,
        )

        results = await scheduler.run_once(symbols=["BTC/USDT:USDT"])

        assert len(results) == 1
        mock_runner.execute.assert_called_once_with("BTC/USDT:USDT", model_override=None)

    @pytest.mark.asyncio
    async def test_run_once_with_model_override(self):
        mock_runner = AsyncMock()
        mock_runner.execute.return_value = MagicMock(status="completed")

        scheduler = PipelineScheduler(
            runner=mock_runner,
            symbols=["BTC/USDT:USDT"],
            interval_minutes=15,
            premium_model="anthropic/claude-opus-4-6",
        )

        results = await scheduler.run_once(
            symbols=["BTC/USDT:USDT"],
            model_override="anthropic/claude-opus-4-6",
        )

        assert len(results) == 1
        mock_runner.execute.assert_called_once_with(
            "BTC/USDT:USDT", model_override="anthropic/claude-opus-4-6"
        )

    @pytest.mark.asyncio
    async def test_daily_premium_uses_premium_model(self):
        mock_runner = AsyncMock()
        mock_runner.execute.return_value = MagicMock(status="completed")

        scheduler = PipelineScheduler(
            runner=mock_runner,
            symbols=["BTC/USDT:USDT"],
            interval_minutes=15,
            premium_model="anthropic/claude-opus-4-6",
        )

        await scheduler._run_daily_premium()

        mock_runner.execute.assert_called_once_with(
            "BTC/USDT:USDT", model_override="anthropic/claude-opus-4-6"
        )


class TestSchedulerExpiry:
    @pytest.mark.asyncio
    async def test_expire_stale_approvals(self):
        runner = MagicMock()
        approval_mgr = MagicMock()
        approval_mgr.expire_stale.return_value = []

        scheduler = PipelineScheduler(
            runner=runner,
            symbols=["BTC/USDT:USDT"],
            interval_minutes=15,
            approval_manager=approval_mgr,
        )
        await scheduler._expire_stale_approvals()
        approval_mgr.expire_stale.assert_called_once()

    @pytest.mark.asyncio
    async def test_expire_without_approval_manager(self):
        runner = MagicMock()
        scheduler = PipelineScheduler(
            runner=runner,
            symbols=["BTC/USDT:USDT"],
            interval_minutes=15,
        )
        # Should not raise
        await scheduler._expire_stale_approvals()


class TestSchedulerLifecycle:
    def test_start_creates_and_starts_scheduler(self):
        runner = MagicMock()
        scheduler = PipelineScheduler(
            runner=runner, symbols=["BTC/USDT:USDT"], interval_minutes=15,
        )
        with patch("orchestrator.pipeline.scheduler.AsyncIOScheduler") as mock_cls:
            scheduler.start()
            mock_cls.return_value.start.assert_called_once()
            mock_cls.return_value.add_job.assert_called_once()

    def test_start_with_premium_model_adds_two_jobs(self):
        runner = MagicMock()
        scheduler = PipelineScheduler(
            runner=runner, symbols=["BTC/USDT:USDT"],
            interval_minutes=15, premium_model="anthropic/claude-opus-4-6",
        )
        with patch("orchestrator.pipeline.scheduler.AsyncIOScheduler") as mock_cls:
            scheduler.start()
            assert mock_cls.return_value.add_job.call_count == 2

    def test_stop_shuts_down_scheduler(self):
        runner = MagicMock()
        scheduler = PipelineScheduler(
            runner=runner, symbols=["BTC/USDT:USDT"], interval_minutes=15,
        )
        with patch("orchestrator.pipeline.scheduler.AsyncIOScheduler") as mock_cls:
            scheduler.start()
            scheduler.stop()
            mock_cls.return_value.shutdown.assert_called_once_with(wait=False)

    def test_stop_without_start_is_safe(self):
        runner = MagicMock()
        scheduler = PipelineScheduler(
            runner=runner, symbols=["BTC/USDT:USDT"], interval_minutes=15,
        )
        scheduler.stop()  # Should not raise


class TestPriceMonitorJob:
    def test_start_adds_price_monitor_job(self):
        runner = MagicMock()
        monitor = AsyncMock()
        scheduler = PipelineScheduler(
            runner=runner,
            symbols=["BTC/USDT:USDT"],
            price_monitor=monitor,
            price_monitor_interval_seconds=30,
        )
        with patch("orchestrator.pipeline.scheduler.AsyncIOScheduler") as mock_sched:
            mock_instance = MagicMock()
            mock_sched.return_value = mock_instance
            scheduler.start()

            job_ids = [call.kwargs["id"] for call in mock_instance.add_job.call_args_list]
            assert "price_monitor" in job_ids

    def test_start_skips_monitor_when_none(self):
        runner = MagicMock()
        scheduler = PipelineScheduler(
            runner=runner,
            symbols=["BTC/USDT:USDT"],
        )
        with patch("orchestrator.pipeline.scheduler.AsyncIOScheduler") as mock_sched:
            mock_instance = MagicMock()
            mock_sched.return_value = mock_instance
            scheduler.start()

            job_ids = [call.kwargs.get("id", "") for call in mock_instance.add_job.call_args_list]
            assert "price_monitor" not in job_ids
