from unittest.mock import AsyncMock, MagicMock

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
