from __future__ import annotations

import pytest

from orchestrator.execution.equity import PaperEquityProvider


@pytest.mark.asyncio
async def test_get_equity_returns_engine_equity():
    class FakeEngine:
        @property
        def equity(self) -> float:
            return 10_000.0

    provider = PaperEquityProvider(engine=FakeEngine())
    assert await provider.get_equity() == 10_000.0


@pytest.mark.asyncio
async def test_get_available_margin_returns_engine_available():
    class FakeEngine:
        @property
        def equity(self) -> float:
            return 10_000.0

        @property
        def used_margin(self) -> float:
            return 1_000.0

    provider = PaperEquityProvider(engine=FakeEngine())
    assert await provider.get_available_margin() == 9_000.0
