from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.eval.consistency import ConsistencyChecker
from orchestrator.eval.dataset import EvalCase, ExpectedOutputs
from orchestrator.models import (
    EntryOrder,
    Side,
    TakeProfit,
    TradeProposal,
)


class TestConsistencyChecker:
    @pytest.mark.asyncio
    async def test_fully_consistent(self):
        """All runs produce same side -> consistency 1.0."""
        case = EvalCase(
            id="test", description="test",
            snapshot={"symbol": "BTC/USDT:USDT", "current_price": 95000.0, "ohlcv": []},
            expected=ExpectedOutputs(),
        )

        proposer = AsyncMock()
        proposer.analyze.return_value = MagicMock(
            output=TradeProposal(
                symbol="BTC/USDT:USDT", side=Side.LONG, entry=EntryOrder(type="market"),
                position_size_risk_pct=1.0, stop_loss=93000.0,
                take_profit=[TakeProfit(price=97000.0, close_pct=100)],
                time_horizon="4h", confidence=0.7, invalid_if=[], rationale="test",
            ),
        )

        checker = ConsistencyChecker(proposer_agent=proposer)
        score = await checker.check(case, runs=3)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_partially_consistent(self):
        """2 out of 3 agree -> consistency 2/3."""
        case = EvalCase(
            id="test", description="test",
            snapshot={"symbol": "BTC/USDT:USDT", "current_price": 95000.0, "ohlcv": []},
            expected=ExpectedOutputs(),
        )

        proposer = AsyncMock()
        long_proposal = MagicMock()
        long_proposal.output = TradeProposal(
            symbol="BTC/USDT:USDT", side=Side.LONG, entry=EntryOrder(type="market"),
            position_size_risk_pct=1.0, stop_loss=93000.0,
            take_profit=[TakeProfit(price=97000.0, close_pct=100)],
            time_horizon="4h", confidence=0.7, invalid_if=[], rationale="test",
        )
        short_proposal = MagicMock()
        short_proposal.output = TradeProposal(
            symbol="BTC/USDT:USDT", side=Side.SHORT, entry=EntryOrder(type="market"),
            position_size_risk_pct=1.0, stop_loss=97000.0,
            take_profit=[TakeProfit(price=93000.0, close_pct=100)],
            time_horizon="4h", confidence=0.6, invalid_if=[], rationale="test",
        )
        proposer.analyze.side_effect = [long_proposal, long_proposal, short_proposal]

        checker = ConsistencyChecker(proposer_agent=proposer)
        score = await checker.check(case, runs=3)
        assert score == pytest.approx(2 / 3)
