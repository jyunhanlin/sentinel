from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import structlog

from orchestrator.eval.dataset import EvalCase
from orchestrator.exchange.data_fetcher import MarketSnapshot

if TYPE_CHECKING:
    from orchestrator.agents.base import BaseAgent
    from orchestrator.models import TradeProposal

logger = structlog.get_logger(__name__)


class ConsistencyChecker:
    def __init__(self, *, proposer_agent: BaseAgent[TradeProposal]) -> None:
        self._proposer_agent = proposer_agent

    async def check(self, case: EvalCase, *, runs: int = 3) -> float:
        snapshot = MarketSnapshot(
            symbol=case.snapshot.get("symbol", "BTC/USDT:USDT"),
            timeframe=case.snapshot.get("timeframe", "4h"),
            current_price=case.snapshot.get("current_price", 0.0),
            volume_24h=case.snapshot.get("volume_24h", 0.0),
            funding_rate=case.snapshot.get("funding_rate", 0.0),
            ohlcv=case.snapshot.get("ohlcv", []),
        )

        sides: list[str] = []
        for i in range(runs):
            result = await self._proposer_agent.analyze(
                snapshot=snapshot,
                sentiment=None,
                market=None,
            )
            sides.append(result.output.side.value)
            logger.info("consistency_run", case_id=case.id, run=i + 1, side=sides[-1])

        if not sides:
            return 0.0

        counter = Counter(sides)
        most_common_count = counter.most_common(1)[0][1]
        return most_common_count / len(sides)
