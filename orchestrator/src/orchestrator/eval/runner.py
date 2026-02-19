from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel

from orchestrator.eval.dataset import EvalCase
from orchestrator.eval.scorers import RuleScorer, ScoreResult
from orchestrator.exchange.data_fetcher import MarketSnapshot

if TYPE_CHECKING:
    from orchestrator.agents.base import BaseAgent

logger = structlog.get_logger(__name__)


class CaseResult(BaseModel, frozen=True):
    case_id: str
    passed: bool
    scores: list[ScoreResult]
    consistency: float | None = None


class EvalReport(BaseModel, frozen=True):
    dataset_name: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    accuracy: float
    consistency_score: float | None = None
    case_results: list[CaseResult]


class EvalRunner:
    def __init__(
        self,
        *,
        sentiment_agent: BaseAgent,
        market_agent: BaseAgent,
        proposer_agent: BaseAgent,
    ) -> None:
        self._sentiment_agent = sentiment_agent
        self._market_agent = market_agent
        self._proposer_agent = proposer_agent
        self._scorer = RuleScorer()

    async def run(
        self, *, cases: list[EvalCase], dataset_name: str
    ) -> EvalReport:
        case_results: list[CaseResult] = []

        for case in cases:
            logger.info("eval_case_start", case_id=case.id)
            result = await self._evaluate_case(case)
            case_results.append(result)
            logger.info("eval_case_done", case_id=case.id, passed=result.passed)

        passed = sum(1 for r in case_results if r.passed)
        total = len(case_results)

        return EvalReport(
            dataset_name=dataset_name,
            total_cases=total,
            passed_cases=passed,
            failed_cases=total - passed,
            accuracy=passed / total if total > 0 else 0.0,
            case_results=case_results,
        )

    async def _evaluate_case(self, case: EvalCase) -> CaseResult:
        snapshot = self._build_snapshot(case.snapshot)
        all_scores: list[ScoreResult] = []

        # Run agents
        sentiment_result = await self._sentiment_agent.analyze(snapshot=snapshot)
        market_result = await self._market_agent.analyze(snapshot=snapshot)
        proposer_result = await self._proposer_agent.analyze(
            snapshot=snapshot,
            sentiment=sentiment_result.output,
            market=market_result.output,
        )

        # Score each agent output
        if case.expected.sentiment is not None:
            all_scores.extend(
                self._scorer.score_sentiment(sentiment_result.output, case.expected.sentiment)
            )

        if case.expected.market is not None:
            all_scores.extend(
                self._scorer.score_market(market_result.output, case.expected.market)
            )

        if case.expected.proposal is not None:
            all_scores.extend(
                self._scorer.score_proposal(proposer_result.output, case.expected.proposal)
            )

        passed = all(s.passed for s in all_scores) if all_scores else True

        return CaseResult(case_id=case.id, passed=passed, scores=all_scores)

    def _build_snapshot(self, raw: dict) -> MarketSnapshot:
        return MarketSnapshot(
            symbol=raw.get("symbol", "BTC/USDT:USDT"),
            timeframe=raw.get("timeframe", "4h"),
            current_price=raw.get("current_price", 0.0),
            volume_24h=raw.get("volume_24h", 0.0),
            funding_rate=raw.get("funding_rate", 0.0),
            ohlcv=raw.get("ohlcv", []),
        )
