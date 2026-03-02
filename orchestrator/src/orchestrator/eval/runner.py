from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel

from orchestrator.eval.dataset import EvalCase, load_dataset
from orchestrator.eval.scorers import RuleScorer, ScoreResult
from orchestrator.exchange.data_fetcher import MarketSnapshot

if TYPE_CHECKING:
    from orchestrator.agents.base import BaseAgent
    from orchestrator.models import (
        CatalystReport,
        CorrelationAnalysis,
        PositioningAnalysis,
        TechnicalAnalysis,
        TradeProposal,
    )

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
        technical_short_agent: BaseAgent[TechnicalAnalysis],
        technical_long_agent: BaseAgent[TechnicalAnalysis],
        positioning_agent: BaseAgent[PositioningAnalysis],
        catalyst_agent: BaseAgent[CatalystReport],
        correlation_agent: BaseAgent[CorrelationAnalysis],
        proposer_agent: BaseAgent[TradeProposal],
    ) -> None:
        self._technical_short_agent = technical_short_agent
        self._technical_long_agent = technical_long_agent
        self._positioning_agent = positioning_agent
        self._catalyst_agent = catalyst_agent
        self._correlation_agent = correlation_agent
        self._proposer_agent = proposer_agent
        self._scorer = RuleScorer()

    async def run_default(self) -> EvalReport:
        """Run evaluation using the built-in golden_v1 dataset."""
        dataset_dir = os.path.join(os.path.dirname(__file__), "datasets")
        dataset_path = os.path.join(dataset_dir, "golden_v1.yaml")
        cases = load_dataset(dataset_path)
        return await self.run(cases=cases, dataset_name="golden_v1")

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

        # Run analysis agents
        tech_short_result = await self._technical_short_agent.analyze(snapshot=snapshot)
        tech_long_result = await self._technical_long_agent.analyze(snapshot=snapshot)
        positioning_result = await self._positioning_agent.analyze(
            symbol=snapshot.symbol, current_price=snapshot.current_price,
            funding_rate_history=[], open_interest=0, oi_change_pct=0.0,
            long_short_ratio=1.0, top_trader_long_short_ratio=1.0,
            order_book_summary={"bid_depth": 0, "ask_depth": 0},
        )
        catalyst_result = await self._catalyst_agent.analyze(
            symbol=snapshot.symbol, current_price=snapshot.current_price,
            economic_calendar=[], exchange_announcements=[],
        )
        correlation_result = await self._correlation_agent.analyze(
            symbol=snapshot.symbol,
            dxy_data={"current": 0, "change_pct": 0, "trend_5d": []},
            sp500_data={"current": 0, "change_pct": 0, "trend_5d": []},
            btc_dominance={"current": 0, "change_7d": 0},
        )

        # Run proposer with all analysis outputs
        proposer_result = await self._proposer_agent.analyze(
            snapshot=snapshot,
            technical_short=tech_short_result.output,
            technical_long=tech_long_result.output,
            positioning=positioning_result.output,
            catalyst=catalyst_result.output,
            correlation=correlation_result.output,
        )

        # Score technical analysis (maps to legacy "market" expectations)
        if case.expected.market is not None:
            all_scores.extend(
                self._scorer.score_market(tech_short_result.output, case.expected.market)
            )

        # Score sentiment (deprecated — skip with warning)
        if case.expected.sentiment is not None:
            logger.warning("eval_sentiment_deprecated", case_id=case.id)

        if case.expected.proposal is not None:
            all_scores.extend(
                self._scorer.score_proposal(proposer_result.output, case.expected.proposal)
            )

        passed = all(s.passed for s in all_scores) if all_scores else True

        return CaseResult(case_id=case.id, passed=passed, scores=all_scores)

    def _build_snapshot(self, raw: dict[str, Any]) -> MarketSnapshot:
        return MarketSnapshot(
            symbol=raw.get("symbol", "BTC/USDT:USDT"),
            timeframe=raw.get("timeframe", "4h"),
            current_price=raw.get("current_price", 0.0),
            volume_24h=raw.get("volume_24h", 0.0),
            funding_rate=raw.get("funding_rate", 0.0),
            ohlcv=raw.get("ohlcv", []),
        )
