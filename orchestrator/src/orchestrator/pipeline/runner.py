from __future__ import annotations

import asyncio
import uuid

import structlog
from pydantic import BaseModel

from orchestrator.agents.base import AgentResult, BaseAgent
from orchestrator.exchange.data_fetcher import DataFetcher
from orchestrator.models import MarketInterpretation, SentimentReport, TradeProposal
from orchestrator.pipeline.aggregator import aggregate_proposal
from orchestrator.storage.repository import (
    LLMCallRepository,
    PipelineRepository,
    TradeProposalRepository,
)

logger = structlog.get_logger(__name__)


class PipelineResult(BaseModel, frozen=True):
    run_id: str
    symbol: str
    status: str  # completed, rejected, failed
    model_used: str = ""
    proposal: TradeProposal | None = None
    rejection_reason: str = ""
    sentiment_degraded: bool = False
    market_degraded: bool = False
    proposer_degraded: bool = False


class PipelineRunner:
    def __init__(
        self,
        *,
        data_fetcher: DataFetcher,
        sentiment_agent: BaseAgent,
        market_agent: BaseAgent,
        proposer_agent: BaseAgent,
        pipeline_repo: PipelineRepository,
        llm_call_repo: LLMCallRepository,
        proposal_repo: TradeProposalRepository,
    ) -> None:
        self._data_fetcher = data_fetcher
        self._sentiment_agent = sentiment_agent
        self._market_agent = market_agent
        self._proposer_agent = proposer_agent
        self._pipeline_repo = pipeline_repo
        self._llm_call_repo = llm_call_repo
        self._proposal_repo = proposal_repo

    async def execute(
        self, symbol: str, *, timeframe: str = "1h", model_override: str | None = None
    ) -> PipelineResult:
        run_id = str(uuid.uuid4())
        log = logger.bind(run_id=run_id, symbol=symbol, model_override=model_override)
        log.info("pipeline_start")

        self._pipeline_repo.create_run(run_id=run_id, symbol=symbol)

        try:
            # Fetch market data
            snapshot = await self._data_fetcher.fetch_snapshot(symbol, timeframe=timeframe)
            log.info("snapshot_fetched", price=snapshot.current_price)

            # Run LLM-1 and LLM-2 in parallel
            sentiment_result, market_result = await asyncio.gather(
                self._sentiment_agent.analyze(snapshot=snapshot, model_override=model_override),
                self._market_agent.analyze(snapshot=snapshot, model_override=model_override),
            )

            self._save_llm_calls(run_id, "sentiment", sentiment_result)
            self._save_llm_calls(run_id, "market", market_result)

            # Run LLM-3 (depends on LLM-1 + LLM-2)
            proposer_result = await self._proposer_agent.analyze(
                snapshot=snapshot,
                sentiment=sentiment_result.output,
                market=market_result.output,
                model_override=model_override,
            )
            self._save_llm_calls(run_id, "proposer", proposer_result)

            # Validate proposal
            aggregation = aggregate_proposal(
                proposer_result.output, current_price=snapshot.current_price
            )

            model_used = model_override or ""

            if aggregation.valid:
                self._proposal_repo.save_proposal(
                    proposal_id=aggregation.proposal.proposal_id,
                    run_id=run_id,
                    proposal_json=aggregation.proposal.model_dump_json(),
                    risk_check_result="approved",
                )
                self._pipeline_repo.update_run_status(run_id, "completed")
                log.info("pipeline_completed", side=aggregation.proposal.side)

                return PipelineResult(
                    run_id=run_id,
                    symbol=symbol,
                    status="completed",
                    model_used=model_used,
                    proposal=aggregation.proposal,
                    sentiment_degraded=sentiment_result.degraded,
                    market_degraded=market_result.degraded,
                    proposer_degraded=proposer_result.degraded,
                )
            else:
                self._proposal_repo.save_proposal(
                    proposal_id=aggregation.proposal.proposal_id,
                    run_id=run_id,
                    proposal_json=aggregation.proposal.model_dump_json(),
                    risk_check_result="rejected",
                    risk_check_reason=aggregation.rejection_reason,
                )
                self._pipeline_repo.update_run_status(run_id, "rejected")
                log.warning("pipeline_rejected", reason=aggregation.rejection_reason)

                return PipelineResult(
                    run_id=run_id,
                    symbol=symbol,
                    status="rejected",
                    model_used=model_used,
                    proposal=aggregation.proposal,
                    rejection_reason=aggregation.rejection_reason,
                    sentiment_degraded=sentiment_result.degraded,
                    market_degraded=market_result.degraded,
                    proposer_degraded=proposer_result.degraded,
                )

        except Exception as e:
            log.error("pipeline_failed", error=str(e))
            self._pipeline_repo.update_run_status(run_id, "failed")
            return PipelineResult(
                run_id=run_id,
                symbol=symbol,
                status="failed",
                rejection_reason=str(e),
            )

    def _save_llm_calls(self, run_id: str, agent_type: str, result: AgentResult) -> None:
        for call in result.llm_calls:
            self._llm_call_repo.save_call(
                run_id=run_id,
                agent_type=agent_type,
                prompt="(see messages)",
                response=call.content,
                model=call.model,
                latency_ms=call.latency_ms,
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
            )
