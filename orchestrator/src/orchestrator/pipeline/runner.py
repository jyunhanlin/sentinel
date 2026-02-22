from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field

from orchestrator.agents.base import AgentResult, BaseAgent
from orchestrator.exchange.data_fetcher import DataFetcher
from orchestrator.exchange.paper_engine import CloseResult
from orchestrator.models import MarketInterpretation, SentimentReport, Side, TradeProposal
from orchestrator.pipeline.aggregator import aggregate_proposal
from orchestrator.risk.checker import RiskResult
from orchestrator.storage.repository import (
    LLMCallRepository,
    PipelineRepository,
    TradeProposalRepository,
)

if TYPE_CHECKING:
    from orchestrator.approval.manager import ApprovalManager
    from orchestrator.exchange.paper_engine import PaperEngine
    from orchestrator.risk.checker import RiskChecker

logger = structlog.get_logger(__name__)


class PipelineResult(BaseModel, frozen=True):
    run_id: str
    symbol: str
    status: str  # completed, rejected, risk_rejected, risk_paused, pending_approval, failed
    model_used: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    proposal: TradeProposal | None = None
    rejection_reason: str = ""
    approval_id: str | None = None
    sentiment_degraded: bool = False
    market_degraded: bool = False
    proposer_degraded: bool = False
    risk_result: RiskResult | None = None
    close_results: list[CloseResult] = []


class PipelineRunner:
    def __init__(
        self,
        *,
        data_fetcher: DataFetcher,
        sentiment_agent: BaseAgent[SentimentReport],
        market_agent: BaseAgent[MarketInterpretation],
        proposer_agent: BaseAgent[TradeProposal],
        pipeline_repo: PipelineRepository,
        llm_call_repo: LLMCallRepository,
        proposal_repo: TradeProposalRepository,
        risk_checker: RiskChecker | None = None,
        paper_engine: PaperEngine | None = None,
        approval_manager: ApprovalManager | None = None,
    ) -> None:
        self._data_fetcher = data_fetcher
        self._sentiment_agent = sentiment_agent
        self._market_agent = market_agent
        self._proposer_agent = proposer_agent
        self._pipeline_repo = pipeline_repo
        self._llm_call_repo = llm_call_repo
        self._proposal_repo = proposal_repo
        self._risk_checker = risk_checker
        self._paper_engine = paper_engine
        self._approval_manager = approval_manager

    async def execute(
        self, symbol: str, *, timeframe: str = "1h", model_override: str | None = None
    ) -> PipelineResult:
        run_id = str(uuid.uuid4())
        log = logger.bind(run_id=run_id, symbol=symbol, model_override=model_override)
        log.info("pipeline_start")

        self._pipeline_repo.create_run(run_id=run_id, symbol=symbol)

        try:
            # Step 1: Fetch market data
            snapshot = await self._data_fetcher.fetch_snapshot(symbol, timeframe=timeframe)
            log.info("snapshot_fetched", price=snapshot.current_price)

            # Step 2: Check SL/TP on existing positions
            close_results: list[CloseResult] = []
            if self._paper_engine is not None:
                close_results = self._paper_engine.check_sl_tp(
                    symbol=symbol, current_price=snapshot.current_price
                )
                for cr in close_results:
                    log.info("position_closed_sltp", trade_id=cr.trade_id, reason=cr.reason)

            # Step 3: Run LLM-1 and LLM-2 in parallel
            sentiment_result, market_result = await asyncio.gather(
                self._sentiment_agent.analyze(snapshot=snapshot, model_override=model_override),
                self._market_agent.analyze(snapshot=snapshot, model_override=model_override),
            )

            self._save_llm_calls(run_id, "sentiment", sentiment_result)
            self._save_llm_calls(run_id, "market", market_result)

            # Step 4: Run LLM-3 (depends on LLM-1 + LLM-2)
            proposer_result = await self._proposer_agent.analyze(
                snapshot=snapshot,
                sentiment=sentiment_result.output,
                market=market_result.output,
                model_override=model_override,
            )
            self._save_llm_calls(run_id, "proposer", proposer_result)

            # Step 5: Validate proposal
            aggregation = aggregate_proposal(
                proposer_result.output, current_price=snapshot.current_price
            )

            model_used = model_override or ""
            if proposer_result.llm_calls:
                model_used = proposer_result.llm_calls[-1].model

            if not aggregation.valid:
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
                    close_results=close_results,
                )

            # Step 6: Risk check
            risk_result: RiskResult | None = None
            if self._risk_checker is not None and self._paper_engine is not None:
                from datetime import UTC, datetime

                paper_trade_repo = self._paper_engine._trade_repo
                daily_pnl = paper_trade_repo.get_daily_pnl(datetime.now(UTC).date())
                daily_loss_pct = (
                    abs(daily_pnl) / self._paper_engine.equity * 100
                    if daily_pnl < 0
                    else 0.0
                )

                risk_result = self._risk_checker.check(
                    proposal=aggregation.proposal,
                    open_positions_risk_pct=self._paper_engine.open_positions_risk_pct,
                    consecutive_losses=paper_trade_repo.count_consecutive_losses(),
                    daily_loss_pct=daily_loss_pct,
                )

                if not risk_result.approved:
                    status = "risk_paused" if risk_result.action == "pause" else "risk_rejected"
                    if risk_result.action == "pause":
                        self._paper_engine.set_paused(True)

                    self._proposal_repo.save_proposal(
                        proposal_id=aggregation.proposal.proposal_id,
                        run_id=run_id,
                        proposal_json=aggregation.proposal.model_dump_json(),
                        risk_check_result=status,
                        risk_check_reason=risk_result.reason,
                    )
                    self._pipeline_repo.update_run_status(run_id, status)
                    log.warning(
                        "pipeline_risk_blocked", status=status, rule=risk_result.rule_violated
                    )

                    return PipelineResult(
                        run_id=run_id,
                        symbol=symbol,
                        status=status,
                        model_used=model_used,
                        proposal=aggregation.proposal,
                        rejection_reason=risk_result.reason,
                        sentiment_degraded=sentiment_result.degraded,
                        market_degraded=market_result.degraded,
                        proposer_degraded=proposer_result.degraded,
                        risk_result=risk_result,
                        close_results=close_results,
                    )

                # Step 7: Open position or create pending approval
                if aggregation.proposal.side != Side.FLAT:
                    if self._approval_manager is not None:
                        # Semi-auto: create pending approval
                        approval = self._approval_manager.create(
                            proposal=aggregation.proposal,
                            run_id=run_id,
                            snapshot_price=snapshot.current_price,
                            model_used=model_used,
                        )
                        self._proposal_repo.save_proposal(
                            proposal_id=aggregation.proposal.proposal_id,
                            run_id=run_id,
                            proposal_json=aggregation.proposal.model_dump_json(),
                            risk_check_result="pending_approval",
                        )
                        self._pipeline_repo.update_run_status(run_id, "pending_approval")
                        log.info("pipeline_pending_approval", approval_id=approval.approval_id)
                        return PipelineResult(
                            run_id=run_id,
                            symbol=symbol,
                            status="pending_approval",
                            model_used=model_used,
                            proposal=aggregation.proposal,
                            risk_result=risk_result,
                            approval_id=approval.approval_id,
                            sentiment_degraded=sentiment_result.degraded,
                            market_degraded=market_result.degraded,
                            proposer_degraded=proposer_result.degraded,
                            close_results=close_results,
                        )
                    else:
                        # Auto mode: execute immediately
                        self._paper_engine.open_position(
                            aggregation.proposal, current_price=snapshot.current_price
                        )

            # Step 8: Save approved proposal and return
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
                risk_result=risk_result,
                close_results=close_results,
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

    def _save_llm_calls(self, run_id: str, agent_type: str, result: AgentResult[BaseModel]) -> None:
        import json

        prompt_json = json.dumps(result.messages, ensure_ascii=False) if result.messages else ""
        for call in result.llm_calls:
            self._llm_call_repo.save_call(
                run_id=run_id,
                agent_type=agent_type,
                prompt=prompt_json,
                response=call.content,
                model=call.model,
                latency_ms=call.latency_ms,
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
            )
