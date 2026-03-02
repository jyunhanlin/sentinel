from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field

from orchestrator.agents.base import AgentResult, BaseAgent
from orchestrator.exchange.data_fetcher import DataFetcher
from orchestrator.exchange.external_data import ExternalDataFetcher
from orchestrator.exchange.paper_engine import CloseResult
from orchestrator.models import (
    CatalystReport,
    CorrelationAnalysis,
    MarketInterpretation,
    PositioningAnalysis,
    SentimentReport,
    Side,
    TechnicalAnalysis,
    TradeProposal,
)
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
    # Legacy degradation flags (kept for backward compat)
    sentiment_degraded: bool = False
    market_degraded: bool = False
    proposer_degraded: bool = False
    # New agent degradation flags
    technical_short_degraded: bool = False
    technical_long_degraded: bool = False
    positioning_degraded: bool = False
    catalyst_degraded: bool = False
    correlation_degraded: bool = False
    risk_result: RiskResult | None = None
    close_results: list[CloseResult] = []
    # Legacy analysis outputs (kept for backward compat)
    sentiment: SentimentReport | None = None
    market: MarketInterpretation | None = None
    # New analysis outputs
    technical_short: TechnicalAnalysis | None = None
    technical_long: TechnicalAnalysis | None = None
    positioning: PositioningAnalysis | None = None
    catalyst: CatalystReport | None = None
    correlation: CorrelationAnalysis | None = None


class PipelineRunner:
    def __init__(
        self,
        *,
        data_fetcher: DataFetcher,
        technical_short_agent: BaseAgent[TechnicalAnalysis],
        technical_long_agent: BaseAgent[TechnicalAnalysis],
        positioning_agent: BaseAgent[PositioningAnalysis],
        catalyst_agent: BaseAgent[CatalystReport],
        correlation_agent: BaseAgent[CorrelationAnalysis],
        proposer_agent: BaseAgent[TradeProposal],
        external_data_fetcher: ExternalDataFetcher,
        pipeline_repo: PipelineRepository,
        llm_call_repo: LLMCallRepository,
        proposal_repo: TradeProposalRepository,
        risk_checker: RiskChecker | None = None,
        paper_engine: PaperEngine | None = None,
        approval_manager: ApprovalManager | None = None,
    ) -> None:
        self._data_fetcher = data_fetcher
        self._technical_short_agent = technical_short_agent
        self._technical_long_agent = technical_long_agent
        self._positioning_agent = positioning_agent
        self._catalyst_agent = catalyst_agent
        self._correlation_agent = correlation_agent
        self._proposer_agent = proposer_agent
        self._external_data = external_data_fetcher
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
        structlog.contextvars.bind_contextvars(symbol=symbol)
        logger.info("pipeline_start", run_id=run_id)

        self._pipeline_repo.create_run(run_id=run_id, symbol=symbol)

        try:
            # Step 1: Fetch all data in parallel
            snapshot, positioning_data, macro_data, dxy_data, sp500_data, btc_dom, calendar, announcements = (
                await asyncio.gather(
                    self._data_fetcher.fetch_snapshot(symbol, timeframe=timeframe),
                    self._data_fetcher.fetch_positioning_data(symbol),
                    self._data_fetcher.fetch_macro_indicators(symbol),
                    self._external_data.fetch_dxy_data(),
                    self._external_data.fetch_sp500_data(),
                    self._external_data.fetch_btc_dominance(),
                    self._external_data.fetch_economic_calendar(),
                    self._external_data.fetch_exchange_announcements(),
                )
            )
            logger.info("snapshot_fetched", price=snapshot.current_price)

            # Step 2: Run 5 analysis agents in parallel
            tech_short_result, tech_long_result, positioning_result, catalyst_result, correlation_result = (
                await asyncio.gather(
                    self._technical_short_agent.analyze(
                        snapshot=snapshot, model_override=model_override,
                    ),
                    self._technical_long_agent.analyze(
                        snapshot=snapshot, macro_data=macro_data, model_override=model_override,
                    ),
                    self._positioning_agent.analyze(
                        symbol=symbol,
                        current_price=snapshot.current_price,
                        model_override=model_override,
                        **positioning_data,
                    ),
                    self._catalyst_agent.analyze(
                        symbol=symbol,
                        current_price=snapshot.current_price,
                        economic_calendar=calendar,
                        exchange_announcements=announcements,
                        model_override=model_override,
                    ),
                    self._correlation_agent.analyze(
                        symbol=symbol,
                        dxy_data=dxy_data,
                        sp500_data=sp500_data,
                        btc_dominance=btc_dom,
                        model_override=model_override,
                    ),
                )
            )

            self._save_llm_calls(run_id, "technical_short", tech_short_result)
            self._save_llm_calls(run_id, "technical_long", tech_long_result)
            self._save_llm_calls(run_id, "positioning", positioning_result)
            self._save_llm_calls(run_id, "catalyst", catalyst_result)
            self._save_llm_calls(run_id, "correlation", correlation_result)

            # Step 3: Run Proposer (depends on all 5 analysis outputs)
            proposer_result = await self._proposer_agent.analyze(
                snapshot=snapshot,
                technical_short=tech_short_result.output,
                technical_long=tech_long_result.output,
                positioning=positioning_result.output,
                catalyst=catalyst_result.output,
                correlation=correlation_result.output,
                model_override=model_override,
            )
            self._save_llm_calls(run_id, "proposer", proposer_result)

            # Step 4: Validate proposal
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
                logger.warning("pipeline_rejected", reason=aggregation.rejection_reason)

                return self._build_result(
                    run_id=run_id, symbol=symbol, status="rejected",
                    model_used=model_used, proposal=aggregation.proposal,
                    rejection_reason=aggregation.rejection_reason,
                    proposer_result=proposer_result,
                    tech_short_result=tech_short_result,
                    tech_long_result=tech_long_result,
                    positioning_result=positioning_result,
                    catalyst_result=catalyst_result,
                    correlation_result=correlation_result,
                )

            # Step 5: Risk check
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
                    logger.warning(
                        "pipeline_risk_blocked", status=status, rule=risk_result.rule_violated
                    )

                    return self._build_result(
                        run_id=run_id, symbol=symbol, status=status,
                        model_used=model_used, proposal=aggregation.proposal,
                        rejection_reason=risk_result.reason,
                        risk_result=risk_result,
                        proposer_result=proposer_result,
                        tech_short_result=tech_short_result,
                        tech_long_result=tech_long_result,
                        positioning_result=positioning_result,
                        catalyst_result=catalyst_result,
                        correlation_result=correlation_result,
                    )

                # Step 6: Open position or create pending approval
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
                        logger.info("pipeline_pending_approval", approval_id=approval.approval_id)
                        return self._build_result(
                            run_id=run_id, symbol=symbol, status="pending_approval",
                            model_used=model_used, proposal=aggregation.proposal,
                            risk_result=risk_result,
                            approval_id=approval.approval_id,
                            proposer_result=proposer_result,
                            tech_short_result=tech_short_result,
                            tech_long_result=tech_long_result,
                            positioning_result=positioning_result,
                            catalyst_result=catalyst_result,
                            correlation_result=correlation_result,
                        )
                    else:
                        # Auto mode: execute immediately
                        self._paper_engine.open_position(
                            aggregation.proposal, current_price=snapshot.current_price
                        )

            # Step 7: Save approved proposal and return
            self._proposal_repo.save_proposal(
                proposal_id=aggregation.proposal.proposal_id,
                run_id=run_id,
                proposal_json=aggregation.proposal.model_dump_json(),
                risk_check_result="approved",
            )
            self._pipeline_repo.update_run_status(run_id, "completed")
            logger.info("pipeline_completed", side=aggregation.proposal.side)

            return self._build_result(
                run_id=run_id, symbol=symbol, status="completed",
                model_used=model_used, proposal=aggregation.proposal,
                risk_result=risk_result,
                proposer_result=proposer_result,
                tech_short_result=tech_short_result,
                tech_long_result=tech_long_result,
                positioning_result=positioning_result,
                catalyst_result=catalyst_result,
                correlation_result=correlation_result,
            )

        except Exception as e:
            logger.error("pipeline_failed", error=str(e))
            self._pipeline_repo.update_run_status(run_id, "failed")
            return PipelineResult(
                run_id=run_id,
                symbol=symbol,
                status="failed",
                rejection_reason=str(e),
            )
        finally:
            structlog.contextvars.unbind_contextvars("symbol")

    @staticmethod
    def _build_result(
        *,
        run_id: str,
        symbol: str,
        status: str,
        model_used: str = "",
        proposal: TradeProposal | None = None,
        rejection_reason: str = "",
        approval_id: str | None = None,
        risk_result: RiskResult | None = None,
        proposer_result: AgentResult[TradeProposal] | None = None,
        tech_short_result: AgentResult[TechnicalAnalysis] | None = None,
        tech_long_result: AgentResult[TechnicalAnalysis] | None = None,
        positioning_result: AgentResult[PositioningAnalysis] | None = None,
        catalyst_result: AgentResult[CatalystReport] | None = None,
        correlation_result: AgentResult[CorrelationAnalysis] | None = None,
    ) -> PipelineResult:
        return PipelineResult(
            run_id=run_id,
            symbol=symbol,
            status=status,
            model_used=model_used,
            proposal=proposal,
            rejection_reason=rejection_reason,
            approval_id=approval_id,
            proposer_degraded=proposer_result.degraded if proposer_result else False,
            technical_short_degraded=tech_short_result.degraded if tech_short_result else False,
            technical_long_degraded=tech_long_result.degraded if tech_long_result else False,
            positioning_degraded=positioning_result.degraded if positioning_result else False,
            catalyst_degraded=catalyst_result.degraded if catalyst_result else False,
            correlation_degraded=correlation_result.degraded if correlation_result else False,
            risk_result=risk_result,
            technical_short=tech_short_result.output if tech_short_result else None,
            technical_long=tech_long_result.output if tech_long_result else None,
            positioning=positioning_result.output if positioning_result else None,
            catalyst=catalyst_result.output if catalyst_result else None,
            correlation=correlation_result.output if correlation_result else None,
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
