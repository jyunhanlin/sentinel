from __future__ import annotations

from sqlmodel import Session, select

from orchestrator.storage.models import LLMCallRecord, PipelineRunRecord, TradeProposalRecord


class PipelineRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create_run(self, *, run_id: str, symbol: str) -> PipelineRunRecord:
        run = PipelineRunRecord(run_id=run_id, symbol=symbol, status="running")
        self._session.add(run)
        self._session.commit()
        self._session.refresh(run)
        return run

    def get_run(self, run_id: str) -> PipelineRunRecord | None:
        statement = select(PipelineRunRecord).where(PipelineRunRecord.run_id == run_id)
        return self._session.exec(statement).first()

    def update_run_status(self, run_id: str, status: str) -> PipelineRunRecord:
        run = self.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")
        run.status = status
        self._session.add(run)
        self._session.commit()
        self._session.refresh(run)
        return run


class LLMCallRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def save_call(
        self,
        *,
        run_id: str,
        agent_type: str,
        prompt: str,
        response: str,
        model: str,
        latency_ms: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> LLMCallRecord:
        record = LLMCallRecord(
            run_id=run_id,
            agent_type=agent_type,
            prompt=prompt,
            response=response,
            model=model,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
        self._session.add(record)
        self._session.commit()
        self._session.refresh(record)
        return record

    def list_by_run(self, run_id: str) -> list[LLMCallRecord]:
        statement = select(LLMCallRecord).where(LLMCallRecord.run_id == run_id)
        return list(self._session.exec(statement).all())


class TradeProposalRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def save_proposal(
        self,
        *,
        proposal_id: str,
        run_id: str,
        proposal_json: str,
        risk_check_result: str = "",
        risk_check_reason: str = "",
    ) -> TradeProposalRecord:
        record = TradeProposalRecord(
            proposal_id=proposal_id,
            run_id=run_id,
            proposal_json=proposal_json,
            risk_check_result=risk_check_result,
            risk_check_reason=risk_check_reason,
        )
        self._session.add(record)
        self._session.commit()
        self._session.refresh(record)
        return record

    def get_latest_by_symbol(self, run_id: str) -> TradeProposalRecord | None:
        statement = (
            select(TradeProposalRecord)
            .where(TradeProposalRecord.run_id == run_id)
            .order_by(TradeProposalRecord.created_at.desc())
        )
        return self._session.exec(statement).first()

    def get_recent(self, *, limit: int = 10) -> list[TradeProposalRecord]:
        statement = (
            select(TradeProposalRecord)
            .order_by(TradeProposalRecord.created_at.desc())
            .limit(limit)
        )
        return list(self._session.exec(statement).all())
