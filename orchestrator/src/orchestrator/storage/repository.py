from __future__ import annotations

from datetime import date

from sqlmodel import Session, select

from orchestrator.storage.models import (
    AccountSnapshotRecord,
    LLMCallRecord,
    PaperTradeRecord,
    PipelineRunRecord,
    TradeProposalRecord,
)


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


class PaperTradeRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def save_trade(
        self,
        *,
        trade_id: str,
        proposal_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        risk_pct: float = 0.0,
    ) -> PaperTradeRecord:
        record = PaperTradeRecord(
            trade_id=trade_id,
            proposal_id=proposal_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            risk_pct=risk_pct,
        )
        self._session.add(record)
        self._session.commit()
        self._session.refresh(record)
        return record

    def get_by_trade_id(self, trade_id: str) -> PaperTradeRecord | None:
        statement = select(PaperTradeRecord).where(PaperTradeRecord.trade_id == trade_id)
        return self._session.exec(statement).first()

    def get_open_positions(self) -> list[PaperTradeRecord]:
        statement = select(PaperTradeRecord).where(PaperTradeRecord.status == "open")
        return list(self._session.exec(statement).all())

    def update_trade_closed(
        self,
        trade_id: str,
        *,
        exit_price: float,
        pnl: float,
        fees: float,
    ) -> PaperTradeRecord:
        from datetime import UTC, datetime

        trade = self.get_by_trade_id(trade_id)
        if trade is None:
            raise ValueError(f"Trade {trade_id} not found")
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.fees = fees
        trade.status = "closed"
        trade.closed_at = datetime.now(UTC)
        self._session.add(trade)
        self._session.commit()
        self._session.refresh(trade)
        return trade

    def get_recent_closed(self, *, limit: int = 10) -> list[PaperTradeRecord]:
        statement = (
            select(PaperTradeRecord)
            .where(PaperTradeRecord.status == "closed")
            .order_by(PaperTradeRecord.closed_at.desc())
            .limit(limit)
        )
        return list(self._session.exec(statement).all())

    def count_consecutive_losses(self) -> int:
        """Count consecutive losses from most recent closed trade backwards."""
        statement = (
            select(PaperTradeRecord)
            .where(PaperTradeRecord.status == "closed")
            .order_by(PaperTradeRecord.closed_at.desc())
        )
        trades = list(self._session.exec(statement).all())
        count = 0
        for trade in trades:
            if trade.pnl < 0:
                count += 1
            else:
                break
        return count

    def get_daily_pnl(self, day: date) -> float:
        """Sum PnL for all closed trades on a given date."""
        statement = (
            select(PaperTradeRecord)
            .where(PaperTradeRecord.status == "closed")
        )
        trades = list(self._session.exec(statement).all())
        return sum(t.pnl for t in trades if t.closed_at and t.closed_at.date() == day)


class AccountSnapshotRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def save_snapshot(
        self,
        *,
        equity: float,
        open_count: int,
        daily_pnl: float,
    ) -> AccountSnapshotRecord:
        record = AccountSnapshotRecord(
            equity=equity,
            open_positions_count=open_count,
            daily_pnl=daily_pnl,
        )
        self._session.add(record)
        self._session.commit()
        self._session.refresh(record)
        return record

    def get_latest(self) -> AccountSnapshotRecord | None:
        statement = (
            select(AccountSnapshotRecord)
            .order_by(AccountSnapshotRecord.created_at.desc())
        )
        return self._session.exec(statement).first()
