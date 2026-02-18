from __future__ import annotations

from sqlmodel import Session, select

from orchestrator.storage.models import PipelineRunRecord


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
