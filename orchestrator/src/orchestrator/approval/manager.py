from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel

from orchestrator.models import TradeProposal

if TYPE_CHECKING:
    from orchestrator.storage.repository import ApprovalRepository

logger = structlog.get_logger(__name__)


class PendingApproval(BaseModel, frozen=True):
    approval_id: str
    proposal: TradeProposal
    run_id: str
    snapshot_price: float
    created_at: datetime
    expires_at: datetime
    status: str = "pending"
    message_id: int | None = None


class ApprovalManager:
    def __init__(self, *, repo: ApprovalRepository, timeout_minutes: int = 15) -> None:
        self._repo = repo
        self._timeout_minutes = timeout_minutes
        self._pending: dict[str, PendingApproval] = {}

    def create(
        self, *, proposal: TradeProposal, run_id: str, snapshot_price: float
    ) -> PendingApproval:
        approval_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        expires_at = now + timedelta(minutes=self._timeout_minutes)

        approval = PendingApproval(
            approval_id=approval_id,
            proposal=proposal,
            run_id=run_id,
            snapshot_price=snapshot_price,
            created_at=now,
            expires_at=expires_at,
        )
        self._pending[approval_id] = approval

        self._repo.save_approval(
            approval_id=approval_id,
            proposal_id=proposal.proposal_id,
            run_id=run_id,
            snapshot_price=snapshot_price,
            expires_at=expires_at,
        )

        logger.info(
            "approval_created",
            approval_id=approval_id,
            symbol=proposal.symbol,
            side=proposal.side,
            expires_at=expires_at.isoformat(),
        )
        return approval

    def approve(self, approval_id: str) -> PendingApproval | None:
        approval = self._pending.get(approval_id)
        if approval is None:
            return None
        if datetime.now(UTC) > approval.expires_at:
            self._expire(approval_id)
            return None

        del self._pending[approval_id]
        self._repo.update_status(approval_id, status="approved")
        logger.info("approval_approved", approval_id=approval_id)
        return approval

    def reject(self, approval_id: str) -> PendingApproval | None:
        approval = self._pending.pop(approval_id, None)
        if approval is None:
            return None
        self._repo.update_status(approval_id, status="rejected")
        logger.info("approval_rejected", approval_id=approval_id)
        return approval

    def get(self, approval_id: str) -> PendingApproval | None:
        return self._pending.get(approval_id)

    def set_message_id(self, approval_id: str, message_id: int) -> None:
        approval = self._pending.get(approval_id)
        if approval is not None:
            updated = PendingApproval(
                **{**approval.model_dump(), "message_id": message_id}
            )
            self._pending[approval_id] = updated
            self._repo.update_message_id(approval_id, message_id=message_id)

    def expire_stale(self) -> list[PendingApproval]:
        now = datetime.now(UTC)
        expired: list[PendingApproval] = []
        for aid, approval in list(self._pending.items()):
            if now > approval.expires_at:
                expired.append(approval)
                self._expire(aid)
        return expired

    def get_pending_count(self) -> int:
        return len(self._pending)

    def _expire(self, approval_id: str) -> None:
        self._pending.pop(approval_id, None)
        self._repo.update_status(approval_id, status="expired")
        logger.info("approval_expired", approval_id=approval_id)
