import time
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from orchestrator.approval.manager import ApprovalManager, PendingApproval
from orchestrator.models import EntryOrder, Side, TradeProposal


def _make_proposal():
    return TradeProposal(
        symbol="BTC/USDT:USDT",
        side=Side.LONG,
        entry=EntryOrder(type="market"),
        position_size_risk_pct=1.5,
        stop_loss=93000.0,
        take_profit=[97000.0],
        time_horizon="4h",
        confidence=0.75,
        invalid_if=[],
        rationale="test",
    )


class TestApprovalManager:
    def test_create_pending(self):
        repo = MagicMock()
        repo.save_approval.return_value = MagicMock(approval_id="a-001")
        mgr = ApprovalManager(repo=repo, timeout_minutes=15)
        approval = mgr.create(
            proposal=_make_proposal(), run_id="run-1", snapshot_price=95200.0
        )
        assert isinstance(approval, PendingApproval)
        assert approval.status == "pending"
        assert (approval.expires_at - approval.created_at).total_seconds() == 900

    def test_approve_valid(self):
        repo = MagicMock()
        repo.save_approval.return_value = MagicMock(approval_id="a-001")
        repo.update_status.return_value = MagicMock(status="approved")
        mgr = ApprovalManager(repo=repo, timeout_minutes=15)
        approval = mgr.create(
            proposal=_make_proposal(), run_id="run-1", snapshot_price=95200.0
        )
        result = mgr.approve(approval.approval_id)
        assert result is not None
        repo.update_status.assert_called_once()

    def test_approve_expired_returns_none(self):
        repo = MagicMock()
        repo.save_approval.return_value = MagicMock(approval_id="a-001")
        mgr = ApprovalManager(repo=repo, timeout_minutes=0)  # instant expiry
        approval = mgr.create(
            proposal=_make_proposal(), run_id="run-1", snapshot_price=95200.0
        )
        time.sleep(0.01)
        result = mgr.approve(approval.approval_id)
        assert result is None

    def test_reject(self):
        repo = MagicMock()
        repo.save_approval.return_value = MagicMock(approval_id="a-001")
        repo.update_status.return_value = MagicMock(status="rejected")
        mgr = ApprovalManager(repo=repo, timeout_minutes=15)
        approval = mgr.create(
            proposal=_make_proposal(), run_id="run-1", snapshot_price=95200.0
        )
        result = mgr.reject(approval.approval_id)
        assert result is not None

    def test_expire_stale(self):
        repo = MagicMock()
        repo.save_approval.return_value = MagicMock(approval_id="a-001")
        repo.update_status.return_value = MagicMock(status="expired")
        mgr = ApprovalManager(repo=repo, timeout_minutes=0)
        mgr.create(
            proposal=_make_proposal(), run_id="run-1", snapshot_price=95200.0
        )
        time.sleep(0.01)
        expired = mgr.expire_stale()
        assert len(expired) == 1

    def test_pending_approval_is_frozen(self):
        approval = PendingApproval(
            approval_id="a-001",
            proposal=_make_proposal(),
            run_id="run-1",
            snapshot_price=95200.0,
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(minutes=15),
        )
        with pytest.raises(Exception):
            approval.status = "approved"
