from __future__ import annotations

import structlog
from pydantic import BaseModel

from orchestrator.models import Side, TradeProposal

logger = structlog.get_logger(__name__)


class RiskResult(BaseModel, frozen=True):
    approved: bool
    rule_violated: str = ""
    reason: str = ""
    action: str = "reject"  # "reject" | "pause"


class RiskChecker:
    def __init__(
        self,
        *,
        max_single_risk_pct: float,
        max_total_exposure_pct: float,
        max_consecutive_losses: int,
        max_daily_loss_pct: float,
    ) -> None:
        self._max_single_risk_pct = max_single_risk_pct
        self._max_total_exposure_pct = max_total_exposure_pct
        self._max_consecutive_losses = max_consecutive_losses
        self._max_daily_loss_pct = max_daily_loss_pct

    def check(
        self,
        *,
        proposal: TradeProposal,
        open_positions_risk_pct: float,
        consecutive_losses: int,
        daily_loss_pct: float,
    ) -> RiskResult:
        if proposal.side == Side.FLAT:
            return RiskResult(approved=True)

        # Rule 1: Max single risk
        if proposal.position_size_risk_pct > self._max_single_risk_pct:
            reason = (
                f"Single risk {proposal.position_size_risk_pct}% "
                f"exceeds {self._max_single_risk_pct}% limit"
            )
            logger.warning("risk_rejected", rule="max_single_risk", reason=reason)
            return RiskResult(
                approved=False,
                rule_violated="max_single_risk",
                reason=reason,
                action="reject",
            )

        # Rule 2: Max total exposure
        total = open_positions_risk_pct + proposal.position_size_risk_pct
        if total > self._max_total_exposure_pct:
            reason = (
                f"Total exposure {total}% "
                f"exceeds {self._max_total_exposure_pct}% limit"
            )
            logger.warning("risk_rejected", rule="max_total_exposure", reason=reason)
            return RiskResult(
                approved=False,
                rule_violated="max_total_exposure",
                reason=reason,
                action="reject",
            )

        # Rule 3: Max consecutive losses
        if consecutive_losses >= self._max_consecutive_losses:
            reason = (
                f"{consecutive_losses} consecutive losses "
                f"reached {self._max_consecutive_losses} limit"
            )
            logger.warning("risk_paused", rule="max_consecutive_losses", reason=reason)
            return RiskResult(
                approved=False,
                rule_violated="max_consecutive_losses",
                reason=reason,
                action="pause",
            )

        # Rule 4: Max daily loss
        if daily_loss_pct > self._max_daily_loss_pct:
            reason = (
                f"Daily loss {daily_loss_pct}% "
                f"exceeds {self._max_daily_loss_pct}% limit"
            )
            logger.warning("risk_paused", rule="max_daily_loss", reason=reason)
            return RiskResult(
                approved=False,
                rule_violated="max_daily_loss",
                reason=reason,
                action="pause",
            )

        return RiskResult(approved=True)
