from __future__ import annotations

from pydantic import BaseModel

from orchestrator.models import Side, TradeProposal


class AggregationResult(BaseModel, frozen=True):
    valid: bool
    proposal: TradeProposal
    rejection_reason: str = ""


def aggregate_proposal(proposal: TradeProposal, *, current_price: float) -> AggregationResult:
    """Validate a TradeProposal for sanity before forwarding."""
    if proposal.side == Side.FLAT:
        return AggregationResult(valid=True, proposal=proposal)

    # Directional trades require a stop loss
    if proposal.stop_loss is None:
        return AggregationResult(
            valid=False,
            proposal=proposal,
            rejection_reason="Directional trade (long/short) requires a stop_loss.",
        )

    # SL must be on the correct side of entry
    if proposal.side == Side.LONG and proposal.stop_loss >= current_price:
        return AggregationResult(
            valid=False,
            proposal=proposal,
            rejection_reason=(
                f"Long stop_loss ({proposal.stop_loss}) must be below "
                f"current price ({current_price})."
            ),
        )

    if proposal.side == Side.SHORT and proposal.stop_loss <= current_price:
        return AggregationResult(
            valid=False,
            proposal=proposal,
            rejection_reason=(
                f"Short stop_loss ({proposal.stop_loss}) must be above "
                f"current price ({current_price})."
            ),
        )

    return AggregationResult(valid=True, proposal=proposal)
