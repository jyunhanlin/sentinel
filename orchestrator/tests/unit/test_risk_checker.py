
from orchestrator.models import EntryOrder, Side, TradeProposal
from orchestrator.risk.checker import RiskChecker


def _make_proposal(
    *,
    side: Side = Side.LONG,
    risk_pct: float = 1.0,
    stop_loss: float = 93000.0,
    symbol: str = "BTC/USDT:USDT",
    invalid_if: list[str] | None = None,
) -> TradeProposal:
    return TradeProposal(
        symbol=symbol,
        side=side,
        entry=EntryOrder(type="market"),
        position_size_risk_pct=risk_pct,
        stop_loss=stop_loss,
        take_profit=[97000.0],
        time_horizon="4h",
        confidence=0.7,
        invalid_if=invalid_if or [],
        rationale="test",
    )


class TestRiskChecker:
    def test_approved_proposal(self):
        checker = RiskChecker(
            max_single_risk_pct=2.0,
            max_total_exposure_pct=20.0,
            max_consecutive_losses=5,
            max_daily_loss_pct=5.0,
        )
        result = checker.check(
            proposal=_make_proposal(risk_pct=1.0),
            open_positions_risk_pct=5.0,
            consecutive_losses=0,
            daily_loss_pct=0.0,
        )
        assert result.approved is True

    def test_reject_single_risk_too_high(self):
        checker = RiskChecker(
            max_single_risk_pct=2.0,
            max_total_exposure_pct=20.0,
            max_consecutive_losses=5,
            max_daily_loss_pct=5.0,
        )
        result = checker.check(
            proposal=_make_proposal(risk_pct=3.0),
            open_positions_risk_pct=0.0,
            consecutive_losses=0,
            daily_loss_pct=0.0,
        )
        assert result.approved is False
        assert result.rule_violated == "max_single_risk"
        assert result.action == "reject"

    def test_reject_total_exposure_exceeded(self):
        checker = RiskChecker(
            max_single_risk_pct=2.0,
            max_total_exposure_pct=20.0,
            max_consecutive_losses=5,
            max_daily_loss_pct=5.0,
        )
        result = checker.check(
            proposal=_make_proposal(risk_pct=1.5),
            open_positions_risk_pct=19.0,
            consecutive_losses=0,
            daily_loss_pct=0.0,
        )
        assert result.approved is False
        assert result.rule_violated == "max_total_exposure"
        assert result.action == "reject"

    def test_pause_consecutive_losses(self):
        checker = RiskChecker(
            max_single_risk_pct=2.0,
            max_total_exposure_pct=20.0,
            max_consecutive_losses=5,
            max_daily_loss_pct=5.0,
        )
        result = checker.check(
            proposal=_make_proposal(risk_pct=1.0),
            open_positions_risk_pct=0.0,
            consecutive_losses=5,
            daily_loss_pct=0.0,
        )
        assert result.approved is False
        assert result.rule_violated == "max_consecutive_losses"
        assert result.action == "pause"

    def test_pause_daily_loss(self):
        checker = RiskChecker(
            max_single_risk_pct=2.0,
            max_total_exposure_pct=20.0,
            max_consecutive_losses=5,
            max_daily_loss_pct=5.0,
        )
        result = checker.check(
            proposal=_make_proposal(risk_pct=1.0),
            open_positions_risk_pct=0.0,
            consecutive_losses=0,
            daily_loss_pct=6.0,
        )
        assert result.approved is False
        assert result.rule_violated == "max_daily_loss"
        assert result.action == "pause"

    def test_flat_proposal_always_approved(self):
        checker = RiskChecker(
            max_single_risk_pct=2.0,
            max_total_exposure_pct=20.0,
            max_consecutive_losses=5,
            max_daily_loss_pct=5.0,
        )
        result = checker.check(
            proposal=_make_proposal(side=Side.FLAT, risk_pct=0.0, stop_loss=None),
            open_positions_risk_pct=100.0,
            consecutive_losses=100,
            daily_loss_pct=100.0,
        )
        assert result.approved is True
