import pytest

from orchestrator.risk.position_sizer import RiskPercentSizer


class TestRiskPercentSizer:
    def test_basic_calculation(self):
        sizer = RiskPercentSizer()
        # equity=$10000, risk=1.5%, entry=$95000, sl=$93000
        # risk_amount = 10000 * 0.015 = $150
        # distance = abs(95000 - 93000) = $2000
        # quantity = 150 / 2000 = 0.075
        qty = sizer.calculate(
            equity=10000.0,
            risk_pct=1.5,
            entry_price=95000.0,
            stop_loss=93000.0,
        )
        assert qty == pytest.approx(0.075)

    def test_short_position(self):
        sizer = RiskPercentSizer()
        # entry=$95000, sl=$97000 (short)
        qty = sizer.calculate(
            equity=10000.0,
            risk_pct=1.0,
            entry_price=95000.0,
            stop_loss=97000.0,
        )
        # risk_amount = 100, distance = 2000, qty = 0.05
        assert qty == pytest.approx(0.05)

    def test_zero_distance_raises(self):
        sizer = RiskPercentSizer()
        with pytest.raises(ValueError, match="stop_loss cannot equal entry_price"):
            sizer.calculate(
                equity=10000.0,
                risk_pct=1.0,
                entry_price=95000.0,
                stop_loss=95000.0,
            )

    def test_zero_risk_returns_zero(self):
        sizer = RiskPercentSizer()
        qty = sizer.calculate(
            equity=10000.0,
            risk_pct=0.0,
            entry_price=95000.0,
            stop_loss=93000.0,
        )
        assert qty == 0.0
