from __future__ import annotations

from abc import ABC, abstractmethod


class PositionSizer(ABC):
    @abstractmethod
    def calculate(
        self, *, equity: float, risk_pct: float, entry_price: float, stop_loss: float
    ) -> float:
        """Return quantity in base currency units."""


class RiskPercentSizer(PositionSizer):
    """quantity = (equity * risk_pct / 100) / abs(entry - stop_loss)"""

    def calculate(
        self, *, equity: float, risk_pct: float, entry_price: float, stop_loss: float
    ) -> float:
        if entry_price == stop_loss:
            raise ValueError("stop_loss cannot equal entry_price")
        if risk_pct == 0.0:
            return 0.0
        risk_amount = equity * (risk_pct / 100)
        price_distance = abs(entry_price - stop_loss)
        return risk_amount / price_distance


class MarginSizer:
    """quantity = margin_usdt × leverage / entry_price"""

    def calculate_from_margin(
        self, *, margin_usdt: float, leverage: int, entry_price: float,
    ) -> float:
        if entry_price <= 0:
            raise ValueError("entry_price must be positive")
        if margin_usdt == 0.0:
            return 0.0
        return margin_usdt * leverage / entry_price
