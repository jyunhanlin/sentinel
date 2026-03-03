from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EquityProvider(Protocol):
    async def get_equity(self) -> float: ...
    async def get_available_margin(self) -> float: ...


class PaperEquityProvider:
    """Reads equity from PaperEngine's simulated account."""

    def __init__(self, engine: object) -> None:
        self._engine = engine

    async def get_equity(self) -> float:
        return self._engine.equity  # type: ignore[attr-defined]

    async def get_available_margin(self) -> float:
        equity: float = self._engine.equity  # type: ignore[attr-defined]
        used: float = self._engine.used_margin  # type: ignore[attr-defined]
        return equity - used
