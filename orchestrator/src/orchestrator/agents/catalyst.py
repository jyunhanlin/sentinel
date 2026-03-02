from __future__ import annotations

from typing import Any

from orchestrator.agents.base import BaseAgent
from orchestrator.models import CatalystReport


class CatalystAgent(BaseAgent[CatalystReport]):
    output_model = CatalystReport
    _skill_name = "catalyst"

    def _build_prompt(self, **kwargs) -> str:
        symbol: str = kwargs["symbol"]
        current_price: float = kwargs["current_price"]
        calendar: list[dict[str, Any]] = kwargs["economic_calendar"]
        announcements: list[str] = kwargs["exchange_announcements"]

        calendar_str = ""
        for entry in calendar:
            evt = entry.get("event", "Unknown")
            time = entry.get("time", "TBD")
            impact = entry.get("impact", "unknown")
            calendar_str += f"- {evt} | {time} | Impact: {impact}\n"
        if not calendar_str:
            calendar_str = "No upcoming events\n"

        announcements_str = ""
        for ann in announcements:
            announcements_str += f"- {ann}\n"
        if not announcements_str:
            announcements_str = "No recent announcements\n"

        data = (
            f"Symbol: {symbol}\n"
            f"Current Price: {current_price}\n\n"
            f"=== Economic Calendar (next 48h) ===\n"
            f"{calendar_str}\n"
            f"=== Exchange Announcements ===\n"
            f"{announcements_str}"
        )

        return (
            f"Use the {self._skill_name} skill.\n\n"
            f"=== Catalyst Data ===\n{data}"
        )

    def _get_default_output(self) -> CatalystReport:
        return CatalystReport(
            upcoming_events=[],
            active_events=[],
            risk_level="low",
            recommendation="proceed",
            confidence=0.1,
        )
