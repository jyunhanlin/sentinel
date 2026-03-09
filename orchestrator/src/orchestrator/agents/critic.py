from __future__ import annotations

from orchestrator.agents.base import BaseAgent
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.models import (
    CatalystReport,
    CorrelationAnalysis,
    CritiqueResult,
    DimensionVerdict,
    PositioningAnalysis,
    TechnicalAnalysis,
    TradeProposal,
)


class CriticAgent(BaseAgent[CritiqueResult]):
    output_model = CritiqueResult
    _skill_name = "critic"

    def _build_prompt(self, **kwargs) -> str:
        proposal: TradeProposal = kwargs["proposal"]
        snapshot: MarketSnapshot = kwargs["snapshot"]
        tech_short: TechnicalAnalysis = kwargs["technical_short"]
        tech_long: TechnicalAnalysis = kwargs["technical_long"]
        positioning: PositioningAnalysis = kwargs["positioning"]
        catalyst: CatalystReport = kwargs["catalyst"]
        correlation: CorrelationAnalysis = kwargs["correlation"]

        def _format_risk_flags(flags: list[str]) -> str:
            return ", ".join(flags) or "none"

        tp_str = ", ".join(
            f"{tp.price} ({tp.close_pct}%)" for tp in proposal.take_profit
        ) or "none"

        data = (
            f"=== Proposal Under Review ===\n"
            f"Symbol: {proposal.symbol}\n"
            f"Side: {proposal.side}\n"
            f"Entry: {proposal.entry.type}"
            f"{f' @ {proposal.entry.price}' if proposal.entry.price else ''}\n"
            f"Stop Loss: {proposal.stop_loss}\n"
            f"Take Profit: {tp_str}\n"
            f"Position Size Risk: {proposal.position_size_risk_pct}%\n"
            f"Leverage: {proposal.suggested_leverage}x\n"
            f"Confidence: {proposal.confidence}\n"
            f"Time Horizon: {proposal.time_horizon}\n"
            f"Rationale: {proposal.rationale}\n\n"
            f"=== Market Context ===\n"
            f"Current Price: {snapshot.current_price}\n"
            f"24h Volume: {snapshot.volume_24h:,.0f}\n"
            f"Funding Rate: {snapshot.funding_rate:.6f}\n\n"
            f"=== Short-Term Technical ({tech_short.label}) ===\n"
            f"Trend: {tech_short.trend} (ADX: {tech_short.trend_strength:.1f})\n"
            f"Momentum: {tech_short.momentum} (RSI: {tech_short.rsi:.1f})\n"
            f"Volatility: {tech_short.volatility_regime} ({tech_short.volatility_pct:.1f}%)\n"
            f"Risk Flags: {_format_risk_flags(tech_short.risk_flags)}\n\n"
            f"=== Long-Term Technical ({tech_long.label}) ===\n"
            f"Trend: {tech_long.trend} (ADX: {tech_long.trend_strength:.1f})\n"
            f"Momentum: {tech_long.momentum} (RSI: {tech_long.rsi:.1f})\n"
            f"Volatility: {tech_long.volatility_regime} ({tech_long.volatility_pct:.1f}%)\n"
            f"Risk Flags: {_format_risk_flags(tech_long.risk_flags)}\n\n"
            f"=== Positioning ===\n"
            f"Funding Trend: {positioning.funding_trend} (extreme: {positioning.funding_extreme})\n"
            f"OI Change: {positioning.oi_change_pct:+.1f}%\n"
            f"Squeeze Risk: {positioning.squeeze_risk}\n"
            f"Risk Flags: {_format_risk_flags(positioning.risk_flags)}\n"
            f"Confidence: {positioning.confidence:.2f}\n\n"
            f"=== Catalyst ===\n"
            f"Risk Level: {catalyst.risk_level}\n"
            f"Recommendation: {catalyst.recommendation}\n"
            f"Confidence: {catalyst.confidence:.2f}\n\n"
            f"=== Cross-Market Correlation ===\n"
            f"DXY: {correlation.dxy_trend} ({correlation.dxy_impact})\n"
            f"S&P 500: {correlation.sp500_regime}\n"
            f"Alignment: {correlation.cross_market_alignment}\n"
            f"Risk Flags: {_format_risk_flags(correlation.risk_flags)}\n"
            f"Confidence: {correlation.confidence:.2f}"
        )

        return (
            f"Use the {self._skill_name} skill.\n\n"
            f"{data}"
        )

    def _get_default_output(self) -> CritiqueResult:
        return CritiqueResult(
            verdicts=[
                DimensionVerdict(
                    dimension="consistency", passed=True, reason="degraded — skipped",
                ),
                DimensionVerdict(
                    dimension="risk_reward", passed=True, reason="degraded — skipped",
                ),
                DimensionVerdict(
                    dimension="input_respect", passed=True, reason="degraded — skipped",
                ),
                DimensionVerdict(
                    dimension="parameter_sanity", passed=True, reason="degraded — skipped",
                ),
            ],
            overall_passed=True,
            suggestions=[],
            summary="Critic degraded — proposal accepted without review",
        )
