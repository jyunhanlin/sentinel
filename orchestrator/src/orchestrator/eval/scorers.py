from __future__ import annotations

from pydantic import BaseModel

from orchestrator.eval.dataset import (
    ExpectedMarket,
    ExpectedProposal,
    ExpectedRange,
    ExpectedSentiment,
)
from orchestrator.models import MarketInterpretation, SentimentReport, TradeProposal


class ScoreResult(BaseModel, frozen=True):
    field: str
    passed: bool
    expected: str
    actual: str
    reason: str = ""


class RuleScorer:
    def score_proposal(
        self, proposal: TradeProposal, expected: ExpectedProposal
    ) -> list[ScoreResult]:
        results: list[ScoreResult] = []

        if expected.side is not None:
            passed = proposal.side.value in expected.side
            results.append(ScoreResult(
                field="side", passed=passed,
                expected=str(expected.side), actual=proposal.side.value,
                reason="" if passed else f"expected one of {expected.side}, got {proposal.side.value}",
            ))

        if expected.confidence is not None:
            results.append(self._check_range(
                "confidence", proposal.confidence, expected.confidence
            ))

        if expected.has_stop_loss is not None:
            has_sl = proposal.stop_loss is not None
            passed = has_sl == expected.has_stop_loss
            results.append(ScoreResult(
                field="has_stop_loss", passed=passed,
                expected=str(expected.has_stop_loss), actual=str(has_sl),
            ))

        if expected.sl_correct_side is not None and proposal.stop_loss is not None:
            if proposal.side.value == "long":
                correct = proposal.stop_loss < (proposal.entry.price or 0)
            elif proposal.side.value == "short":
                correct = proposal.stop_loss > (proposal.entry.price or float("inf"))
            else:
                correct = True
            results.append(ScoreResult(
                field="sl_correct_side", passed=correct == expected.sl_correct_side,
                expected=str(expected.sl_correct_side), actual=str(correct),
            ))

        return results

    def score_sentiment(
        self, output: SentimentReport, expected: ExpectedSentiment
    ) -> list[ScoreResult]:
        results: list[ScoreResult] = []

        if expected.sentiment_score is not None:
            results.append(self._check_range(
                "sentiment_score", float(output.sentiment_score), expected.sentiment_score
            ))

        if expected.confidence is not None:
            results.append(self._check_range(
                "confidence", output.confidence, expected.confidence
            ))

        return results

    def score_market(
        self, output: MarketInterpretation, expected: ExpectedMarket
    ) -> list[ScoreResult]:
        results: list[ScoreResult] = []

        if expected.trend is not None:
            passed = output.trend.value in expected.trend
            results.append(ScoreResult(
                field="trend", passed=passed,
                expected=str(expected.trend), actual=output.trend.value,
                reason="" if passed else f"expected one of {expected.trend}, got {output.trend.value}",
            ))

        if expected.volatility_regime is not None:
            passed = output.volatility_regime.value in expected.volatility_regime
            results.append(ScoreResult(
                field="volatility_regime", passed=passed,
                expected=str(expected.volatility_regime),
                actual=output.volatility_regime.value,
            ))

        return results

    def _check_range(
        self, field: str, value: float, expected: ExpectedRange
    ) -> ScoreResult:
        passed = True
        reasons: list[str] = []

        if expected.min is not None and value < expected.min:
            passed = False
            reasons.append(f"{value} < min {expected.min}")
        if expected.max is not None and value > expected.max:
            passed = False
            reasons.append(f"{value} > max {expected.max}")

        return ScoreResult(
            field=field, passed=passed,
            expected=f"[{expected.min}, {expected.max}]", actual=str(value),
            reason="; ".join(reasons),
        )
