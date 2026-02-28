# Evaluation & Testing Codemap

**Last Updated:** 2026-02-28

## Overview

Evaluation framework validates LLM agent outputs against golden dataset test cases.

## Architecture

```
EvalRunner
    │
    ├─ Load golden dataset (YAML)
    │
    ├─ For each test case:
    │   ├─ Run sentiment agent
    │   ├─ Run market agent
    │   ├─ Run proposer agent
    │   │
    │   ├─ Score each output (RuleScorer)
    │   └─ Calculate consistency score
    │
    └─ Generate EvalReport (accuracy, case results)
```

## Evaluation Runner

**File:** `orchestrator/src/orchestrator/eval/runner.py`

Main evaluation orchestrator.

### Interface

```python
class EvalRunner:
    def __init__(
        self,
        *,
        sentiment_agent: BaseAgent[SentimentReport],
        market_agent: BaseAgent[MarketInterpretation],
        proposer_agent: BaseAgent[TradeProposal],
    ) -> None:
        self._sentiment_agent = sentiment_agent
        self._market_agent = market_agent
        self._proposer_agent = proposer_agent
        self._scorer = RuleScorer()

    async def run_default(self) -> EvalReport:
        """Run evaluation on built-in golden_v1 dataset."""
        dataset_dir = os.path.join(os.path.dirname(__file__), "datasets")
        dataset_path = os.path.join(dataset_dir, "golden_v1.yaml")
        cases = load_dataset(dataset_path)
        return await self.run(cases=cases, dataset_name="golden_v1")

    async def run(
        self, *, cases: list[EvalCase], dataset_name: str
    ) -> EvalReport:
        """Run evaluation on arbitrary dataset."""
        case_results: list[CaseResult] = []

        for case in cases:
            logger.info("eval_case_start", case_id=case.id)
            result = await self._evaluate_case(case)
            case_results.append(result)
            logger.info("eval_case_done", case_id=case.id, passed=result.passed)

        passed = sum(1 for r in case_results if r.passed)
        total = len(case_results)

        return EvalReport(
            dataset_name=dataset_name,
            total_cases=total,
            passed_cases=passed,
            failed_cases=total - passed,
            accuracy=passed / total if total > 0 else 0.0,
            case_results=case_results,
        )

    async def _evaluate_case(self, case: EvalCase) -> CaseResult:
        """Evaluate single test case against expected outputs."""
        # Build MarketSnapshot from case data
        snapshot = self._build_snapshot(case.snapshot)

        # Run agents in parallel
        sentiment_result = await self._sentiment_agent.analyze(snapshot=snapshot)
        market_result = await self._market_agent.analyze(snapshot=snapshot)
        proposer_result = await self._proposer_agent.analyze(
            snapshot=snapshot,
            sentiment=sentiment_result.output,
            market=market_result.output,
        )

        # Score each agent output
        all_scores: list[ScoreResult] = []

        if case.expected.sentiment is not None:
            all_scores.extend(
                self._scorer.score_sentiment(
                    sentiment_result.output,
                    case.expected.sentiment
                )
            )

        if case.expected.market is not None:
            all_scores.extend(
                self._scorer.score_market(
                    market_result.output,
                    case.expected.market
                )
            )

        if case.expected.proposal is not None:
            all_scores.extend(
                self._scorer.score_proposal(
                    proposer_result.output,
                    case.expected.proposal
                )
            )

        # Check if all scores passed
        passed = all(score.passed for score in all_scores)

        # Calculate consistency
        consistency = await self._calculate_consistency(
            sentiment=sentiment_result.output,
            market=market_result.output,
            proposal=proposer_result.output,
        )

        return CaseResult(
            case_id=case.id,
            passed=passed,
            scores=all_scores,
            consistency=consistency,
        )

    async def _calculate_consistency(
        self,
        sentiment: SentimentReport,
        market: MarketInterpretation,
        proposal: TradeProposal,
    ) -> float:
        """Calculate consistency score across agents."""
        from orchestrator.eval.consistency import ConsistencyChecker

        checker = ConsistencyChecker()
        return checker.score(sentiment, market, proposal)
```

### EvalReport

```python
class EvalReport(BaseModel, frozen=True):
    dataset_name: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    accuracy: float  # 0.0-1.0
    consistency_score: float | None = None
    case_results: list[CaseResult]

class CaseResult(BaseModel, frozen=True):
    case_id: str
    passed: bool
    scores: list[ScoreResult]
    consistency: float | None = None

class ScoreResult(BaseModel, frozen=True):
    metric: str  # e.g., "sentiment_score_range"
    expected: str  # Expected value
    actual: str  # Actual value
    passed: bool
    severity: str  # "critical", "high", "low"
```

## Golden Dataset

**File:** `orchestrator/src/orchestrator/eval/datasets/golden_v1.yaml`

YAML file with test cases.

### Case Format

```yaml
- id: case_001
  description: BTC strong uptrend with high volume
  snapshot:
    symbol: BTC/USDT:USDT
    current_price: 45000.50
    volume_24h: 28500000000
    funding_rate: 0.000865
    ohlcv:
      - [1234567890, 44750, 45100, 44600, 45000, 1234567]  # [timestamp, O, H, L, C, V]
      - [1234567891, 45000, 45500, 44900, 45400, 1345678]
      - ...
  expected:
    sentiment:
      sentiment_score: 70  # Expect bullish
      confidence_min: 0.65
    market:
      trend: up
      volatility_regime: medium
    proposal:
      side: long
      position_size_risk_pct_min: 0.5
      position_size_risk_pct_max: 2.0
```

### Dataset Loading

**File:** `orchestrator/src/orchestrator/eval/dataset.py`

```python
class EvalCase(BaseModel, frozen=True):
    id: str
    description: str
    snapshot: SnapshotData
    expected: ExpectedOutputs

class SnapshotData(BaseModel, frozen=True):
    symbol: str
    current_price: float
    volume_24h: float
    funding_rate: float
    ohlcv: list[tuple[float, float, float, float, float]]

class ExpectedOutputs(BaseModel, frozen=True):
    sentiment: SentimentExpectation | None = None
    market: MarketExpectation | None = None
    proposal: ProposalExpectation | None = None

def load_dataset(path: str) -> list[EvalCase]:
    """Load test cases from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)

    cases = []
    for case_data in data:
        case = EvalCase(
            id=case_data["id"],
            description=case_data["description"],
            snapshot=SnapshotData(**case_data["snapshot"]),
            expected=ExpectedOutputs(**case_data.get("expected", {})),
        )
        cases.append(case)

    return cases
```

## Scoring

**File:** `orchestrator/src/orchestrator/eval/scorers.py`

Rule-based scoring for agent outputs.

### RuleScorer

```python
class RuleScorer:
    """Score agent outputs against expected values."""

    def score_sentiment(
        self,
        actual: SentimentReport,
        expected: SentimentExpectation,
    ) -> list[ScoreResult]:
        """Score sentiment report."""
        scores = []

        # Rule 1: Score in expected range
        if expected.sentiment_score is not None:
            passed = actual.sentiment_score == expected.sentiment_score
            scores.append(ScoreResult(
                metric="sentiment_score",
                expected=str(expected.sentiment_score),
                actual=str(actual.sentiment_score),
                passed=passed,
                severity="critical" if not passed else "low",
            ))

        # Rule 2: Confidence in expected range
        if expected.confidence_min is not None:
            passed = actual.confidence >= expected.confidence_min
            scores.append(ScoreResult(
                metric="confidence_min",
                expected=f">= {expected.confidence_min}",
                actual=f"{actual.confidence}",
                passed=passed,
                severity="high" if not passed else "low",
            ))

        # Rule 3: Must have key_events if bullish/bearish
        if expected.sentiment_score and (expected.sentiment_score < 40 or expected.sentiment_score > 60):
            passed = len(actual.key_events) > 0
            scores.append(ScoreResult(
                metric="key_events_present",
                expected="at least 1",
                actual=str(len(actual.key_events)),
                passed=passed,
                severity="high",
            ))

        return scores

    def score_market(
        self,
        actual: MarketInterpretation,
        expected: MarketExpectation,
    ) -> list[ScoreResult]:
        """Score market interpretation."""
        scores = []

        # Rule 1: Trend must match
        if expected.trend is not None:
            passed = actual.trend == expected.trend
            scores.append(ScoreResult(
                metric="trend",
                expected=expected.trend.value,
                actual=actual.trend.value,
                passed=passed,
                severity="critical",
            ))

        # Rule 2: Volatility regime
        if expected.volatility_regime is not None:
            passed = actual.volatility_regime == expected.volatility_regime
            scores.append(ScoreResult(
                metric="volatility_regime",
                expected=expected.volatility_regime.value,
                actual=actual.volatility_regime.value,
                passed=passed,
                severity="high",
            ))

        # Rule 3: Key levels present
        if expected.min_key_levels is not None:
            passed = len(actual.key_levels) >= expected.min_key_levels
            scores.append(ScoreResult(
                metric="key_levels_count",
                expected=f">= {expected.min_key_levels}",
                actual=str(len(actual.key_levels)),
                passed=passed,
                severity="high",
            ))

        return scores

    def score_proposal(
        self,
        actual: TradeProposal,
        expected: ProposalExpectation,
    ) -> list[ScoreResult]:
        """Score trade proposal."""
        scores = []

        # Rule 1: Side must match
        if expected.side is not None:
            passed = actual.side == expected.side
            scores.append(ScoreResult(
                metric="side",
                expected=expected.side.value,
                actual=actual.side.value,
                passed=passed,
                severity="critical" if expected.side != Side.FLAT else "high",
            ))

        # Rule 2: Position size in range
        if expected.position_size_risk_pct_min is not None:
            passed = actual.position_size_risk_pct >= expected.position_size_risk_pct_min
            scores.append(ScoreResult(
                metric="position_size_risk_pct_min",
                expected=f">= {expected.position_size_risk_pct_min}",
                actual=str(actual.position_size_risk_pct),
                passed=passed,
                severity="high",
            ))

        if expected.position_size_risk_pct_max is not None:
            passed = actual.position_size_risk_pct <= expected.position_size_risk_pct_max
            scores.append(ScoreResult(
                metric="position_size_risk_pct_max",
                expected=f"<= {expected.position_size_risk_pct_max}",
                actual=str(actual.position_size_risk_pct),
                passed=passed,
                severity="high",
            ))

        # Rule 3: Confidence in range
        if expected.confidence_min is not None:
            passed = actual.confidence >= expected.confidence_min
            scores.append(ScoreResult(
                metric="confidence_min",
                expected=f">= {expected.confidence_min}",
                actual=str(actual.confidence),
                passed=passed,
                severity="high",
            ))

        # Rule 4: SL must be valid
        if actual.side != Side.FLAT and actual.stop_loss is not None:
            if actual.side == Side.LONG:
                passed = actual.stop_loss < actual.entry.price
            else:  # SHORT
                passed = actual.stop_loss > actual.entry.price

            scores.append(ScoreResult(
                metric="stop_loss_valid",
                expected="SL on correct side of entry",
                actual=f"SL: {actual.stop_loss}, Entry: {actual.entry.price}",
                passed=passed,
                severity="critical",
            ))

        # Rule 5: TP prices must be valid
        if actual.side != Side.FLAT and actual.take_profit:
            for i, tp in enumerate(actual.take_profit):
                if actual.side == Side.LONG:
                    passed = tp.price > actual.entry.price
                else:  # SHORT
                    passed = tp.price < actual.entry.price

                scores.append(ScoreResult(
                    metric=f"take_profit_{i}_valid",
                    expected="TP on correct side of entry",
                    actual=f"TP: {tp.price}, Entry: {actual.entry.price}",
                    passed=passed,
                    severity="critical",
                ))

        return scores
```

## Consistency Checking

**File:** `orchestrator/src/orchestrator/eval/consistency.py`

Validates that sentiment, market, and proposal are aligned.

### ConsistencyChecker

```python
class ConsistencyChecker:
    """Check if outputs are internally consistent."""

    def score(
        self,
        sentiment: SentimentReport,
        market: MarketInterpretation,
        proposal: TradeProposal,
    ) -> float:
        """
        Score consistency 0.0-1.0.
        Higher = more aligned outputs.
        """
        checks = []

        # Check 1: Sentiment and trend alignment
        if sentiment.sentiment_score > 60 and market.trend == Trend.UP:
            checks.append(True)  # Aligned
        elif sentiment.sentiment_score < 40 and market.trend == Trend.DOWN:
            checks.append(True)  # Aligned
        elif sentiment.sentiment_score >= 40 and sentiment.sentiment_score <= 60 and market.trend == Trend.RANGE:
            checks.append(True)  # Aligned
        else:
            checks.append(False)  # Misaligned

        # Check 2: Proposal side matches trend
        if proposal.side == Side.FLAT:
            checks.append(True)  # Flat is always consistent
        elif proposal.side == Side.LONG and market.trend in (Trend.UP, Trend.RANGE):
            checks.append(True)
        elif proposal.side == Side.SHORT and market.trend in (Trend.DOWN, Trend.RANGE):
            checks.append(True)
        else:
            checks.append(False)

        # Check 3: Confidence levels reasonable
        avg_confidence = (sentiment.confidence + proposal.confidence) / 2
        if avg_confidence >= 0.5:
            checks.append(True)
        else:
            checks.append(False)

        # Check 4: Risk sizing reasonable
        if proposal.side == Side.FLAT:
            checks.append(proposal.position_size_risk_pct == 0.0)
        else:
            checks.append(0.0 < proposal.position_size_risk_pct <= 2.0)

        # Calculate score
        passed = sum(checks)
        total = len(checks)
        return passed / total
```

## CLI Usage

From command line:

```bash
# Run default evaluation
uv run python -m orchestrator eval

# Output
# Dataset: golden_v1
# Total cases: 10
# Passed: 8 / 10
# Accuracy: 80.0%
#
# Case Results:
# - case_001: PASS (3/3 agents passed)
# - case_002: FAIL (proposer side mismatch)
# ...
```

## Testing

**File:** `tests/unit/test_eval_consistency.py`

Tests consistency scoring:

```python
def test_consistency_bullish_uptrend():
    sentiment = SentimentReport(sentiment_score=75, confidence=0.8)
    market = MarketInterpretation(trend=Trend.UP, ...)
    proposal = TradeProposal(side=Side.LONG, ...)

    checker = ConsistencyChecker()
    score = checker.score(sentiment, market, proposal)

    assert score == 1.0  # Perfect alignment
```

**File:** `tests/unit/test_eval_dataset.py`

Tests dataset loading:

```python
def test_load_golden_dataset():
    cases = load_dataset("orchestrator/src/orchestrator/eval/datasets/golden_v1.yaml")
    assert len(cases) > 0
    assert cases[0].id == "case_001"
```

## Future Enhancements

- **Fuzzy matching** — Accept sentiment scores ±5 instead of exact match
- **Weighted scoring** — Critical metrics weighted higher
- **Historical performance** — Track accuracy over time
- **Custom datasets** — Load arbitrary YAML for regression testing
- **Backtesting** — Run evaluation over historical price data

## Related

- [agents-skills.md](agents-skills.md) — Agents being evaluated
- [pipeline.md](pipeline.md) — Pipeline integration
