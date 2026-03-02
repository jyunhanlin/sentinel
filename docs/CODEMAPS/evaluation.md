<!-- Generated: 2026-03-02 | Files scanned: 6 | Token estimate: ~500 -->
# Evaluation & Testing Codemap

## Overview

Evaluation framework validates LLM agent outputs against golden dataset test cases.

## Architecture

```
EvalRunner
    │
    ├─ Load golden dataset (YAML)
    │
    ├─ For each test case:
    │   ├─ Run 5 analysis agents (technical_short, technical_long,
    │   │   positioning, catalyst, correlation) with default external data
    │   ├─ Run proposer agent with analysis outputs
    │   │
    │   ├─ Score market expectations against technical_short output
    │   │   (both share trend + volatility_regime fields)
    │   ├─ Score proposal expectations
    │   └─ Deprecated: sentiment scoring (warns if present in dataset)
    │
    └─ Generate EvalReport (accuracy, case results)
```

## Evaluation Runner

**File:** `eval/runner.py`

```python
class EvalRunner:
    def __init__(self, *, technical_short_agent, technical_long_agent,
                 positioning_agent, catalyst_agent, correlation_agent,
                 proposer_agent) -> None:
```

### _evaluate_case flow:
1. Build `MarketSnapshot` from case data
2. Run 5 analysis agents with default data for positioning/catalyst/correlation
3. Run proposer with all 5 outputs
4. Score `expected.market` against `technical_short` output (trend, volatility_regime)
5. Score `expected.proposal` against proposer output (side, risk_pct, confidence)
6. If `expected.sentiment` present → log deprecation warning, skip scoring

## Scoring

**File:** `eval/scorers.py`

| Method | Compares | Key fields |
|--------|----------|------------|
| `score_market(TechnicalAnalysis, ExpectedMarket)` | trend, volatility_regime | critical severity |
| `score_proposal(TradeProposal, ExpectedProposal)` | side, position_size_risk_pct range, confidence_min, SL/TP validity | critical for side |

## Golden Dataset

**File:** `eval/datasets/golden_v1.yaml`

```yaml
- id: case_001
  description: BTC strong uptrend with high volume
  snapshot:
    symbol: BTC/USDT:USDT
    current_price: 45000.50
    ...
  expected:
    market:
      trend: up
      volatility_regime: medium
    proposal:
      side: long
      position_size_risk_pct_min: 0.5
```

## Testing

- `tests/unit/test_eval_runner.py` — EvalRunner with 6 mocked agents
- `tests/unit/test_eval_scorers.py` — Score functions for TechnicalAnalysis and TradeProposal
- `tests/unit/test_eval_dataset.py` — YAML loading
- `tests/unit/test_eval_consistency.py` — Cross-agent consistency scoring
