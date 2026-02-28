# M3 Design: Eval Framework + Performance Statistics + Monitoring

**Date:** 2026-02-19
**Status:** Approved
**Depends on:** M2 (Risk Management + Paper Trading)

---

## Goal

Add the observability and evaluation layer: measure LLM output quality (eval framework), track trading performance (5 key metrics), expose results via TG commands and CLI, and bring test coverage to 80%+ across all modules.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Eval approach | Golden dataset + self-consistency | Golden for accuracy regression, consistency for stability |
| Eval scoring | Mixed: rule-based + LLM-as-judge | Discrete/numeric fields use rules, free text uses LLM |
| Golden dataset size | 5 initial cases | Bull, bear, sideways, high-vol, funding-anomaly; expandable |
| Perf metrics | All 5: PnL, win rate, profit factor, max drawdown, Sharpe | Complete trading performance picture |
| Stats calculation | Real-time on position close | Immediate visibility, stored in DB |
| Dashboard tech | TG commands + CLI | Zero infra cost, consistent with existing UX |
| Test coverage gaps | Fix in M3 | bot.py and scheduler.py to 80%+ |

## Architecture

### New / Modified Modules

```
stats/
  calculator.py         Performance metrics calculator (5 indicators)

eval/
  dataset.py            Golden dataset loader (YAML → typed models)
  runner.py             Eval pipeline: load cases → run agents → score → report
  scorers.py            Rule-based scorer + LLM-as-judge scorer
  consistency.py        Self-consistency checker (N runs per case)
  report.py             Eval report generator

eval/datasets/
  golden_v1.yaml        Initial 5 golden test cases

pipeline/
  runner.py             + call StatsCalculator on position close

telegram/
  bot.py                + /perf, /eval commands
  formatters.py         + format_perf_report(), format_eval_report()

storage/
  models.py             + extend AccountSnapshotRecord with stats fields
  repository.py         + stats-related queries

__main__.py             + subcommand support (eval, perf)
```

### Dependency Flow

```
stats      ← depends on storage (closed trades, snapshots)
eval       ← depends on agents, llm, models (runs agents against golden data)
pipeline   ← now calls stats on position close
telegram   ← depends on stats, eval (new commands)
__main__   ← wires everything, adds subcommands
```

## Module A: Performance Statistics

### StatsCalculator (`stats/calculator.py`)

Pure calculation module. Computes from closed trades stored in DB.

```python
class PerformanceStats(BaseModel, frozen=True):
    total_pnl: float
    total_pnl_pct: float          # relative to initial equity
    win_rate: float               # wins / total (0.0 - 1.0)
    total_trades: int
    winning_trades: int
    losing_trades: int
    profit_factor: float          # sum(wins) / abs(sum(losses)), inf if no losses
    max_drawdown_pct: float       # peak-to-trough from equity curve
    sharpe_ratio: float           # annualized: mean(daily_returns) / std * sqrt(365)

class StatsCalculator:
    def calculate(self, closed_trades: list[PaperTradeRecord],
                  initial_equity: float) -> PerformanceStats:
        """Compute all 5 metrics from closed trade history."""
```

**Metric Definitions:**

| Metric | Formula | Edge Case |
|--------|---------|-----------|
| Total PnL | `sum(trade.pnl for all closed)` | 0 if no trades |
| Win Rate | `count(pnl > 0) / count(all)` | 0.0 if no trades |
| Profit Factor | `sum(pnl where pnl > 0) / abs(sum(pnl where pnl < 0))` | `inf` if no losses, 0.0 if no wins |
| Max Drawdown | `max((peak - trough) / peak)` from equity curve | 0.0 if equity never declined |
| Sharpe Ratio | `mean(daily_returns) / std(daily_returns) * sqrt(365)` | 0.0 if < 2 days of data |

**Trigger:** Called by PaperEngine after each `close_position()`. Result saved to `account_snapshots` (extended with stats fields).

### Storage Extensions

Extend `AccountSnapshotRecord` with new fields:

```python
# New fields in account_snapshots table
total_pnl: float = 0.0
win_rate: float = 0.0
profit_factor: float = 0.0
max_drawdown_pct: float = 0.0
sharpe_ratio: float = 0.0
total_trades: int = 0
```

## Module B: Eval Framework

### Golden Dataset (`eval/datasets/golden_v1.yaml`)

5 representative market scenarios:

| ID | Scenario | Expected Side |
|----|----------|--------------|
| bull_breakout | Strong uptrend, breakout above resistance, normal funding | long |
| bear_divergence | Price rising but OI dropping, funding elevated | short or flat |
| sideways_range | Low volatility, price oscillating in range | flat |
| high_volatility | Large candles, wide range, elevated funding | flat or reduced confidence |
| funding_anomaly | Extreme funding rate (>0.05%), otherwise bullish | flat or cautious long |

Each case contains:
- `snapshot`: Full MarketSnapshot data (OHLCV, funding, OI, price)
- `expected.sentiment`: Expected SentimentReport constraints
- `expected.market`: Expected MarketInterpretation constraints
- `expected.proposal`: Expected TradeProposal constraints

### Dataset Model (`eval/dataset.py`)

```python
class ExpectedRange(BaseModel, frozen=True):
    min: float | None = None
    max: float | None = None

class ExpectedSentiment(BaseModel, frozen=True):
    sentiment_score: ExpectedRange | None = None
    confidence: ExpectedRange | None = None

class ExpectedMarket(BaseModel, frozen=True):
    trend: list[str] | None = None           # acceptable values
    volatility_regime: list[str] | None = None

class ExpectedProposal(BaseModel, frozen=True):
    side: list[str] | None = None            # acceptable values
    confidence: ExpectedRange | None = None
    has_stop_loss: bool | None = None
    sl_correct_side: bool | None = None      # SL below entry for long, above for short

class EvalCase(BaseModel, frozen=True):
    id: str
    description: str
    snapshot: dict                            # raw snapshot data
    expected: ExpectedOutputs

class ExpectedOutputs(BaseModel, frozen=True):
    sentiment: ExpectedSentiment | None = None
    market: ExpectedMarket | None = None
    proposal: ExpectedProposal | None = None
```

### Scorers (`eval/scorers.py`)

**Rule-Based Scorer:**
- Discrete fields: exact match against allowed values list
- Numeric fields: range check (min/max)
- Boolean conditions: direct check

```python
class ScoreResult(BaseModel, frozen=True):
    field: str
    passed: bool
    expected: str
    actual: str
    reason: str = ""

class RuleScorer:
    def score_sentiment(self, output: SentimentReport, expected: ExpectedSentiment) -> list[ScoreResult]
    def score_market(self, output: MarketInterpretation, expected: ExpectedMarket) -> list[ScoreResult]
    def score_proposal(self, output: TradeProposal, expected: ExpectedProposal) -> list[ScoreResult]
```

**LLM-as-Judge Scorer:**
- Used only for `rationale` field
- Prompt: "Does this rationale align with the expected scenario? Scenario: {description}. Rationale: {rationale}. Score 1-5."
- Pass threshold: score >= 3

```python
class LLMJudgeScorer:
    async def score_rationale(self, rationale: str, scenario_description: str) -> ScoreResult
```

### Eval Runner (`eval/runner.py`)

```python
class EvalRunner:
    async def run(self, dataset_path: str) -> EvalReport:
        """Load dataset → run each case through agents → score → return report."""

class EvalReport(BaseModel, frozen=True):
    dataset_name: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    accuracy: float                          # passed / total
    consistency_score: float | None = None   # from self-consistency (if run)
    case_results: list[CaseResult]

class CaseResult(BaseModel, frozen=True):
    case_id: str
    passed: bool
    scores: list[ScoreResult]
    consistency: float | None = None
```

### Self-Consistency (`eval/consistency.py`)

```python
class ConsistencyChecker:
    async def check(self, case: EvalCase, agent: BaseAgent, runs: int = 3) -> float:
        """Run same input N times, return consistency score (0.0-1.0)."""
        # For discrete fields: most_common_count / N
        # For numeric fields: 1 - (std / mean) capped at [0, 1]
```

## Module C: TG + CLI Enhancements

### New TG Commands

| Command | Function |
|---------|----------|
| `/perf` | Output performance report (5 metrics) |
| `/perf 7d` | Performance for last 7 days (optional filter) |
| `/eval` | Trigger eval run and output summary |

### New Formatters

`format_perf_report(stats: PerformanceStats) -> str`:
```
📊 Performance Report
─────────────────────
Total PnL:      +$1,250.00 (+12.5%)
Win Rate:       62.5% (10/16)
Profit Factor:  1.85
Max Drawdown:   -4.2%
Sharpe Ratio:   1.32
─────────────────────
Period: 2026-02-01 ~ 2026-02-19
```

`format_eval_report(report: EvalReport) -> str`:
```
🧪 Eval Report (golden_v1)
──────────────────────────
Cases: 5 | Passed: 4 | Failed: 1
Accuracy: 80%
Consistency: 93.3%
──────────────────────────
❌ bear_divergence: expected SHORT, got LONG
```

### CLI Subcommands

Extend `__main__.py` with argparse subcommands:

```bash
python -m orchestrator              # default: start bot + scheduler
python -m orchestrator eval         # run eval dataset
python -m orchestrator eval --dataset golden_v1
python -m orchestrator perf         # output performance stats
python -m orchestrator perf --days 7
```

### Test Coverage Fixes

| Module | Current | Target | Strategy |
|--------|---------|--------|----------|
| `bot.py` | ~30% | 80% | Mock Update/Context objects, test each handler's logic branch |
| `scheduler.py` | 76% | 80% | Mock APScheduler, test start/stop/graceful shutdown |

## Out of Scope

- Web dashboard / Streamlit UI (future)
- Backtesting on historical data (future — different from eval)
- Automated prompt optimization based on eval results (future)
- Multi-model comparison eval (future)
- Real-time monitoring alerts (future)
