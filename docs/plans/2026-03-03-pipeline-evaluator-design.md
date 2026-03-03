# Pipeline Evaluator Design

**Date:** 2026-03-03
**Status:** Approved

## Goal

Build a `PipelineEvaluator` module that compares pipeline proposals against actual trade results, providing accuracy metrics and feedback. Pure computation — no LLM agent.

## Motivation

- Pipeline generates `TradeProposal` → trades execute → results stored
- Currently no way to measure "how good are pipeline suggestions?"
- Need a feedback loop: proposal accuracy → identify weak spots → improve over time

## Non-Goals

- No LLM-based qualitative analysis (future phase)
- No automatic parameter tuning (future phase)
- No changes to existing pipeline flow

## Data Flow

```
TradeProposalRecord + PaperTradeRecord
        ↓  (join by proposal_id)
  PipelineEvaluator.evaluate()
        ↓
  EvaluationReport
        ↓
  format_evaluation_report() → Telegram /evaluate
  format_trade_evaluation() → appended to close report
```

## Data Models

### EvaluationReport (frozen Pydantic)

```python
class TradeEvaluation(BaseModel, frozen=True):
    """Single trade vs proposal comparison."""
    trade_id: str
    proposal_id: str
    symbol: str
    direction_correct: bool          # proposal.side matched profitable outcome
    entry_deviation_pct: float | None  # (actual - proposed) / proposed * 100
    close_reason: str                # sl, tp, liquidation, manual
    confidence: int                  # proposal confidence (1-100)
    pnl: float

class SymbolStats(BaseModel, frozen=True):
    symbol: str
    total_trades: int
    direction_accuracy: float        # 0-1
    avg_entry_deviation_pct: float
    sl_hit_rate: float
    tp_hit_rate: float
    total_pnl: float

class ConfidenceBucket(BaseModel, frozen=True):
    bucket: str                      # "low (1-33)", "mid (34-66)", "high (67-100)"
    total_trades: int
    direction_accuracy: float
    avg_pnl: float

class PeriodStats(BaseModel, frozen=True):
    period: str                      # "2026-W09" or "2026-03"
    total_trades: int
    direction_accuracy: float
    total_pnl: float

class EvaluationReport(BaseModel, frozen=True):
    # Overall
    total_evaluated: int
    total_unmatched: int             # trades without proposal
    direction_accuracy: float
    avg_entry_deviation_pct: float
    sl_hit_rate: float
    tp_hit_rate: float
    liquidation_rate: float
    # Breakdowns
    by_symbol: list[SymbolStats]
    by_confidence: list[ConfidenceBucket]
    by_period: list[PeriodStats]
    # Raw evaluations
    evaluations: list[TradeEvaluation]
```

## Module: `orchestrator/stats/evaluator.py`

### PipelineEvaluator

```python
class PipelineEvaluator:
    def __init__(
        self,
        trade_repo: PaperTradeRepository,
        proposal_repo: TradeProposalRepository,
    ): ...

    def evaluate(
        self,
        symbol: str | None = None,
        since: datetime | None = None,
    ) -> EvaluationReport:
        """Evaluate all closed trades, optionally filtered."""

    def evaluate_single(
        self,
        trade_id: str,
    ) -> TradeEvaluation | None:
        """Evaluate a single trade against its proposal."""
```

### Logic

1. Fetch closed trades (filtered by symbol/since if provided)
2. For each trade, look up proposal by `proposal_id`
3. Compare:
   - **Direction accuracy:** proposal.side == LONG and pnl > 0, or side == SHORT and pnl > 0
   - **Entry deviation:** `(trade.entry_price - proposal.entry.price) / proposal.entry.price * 100` (None if market order with no price)
   - **SL/TP/Liquidation:** from `trade.close_reason`
   - **Confidence:** from `proposal.confidence`
4. Aggregate into breakdowns

## Presentation

### 1. Telegram `/evaluate` Command

Full report with all 6 indicator groups. Format similar to `/perf`.

### 2. Close Report Attachment

When a position closes (via SL/TP/manual/reduce), append a one-line pipeline comparison:

```
📊 Pipeline: ✅ Direction correct · Entry dev +0.3% · Confidence 75%
```

or

```
📊 Pipeline: ❌ Direction wrong · Entry dev -1.2% · Confidence 40%
```

### Injection Points

- `bot.push_close_report()` — for automatic SL/TP closes
- `_handle_confirm_close()` — for manual closes
- `_handle_confirm_reduce()` — for partial reduces

All three call `evaluate_single(trade_id)` and append to the message.

## File Structure

```
orchestrator/src/orchestrator/
├── stats/
│   ├── calculator.py          # existing — unchanged
│   └── evaluator.py           # NEW — PipelineEvaluator
├── telegram/
│   ├── bot.py                 # MODIFIED — /evaluate command, close report injection
│   └── formatters.py          # MODIFIED — format_evaluation_report(), format_trade_evaluation()
```

## Dependencies

- Uses existing `PaperTradeRepository` and `TradeProposalRepository`
- No new DB tables or migrations needed
- No new external dependencies
