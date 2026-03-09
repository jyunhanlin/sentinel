# Proposer Refinement Loop Design

**Date:** 2026-03-09
**Status:** Approved

## Problem

The proposer agent generates a single-pass trade proposal. There is no mechanism to
evaluate proposal quality beyond basic sanity checks (SL direction, missing SL on
directional trades). Higher-quality proposals require iterative refinement — a critic
reviews the proposal against analysis inputs and the proposer revises based on feedback.

## Solution

Add a **CriticAgent** and a **RefinementLoop** that coordinates Proposer ↔ Critic
iterations. The loop is **opt-in** — only enabled for specific pipeline runs (e.g.,
daily premium runs or manual `/run` with refinement flag).

## Architecture

```
Pipeline Step 3 (current):
  Proposer → TradeProposal → Aggregator → done

Pipeline Step 3 (with refinement):
  Proposer → TradeProposal (v1)
      ↓
  [refinement_enabled?] ── no → Aggregator → done
      │ yes
      ↓
  CriticAgent → CritiqueResult
      ↓
    pass? → Aggregator → done
    fail? → Proposer re-generate (with critique feedback) → CriticAgent again
      ↓
    max_rounds reached → accept last proposal + flag refinement_exhausted
                        → Aggregator → done
```

## New Components

### 1. CritiqueResult Model (`models.py`)

```python
class DimensionVerdict(BaseModel, frozen=True):
    dimension: str        # "consistency" | "risk_reward" | "input_respect" | "parameter_sanity"
    passed: bool
    reason: str

class CritiqueResult(BaseModel, frozen=True):
    verdicts: list[DimensionVerdict]
    overall_passed: bool
    suggestions: list[str]   # concrete improvements for proposer
    summary: str              # one-line summary for logging/display
```

### 2. CriticAgent (`agents/critic.py`)

- Inherits `BaseAgent[CritiqueResult]`
- Has its own SKILL.md at `.claude/skills/critic/SKILL.md`
- Receives: `TradeProposal` + all 5 analysis outputs + `MarketSnapshot`
- Evaluates 4 dimensions (5th — historical calibration — reserved for phase 2):

| Dimension | What it checks |
|-----------|---------------|
| **Consistency** | Side/entry/SL/TP matches rationale and analysis results |
| **Risk/Reward** | R:R ratio reasonable, position size proportional to confidence |
| **Input Respect** | No ignored warnings (e.g., catalyst=wait but still opening) |
| **Parameter Sanity** | SL/TP distances reasonable for volatility, leverage appropriate |

### 3. RefinementLoop (`pipeline/refinement.py`)

Coordinates the Proposer ↔ Critic iteration:

```python
class RefinementLoop:
    def __init__(
        self,
        proposer: BaseAgent[TradeProposal],
        critic: BaseAgent[CritiqueResult],
        max_rounds: int = 2,
    ) -> None: ...

    async def run(
        self,
        *,
        model_override: str | None = None,
        **proposer_kwargs,
    ) -> RefinementResult: ...
```

```python
class RefinementResult(BaseModel, frozen=True):
    proposal: TradeProposal
    critique: CritiqueResult | None    # last critique (None if refinement disabled)
    rounds: int                        # 0 = no refinement, 1 = first critique passed, etc.
    exhausted: bool                    # True if max_rounds reached without passing
    all_llm_calls: list[LLMCallResult] # from both proposer + critic across all rounds
    proposer_degraded: bool
```

**Loop logic:**

1. Call Proposer → get TradeProposal
2. Call Critic with (proposal + analysis inputs) → get CritiqueResult
3. If `overall_passed` → return proposal
4. If not passed and rounds < max_rounds:
   - Build new Proposer prompt with original data + critique feedback
   - Go to step 1
5. If rounds >= max_rounds → return last proposal with `exhausted=True`

**Re-prompting strategy:** When the Proposer is re-invoked after a failed critique, the
prompt includes the original market data, the previous proposal, and the Critic's
`suggestions` list. The Proposer is instructed to address each suggestion specifically.

## Integration

### runner.py Changes

Replace the current single Proposer call (Step 3) with:

```python
# Step 3: Run Proposer (with optional refinement)
if self._refinement_loop is not None:
    refinement_result = await self._refinement_loop.run(
        snapshot=snapshot,
        technical_short=tech_short_result.output,
        ...,
        model_override=model_override,
    )
    # Extract proposal, save all LLM calls, set refinement metadata
else:
    # Existing single-pass behavior
    proposer_result = await self._proposer_agent.analyze(...)
```

### PipelineResult Changes

New fields:

```python
class PipelineResult(BaseModel, frozen=True):
    # ... existing fields ...
    refinement_rounds: int = 0
    refinement_exhausted: bool = False
    critique: CritiqueResult | None = None
```

### PipelineRunner.__init__ Changes

Accept optional `RefinementLoop`:

```python
def __init__(self, ..., refinement_loop: RefinementLoop | None = None) -> None:
```

### Config Changes

```python
class Settings(BaseSettings):
    # ... existing fields ...
    refinement_enabled: bool = False
    refinement_max_rounds: int = 2
```

### Scheduler Integration

- **Interval runs (Sonnet):** `refinement_enabled=False` — no change to current behavior
- **Daily premium runs (Opus):** `refinement_enabled=True` — Critic loop enabled
- **Manual `/run` with refinement flag:** e.g., `/run BTC opus refine`

The `PipelineRunner.execute()` accepts a new `refinement` parameter. The scheduler
passes this based on run type. The `RefinementLoop` is always injected into the runner,
but only activated when `refinement=True` is passed to `execute()`.

## Cost Analysis

| Scenario | Proposer Calls | Critic Calls | Total Extra |
|----------|---------------|-------------|-------------|
| Refinement disabled | 1 | 0 | 0 |
| First critique passes | 1 | 1 | +1 call |
| 1 revision needed | 2 | 2 | +3 calls |
| max_rounds=2 exhausted | 3 | 2 | +4 calls |

With `max_rounds=2`, worst case is 5 total LLM calls for the proposer+critic stage
(vs. 1 today). This only applies to premium runs.

## Future Extensions (Not in Scope)

- **Historical Calibration dimension:** Add a 5th evaluation dimension using past trade
  records once enough data is accumulated. Critic SKILL.md already has a placeholder.
- **Critique analytics:** Store CritiqueResult in DB to analyze which dimensions fail
  most frequently.
- **Adaptive max_rounds:** Dynamically adjust based on historical pass rates.

## Testing Strategy

- **Unit tests:** CriticAgent with mock LLM responses, RefinementLoop with mock agents
- **Integration test:** Full pipeline run with refinement enabled using fixtures
- **Test scenarios:**
  - Critic passes on first round → 1 proposer + 1 critic call
  - Critic fails, proposer revises successfully → 2 proposer + 2 critic calls
  - Critic fails all rounds → exhausted flag set, last proposal returned
  - Proposer degrades → refinement skipped, degraded result returned
