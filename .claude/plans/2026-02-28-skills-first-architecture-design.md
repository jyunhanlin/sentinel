# Skills-First Architecture Design

**Date:** 2026-02-28
**Status:** Approved
**Reference:** [Don't Build Agents, Build Skills](https://blog.aihao.tw/2026/02/24/dont-build-agents-build-skills/)

## Problem

Agent prompts are embedded inside Python classes (`orchestrator/src/orchestrator/agents/`).
This couples domain knowledge (how to analyze) with orchestration logic (what data to pass, how to parse results).

Current issues:
- Prompts force "Respond with ONLY JSON" which limits LLM thinking quality
- Prompt content changes require Python code changes
- No separation between analysis methodology and data pipeline
- No room for few-shot examples or chain-of-thought guidance

## Goals

1. **Separation of concerns** — Python handles orchestration (data fetching, result parsing); skills own domain knowledge (analysis methodology)
2. **Better analysis quality** — Skills guide LLM through structured thinking before outputting JSON
3. **Independent iteration** — Skills can be tuned without touching Python code
4. **RAG extension point** — Architecture supports injecting historical trade context in the future
5. **Token efficiency** — Output schemas include only fields consumed downstream

## Architecture

### Before

```
Python Agent (prompt + data formatting + parsing)
  → claude -p --system-prompt "You are..." < formatted_data
  → Parse pure JSON response
```

### After

```
Python Agent (data formatting + parsing only)
  → claude -p "Use the {skill_name} skill to analyze: {data}"
  → Claude reads .claude/skills/{name}/SKILL.md via Read tool
  → Claude thinks through methodology
  → Claude outputs analysis + JSON block
  → Python extracts JSON from fenced code block
```

### Data Flow

```
MarketSnapshot
  ├──→ [sentiment skill] → SentimentReport ──┐
  ├──→ [market skill]    → MarketInterpretation ──┤
  │                                               ▼
  └──────────────────────→ [proposer skill] → TradeProposal
```

## Skills Structure

```
.claude/skills/
├── sentiment/
│   ├── SKILL.md        # Role, methodology, decision criteria, output schema
│   └── examples.md     # 2-3 input→reasoning→output examples
├── market/
│   ├── SKILL.md
│   └── examples.md
└── proposer/
    ├── SKILL.md
    └── examples.md
```

### SKILL.md Template

Each skill follows this structure:

```markdown
---
name: <name>
description: <one-line purpose — what this skill feeds into>
---

# <Role Title>

## Context
Why this skill exists and where its output goes in the pipeline.

## Input Description
| Field | Type | Meaning |
|-------|------|---------|

## Methodology
### Step 1: <Name>
### Step 2: <Name>
...

## Decision Criteria
| Condition | Interpretation |
|-----------|---------------|

## Output
After analysis, output a single fenced JSON block.
Only fields consumed downstream are included.

## Historical Context
Conditional section — if historical data is provided, factor it in.
(RAG extension point, not implemented yet)
```

### Design Rationale

| Section | For Humans | For LLM | Token Efficiency |
|---------|-----------|---------|-----------------|
| Context | Understand skill's role in pipeline | Understand output purpose | Few lines |
| Input Description | Know what to change when data format changes | Reduce input misinterpretation | Table, compact |
| Methodology | Tune analysis steps independently | Chain-of-thought guidance | Core value |
| Decision Criteria | Centralized thresholds, easy to audit | Concrete rules, less guessing | Table, compact |
| Output | Understand schema design intent | Know exactly what to output | Only downstream-consumed fields |
| Historical Context | Future RAG placeholder | Conditional activation | 0 tokens if unused |

## Output Schema Optimization

Fields consumed by downstream agents (proposer):

**SentimentReport:**
- `sentiment_score` — used by proposer
- `key_events[].event` + `key_events[].impact` — used by proposer
- `confidence` — used by proposer
- ~~`sources`~~ — NOT consumed downstream, removed from skill output

**MarketInterpretation:**
- All fields consumed by proposer (trend, volatility_regime, volatility_pct, key_levels, risk_flags)

**TradeProposal:**
- All fields consumed by risk checker, paper engine, storage, Telegram bot

## Python Changes

### BaseAgent Refactoring

Current `_build_messages()` replaced with `_build_skill_prompt()`:

```python
class BaseAgent[T](ABC):
    _skill_name: str  # e.g., "sentiment"

    def _build_skill_prompt(self, **kwargs) -> str:
        """Build prompt that instructs Claude to read the skill and analyze data."""
        data = self._format_data(**kwargs)
        return (
            f"Use the {self._skill_name} skill to analyze the following data.\n"
            f"Read .claude/skills/{self._skill_name}/SKILL.md for instructions.\n\n"
            f"{data}"
        )

    @abstractmethod
    def _format_data(self, **kwargs) -> str:
        """Format input data as text. Subclasses implement this."""

    def _extract_json(self, response: str) -> str:
        """Extract JSON from fenced code block in response."""
```

### Agent Classes Simplified

Each agent class reduces to:
- `_skill_name` declaration
- `_format_data()` implementation (data formatting only)
- `_get_default_output()` for degraded fallback

### ClaudeCLIBackend Simplification

- Remove `--system-prompt` parameter construction
- Single prompt via stdin
- `--output-format json` still used for CLI envelope parsing

### Response Parsing

New JSON extraction from mixed content:

```python
import re

def extract_json_block(text: str) -> str:
    """Extract first ```json ... ``` block from response text."""
    match = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: try parsing entire text as JSON
    return text.strip()
```

## CLAUDE.md Update

Add skills registration:

```markdown
## Trading Skills
When running pipeline analysis:
- sentiment: .claude/skills/sentiment/SKILL.md — market sentiment analysis
- market: .claude/skills/market/SKILL.md — technical analysis
- proposer: .claude/skills/proposer/SKILL.md — trade proposal generation
```

## RAG Extension Point

Future historical context injection in Python:

```python
def _build_skill_prompt(self, *, history_context: str | None = None, **kwargs) -> str:
    data = self._format_data(**kwargs)
    prompt = f"Use the {self._skill_name} skill...\n\n{data}"
    if history_context:
        prompt += f"\n\n=== Historical Context ===\n{history_context}"
    return prompt
```

Each skill's "Historical Context" section provides instructions for using this data.

## Change Summary

| Component | Change |
|-----------|--------|
| `.claude/skills/sentiment/` | New: SKILL.md + examples.md |
| `.claude/skills/market/` | New: SKILL.md + examples.md |
| `.claude/skills/proposer/` | New: SKILL.md + examples.md |
| `agents/base.py` | Refactor: skill-based prompt + JSON block extraction |
| `agents/sentiment.py` | Simplify: remove prompt, keep data formatting |
| `agents/market.py` | Simplify: remove prompt, keep data formatting |
| `agents/proposer.py` | Simplify: remove prompt, keep data formatting |
| `llm/backend.py` | Simplify: remove system prompt construction |
| `CLAUDE.md` | Update: register skills |
| Tests | Update: match new prompt/parsing behavior |

## Risks

1. **Extra tool call overhead** — Claude reads skill file via Read tool each invocation. Mitigated by: skill files are small (<1KB), read is fast.
2. **Non-deterministic skill loading** — Claude might not read the skill correctly. Mitigated by: explicit file path in prompt, retry mechanism in BaseAgent.
3. **JSON extraction fragility** — Mixed text+JSON needs reliable parsing. Mitigated by: regex extraction with fallback to raw JSON parse.
