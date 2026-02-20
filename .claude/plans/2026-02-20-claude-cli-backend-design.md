# Claude CLI Backend Design

**Date:** 2026-02-20
**Status:** Approved

## Problem

LLM API calls via LiteLLM require a separate Anthropic API key with per-token billing. The user has a Claude Max ($100) subscription that includes CLI usage, making `claude -p` a cost-effective alternative.

## Goals

- Replace LiteLLM API calls with `claude -p` subprocess calls
- Maintain existing agent/pipeline interfaces unchanged
- Support gradual migration (per-agent backend selection)
- Keep LiteLLM as a switchable fallback

## Non-Goals

- Cron-based scheduling (use existing interval config)
- Removing LiteLLM dependency (keep as fallback)

## Architecture: Strategy Pattern

```
LLMClient (facade, unchanged call() interface)
  +-- LLMBackend (ABC)
        +-- LiteLLMBackend (existing API logic)
        +-- ClaudeCLIBackend (new, asyncio subprocess)
```

### LLMBackend Interface

```python
class LLMBackend(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMCallResult: ...
```

### LiteLLMBackend

Extracts existing `acompletion` logic from `LLMClient.call()` into a standalone backend class. No behavioral changes.

### ClaudeCLIBackend

- Uses `asyncio.create_subprocess_exec` for async subprocess calls
- Passes prompt via stdin pipe (avoids ARG_MAX limits on large prompts)
- Uses `--system-prompt` flag for system messages
- Uses `--output-format json` to get structured response envelope
- Maps model names: `anthropic/claude-sonnet-4-6` -> `sonnet`
- Configurable subprocess timeout (default 120s)

**Message conversion:**
- `system` role -> `--system-prompt` flag
- `user` + `assistant` roles -> flattened into prompt text (supports retry multi-turn)

**Output parsing:**
- Parse CLI JSON envelope to extract response content
- Feed content through existing `schema_validator` for Pydantic validation

### LLMClient Changes

`LLMClient` becomes a thin facade accepting an `LLMBackend`:

```python
class LLMClient:
    def __init__(self, backend: LLMBackend, model: str, temperature: float, max_tokens: int): ...
    async def call(self, messages, *, model=None, ...) -> LLMCallResult:
        return await self._backend.complete(messages, model=effective_model, ...)
```

## Config Changes

```python
class Settings(BaseSettings):
    # LLM (modified)
    anthropic_api_key: str = ""     # Optional (not needed for CLI mode)

    # LLM Backend (new)
    llm_backend: str = "api"        # "api" | "cli"
    claude_cli_path: str = "claude"
    claude_cli_timeout: int = 120   # seconds
```

Pipeline interval adjusted for CLI mode: `pipeline_interval_minutes: 720` (2x daily).

## Migration Strategy

| Phase | Scope | Backend |
|-------|-------|---------|
| 1 | SentimentAgent + MarketAgent | CLI |
| 1 | ProposerAgent | API (critical decision agent) |
| 2 | All agents | CLI |
| 3 | Remove LiteLLM (optional) | CLI only |

## File Changes

| File | Action |
|------|--------|
| `llm/backend.py` | **New** -- ABC + LiteLLMBackend + ClaudeCLIBackend |
| `llm/client.py` | **Modified** -- Accept backend injection |
| `config.py` | **Modified** -- New backend settings |
| Pipeline setup code | **Modified** -- Backend construction logic |
| `schema_validator.py` | **Unchanged** |
| `agents/*` | **Unchanged** |

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Pattern | Strategy (LLMBackend ABC) | Matches project convention (PositionSizer), clean separation |
| Subprocess I/O | stdin pipe | Avoids ARG_MAX for large prompts |
| System prompt | `--system-prompt` flag | Preserves semantic structure |
| Output parsing | `--output-format json` + schema_validator | Combines CLI envelope parsing with existing validation |
| Model mapping | Strip `anthropic/` prefix, use alias | CLI accepts `sonnet`, `opus` directly |
| Parallelism | `asyncio.create_subprocess_exec` | Maintains async gather for LLM-1 + LLM-2 |
| API key | Optional (default `""`) | CLI mode doesn't need it |

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| CLI rate limiting | Low frequency (2x/day), Claude Max has high limits |
| CLI output format changes | Pin claude CLI version, add integration tests |
| Subprocess hangs | Configurable timeout, process kill on timeout |
| Retry multi-turn degradation | Flatten conversation history preserves context |
| CLAUDECODE env var blocking | Unset env var before subprocess call |
