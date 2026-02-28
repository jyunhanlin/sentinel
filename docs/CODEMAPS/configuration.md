# Configuration Codemap

**Last Updated:** 2026-02-28

## Overview

All configuration via Pydantic BaseSettings with environment variables and `.env` file support.

## Settings Class

**File:** `orchestrator/src/orchestrator/config.py`

```python
class Settings(BaseSettings):
    model_config = {
        "env_prefix": "",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

    # All fields read from environment variables
    # Use env_file for local development
```

## Environment Variables

### Required

| Variable | Type | Description |
|----------|------|-------------|
| `TELEGRAM_BOT_TOKEN` | str | Telegram bot token from @BotFather |
| `TELEGRAM_ADMIN_CHAT_IDS` | str (comma-sep) | Chat IDs with access (e.g., "123,456") |

### Optional with Defaults

#### Telegram
```python
telegram_bot_token: str  # Required
telegram_admin_chat_ids: list[int]  # Required, parsed from comma-separated string
```

#### Exchange
```python
exchange_id: str = "binance"  # CCXT exchange ID
exchange_api_key: str = ""  # Required for live trading
exchange_api_secret: str = ""  # Required for live trading
```

#### LLM
```python
anthropic_api_key: str = ""  # Required for API backend
llm_model: str = "anthropic/claude-sonnet-4-6"
llm_model_premium: str = "anthropic/claude-opus-4-6"
llm_temperature: float = 0.2  # Lower = more deterministic
llm_max_tokens: int = 2000
llm_max_retries: int = 1  # Agent validation retries
```

#### LLM Backend
```python
llm_backend: str = "cli"  # "api" | "cli"
claude_cli_path: str = "claude"  # Path to claude binary
claude_cli_timeout: int = 120  # Seconds
```

#### Logging
```python
log_json: bool = False  # True for structured JSON, False for console
```

#### Database
```python
database_url: str = "sqlite:///data/sentinel.db"
```

#### Pipeline
```python
pipeline_interval_minutes: int = 720  # Default: 12 hours
pipeline_symbols: list[str] = Field(default=[
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
])
```

#### Risk
```python
max_single_risk_pct: float = 2.0  # Max per-trade risk
max_total_exposure_pct: float = 20.0  # Max open exposure
max_daily_loss_pct: float = 5.0  # Daily loss circuit breaker
max_consecutive_losses: int = 5  # Consecutive loss limit
```

#### Paper Trading
```python
paper_initial_equity: float = 10000.0
paper_taker_fee_rate: float = 0.0005  # 0.05%
paper_maker_fee_rate: float = 0.0002  # 0.02%
paper_default_leverage: int = 10
paper_maintenance_margin_rate: float = 0.5  # Liquidation margin %
paper_leverage_options: list[int] = Field(default=[5, 10, 20, 50])
```

#### Price Monitor
```python
price_monitor_interval_seconds: int = 300  # Check SL/TP every 5 min
price_monitor_enabled: bool = True
```

#### Semi-Auto Trading
```python
trading_mode: str = "paper"  # "paper" | "live"
approval_timeout_minutes: int = 15
price_deviation_threshold: float = 0.01  # 1%
```

## Loading Configuration

### At Startup

**File:** `orchestrator/src/orchestrator/__main__.py`

```python
from orchestrator.config import Settings

# Settings reads from environment + .env file
settings = Settings()

# Access values
print(settings.telegram_bot_token)
print(settings.pipeline_symbols)
print(settings.max_single_risk_pct)
```

### In Components

Configuration passed to component factories:

```python
def create_app_components(
    *,
    telegram_bot_token: str,
    telegram_admin_chat_ids: list[int],
    exchange_id: str,
    database_url: str,
    anthropic_api_key: str = "",
    llm_model: str = "anthropic/claude-sonnet-4-6",
    ...
) -> tuple[...]:
    """Create application components from config."""

    # Create LLM client
    llm_client = LLMClient(
        api_key=anthropic_api_key,
        model=llm_model,
        temperature=llm_temperature,
    )

    # Create exchange client
    exchange_client = ExchangeClient(
        exchange_id=exchange_id,
        api_key=exchange_api_key,
        api_secret=exchange_api_secret,
    )

    # Create risk checker
    risk_checker = RiskChecker(
        max_single_risk_pct=max_single_risk_pct,
        max_total_exposure_pct=max_total_exposure_pct,
        max_consecutive_losses=max_consecutive_losses,
        max_daily_loss_pct=max_daily_loss_pct,
    )

    # ... more components
```

## Environment Variable Parsing

### List Parsing

Comma-separated strings parsed to lists:

```python
# In .env
TELEGRAM_ADMIN_CHAT_IDS=123456789,987654321
PIPELINE_SYMBOLS=BTC/USDT:USDT,ETH/USDT:USDT

# In Python
settings = Settings()
assert settings.telegram_admin_chat_ids == [123456789, 987654321]
assert settings.pipeline_symbols == ["BTC/USDT:USDT", "ETH/USDT:USDT"]
```

### Type Conversion

Pydantic handles conversions:

```python
# String to int
max_single_risk_pct: float = 2.0

# String to bool
log_json: bool = False

# String to int list
paper_leverage_options: list[int] = [5, 10, 20, 50]
```

## Example .env File

```bash
# .env

# Required
ANTHROPIC_API_KEY=sk-ant-...
TELEGRAM_BOT_TOKEN=1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh
TELEGRAM_ADMIN_CHAT_IDS=123456789,987654321

# Exchange (live mode only)
EXCHANGE_ID=binance
EXCHANGE_API_KEY=
EXCHANGE_API_SECRET=

# LLM
LLM_MODEL=anthropic/claude-sonnet-4-6
LLM_MODEL_PREMIUM=anthropic/claude-opus-4-6
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=2000

# LLM Backend
LLM_BACKEND=cli
CLAUDE_CLI_PATH=claude
CLAUDE_CLI_TIMEOUT=120

# Database
DATABASE_URL=sqlite:///data/sentinel.db

# Pipeline
PIPELINE_INTERVAL_MINUTES=15
PIPELINE_SYMBOLS=BTC/USDT:USDT,ETH/USDT:USDT

# Risk
MAX_SINGLE_RISK_PCT=2.0
MAX_TOTAL_EXPOSURE_PCT=20.0
MAX_DAILY_LOSS_PCT=5.0
MAX_CONSECUTIVE_LOSSES=5

# Paper Trading
PAPER_INITIAL_EQUITY=10000.0
PAPER_TAKER_FEE_RATE=0.0005
PAPER_MAKER_FEE_RATE=0.0002
PAPER_DEFAULT_LEVERAGE=10
PAPER_LEVERAGE_OPTIONS=5,10,20,50

# Price Monitor
PRICE_MONITOR_INTERVAL_SECONDS=300
PRICE_MONITOR_ENABLED=true

# Semi-Auto Trading
TRADING_MODE=paper
APPROVAL_TIMEOUT_MINUTES=15
PRICE_DEVIATION_THRESHOLD=0.01

# Logging
LOG_JSON=false
```

## LLM Backend Selection

### API Backend (LiteLLM)

Uses Anthropic API directly:

```bash
LLM_BACKEND=api
ANTHROPIC_API_KEY=sk-ant-...
LLM_MODEL=anthropic/claude-sonnet-4-6
```

Requires:
- ANTHROPIC_API_KEY set
- Network access to api.anthropic.com

Pros:
- Works anywhere
- Standard LLM pricing

Cons:
- Requires API key in environment
- Network latency

### CLI Backend (Claude CLI)

Uses `claude` command-line tool:

```bash
LLM_BACKEND=cli
CLAUDE_CLI_PATH=claude
CLAUDE_CLI_TIMEOUT=120
```

Requires:
- Claude CLI installed and in PATH: `brew install anthropic/brew/claude-cli`
- Configured with `claude login`

Pros:
- No API key in environment
- May have better caching
- Uses local credentials

Cons:
- Requires CLI installation
- Timeout at 120 seconds

Switching:

```python
# Check which backend is active
if settings.llm_backend == "cli":
    backend = ClaudeCLIBackend(
        path=settings.claude_cli_path,
        timeout=settings.claude_cli_timeout,
    )
else:
    backend = LiteLLMBackend()

client = LLMClient(backend=backend)
```

## Trading Mode

### Paper Mode (Default)

```bash
TRADING_MODE=paper
PAPER_INITIAL_EQUITY=10000.0
```

Simulates trading on PaperEngine. No real orders placed.

Use for:
- Development
- Testing
- Demo

### Live Mode

```bash
TRADING_MODE=live
EXCHANGE_API_KEY=...
EXCHANGE_API_SECRET=...
```

Places real orders on exchange. Requires API credentials.

Use for:
- Production trading

Changing modes:
- **Don't mix in same run** — Choose one, restart application
- **Paper positions don't carry over** — Fresh start each mode switch

## Database Configuration

### SQLite (Default)

```bash
DATABASE_URL=sqlite:///data/sentinel.db
```

Location relative to current working directory. Creates `data/` directory if missing.

### Absolute Path

```bash
DATABASE_URL=sqlite:////var/sentinel/data.db
```

Note: 4 slashes for absolute paths (`/var/...`).

## Validation

Configuration is validated at startup:

```python
settings = Settings()
# Raises ValidationError if:
# - Required fields missing
# - Type conversion fails
# - Custom validators fail
```

Example error:

```
ValidationError: 1 validation error for Settings
anthropic_api_key
  Field required [type=missing, input_value={...}]
```

### Custom Validators

```python
# Example: Max daily loss must be < 10%
@field_validator("max_daily_loss_pct")
@classmethod
def validate_max_daily_loss(cls, v):
    if v > 10.0:
        raise ValueError("Max daily loss cannot exceed 10%")
    return v
```

## Accessing Configuration

### In __main__.py

```python
settings = Settings()

# Pass to components
runner = PipelineRunner(
    max_single_risk_pct=settings.max_single_risk_pct,
    max_total_exposure_pct=settings.max_total_exposure_pct,
)
```

### In Components

Components receive config as constructor args, not global:

```python
class RiskChecker:
    def __init__(
        self,
        *,
        max_single_risk_pct: float,
        max_total_exposure_pct: float,
        ...
    ):
        self._max_single_risk_pct = max_single_risk_pct
        # Store as instance variables
```

**NOT:**
```python
# Bad: Global config access
from orchestrator.config import Settings
settings = Settings()
```

This keeps components testable with mocked config.

## Development vs Production

### Development (.env)

```bash
TRADING_MODE=paper
PIPELINE_INTERVAL_MINUTES=1
LLM_TEMPERATURE=0.5
PAPER_INITIAL_EQUITY=1000.0  # Smaller account
```

### Production

```bash
TRADING_MODE=live
PIPELINE_INTERVAL_MINUTES=15
LLM_TEMPERATURE=0.2
PAPER_INITIAL_EQUITY=10000.0
MAX_DAILY_LOSS_PCT=5.0  # Strict limits
```

## Secrets Management

### Local Development

Use `.env` file (git-ignored):

```bash
# .env (not committed)
ANTHROPIC_API_KEY=sk-ant-...
TELEGRAM_BOT_TOKEN=...
EXCHANGE_API_SECRET=...
```

```bash
# .gitignore
.env
.env.local
```

### Production

Use environment variables:

```bash
# Set in deployment system (GitHub Actions, Docker, etc.)
export ANTHROPIC_API_KEY=sk-ant-...
export TELEGRAM_BOT_TOKEN=...
export EXCHANGE_API_SECRET=...

# Start app
uv run python -m orchestrator
```

Or use secrets manager:

```bash
# Docker
docker run \
  -e ANTHROPIC_API_KEY=$(vault kv get secret/sentinel/api_key) \
  sentinel
```

## Testing Configuration

**File:** `tests/conftest.py`

Override config for tests:

```python
@pytest.fixture
def settings():
    """Test settings with minimal config."""
    return Settings(
        telegram_bot_token="test_token",
        telegram_admin_chat_ids=[123],
        exchange_id="binance",
        database_url="sqlite:///:memory:",
        anthropic_api_key="test_key",
        trading_mode="paper",
        paper_initial_equity=1000.0,
    )
```

Use in tests:

```python
def test_pipeline(settings):
    risk_checker = RiskChecker(
        max_single_risk_pct=settings.max_single_risk_pct,
        ...
    )
    # Test with settings from fixture
```

## Debugging Configuration

Print active settings:

```python
settings = Settings()
print(settings.model_dump(exclude={"anthropic_api_key", "exchange_api_secret"}))

# Output
# {
#     'telegram_bot_token': '1234567890:ABC...',
#     'telegram_admin_chat_ids': [123456789],
#     'exchange_id': 'binance',
#     'llm_model': 'anthropic/claude-sonnet-4-6',
#     'trading_mode': 'paper',
#     'pipeline_interval_minutes': 15,
#     'max_single_risk_pct': 2.0,
#     ...
# }
```

## Related

- [pipeline.md](pipeline.md) — Configuration usage
- [exchange.md](exchange.md) — Exchange settings
- [risk.md](risk.md) — Risk settings
- [approval-telegram.md](approval-telegram.md) — Telegram settings
