from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "", "env_file": ".env", "env_file_encoding": "utf-8"}

    # Telegram
    telegram_bot_token: str
    telegram_admin_chat_ids: list[int]

    # Exchange
    exchange_id: str = "binance"
    exchange_api_key: str = ""
    exchange_api_secret: str = ""

    # LLM
    anthropic_api_key: str = ""
    llm_model: str = "anthropic/claude-sonnet-4-6"
    llm_model_premium: str = "anthropic/claude-opus-4-6"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2000
    llm_max_retries: int = 1

    # LLM Backend
    llm_backend: str = "api"                         # "api" | "cli"
    claude_cli_path: str = "claude"
    claude_cli_timeout: int = 120                    # seconds

    # Database
    database_url: str = "sqlite:///data/sentinel.db"

    # Pipeline
    pipeline_interval_minutes: int = 15
    pipeline_symbols: list[str] = Field(default=["BTC/USDT:USDT", "ETH/USDT:USDT"])

    # Risk
    max_single_risk_pct: float = 2.0
    max_total_exposure_pct: float = 20.0
    max_daily_loss_pct: float = 5.0
    max_consecutive_losses: int = 5

    # Paper Trading
    paper_initial_equity: float = 10000.0
    paper_taker_fee_rate: float = 0.0005   # 0.05%
    paper_maker_fee_rate: float = 0.0002   # 0.02%

    # Semi-auto Trading
    trading_mode: str = "paper"                    # "paper" | "live"
    approval_timeout_minutes: int = 15
    price_deviation_threshold: float = 0.01        # 1%
