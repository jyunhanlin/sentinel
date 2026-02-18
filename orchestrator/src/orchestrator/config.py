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
    anthropic_api_key: str

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
