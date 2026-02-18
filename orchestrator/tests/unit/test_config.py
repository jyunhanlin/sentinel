import pytest
from orchestrator.config import Settings


def test_settings_loads_defaults():
    settings = Settings(
        telegram_bot_token="test-token",
        telegram_admin_chat_ids=[123],
        anthropic_api_key="test-key",
    )
    assert settings.exchange_id == "binance"
    assert settings.pipeline_interval_minutes == 15
    assert settings.max_single_risk_pct == 2.0
    assert settings.database_url == "sqlite:///data/sentinel.db"


def test_settings_requires_telegram_token():
    with pytest.raises(Exception):
        Settings(
            telegram_admin_chat_ids=[123],
            anthropic_api_key="test-key",
        )


def test_settings_parses_symbols():
    settings = Settings(
        telegram_bot_token="test-token",
        telegram_admin_chat_ids=[123],
        anthropic_api_key="test-key",
        pipeline_symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
    )
    assert len(settings.pipeline_symbols) == 2
    assert settings.pipeline_symbols[0] == "BTC/USDT:USDT"
