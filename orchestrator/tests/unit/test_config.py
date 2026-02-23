import pytest

from orchestrator.config import Settings


def test_settings_loads_defaults():
    settings = Settings(
        telegram_bot_token="test-token",
        telegram_admin_chat_ids=[123],
        anthropic_api_key="test-key",
    )
    assert settings.exchange_id == "binance"
    assert settings.pipeline_interval_minutes == 720
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


def test_settings_semi_auto_defaults():
    """Settings should have semi-auto trading config fields."""
    settings = Settings(
        telegram_bot_token="test",
        telegram_admin_chat_ids=[123],
        anthropic_api_key="test-key",
    )
    assert settings.trading_mode == "paper"
    assert settings.approval_timeout_minutes == 15
    assert settings.price_deviation_threshold == 0.01


def test_settings_llm_backend_defaults():
    settings = Settings(
        telegram_bot_token="test",
        telegram_admin_chat_ids=[123],
    )
    assert settings.llm_backend == "cli"
    assert settings.claude_cli_path == "claude"
    assert settings.claude_cli_timeout == 120


def test_settings_api_key_optional_for_cli():
    """anthropic_api_key should default to empty string for CLI mode."""
    settings = Settings(
        telegram_bot_token="test",
        telegram_admin_chat_ids=[123],
        llm_backend="cli",
    )
    assert settings.anthropic_api_key == ""
    assert settings.llm_backend == "cli"


def test_settings_paper_trading_defaults(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")
    monkeypatch.setenv("TELEGRAM_ADMIN_CHAT_IDS", "[123]")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    settings = Settings()
    assert settings.paper_initial_equity == 10000.0
    assert settings.paper_taker_fee_rate == 0.0005
    assert settings.paper_maker_fee_rate == 0.0002


def test_paper_leverage_defaults(monkeypatch):
    """New leverage config fields have sensible defaults."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")
    monkeypatch.setenv("TELEGRAM_ADMIN_CHAT_IDS", "[123]")
    s = Settings()
    assert s.paper_default_leverage == 10
    assert s.paper_maintenance_margin_rate == 0.5
    assert s.paper_leverage_options == [5, 10, 20, 50]


def test_price_monitor_defaults(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")
    monkeypatch.setenv("TELEGRAM_ADMIN_CHAT_IDS", "[123]")
    s = Settings()
    assert s.price_monitor_interval_seconds == 300
    assert s.price_monitor_enabled is True
