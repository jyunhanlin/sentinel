from orchestrator.__main__ import create_app_components


def test_create_app_components():
    """Verify we can construct core components without starting services."""
    components = create_app_components(
        telegram_bot_token="test-token",
        telegram_admin_chat_ids=[123],
        exchange_id="binance",
        database_url="sqlite:///:memory:",
        anthropic_api_key="test-key",
        llm_model="anthropic/claude-sonnet-4-6",
        llm_model_premium="anthropic/claude-opus-4-6",
        llm_temperature=0.2,
        llm_max_tokens=2000,
        llm_max_retries=1,
        pipeline_symbols=["BTC/USDT:USDT"],
        pipeline_interval_minutes=15,
    )
    assert "bot" in components
    assert "exchange_client" in components
    assert "db_engine" in components
    assert "scheduler" in components
    assert "runner" in components


def test_create_app_components_includes_m2():
    """Verify M2 components (risk_checker, paper_engine) are in output."""
    components = create_app_components(
        telegram_bot_token="test-token",
        telegram_admin_chat_ids=[123],
        exchange_id="binance",
        database_url="sqlite:///:memory:",
        anthropic_api_key="test-key",
    )
    assert "paper_engine" in components
    assert "risk_checker" in components


def test_create_app_components_includes_m4():
    """Verify M4 components (approval_manager, executor) are in output."""
    components = create_app_components(
        telegram_bot_token="test-token",
        telegram_admin_chat_ids=[123],
        exchange_id="binance",
        database_url="sqlite:///:memory:",
        anthropic_api_key="test-key",
        trading_mode="paper",
    )
    assert "approval_manager" in components
    assert "executor" in components


def test_create_app_components_includes_m3():
    """Verify M3 components (stats_calculator, eval_runner) are in output."""
    components = create_app_components(
        telegram_bot_token="test-token",
        telegram_admin_chat_ids=[123],
        exchange_id="binance",
        database_url="sqlite:///:memory:",
        anthropic_api_key="test-key",
    )
    assert "stats_calculator" in components
    assert "eval_runner" in components
