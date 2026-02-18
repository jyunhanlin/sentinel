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
