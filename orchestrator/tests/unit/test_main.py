from orchestrator.__main__ import create_app_components


def test_create_app_components():
    """Verify we can construct core components without starting services."""
    components = create_app_components(
        telegram_bot_token="test-token",
        telegram_admin_chat_ids=[123],
        exchange_id="binance",
        database_url="sqlite:///:memory:",
        anthropic_api_key="test-key",
    )
    assert "bot" in components
    assert "exchange_client" in components
    assert "db_engine" in components
