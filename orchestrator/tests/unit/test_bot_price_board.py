from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.exchange.data_fetcher import TickerSummary


class TestPriceBoardUpdate:
    @pytest.mark.asyncio
    async def test_update_price_board_sends_and_pins_first_time(self):
        """First call should send a new message and pin it."""
        from orchestrator.telegram.bot import SentinelBot

        bot = SentinelBot(token="test", admin_chat_ids=[123, 456])
        mock_app = MagicMock()
        mock_bot = AsyncMock()
        mock_app.bot = mock_bot
        bot._app = mock_app

        sent_msg = MagicMock()
        sent_msg.message_id = 99
        mock_bot.send_message.return_value = sent_msg

        summaries = [
            TickerSummary(symbol="BTC/USDT:USDT", price=69000.0, change_24h_pct=1.0),
        ]

        await bot.update_price_board(summaries)

        # Should send to both admin chats
        assert mock_bot.send_message.call_count == 2
        # Should pin both messages
        assert mock_bot.pin_chat_message.call_count == 2
        # Should store message ids
        assert bot._price_board_msg_ids[123] == 99
        assert bot._price_board_msg_ids[456] == 99

    @pytest.mark.asyncio
    async def test_update_price_board_edits_existing_message(self):
        """Subsequent calls should edit the existing pinned message."""
        from orchestrator.telegram.bot import SentinelBot

        bot = SentinelBot(token="test", admin_chat_ids=[123])
        mock_app = MagicMock()
        mock_bot = AsyncMock()
        mock_app.bot = mock_bot
        bot._app = mock_app

        # Simulate existing pinned message
        bot._price_board_msg_ids = {123: 99}

        summaries = [
            TickerSummary(symbol="BTC/USDT:USDT", price=69500.0, change_24h_pct=1.5),
        ]

        await bot.update_price_board(summaries)

        # Should edit, not send new
        mock_bot.send_message.assert_not_called()
        mock_bot.edit_message_text.assert_called_once()
        call_kwargs = mock_bot.edit_message_text.call_args.kwargs
        assert call_kwargs["chat_id"] == 123
        assert call_kwargs["message_id"] == 99

    @pytest.mark.asyncio
    async def test_update_price_board_resends_if_edit_fails(self):
        """If edit fails (message deleted), send a new one and re-pin."""
        from telegram.error import BadRequest

        from orchestrator.telegram.bot import SentinelBot

        bot = SentinelBot(token="test", admin_chat_ids=[123])
        mock_app = MagicMock()
        mock_bot = AsyncMock()
        mock_app.bot = mock_bot
        bot._app = mock_app

        bot._price_board_msg_ids = {123: 99}
        mock_bot.edit_message_text.side_effect = BadRequest("Message to edit not found")

        sent_msg = MagicMock()
        sent_msg.message_id = 200
        mock_bot.send_message.return_value = sent_msg

        summaries = [
            TickerSummary(symbol="BTC/USDT:USDT", price=69000.0, change_24h_pct=1.0),
        ]

        await bot.update_price_board(summaries)

        # Should fall back to send + pin
        mock_bot.send_message.assert_called_once()
        mock_bot.pin_chat_message.assert_called_once()
        assert bot._price_board_msg_ids[123] == 200

    @pytest.mark.asyncio
    async def test_update_price_board_noop_when_app_is_none(self):
        from orchestrator.telegram.bot import SentinelBot

        bot = SentinelBot(token="test", admin_chat_ids=[123])
        # _app is None by default

        summaries = [
            TickerSummary(symbol="BTC/USDT:USDT", price=69000.0, change_24h_pct=1.0),
        ]

        # Should not raise
        await bot.update_price_board(summaries)
