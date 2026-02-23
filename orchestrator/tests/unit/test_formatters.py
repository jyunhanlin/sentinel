from orchestrator.exchange.data_fetcher import TickerSummary
from orchestrator.telegram.formatters import format_price_board


class TestFormatPriceBoard:
    def test_formats_multiple_symbols(self):
        summaries = [
            TickerSummary(symbol="BTC/USDT:USDT", price=69123.5, change_24h_pct=1.2),
            TickerSummary(symbol="ETH/USDT:USDT", price=2530.8, change_24h_pct=-0.5),
        ]
        result = format_price_board(summaries)

        assert "━━ Price Board ━━" in result
        assert "BTC/USDT" in result
        assert "$69,123.5" in result
        assert "+1.20%" in result
        assert "ETH/USDT" in result
        assert "$2,530.8" in result
        assert "-0.50%" in result
        assert "Updated:" in result

    def test_formats_empty_list(self):
        result = format_price_board([])
        assert "━━ Price Board ━━" in result
        assert "No symbols" in result

    def test_positive_change_has_plus_sign(self):
        summaries = [
            TickerSummary(symbol="SOL/USDT:USDT", price=142.35, change_24h_pct=3.1),
        ]
        result = format_price_board(summaries)
        assert "+3.10%" in result

    def test_negative_change_has_minus_sign(self):
        summaries = [
            TickerSummary(symbol="SOL/USDT:USDT", price=142.35, change_24h_pct=-2.5),
        ]
        result = format_price_board(summaries)
        assert "-2.50%" in result
