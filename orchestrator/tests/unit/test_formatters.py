from orchestrator.exchange.data_fetcher import TickerSummary
from orchestrator.telegram.formatters import _compact_price, format_price_board


class TestCompactPrice:
    def test_thousands(self):
        assert _compact_price(69123.5) == "69.1K"

    def test_low_thousands(self):
        assert _compact_price(2530.8) == "2.5K"

    def test_below_thousand(self):
        assert _compact_price(142.35) == "142"

    def test_exact_thousand(self):
        assert _compact_price(1000.0) == "1.0K"


class TestFormatPriceBoard:
    def test_formats_multiple_symbols(self):
        summaries = [
            TickerSummary(symbol="BTC/USDT:USDT", price=69123.5, change_24h_pct=1.2),
            TickerSummary(symbol="ETH/USDT:USDT", price=2530.8, change_24h_pct=-0.5),
        ]
        result = format_price_board(summaries)

        # Compact summary line (first line, visible in pin preview)
        first_line = result.split("\n")[0]
        assert "BTC 69.1K(+1.2%)" in first_line
        assert "ETH 2.5K(-0.5%)" in first_line

        # Detailed section
        assert "Price Board" in result
        assert "BTC/USDT" in result
        assert "$69,123.5" in result
        assert "+1.20%" in result
        assert "ETH/USDT" in result
        assert "$2,530.8" in result
        assert "-0.50%" in result
        assert "Updated:" in result

    def test_formats_empty_list(self):
        result = format_price_board([])
        assert "Price Board" in result
        assert "No symbols" in result

    def test_positive_change_has_plus_sign(self):
        summaries = [
            TickerSummary(symbol="SOL/USDT:USDT", price=142.35, change_24h_pct=3.1),
        ]
        result = format_price_board(summaries)
        assert "+3.10%" in result
        assert "SOL 142(+3.1%)" in result.split("\n")[0]

    def test_negative_change_has_minus_sign(self):
        summaries = [
            TickerSummary(symbol="SOL/USDT:USDT", price=142.35, change_24h_pct=-2.5),
        ]
        result = format_price_board(summaries)
        assert "-2.50%" in result
        assert "SOL 142(-2.5%)" in result.split("\n")[0]

    def test_compact_line_is_first(self):
        """First line should be the compact summary, not the header."""
        summaries = [
            TickerSummary(symbol="BTC/USDT:USDT", price=69000.0, change_24h_pct=0.0),
        ]
        result = format_price_board(summaries)
        first_line = result.split("\n")[0]
        assert "BTC" in first_line
        assert "━━" not in first_line
