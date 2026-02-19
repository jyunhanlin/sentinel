from orchestrator.agents.utils import summarize_ohlcv


class TestSummarizeOhlcv:
    def test_empty_ohlcv(self):
        assert summarize_ohlcv([], max_candles=10) == "No OHLCV data available"

    def test_summarizes_last_n(self):
        candles = [
            [1700000000000, 95000.0, 95500.0, 94800.0, 95200.0, 1000.0],
            [1700003600000, 95200.0, 95800.0, 95100.0, 95600.0, 800.0],
        ]
        text = summarize_ohlcv(candles, max_candles=5)
        assert "O=95000.0" in text
        assert "O=95200.0" in text

    def test_limits_to_max_candles(self):
        candles = [[i, float(i), float(i), float(i), float(i), 100.0] for i in range(20)]
        text = summarize_ohlcv(candles, max_candles=3)
        lines = [l for l in text.strip().split("\n") if l.strip()]
        assert len(lines) == 3
