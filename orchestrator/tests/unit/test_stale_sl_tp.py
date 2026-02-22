from orchestrator.telegram.bot import _check_stale_sl_tp


class TestCheckStaleSLTP:
    """Validate stale SL/TP detection — mirrors exchange behavior
    (Binance -2021 'Order would immediately trigger')."""

    # -- LONG positions --

    def test_long_price_below_sl_is_stale(self):
        result = _check_stale_sl_tp(
            side="long", current_price=92000.0, stop_loss=93000.0, take_profit=[97000.0]
        )
        assert result is not None
        assert "SL" in result

    def test_long_price_equal_sl_is_stale(self):
        result = _check_stale_sl_tp(
            side="long", current_price=93000.0, stop_loss=93000.0, take_profit=[97000.0]
        )
        assert result is not None
        assert "SL" in result

    def test_long_price_above_tp_is_stale(self):
        result = _check_stale_sl_tp(
            side="long", current_price=98000.0, stop_loss=93000.0, take_profit=[97000.0]
        )
        assert result is not None
        assert "TP" in result

    def test_long_price_equal_tp_is_stale(self):
        result = _check_stale_sl_tp(
            side="long", current_price=97000.0, stop_loss=93000.0, take_profit=[97000.0]
        )
        assert result is not None
        assert "TP" in result

    def test_long_valid_price_returns_none(self):
        result = _check_stale_sl_tp(
            side="long", current_price=95000.0, stop_loss=93000.0, take_profit=[97000.0]
        )
        assert result is None

    # -- SHORT positions --

    def test_short_price_above_sl_is_stale(self):
        result = _check_stale_sl_tp(
            side="short", current_price=69500.0, stop_loss=69000.0, take_profit=[67000.0]
        )
        assert result is not None
        assert "SL" in result

    def test_short_price_equal_sl_is_stale(self):
        result = _check_stale_sl_tp(
            side="short", current_price=69000.0, stop_loss=69000.0, take_profit=[67000.0]
        )
        assert result is not None
        assert "SL" in result

    def test_short_price_below_tp_is_stale(self):
        result = _check_stale_sl_tp(
            side="short", current_price=66500.0, stop_loss=69000.0, take_profit=[67000.0]
        )
        assert result is not None
        assert "TP" in result

    def test_short_price_equal_tp_is_stale(self):
        result = _check_stale_sl_tp(
            side="short", current_price=67000.0, stop_loss=69000.0, take_profit=[67000.0]
        )
        assert result is not None
        assert "TP" in result

    def test_short_valid_price_returns_none(self):
        result = _check_stale_sl_tp(
            side="short", current_price=68000.0, stop_loss=69000.0, take_profit=[67000.0]
        )
        assert result is None

    # -- Edge cases --

    def test_no_stop_loss_skips_sl_check(self):
        result = _check_stale_sl_tp(
            side="long", current_price=95000.0, stop_loss=None, take_profit=[97000.0]
        )
        assert result is None

    def test_no_take_profit_skips_tp_check(self):
        result = _check_stale_sl_tp(
            side="long", current_price=95000.0, stop_loss=93000.0, take_profit=[]
        )
        assert result is None

    def test_flat_side_returns_none(self):
        result = _check_stale_sl_tp(
            side="flat", current_price=95000.0, stop_loss=93000.0, take_profit=[97000.0]
        )
        assert result is None
