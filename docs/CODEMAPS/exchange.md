# Exchange & Data Codemap

**Last Updated:** 2026-02-28

## Overview

Exchange integration handles:
1. **Data fetching** — OHLCV, current price, funding rate via CCXT
2. **Paper trading engine** — Simulates positions with margin/leverage
3. **Price monitoring** — Tracks SL/TP fills asynchronously

## Architecture

```
ExchangeClient (CCXT wrapper)
    │
    ├─ create_order(symbol, side, amount, price)
    ├─ cancel_order(order_id)
    ├─ set_leverage(symbol, leverage)
    └─ fetch_balance()

DataFetcher (High-level data API)
    │
    ├─ fetch_snapshot(symbol, timeframe)
    │   └─ Returns MarketSnapshot (OHLCV, price, funding, volume)
    │
    └─ fetch_ohlcv(symbol, timeframe)
        └─ Low-level CCXT wrapper with retry

PaperEngine (Simulated trading)
    │
    ├─ open_position(proposal, current_price, leverage)
    ├─ close_position(trade_id, close_price, close_reason)
    └─ check_liquidation(trade_id, current_price)

PriceMonitor (Background SL/TP tracking)
    │
    └─ Async loop: fetch price → check SL/TP → update trades
```

## Exchange Client

**File:** `orchestrator/src/orchestrator/exchange/client.py`

CCXT wrapper for exchange API calls.

### Interface

```python
class ExchangeClient:
    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: str = "",
        api_secret: str = "",
    ) -> None:
        self._exchange = ccxt.async_support.binance({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        })

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
    ) -> list[tuple[float, float, float, float, float]]:
        """Fetch OHLCV candles from exchange."""

    async def fetch_ticker(self, symbol: str) -> dict:
        """Get current price and 24h volume."""

    async def fetch_funding_rate(self, symbol: str) -> float:
        """Get perpetual futures funding rate."""

    async def create_order(
        self,
        symbol: str,
        order_type: str,  # "market", "limit", "stop_market"
        side: str,
        amount: float,
        price: float | None = None,
    ) -> dict:
        """Place order, return filled order info."""

    async def cancel_order(self, symbol: str, order_id: str) -> dict:
        """Cancel order."""

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        """Set leverage for symbol (Binance, etc.)."""

    async def fetch_balance(self) -> dict:
        """Get account balance."""
```

### Retry Logic

Wraps CCXT calls with automatic retry:

```python
async def _retry_request(
    self,
    coro,
    max_retries: int = 3,
) -> Any:
    """Call async function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return await coro
        except ccxt.NetworkError as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # 1s, 2s, 4s
        except ccxt.ExchangeError:
            raise  # Don't retry on exchange errors
```

## Data Fetcher

**File:** `orchestrator/src/orchestrator/exchange/data_fetcher.py`

High-level API for fetching market snapshots used by agents.

### MarketSnapshot

```python
class MarketSnapshot(BaseModel, frozen=True):
    symbol: str
    current_price: float
    ohlcv: list[tuple[float, float, float, float, float]]  # [O, H, L, C, V]
    volume_24h: float
    funding_rate: float  # 8h perpetual funding rate
    timeframe: str  # e.g., "1h"
```

### Interface

```python
class DataFetcher:
    def __init__(self, exchange_client: ExchangeClient) -> None:
        self._exchange_client = exchange_client

    async def fetch_snapshot(
        self,
        symbol: str,
        timeframe: str = "1h",
    ) -> MarketSnapshot:
        """
        Fetch all data for an agent analysis.
        Returns: MarketSnapshot with OHLCV, price, funding, volume.
        """
        # Fetch OHLCV
        ohlcv = await self.fetch_ohlcv(symbol, timeframe, limit=10)

        # Fetch ticker (price, volume)
        ticker = await self._exchange_client.fetch_ticker(symbol)

        # Fetch funding rate (perpetuals only, 0 for spot)
        try:
            funding_rate = await self._exchange_client.fetch_funding_rate(symbol)
        except:
            funding_rate = 0.0

        return MarketSnapshot(
            symbol=symbol,
            current_price=ticker["last"],
            ohlcv=ohlcv,
            volume_24h=ticker["quoteVolume"],
            funding_rate=funding_rate,
            timeframe=timeframe,
        )

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 10,
    ) -> list[tuple[float, float, float, float, float]]:
        """Fetch OHLCV with retry."""
        return await self._exchange_client.fetch_ohlcv(
            symbol, timeframe, limit=limit
        )
```

## Paper Engine

**File:** `orchestrator/src/orchestrator/exchange/paper_engine.py`

Simulates trading with realistic margin, leverage, and fees.

### Core Classes

#### OpenPosition

```python
class OpenPosition(BaseModel, frozen=True):
    trade_id: str
    symbol: str
    side: Side  # "long" or "short"
    entry_price: float
    current_price: float
    quantity: float
    leverage: int
    margin: float  # USDT margin used
    liquidation_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
```

#### CloseResult

```python
class CloseResult(BaseModel, frozen=True):
    trade_id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float  # Realized PnL in USDT
    pnl_pct: float
    fees: float
    close_reason: str  # "sl", "tp", "manual", "liquidation"
```

### PaperEngine API

```python
class PaperEngine:
    def __init__(
        self,
        *,
        initial_equity: float = 10000.0,
        taker_fee_rate: float = 0.0005,
        maker_fee_rate: float = 0.0002,
        default_leverage: int = 10,
        maintenance_margin_rate: float = 0.5,  # %
    ) -> None:
        self._equity = initial_equity
        self._initial_equity = initial_equity
        self._positions: dict[str, OpenPosition] = {}
        self._closed_positions: list[CloseResult] = []

    # Position Management
    def open_position(
        self,
        proposal: TradeProposal,
        current_price: float,
        leverage: int,
    ) -> OpenPosition:
        """
        Open position using risk % sizer.
        1. Calculate quantity from risk_pct
        2. Check margin requirement
        3. Add to open positions
        4. Return OpenPosition
        """
        from orchestrator.risk.position_sizer import RiskPercentSizer

        sizer = RiskPercentSizer()
        quantity = sizer.calculate(
            side=proposal.side,
            entry_price=current_price,
            stop_loss=proposal.stop_loss,
            account_equity=self._equity,
            risk_pct=proposal.position_size_risk_pct,
        )

        return self._open_with_quantity(
            proposal, current_price, leverage, quantity
        )

    def open_position_with_quantity(
        self,
        proposal: TradeProposal,
        current_price: float,
        leverage: int,
        quantity: float,
        margin: float | None = None,
    ) -> OpenPosition:
        """Open position with explicit quantity."""
        return self._open_with_quantity(
            proposal, current_price, leverage, quantity, margin
        )

    def close_position(
        self,
        trade_id: str,
        close_price: float,
        close_reason: str,
    ) -> CloseResult:
        """
        Close position at price.
        1. Calculate PnL
        2. Deduct fees
        3. Update equity
        4. Move to closed list
        5. Return CloseResult
        """

    def check_and_close_sl_tp(
        self,
        trade_id: str,
        current_price: float,
    ) -> CloseResult | None:
        """Check if SL/TP triggered, close if yes."""
        position = self._positions.get(trade_id)
        if not position:
            return None

        # Check SL
        if position.proposal.stop_loss:
            if self._is_sl_triggered(position, current_price):
                return self.close_position(
                    trade_id, current_price, "sl"
                )

        # Check TP (iterate, close_pct at a time)
        if position.proposal.take_profit:
            for tp in position.proposal.take_profit:
                if self._is_tp_triggered(position, current_price, tp.price):
                    qty = position.quantity * tp.close_pct / 100
                    return self.close_position(
                        trade_id, current_price, "tp",
                        quantity_to_close=qty
                    )

        return None

    def check_liquidation(
        self,
        trade_id: str,
        current_price: float,
    ) -> bool:
        """Check if position is liquidated."""
        position = self._positions.get(trade_id)
        if not position:
            return False

        if position.side == Side.LONG:
            return current_price <= position.liquidation_price
        else:  # SHORT
            return current_price >= position.liquidation_price

    # Queries
    def get_open_positions(self) -> list[OpenPosition]:
        """Get all open positions."""
        return list(self._positions.values())

    def get_closed_trades(self) -> list[CloseResult]:
        """Get all closed trades."""
        return self._closed_positions

    def get_equity(self) -> float:
        """Get current account equity."""
        return self._equity

    def get_open_exposure(self) -> float:
        """Get total open position exposure as % of equity."""
        total_risk = sum(p.risk_pct for p in self._positions.values())
        return total_risk
```

### Liquidation Calculation

```python
def _calculate_liquidation_price(
    self,
    side: Side,
    entry_price: float,
    leverage: int,
) -> float:
    """
    Liquidation price = point where margin is exhausted.

    For long:
        liq = entry * (1 - 1/leverage + mm_rate/leverage)

    For short:
        liq = entry * (1 + 1/leverage - mm_rate/leverage)

    Example (leverage=10, mm=0.5%):
        Long @ $100:  liq = 100 * (1 - 0.1 + 0.005) = $90.50
        Short @ $100: liq = 100 * (1 + 0.1 - 0.005) = $109.50
    """
    if side == Side.LONG:
        return entry_price * (
            1 - 1 / leverage + self._maintenance_margin_rate / leverage / 100
        )
    else:  # SHORT
        return entry_price * (
            1 + 1 / leverage - self._maintenance_margin_rate / leverage / 100
        )
```

### Fee Calculation

```python
def _calculate_fees(
    self,
    quantity: float,
    price: float,
    fee_rate: float,
) -> float:
    """Entry and exit fees combined."""
    return quantity * price * fee_rate * 2  # Entry + exit
```

## Price Monitor

**File:** `orchestrator/src/orchestrator/exchange/price_monitor.py`

Background task that monitors open positions for SL/TP fills.

### Interface

```python
class PriceMonitor:
    def __init__(
        self,
        *,
        data_fetcher: DataFetcher,
        paper_engine: PaperEngine,
        trade_repo: PaperTradeRepository,
        interval_seconds: int = 300,  # Check every 5 minutes
    ) -> None:
        self._data_fetcher = data_fetcher
        self._paper_engine = paper_engine
        self._trade_repo = trade_repo
        self._interval_seconds = interval_seconds
        self._running = False

    async def start(self) -> None:
        """Start background monitoring loop."""
        self._running = True
        while self._running:
            await self._check_prices()
            await asyncio.sleep(self._interval_seconds)

    async def stop(self) -> None:
        """Stop monitoring."""
        self._running = False

    async def _check_prices(self) -> None:
        """Fetch current prices, check SL/TP, update database."""
        open_positions = self._paper_engine.get_open_positions()

        for position in open_positions:
            # Fetch current price
            snapshot = await self._data_fetcher.fetch_snapshot(
                position.symbol, timeframe="1m"
            )

            # Check liquidation
            if self._paper_engine.check_liquidation(
                position.trade_id, snapshot.current_price
            ):
                self._paper_engine.close_position(
                    position.trade_id, snapshot.current_price, "liquidation"
                )
                # Update database
                self._trade_repo.update(...)

            # Check SL/TP
            close_result = self._paper_engine.check_and_close_sl_tp(
                position.trade_id, snapshot.current_price
            )
            if close_result:
                self._trade_repo.update(
                    position.trade_id,
                    status="closed",
                    exit_price=close_result.exit_price,
                    pnl=close_result.pnl,
                    close_reason=close_result.close_reason,
                )
                logger.info(
                    "position_closed",
                    trade_id=position.trade_id,
                    reason=close_result.close_reason,
                )
```

### Startup

In `__main__.py`:

```python
price_monitor = PriceMonitor(
    data_fetcher=data_fetcher,
    paper_engine=paper_engine,
    trade_repo=trade_repo,
    interval_seconds=300,
)

# Start in background
asyncio.create_task(price_monitor.start())

# On shutdown
price_monitor.stop()
```

## Configuration

Environment variables:

```python
exchange_id: str = "binance"
exchange_api_key: str = ""  # Required for live
exchange_api_secret: str = ""  # Required for live

# Paper Trading
paper_initial_equity: float = 10000.0
paper_taker_fee_rate: float = 0.0005  # 0.05%
paper_maker_fee_rate: float = 0.0002  # 0.02%
paper_default_leverage: int = 10
paper_maintenance_margin_rate: float = 0.5  # %
paper_leverage_options: list[int] = [5, 10, 20, 50]

# Price Monitor
price_monitor_interval_seconds: int = 300  # 5 minutes
price_monitor_enabled: bool = True
```

## Testing

**File:** `tests/unit/test_exchange.py`

Mocks CCXT:

```python
@patch("ccxt.async_support.binance")
async def test_fetch_ohlcv(mock_exchange):
    mock_exchange.return_value.fetch_ohlcv.return_value = [
        [1000, 100, 110, 105, 1000],  # [timestamp, O, H, L, C, V]
    ]

    client = ExchangeClient()
    ohlcv = await client.fetch_ohlcv("BTC/USDT", "1h")
    assert len(ohlcv) == 1
```

**File:** `tests/unit/test_paper_trading_flow.py`

Full flow:

```python
async def test_open_and_close_position():
    engine = PaperEngine(initial_equity=10000.0)
    proposal = TradeProposal(symbol="BTC/USDT:USDT", side=Side.LONG, ...)

    # Open at $45,000
    position = engine.open_position(proposal, 45000.0, leverage=10)
    assert position.entry_price == 45000.0
    assert position.liquidation_price < 45000.0

    # Close at $46,000 (profit)
    result = engine.close_position(position.trade_id, 46000.0, "tp")
    assert result.pnl > 0
    assert engine.get_equity() > 10000.0
```

## Related

- [pipeline.md](pipeline.md) — DataFetcher usage
- [execution.md](execution.md) — OrderExecutor integration
- [risk.md](risk.md) — Paper engine margin/liquidation
- [storage.md](storage.md) — Trade persistence
