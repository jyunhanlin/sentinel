<!-- Generated: 2026-03-02 | Files scanned: 50 | Token estimate: ~700 -->

# Data — Storage & Models

## Database: SQLite (aiosqlite + SQLModel)

### Tables (`storage/models.py`, 103L)

```
pipeline_runs
  ├── run_id (PK, UUID)
  ├── symbol, status, created_at

llm_calls
  ├── id (PK, auto)
  ├── run_id (FK) → pipeline_runs
  ├── agent_type, prompt, response
  ├── model, latency_ms, input_tokens, output_tokens

trade_proposals
  ├── proposal_id (PK, UUID)
  ├── run_id (FK) → pipeline_runs
  ├── proposal_json (serialized TradeProposal)
  ├── risk_check_result, risk_check_reason

paper_trades
  ├── trade_id (PK, UUID)
  ├── proposal_id (FK) → trade_proposals
  ├── symbol, side, entry_price, exit_price
  ├── quantity, pnl, fees, leverage, margin
  ├── liquidation_price, stop_loss, take_profit_json
  ├── close_reason, status

approval_records
  ├── approval_id (PK, UUID)
  ├── proposal_id (FK), run_id
  ├── snapshot_price, status, message_id, expires_at

account_snapshots
  ├── id (PK, auto)
  ├── equity, daily_pnl, total_pnl
  ├── win_rate, profit_factor, max_drawdown_pct
  ├── sharpe_ratio, total_trades
```

### Repositories (`storage/repository.py`, 441L)

| Repository | Key Methods |
|-----------|-------------|
| PipelineRepository | create_run, get_run, update_run_status |
| LLMCallRepository | save_call, list_by_run |
| TradeProposalRepository | save_proposal, get_recent(limit), get_by_proposal_id |
| PaperTradeRepository | save_trade, count_consecutive_losses, get_daily_pnl, get_closed_paginated |
| AccountSnapshotRepository | save_snapshot, get_latest |
| ApprovalRepository | save_approval, update_status, get_pending |

### Migrations (`storage/migrations.py`, 115L)

Version-tracked via `SchemaMigration` table. `@_register(version, name)` decorator.
- v1: adds leverage/margin/liquidation_price/close_reason/stop_loss/take_profit_json to paper_trades

## Domain Models (`models.py`, 132L)

All `frozen=True` Pydantic models:

| Model | Key Fields |
|-------|-----------|
| TechnicalAnalysis | trend, trend_strength (ADX), volatility_regime, momentum, rsi, key_levels, risk_flags, above_200w_ma, bull_support_band_status |
| PositioningAnalysis | funding_trend/extreme, oi_change_pct, retail/smart_money bias, squeeze_risk, confidence |
| CatalystReport | upcoming/active events, risk_level, recommendation, confidence |
| CorrelationAnalysis | dxy_trend/impact, sp500_regime, btc_dominance_trend, cross_market_alignment, confidence |
| TradeProposal | symbol, side, entry, position_size_risk_pct, stop_loss, take_profit[], suggested_leverage, confidence, invalid_if, rationale |
| MarketSnapshot | symbol, timeframe, ohlcv, funding_rate, last_price |
| TickerSummary | symbol, last_price, change_24h_pct, volume_24h |
| TradeEvaluation | trade_id, proposal_id, symbol, direction_correct, entry_deviation_pct, close_reason, confidence, pnl |
| EvaluationReport | total_evaluated, total_unmatched, direction_accuracy, sl/tp/liq rates, by_symbol, by_confidence |

### Enums

`Side`: long, short, flat
`Trend`: strong_up, up, neutral, down, strong_down
`VolatilityRegime`: low, normal, high, extreme
`Momentum`: strong_bullish, bullish, neutral, bearish, strong_bearish

## Exchange Data Flow

```
ExchangeClient (ccxt async wrapper)
  ├── DataFetcher.fetch_snapshot() → MarketSnapshot
  ├── DataFetcher.fetch_positioning_data() → funding, OI, L/S ratios
  ├── DataFetcher.fetch_macro_indicators() → 200W SMA, bull support band
  └── DataFetcher.fetch_ticker_summary() → TickerSummary (price board)

ExternalDataFetcher (aiohttp)
  ├── fetch_dxy_data() → Yahoo Finance DX-Y.NYB
  ├── fetch_sp500_data() → Yahoo Finance ^GSPC
  ├── fetch_btc_dominance() → CoinGecko API
  ├── fetch_economic_calendar() → stub []
  └── fetch_exchange_announcements() → stub []
```
