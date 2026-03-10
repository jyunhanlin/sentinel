# Tunable Thresholds

These thresholds are calibrated for major crypto futures (BTC, ETH) on Binance.
Adjust for different market conditions or lower-liquidity altcoins.

## ADX Regime Boundaries

| Regime | Threshold | Notes |
|--------|-----------|-------|
| No trend | ADX < 20 | For short_term, consider raising to < 25 |
| Moderate | 20-40 | |
| Strong | 40-60 | |
| Very strong | > 60 | |

## RSI Thresholds

| Level | Value | Context |
|-------|-------|---------|
| Overbought (risk flag) | > 75 | Only flag when ADX < 40 |
| Oversold (risk flag) | < 25 | Only flag when ADX < 40 |
| Bullish bias | > 50 | Used in momentum synthesis |
| Bearish bias | < 50 | Used in momentum synthesis |

## Volatility Regime (ATR / price × 100)

| Regime | Range | Adjustment for altcoins |
|--------|-------|------------------------|
| Low | < 1.5% | Altcoins: < 2.5% (structurally more volatile) |
| Medium | 1.5% - 3.5% | Altcoins: 2.5% - 6% |
| High | > 3.5% | Altcoins: > 6% |

## Risk Flag Triggers

| Flag | Default | Altcoin adjustment |
|------|---------|-------------------|
| `funding_elevated` | abs(rate) > 0.05% | Same — funding mechanics are exchange-level |
| `volume_declining` | 3 consecutive lower-volume candles | Same |
| `high_volatility` | volatility_pct > 5% | > 8% for altcoins |
| `near_key_level` | within 0.3% of S/R | within 0.5% for altcoins (wider spreads) |
| `trend_exhaustion` | > 8 same-color candles | > 6 for altcoins (shorter cycles) |
| `bollinger_squeeze` | band width < 50% of 20-period avg | Same |

## Timeframe-Specific Adjustments

| Parameter | short_term (1h-4h) | long_term (1d-1w) |
|-----------|-------------------|-------------------|
| ADX "meaningful" | > 25 | > 20 |
| S/R range | ±3% of price | ±5% of price |
| Volume baseline | 24h average | 20-day average |

## When to Adjust

- **High-volatility regime (VIX-like spikes, post-halving)**: Widen all percentage thresholds by ~50%
- **Low-liquidity altcoins (< $10M daily volume)**: Use altcoin columns above; volume indicators
  become less reliable — weight price structure and funding more heavily
- **Weekend/holiday trading**: Volume naturally drops; `volume_declining` may false-trigger.
  Consider disabling or raising the threshold for consecutive declining candles to 5.
