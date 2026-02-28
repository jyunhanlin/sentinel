# Market Skill Examples

## Example 1: Uptrend with medium volatility

**Input:**
```
Symbol: BTC/USDT:USDT
Current Price: 95200.0
24h Volume: 2,500,000,000
Funding Rate: 0.000300
Timeframe: 1h

OHLCV Data (10 candles):
  O=93500.0 H=93900.0 L=93300.0 C=93800.0 V=400000
  O=93800.0 H=94200.0 L=93600.0 C=94100.0 V=450000
  O=94100.0 H=94500.0 L=93900.0 C=94400.0 V=420000
  O=94400.0 H=94600.0 L=94200.0 C=94300.0 V=380000
  O=94300.0 H=94700.0 L=94100.0 C=94600.0 V=410000
  O=94600.0 H=95000.0 L=94500.0 C=94900.0 V=440000
  O=94900.0 H=95200.0 L=94700.0 C=95100.0 V=460000
  O=95100.0 H=95400.0 L=94900.0 C=95300.0 V=430000
  O=95300.0 H=95500.0 L=95100.0 C=95200.0 V=400000
  O=95200.0 H=95400.0 L=95000.0 C=95200.0 V=380000
```

**Reasoning:**
- Trend: price moved from 93500 to 95200, series of higher lows → uptrend
- Volatility: average true range ~400 per candle, /95200 * 100 ≈ 0.42% per candle. For 14-candle ATR this extrapolates to ~2.1% → medium
- Support: cluster of lows around 94100-94200 area → support at 94100
- Resistance: highs clustering at 95400-95500 → resistance at 95500
- Risk flags: funding at 0.03% is normal, volume relatively stable, no flags

**Output:**
```json
{
  "trend": "up",
  "volatility_regime": "medium",
  "volatility_pct": 2.1,
  "key_levels": [
    {"type": "support", "price": 94100},
    {"type": "resistance", "price": 95500}
  ],
  "risk_flags": []
}
```

## Example 2: Range-bound with declining volume

**Input:**
```
Symbol: ETH/USDT:USDT
Current Price: 3000.0
24h Volume: 800,000,000
Funding Rate: 0.000050
Timeframe: 1h

OHLCV Data (8 candles):
  O=3010.0 H=3040.0 L=2980.0 C=2990.0 V=200000
  O=2990.0 H=3020.0 L=2975.0 C=3015.0 V=180000
  O=3015.0 H=3035.0 L=2990.0 C=2995.0 V=160000
  O=2995.0 H=3025.0 L=2985.0 C=3010.0 V=150000
  O=3010.0 H=3030.0 L=2980.0 C=2985.0 V=140000
  O=2985.0 H=3015.0 L=2970.0 C=3005.0 V=130000
  O=3005.0 H=3025.0 L=2985.0 C=2995.0 V=120000
  O=2995.0 H=3010.0 L=2980.0 C=3000.0 V=110000
```

**Reasoning:**
- Trend: price oscillating between ~2970 and ~3040, no clear direction → range
- Volatility: true range ~40-50 per candle, /3000 * 100 ≈ 1.5% → low/medium boundary
- Support: multiple lows around 2975-2985 → support at 2975
- Resistance: highs around 3035-3040 → resistance at 3040
- Risk flags: last 3 candles volume declining (130k → 120k → 110k) → volume_declining

**Output:**
```json
{
  "trend": "range",
  "volatility_regime": "low",
  "volatility_pct": 1.4,
  "key_levels": [
    {"type": "support", "price": 2975},
    {"type": "resistance", "price": 3040}
  ],
  "risk_flags": ["volume_declining"]
}
```
