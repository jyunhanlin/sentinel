# Proposer Skill Examples

## Example 1: Bullish setup → long trade

**Input:**
```
=== Market Data ===
Symbol: BTC/USDT:USDT
Current Price: 95200.0
24h Volume: 2,500,000,000
Funding Rate: 0.000300

=== Sentiment Analysis ===
Sentiment Score: 68/100
Confidence: 0.7
Key Events: rising volume on breakout above 95000

=== Technical Analysis ===
Trend: up
Volatility: medium (2.1%)
Key Levels: support=94100, resistance=96500
Risk Flags: none
```

**Reasoning:**
- Edge: sentiment bullish (68) + trend up → directional edge exists
- Direction: long
- Entry: market, since price already above key level at 95000
- Stop loss: below support at 94100, that's ~1.2% below entry → ok
- Take profit: stop distance = 95200 - 94100 = 1100
  - TP1 at 95200 + 1650 (1.5x) = 96850, close 50%
  - TP2 at 96500 (resistance), close 100%
  - Actually 96500 < 96850, so use: TP1 at 96500 close 50%, TP2 at 97500 close 100%
- Risk: confidence 0.7 + medium volatility → 1.0%
- Leverage: volatility 2.1% → up to 10x. Confidence 0.7 → ok. No flags → 10x

**Output:**
```json
{
  "symbol": "BTC/USDT:USDT",
  "side": "long",
  "entry": {"type": "market"},
  "position_size_risk_pct": 1.0,
  "stop_loss": 94100,
  "take_profit": [
    {"price": 96500, "close_pct": 50},
    {"price": 97500, "close_pct": 100}
  ],
  "suggested_leverage": 10,
  "time_horizon": "4h",
  "confidence": 0.65,
  "invalid_if": ["price closes below 94000", "funding rate exceeds 0.1%"],
  "rationale": "Bullish sentiment (68) aligns with uptrend. Support at 94100 provides clear stop. Targeting resistance at 96500 for first exit."
}
```

## Example 2: Mixed signals → flat

**Input:**
```
=== Market Data ===
Symbol: ETH/USDT:USDT
Current Price: 3000.0
24h Volume: 800,000,000
Funding Rate: 0.000050

=== Sentiment Analysis ===
Sentiment Score: 52/100
Confidence: 0.4
Key Events: none

=== Technical Analysis ===
Trend: range
Volatility: low (1.4%)
Key Levels: support=2975, resistance=3040
Risk Flags: volume_declining
```

**Reasoning:**
- Edge: sentiment neutral (52), trend range, low confidence (0.4) → no directional edge
- Multiple disqualifiers: neutral sentiment, range-bound, declining volume
- Decision: flat

**Output:**
```json
{
  "symbol": "ETH/USDT:USDT",
  "side": "flat",
  "entry": {"type": "market"},
  "position_size_risk_pct": 0,
  "stop_loss": null,
  "take_profit": [],
  "suggested_leverage": 1,
  "time_horizon": "4h",
  "confidence": 0.3,
  "invalid_if": [],
  "rationale": "Neutral sentiment (52) in range-bound market with declining volume. No clear edge to trade."
}
```
