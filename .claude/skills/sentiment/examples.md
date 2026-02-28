# Sentiment Skill Examples

## Example 1: Bullish with high funding (conflicting signals)

**Input:**
```
Symbol: BTC/USDT:USDT
Current Price: 95200.0
24h Volume: 2,500,000,000
Funding Rate: 0.000800
Timeframe: 1h

Recent OHLCV (5 candles):
  O=94000.0 H=94800.0 L=93800.0 C=94700.0 V=450000
  O=94700.0 H=95100.0 L=94500.0 C=95000.0 V=520000
  O=95000.0 H=95400.0 L=94900.0 C=95300.0 V=480000
  O=95300.0 H=95600.0 L=95100.0 C=95200.0 V=390000
  O=95200.0 H=95500.0 L=95000.0 C=95200.0 V=350000
```

**Reasoning:**
- Funding rate at 0.08% is elevated — crowded longs, squeeze risk
- Price trending up with 5 consecutive green candles — bullish momentum
- Volume declining on last 2 candles — momentum weakening
- Conflict: price bullish but funding overheated and volume fading

**Output:**
```json
{
  "sentiment_score": 58,
  "key_events": [
    {"event": "funding rate at 0.08% signals crowded longs", "impact": "negative"},
    {"event": "5 consecutive green candles with declining volume", "impact": "neutral"}
  ],
  "confidence": 0.5
}
```

## Example 2: Capitulation / extreme fear

**Input:**
```
Symbol: ETH/USDT:USDT
Current Price: 2800.0
24h Volume: 5,000,000,000
Funding Rate: -0.000600
Timeframe: 1h

Recent OHLCV (5 candles):
  O=3050.0 H=3060.0 L=2950.0 C=2960.0 V=800000
  O=2960.0 H=2970.0 L=2880.0 C=2890.0 V=1200000
  O=2890.0 H=2910.0 L=2820.0 C=2830.0 V=1500000
  O=2830.0 H=2840.0 L=2790.0 C=2810.0 V=900000
  O=2810.0 H=2820.0 L=2795.0 C=2800.0 V=600000
```

**Reasoning:**
- Funding rate at -0.06% is deeply negative — shorts dominating, extreme fear
- Price dropped ~8% over 5 candles — strong bearish move
- Volume spiked on candles 2-3 then declined — selling climax may be passing
- Final candle: small body, low volume → selling exhaustion

**Output:**
```json
{
  "sentiment_score": 25,
  "key_events": [
    {"event": "funding rate at -0.06% indicates capitulation", "impact": "negative"},
    {"event": "volume spike then decline suggests selling climax passing", "impact": "positive"}
  ],
  "confidence": 0.7
}
```
