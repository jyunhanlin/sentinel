# Proposer Skill Examples

## Example 1: Strong convergence → long trade

**Input:**
```
=== Market Data ===
Symbol: BTC/USDT:USDT
Current Price: 95200.0
24h Volume: 2,500,000,000
Funding Rate: 0.000300

=== Short-Term Technical (4h) ===
Trend: up
Trend Strength (ADX): 32
Momentum: bullish
RSI: 58
Volatility Regime: medium
Volatility Pct: 2.1%
Key Levels: support=94100, resistance=96500
Risk Flags: none

=== Long-Term Technical (1d) ===
Trend: up
Trend Strength (ADX): 28
Momentum: bullish
RSI: 55
Volatility Regime: medium
Key Levels: support=92000, resistance=98000
Risk Flags: none
Above 200W MA: true
Bull Support Band Status: above

=== Positioning ===
Funding Trend: rising
Funding Extreme: false
OI Change Pct: +3.2%
OI Interpretation: new_longs
Retail Bias: long
Smart Money Bias: long
Squeeze Risk: none
Liquidity Assessment: normal
Risk Flags: none
Confidence: 0.65
Data Caveats: []

=== Catalyst ===
Upcoming Events: none within 48h
Risk Level: low
Recommendation: proceed
Confidence: 0.7
Data Caveats: []

=== Correlation ===
DXY Trend: weakening
DXY Impact: tailwind
S&P 500 Regime: risk_on
BTC Dominance Trend: rising
Cross-Market Alignment: favorable
Risk Flags: [dxy_tailwind]
Confidence: 0.65
Data Caveats: []
```

**Reasoning:**
- **Catalyst gate**: recommendation = "proceed" → pass
- **Signal convergence**: 5/5 agree — ST up, LT up, positioning new_longs + smart money
  long, catalyst clear, correlation favorable. This is about as clean as it gets.
- **Direction**: both trends up + bullish momentum → long
- **Entry**: market entry — price is away from support, momentum active
- **Stop loss**: ST support at 94100, place slightly beyond → 94050.
  Distance: (95200 - 94050) / 95200 = 1.2% — within acceptable range.
- **Take profit**:
  - Stop distance = 95200 - 94050 = 1150
  - TP1 at 1.5x stop = 95200 + 1725 = 96925, but ST resistance at 96500 is closer.
    Use 96500 as TP1 (close 50%). R:R = (96500-95200)/(95200-94050) = 1.13 — below 1.5.
  - Hmm, TP1 at key level gives only 1.13 R:R. Use 96925 (1.5x) as TP1 instead,
    then TP2 at LT resistance 98000 (close 100%). R:R TP1 = 1.5, R:R TP2 = 2.4 ✓
- **Confidence**: base 0.5 + 0.20 (5/5 agree) + 0.05 (LT aligned) + 0.05 (positioning
  supports) = 0.80 → clamped to 0.75 (cap)
- **Risk %**: confidence 0.75 → 2.0% base. No adjustments needed (catalyst proceed,
  correlation favorable, liquidity normal, no squeeze, 1 risk flag only). → 2.0%
  But note: dxy_tailwind is a positive flag, not a risk flag. Total risk flags = 0. → 2.0%
- **Leverage**: volatility 2.1% → max 10x. No squeeze, no funding extreme,
  confidence > 0.55, normal liquidity. → 10x

**Output:**
```json
{
  "symbol": "BTC/USDT:USDT",
  "side": "long",
  "entry": {"type": "market"},
  "position_size_risk_pct": 2.0,
  "stop_loss": 94050,
  "take_profit": [
    {"price": 96925, "close_pct": 50},
    {"price": 98000, "close_pct": 100}
  ],
  "suggested_leverage": 10,
  "time_horizon": "4h-12h",
  "confidence": 0.75,
  "invalid_if": [
    "BTC 4h close below 94000",
    "Funding rate flips negative"
  ],
  "rationale": "All 5 sources aligned: ST/LT uptrend (ADX 32/28), new longs entering with smart money confirmation, DXY tailwind with favorable macro alignment, no catalyst risk. Stop below ST support at 94050, targeting 1.5x R:R at 96925 and LT resistance at 98000.",
  "data_caveats": []
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

=== Short-Term Technical (4h) ===
Trend: range
Trend Strength (ADX): 16
Momentum: neutral
RSI: 50
Volatility Regime: low
Volatility Pct: 1.4%
Key Levels: support=2975, resistance=3040
Risk Flags: [volume_declining]

=== Long-Term Technical (1d) ===
Trend: up
Trend Strength (ADX): 22
Momentum: neutral
RSI: 48
Volatility Regime: low
Key Levels: support=2850, resistance=3200
Risk Flags: none
Above 200W MA: true
Bull Support Band Status: at_band

=== Positioning ===
Funding Trend: stable
Funding Extreme: false
OI Change Pct: -1.2%
OI Interpretation: unclear
Retail Bias: neutral
Smart Money Bias: neutral
Squeeze Risk: none
Liquidity Assessment: normal
Risk Flags: none
Confidence: 0.45
Data Caveats: ["low_volume_weekend_data"]

=== Catalyst ===
Upcoming Events: none within 48h
Risk Level: low
Recommendation: proceed
Confidence: 0.6
Data Caveats: []

=== Correlation ===
DXY Trend: stable
DXY Impact: neutral
S&P 500 Regime: neutral
BTC Dominance Trend: stable
Cross-Market Alignment: mixed
Risk Flags: none
Confidence: 0.5
Data Caveats: ["weekend_macro_data"]
```

**Reasoning:**
- **Catalyst gate**: recommendation = "proceed" → pass
- **Signal convergence**: ST range + LT weak up + positioning neutral + correlation mixed.
  At best 1-2/5 lean long (LT trend up, but ADX only 22 and momentum neutral). No edge.
- **Additional disqualifiers**: ST ADX 16 (no trend), declining volume, positioning
  confidence only 0.45, weekend data caveats on 2 sources.
- **Decision**: flat — no directional edge exists
- **Confidence**: 0.45 — confident there is NO trade (signal is clear: nothing to do)

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
  "confidence": 0.45,
  "invalid_if": [],
  "rationale": "No directional edge: ST range-bound (ADX 16), declining volume, positioning neutral with low confidence (0.45), correlation mixed on weekend data. LT trend is mildly up but lacks momentum confirmation. Sitting out.",
  "data_caveats": ["low_volume_weekend_data", "weekend_macro_data"]
}
```

## Example 3: Conflicting signals → reduced size trade

**Input:**
```
=== Market Data ===
Symbol: BTC/USDT:USDT
Current Price: 88500.0
24h Volume: 3,200,000,000
Funding Rate: -0.000400

=== Short-Term Technical (4h) ===
Trend: down
Trend Strength (ADX): 35
Momentum: bearish
RSI: 32
Volatility Regime: high
Volatility Pct: 4.5%
Key Levels: support=86000, resistance=90000
Risk Flags: [oversold_rsi]

=== Long-Term Technical (1d) ===
Trend: up
Trend Strength (ADX): 25
Momentum: neutral
RSI: 44
Volatility Regime: medium
Key Levels: support=85000, resistance=92000
Risk Flags: none
Above 200W MA: true
Bull Support Band Status: below

=== Positioning ===
Funding Trend: falling
Funding Extreme: true
OI Change Pct: +8.5%
OI Interpretation: new_shorts
Retail Bias: short
Smart Money Bias: neutral
Squeeze Risk: short_squeeze
Squeeze Severity: high
Liquidity Assessment: thin
Risk Flags: [funding_elevated, crowded_short, thin_liquidity]
Confidence: 0.7
Data Caveats: []

=== Catalyst ===
Upcoming Events: [{"event": "CPI Release", "time": "2026-03-12T13:30:00Z", "impact": "high"}]
Risk Level: medium
Recommendation: reduce_size
Confidence: 0.6
Data Caveats: []

=== Correlation ===
DXY Trend: strengthening
DXY Impact: headwind
S&P 500 Regime: mild_risk_off
BTC Dominance Trend: rising
Cross-Market Alignment: unfavorable
Risk Flags: [dxy_headwind]
Confidence: 0.6
Data Caveats: []
```

**Reasoning:**
- **Catalyst gate**: recommendation = "reduce_size" → proceed with size reduction
- **Signal convergence — it's complicated**:
  - ST: bearish (ADX 35, strong down)
  - LT: bullish (trend up, above 200W MA), but below bull support band
  - Positioning: extreme short crowding → short_squeeze risk HIGH. Funding extremely
    negative. New shorts flooding in. This conflicts with taking a short.
  - Catalyst: CPI upcoming, reduce_size
  - Correlation: unfavorable (DXY headwind + mild risk-off)
- **Conflict resolution**: ST says short, but positioning shows crowded shorts with
  high squeeze risk. Going short HERE means joining the crowd that's about to get
  squeezed. LT trend is still up (structural bull). The squeeze risk is the dominant
  signal — it overrides the ST bearish setup.
- **Decision**: Despite the ST downtrend, the short squeeze setup actually suggests a
  LONG contrarian play. But: correlation is unfavorable, catalyst says reduce size,
  and liquidity is thin. This limits conviction.
- **Direction**: cautious long (contrarian squeeze play into LT trend direction)
- **Entry**: limit at ST support 86000 area — let price come to us given the chaos
- **Stop**: below LT support at 85000, place at 84750. Distance from 86000: 1.5%
- **Take profit**: stop distance = 86000 - 84750 = 1250
  - TP1: 86000 + 1875 (1.5x) = 87875, close 60%
  - TP2: 90000 (ST resistance), close 100%. R:R = 3.2x ✓
- **Confidence**: base 0.5 + 0.10 (3/5: LT trend + positioning squeeze + BTC.D rising)
  - 0.60 − 0.05 (correlation unfavorable has low confidence) = 0.55
  But: thin liquidity -0.10 → 0.45. Risk flags > 2: -0.10 → 0.35.
  This is below 0.40 threshold → technically should be flat.
- **Revised decision**: confidence too low for a trade. The squeeze setup is tempting but
  there are too many headwinds (DXY, risk-off, thin liquidity, CPI upcoming, 4 risk flags).
  **Go flat.** The disciplined call is to wait for the squeeze to resolve and CPI to pass.

**Output:**
```json
{
  "symbol": "BTC/USDT:USDT",
  "side": "flat",
  "entry": {"type": "market"},
  "position_size_risk_pct": 0,
  "stop_loss": null,
  "take_profit": [],
  "suggested_leverage": 1,
  "time_horizon": "1d",
  "confidence": 0.35,
  "invalid_if": [],
  "rationale": "ST bearish (ADX 35) conflicts with LT bullish structure and high short squeeze risk (crowded shorts, extreme negative funding). Contrarian long is tempting but aggregate headwinds too strong: DXY headwind, mild risk-off, thin liquidity, 4 risk flags, and CPI within 48h. Confidence 0.35 is below trading threshold. Wait for squeeze resolution and CPI to pass.",
  "data_caveats": ["thin_liquidity_amplifies_slippage", "cpi_within_48h_reduces_signal_reliability"]
}
```

**Key takeaway**: This example shows the discipline of going flat even when a setup
looks tempting. The squeeze was real, but the aggregate risk picture (thin liquidity +
macro headwind + upcoming CPI + 4 risk flags) dropped confidence below the trading
threshold. The model should not force trades when the math says no.
