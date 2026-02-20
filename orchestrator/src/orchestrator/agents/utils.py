from __future__ import annotations


def summarize_ohlcv(ohlcv: list[list[float]], *, max_candles: int = 10) -> str:
    if not ohlcv:
        return "No OHLCV data available"

    lines = []
    for candle in ohlcv[-max_candles:]:
        o, h, lo, c, v = candle[1], candle[2], candle[3], candle[4], candle[5]
        lines.append(f"  O={o:.1f} H={h:.1f} L={lo:.1f} C={c:.1f} V={v:.0f}")
    return "\n".join(lines)
