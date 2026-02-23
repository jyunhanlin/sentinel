# Price Display Design

**Goal:** Show current prices in two places: (A) all confirmation cards before trade execution, (B) a pinned price board message updated every 60s.

---

## A. Confirmation Cards — Show Current Price

**Rule:** Every confirmation card fetches and displays `Now: $XX,XXX` before the user confirms.

### Affected Confirmation Cards

| Card | Handler | Current State |
|------|---------|---------------|
| Leverage confirm (approve flow) | `_handle_leverage_preview` | Already fetches `current_price`, already shows `Entry: ~$XX` — just need consistent `Now:` label |
| Close confirm | `_handle_close` | Uses `format_position_card()` which needs `current_price` from `get_position_with_pnl()` — already has it |
| Reduce confirm | `_handle_select_reduce` | Already fetches `current_price`, shows `at ~$XX` — just need consistent `Now:` label |
| Add confirm | `_handle_select_add` | Already fetches `current_price`, shows `Current Price: ~$XX` — just need consistent label |

### Changes

Minimal — most confirmation cards already fetch and display `current_price`. Unify the label format:
- Leverage confirm: change `Entry: ~$XX` → `Now: $XX,XXX | Est. Entry: ~$XX`
- Close confirm: add `Now: $XX,XXX` line to the position card display
- Reduce confirm: already shows `at ~$XX` — rename to `Now: $XX,XXX`
- Add confirm: already shows `Current Price: ~$XX` — rename to `Now: $XX,XXX`

---

## B. Pinned Message — Price Board

### Display Format

```
━━ Price Board ━━
BTC/USDT  $69,123.50  +1.2%
ETH/USDT  $2,530.80   -0.5%
SOL/USDT  $142.35     +3.1%

Updated: 14:32:05
```

### Data Source

CCXT `fetch_ticker()` returns a dict with:
- `last` — current price (already used by `fetch_current_price()`)
- `percentage` — 24h change % (new, need to extract)

Add a new method to `DataFetcher`:
```python
async def fetch_ticker_summary(self, symbol: str) -> TickerSummary:
    """Fetch price + 24h change for price board display."""
    ticker = await self._client.fetch_ticker(symbol)
    return TickerSummary(
        symbol=symbol,
        price=ticker.get("last", 0.0),
        change_24h_pct=ticker.get("percentage", 0.0),
    )
```

`TickerSummary` is a simple frozen Pydantic model in `models.py`.

### Update Mechanism

- **Trigger:** PriceMonitor `check()` already runs every 60s. After checking SL/TP, also update the pinned message.
- **Pin on startup:** Bot sends a price board message and pins it on first run. Store `message_id` in memory.
- **Edit message:** Each cycle, `bot.edit_message_text(chat_id, message_id, new_text)`.
- **Per admin chat:** Each admin chat gets its own pinned message. Store `{chat_id: message_id}` mapping.
- **Rate limit safety:** Telegram allows ~20 edits/min per message. 1 edit/60s is well within limits.

### Integration with PriceMonitor

PriceMonitor gets a second callback:

```python
class PriceMonitor:
    def __init__(
        self,
        *,
        paper_engine: PaperEngine,
        data_fetcher: DataFetcher,
        on_close: CloseCallback | None = None,
        on_tick: TickCallback | None = None,  # NEW
    ) -> None:
```

`on_tick` is called every cycle with a list of `TickerSummary` for all monitored symbols. The bot uses this to update the pinned message.

### Symbols

Uses `symbols` from config (the same list used for pipeline scheduling). All monitored symbols are shown regardless of position status.

### Edge Cases

- **Bot restart:** Pin a new message (old one becomes stale but harmless)
- **No symbols configured:** Don't pin anything
- **Fetch failure for one symbol:** Show last known price with a stale indicator, or skip that symbol
- **Chat not found:** Log warning, skip that chat

---

## Non-Goals

- No auto-update of `/status` position cards (user can re-run `/status`)
- No `/price` command (pinned board covers this)
- No toast notifications on button click
