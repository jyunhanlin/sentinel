# TG Message Format Unification

**Date:** 2026-03-03
**Status:** Approved

## Goal

Unify all Telegram bot messages to a consistent English + emoji label style. Remove all Chinese text from buttons and prompts.

## Style Rules

1. **Language**: All English, no Chinese
2. **Line format**: `emoji Label: value` (e.g., `🎯 Entry: $69,000 (limit)`)
3. **Title format**: `side_emoji SYMBOL SIDE · Key Info`
4. **Section separation**: Blank lines only, no divider lines (`── X ──`)
5. **Buttons**: All English labels

## Changes

### A. `bot.py` — Chinese → English (~15 locations)

| Current | New |
|---------|-----|
| `⚙️ 管理` | `⚙️ Manage` |
| `🚀 確認開倉` | `🚀 Confirm` |
| `❌ 取消` | `❌ Cancel` |
| `⬅️ 返回` (×2) | `⬅️ Back` |
| `✏️ 要調整哪個參數？` | `✏️ Which parameter to adjust?` |
| `目前 Leverage: Nx\n選擇新的倍數：` | `Current Leverage: Nx\nSelect new leverage:` |
| `⛔ 輸入新的 Stop Loss 價格：\n例: 92500\n直接輸入數字即可。` (×2) | `⛔ Enter new Stop Loss price:\ne.g. 92500` |
| `✅ 輸入新的 Take Profit 價格：\n例: 97000 50%, 99000 100%\n格式: 價格 平倉比例, 價格 平倉比例` (×2) | `✅ Enter new Take Profit levels:\ne.g. 97000 50%, 99000 100%\nFormat: price close%, price close%` |
| `💰 輸入新的保證金金額 (USDT)：\n例: 300\n直接輸入數字即可。` | `💰 Enter new margin amount (USDT):\ne.g. 300` |
| `❌ 調整已取消。` | `❌ Adjustment cancelled.` |
| `移 SL` | `Move SL` |
| `調 TP` | `Adjust TP` |
| `加倉` | `Add` |
| `減倉` | `Reduce` |
| `平倉` | `Close` |

### B. `formatters.py` — Emoji label style unification

Unify all formatters to use the same emoji + label style as `format_execution_plan()`.

Key emoji assignments:
- `🎯` Entry
- `📦` Quantity
- `💵` Margin
- `💀` Liquidation
- `⛔` Stop Loss
- `✅` Take Profit
- `⚠️` Max Loss / Warnings
- `💰` Est. Profit / PnL
- `📊` Risk/Reward / Technical
- `📈` Trend / Gains
- `📉` Losses / Positioning
- `📅` Catalyst
- `🌍` Correlation
- `💡` Rationale

Functions to update:
1. `format_proposal()` — `▶` → `🎯` for entry, add emoji labels
2. `format_pending_approval()` — `▶` → `🎯`, unify emoji usage
3. `format_execution_result()` — add emoji labels
4. `format_trade_report()` — add emoji labels
5. `format_risk_rejection()` — minor style alignment
6. `format_risk_pause()` — minor style alignment
7. `format_position_card()` — add emoji labels
8. `format_account_overview()` — add emoji labels
9. `format_perf_report()` — add emoji labels
10. `format_history_paginated()` — add emoji labels
11. `format_execution_plan()` — remove `── Analysis ──` divider

### C. Not changed

- `format_welcome()` / `format_help()` — already English
- `format_price_board()` — already English
- Translate button mechanism — preserved
- `callback_data` values — unchanged
- `format_status()` / `format_status_from_records()` — already English
- `format_eval_report()` — already English

## Test Impact

- Update `test_formatters_execution_plan.py` for divider removal
- Update `test_telegram.py` for any assertion on Chinese text
