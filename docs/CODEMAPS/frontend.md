<!-- Generated: 2026-03-11 | Files scanned: 49 | Token estimate: ~700 -->

# Frontend — Telegram Bot UI

## SentinelBot (`telegram/bot.py`, 2211L)

### Commands

| Command | Handler | Description |
|---------|---------|-------------|
| `/start` | start_cmd | Welcome message |
| `/help` | help_cmd | Command listing |
| `/status` | status_cmd | Account overview + open positions + recent proposals |
| `/coin <sym>` | coin_cmd | On-demand pipeline run for symbol |
| `/run [sym] [model]` | run_cmd | Trigger pipeline (model: sonnet/opus, supports refine flag) |
| `/history [sym] [page]` | history_cmd | Paginated closed trades |
| `/perf` | perf_cmd | Performance stats (win rate, Sharpe, drawdown) |
| `/evaluate` | evaluate_cmd | Pipeline accuracy report (direction, entry deviation, confidence buckets) |

### Push Notifications

```
push_to_admins_with_approval(PipelineResult)
  ├── directional trade → approval card + ExecutionPlan + Approve/Reject/Cancel buttons
  ├── FLAT signal → plain notification
  ├── rejected/failed → status notification
  └── shows refinement status (rounds, exhausted) if applicable

push_close_report(CloseResult) → SL/TP/liquidation alert + pipeline evaluation feedback
update_price_board(TickerSummary[]) → pinned message, auto-edit
```

### Approval Inline Keyboard Flow

```
[Approve] → select leverage (1x–20x grid)
         → select margin (25%–100% of available)
         → confirm → OrderExecutor.execute_entry() + place_sl_tp()

[Reject]  → marks rejected, edits message
[Cancel]  → removes keyboard
```

### Position Management Buttons

```
[Close]  → confirm → paper_engine.close_position()
[Reduce] → select % (25/50/75) → confirm → paper_engine.reduce_position()
[Add]    → select risk % (0.5/1.0/1.5/2.0) → confirm → paper_engine.add_to_position()
```

### Translation

`[🇹🇼 中文]` button on proposals → LLM translates to Traditional Chinese
Cached via `_MessageCache` (OrderedDict LRU, 50 entries)

### Model Aliases

`sonnet → anthropic/claude-sonnet-4-6`, `opus → anthropic/claude-opus-4-6`

## Formatters (`telegram/formatters.py`, 838L)

Pure functions, no I/O. All return Telegram MarkdownV2 strings:

| Function | Input | Output |
|----------|-------|--------|
| format_proposal | PipelineResult | Full analysis + proposal card |
| format_pending_approval | PendingApproval | Approval card with leverage/margin info |
| format_execution_plan | ExecutionPlan | Two-section execution plan (order + risk) |
| format_execution_result | ExecutionResult | Entry confirmation |
| format_trade_report | CloseResult | Close summary with PnL |
| format_status | list | Account overview + positions |
| format_perf_report | PerformanceStats | Win rate, Sharpe, drawdown |
| format_position_card | PositionInfo | Single position detail |
| format_account_overview | equity, margin, etc | Balance summary |
| format_history_paginated | trades, page | Paginated trade list |
| format_price_board | TickerSummary[] | Compact price ticker |
| format_trade_evaluation | TradeEvaluation | One-line pipeline feedback (direction/deviation/confidence) |
| format_evaluation_report | EvaluationReport | Full evaluation report (by symbol, by confidence bucket) |

## Translations (`telegram/translations.py`, 37L)

`to_chinese(text, llm_client)` → system prompt: Traditional Chinese, preserve numbers/symbols/trading terms
