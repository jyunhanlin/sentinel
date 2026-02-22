"""English → Chinese translation for Telegram messages.

Uses line-prefix replacement: each label at the start of a line (or in known
positions) is swapped.  Trading abbreviations (SL, TP, PnL) stay as-is since
they're universal in crypto.
"""

from __future__ import annotations

# --- Label replacements (order matters: longer prefixes first) ---

_LINE_PREFIX_MAP: list[tuple[str, str]] = [
    # Status / pipeline
    ("Latest pipeline results:", "最新 Pipeline 結果："),
    ("Latest proposals (from DB):", "最新提案（來自 DB）："),
    ("Running pipeline:", "Pipeline 執行中："),
    ("No pipeline results yet. Use /run to trigger analysis.",
     "尚無 Pipeline 結果。使用 /run 開始分析。"),
    ("No recent analysis for", "找不到近期分析："),
    (". Use /run to trigger analysis.", "。使用 /run 開始分析。"),
    # Proposal labels
    ("[NEW]", "[新提案]"),
    ("[REJECTED]", "[已拒絕]"),
    ("[FAILED]", "[失敗]"),
    ("[PENDING APPROVAL]", "[待審核]"),
    ("[CLOSED]", "[已平倉]"),
    ("[EXECUTED]", "[已執行]"),
    ("[RISK PAUSED]", "[風控暫停]"),
    ("[RISK REJECTED]", "[風控拒絕]"),
    ("[parse error]", "[解析錯誤]"),
    # Field labels
    ("Side:", "方向："),
    ("Entry:", "進場："),
    ("Risk:", "風險："),
    ("Horizon:", "時間範圍："),
    ("Confidence:", "信心度："),
    ("Rationale:", "理由："),
    ("Model:", "模型："),
    ("Degraded:", "降級代理："),
    ("Quantity:", "數量："),
    ("Proposed:", "提案："),
    ("Rule:", "規則："),
    ("Reason:", "原因："),
    ("Mode:", "模式："),
    ("Qty:", "數量："),
    ("Fees:", "手續費："),
    ("SL order:", "SL 訂單："),
    ("TP order:", "TP 訂單："),
    ("Expires in", "將在"),
    ("minutes", "分鐘後過期"),
    # Trade report
    ("PnL:", "PnL："),
    ("fees:", "手續費："),
    # Performance report
    ("Performance Report", "績效報告"),
    ("Total PnL:", "總 PnL："),
    ("Win Rate:", "勝率："),
    ("Profit Factor:", "獲利因子："),
    ("Max Drawdown:", "最大回撤："),
    ("Sharpe Ratio:", "Sharpe Ratio："),
    ("No trades yet. Performance report will be available after closing positions.",
     "尚無交易紀錄。平倉後即可查看績效報告。"),
    ("No performance data yet. Close some positions first.",
     "尚無績效資料。請先平倉後再查看。"),
    # Eval report
    ("Eval Report", "評估報告"),
    ("Cases:", "案例："),
    ("Passed:", "通過："),
    ("Failed:", "失敗："),
    ("Accuracy:", "準確率："),
    ("Consistency:", "一致性："),
    # History
    ("Recent closed trades:", "近期已平倉交易："),
    ("No closed trades yet.", "尚無已平倉交易。"),
    # Welcome / help
    ("Welcome to Sentinel Orchestrator!", "歡迎使用 Sentinel Orchestrator！"),
    ("I analyze crypto markets using multiple AI models and generate "
     "trade proposals with risk management.",
     "我會透過多個 AI 模型分析加密貨幣市場，並產生含風險管理的交易提案。"),
    ("Use /help to see available commands.", "輸入 /help 查看可用指令。"),
    ("Available commands:", "可用指令："),
    ("Welcome message", "歡迎訊息"),
    ("Account overview & latest proposals", "帳戶總覽與最新提案"),
    ("Detailed analysis for a symbol (e.g. /coin BTC)", "單一幣種詳細分析（例如 /coin BTC）"),
    ("Trigger pipeline for all symbols", "對所有幣種執行 pipeline"),
    ("Trigger pipeline for specific symbol", "對特定幣種執行 pipeline"),
    ("Trigger with specific model", "指定模型執行"),
    ("Recent trade records", "近期交易紀錄"),
    ("Performance report (PnL, win rate, Sharpe, etc.)", "績效報告（PnL、勝率、Sharpe 等）"),
    ("Run LLM evaluation and show results", "執行 LLM 評估並顯示結果"),
    ("Un-pause pipeline after risk pause", "風控暫停後恢復 pipeline"),
    ("Show this message", "顯示此訊息"),
    # Bot inline messages
    ("Running pipeline (model:", "正在執行 pipeline（模型："),
    ("Pipeline not configured.", "Pipeline 尚未設定。"),
    ("Paper trading not configured.", "模擬交易尚未設定。"),
    ("Pipeline resumed. Trading un-paused.", "Pipeline 已恢復，交易已解除暫停。"),
    ("Stats not configured.", "統計功能尚未設定。"),
    ("Eval not configured.", "評估功能尚未設定。"),
    ("Running evaluation...", "正在執行評估..."),
    ("Unknown symbol:", "找不到幣種："),
    ("Usage: /coin <symbol> (e.g. /coin BTC)", "用法：/coin <symbol>（例如 /coin BTC）"),
    ("REJECTED:", "拒絕原因："),
    ("No proposal generated", "未產生提案"),
    ("Exit:", "出場："),
]


def to_chinese(text: str) -> str:
    """Translate structured English bot output to Chinese via label replacement."""
    result = text
    for en, zh in _LINE_PREFIX_MAP:
        result = result.replace(en, zh)
    return result
