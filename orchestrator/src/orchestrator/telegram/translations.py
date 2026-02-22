"""English → Chinese translation for Telegram messages via LLM.

Sends structured English bot output to a fast model (Sonnet) for natural
Traditional Chinese translation, preserving formatting and trading terms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.llm.client import LLMClient

_SYSTEM_PROMPT = (
    "You are a translator for a crypto trading Telegram bot. "
    "Translate the following message from English to Traditional Chinese (繁體中文).\n\n"
    "Rules:\n"
    "- Preserve all numbers, prices, percentages, and timestamps exactly\n"
    "- Keep trading abbreviations as-is: SL, TP, PnL, LONG, SHORT, FLAT\n"
    "- Keep symbol names as-is (e.g. BTC/USDT:USDT)\n"
    "- Keep model names as-is (e.g. claude-sonnet-4-6)\n"
    "- Preserve all formatting: line breaks, indentation, separators (─), brackets\n"
    "- Keep /command names as-is (e.g. /run, /status, /help)\n"
    "- Output ONLY the translated text, no explanations"
)


async def to_chinese(text: str, llm_client: LLMClient) -> str:
    """Translate structured English bot output to Chinese via LLM."""
    result = await llm_client.call(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        max_tokens=2000,
    )
    return result.content
