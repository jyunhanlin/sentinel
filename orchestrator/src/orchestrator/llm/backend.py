from __future__ import annotations

import asyncio
import json
import os
import re
import time
from abc import ABC, abstractmethod

import structlog
from litellm import acompletion

from orchestrator.llm.client import LLMCallResult

logger = structlog.get_logger(__name__)


class LLMBackend(ABC):
    """Abstract interface for LLM call execution."""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMCallResult: ...


class LiteLLMBackend(LLMBackend):
    """API-based backend via LiteLLM."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMCallResult:
        start = time.monotonic()

        response = await acompletion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self._api_key,
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        content = response.choices[0].message.content or ""
        usage = response.usage

        return LLMCallResult(
            content=content,
            model=model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=elapsed_ms,
        )


# ---------------------------------------------------------------------------
# Message conversion helpers
# ---------------------------------------------------------------------------


def _extract_system_prompt(
    messages: list[dict[str, str]],
) -> tuple[str | None, list[dict[str, str]]]:
    """Separate system message from the rest."""
    if messages and messages[0]["role"] == "system":
        return messages[0]["content"], messages[1:]
    return None, messages


def _flatten_messages(messages: list[dict[str, str]]) -> str:
    """Flatten user/assistant messages into a single prompt string."""
    parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            parts.append(content)
        elif role == "assistant":
            parts.append(f"[Assistant]: {content}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Model name mapping
# ---------------------------------------------------------------------------


def _map_model_name(model: str) -> str:
    """Map LiteLLM model names to Claude CLI aliases."""
    match = re.search(r"claude-(sonnet|opus|haiku)", model)
    if match:
        return match.group(1)
    return model


# ---------------------------------------------------------------------------
# Claude CLI backend
# ---------------------------------------------------------------------------


class ClaudeCLIBackend(LLMBackend):
    """Claude CLI backend via subprocess."""

    def __init__(
        self,
        cli_path: str = "claude",
        timeout: int = 120,
    ) -> None:
        self._cli_path = cli_path
        self._timeout = timeout

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMCallResult:
        system_prompt, remaining = _extract_system_prompt(messages)
        prompt_text = _flatten_messages(remaining)
        cli_model = _map_model_name(model)

        cmd = [
            self._cli_path,
            "-p",
            "--output-format",
            "json",
            "--model",
            cli_model,
        ]
        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])

        # Remove env vars that interfere with CLI subscription auth
        _exclude_env = {"CLAUDECODE", "ANTHROPIC_API_KEY"}
        env = {k: v for k, v in os.environ.items() if k not in _exclude_env}

        start = time.monotonic()

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=prompt_text.encode()),
                timeout=self._timeout,
            )
        except TimeoutError:
            process.kill()
            raise

        if process.returncode != 0:
            error_msg = stderr.decode().strip()
            stdout_msg = stdout.decode().strip()
            logger.error(
                "claude_cli_failed",
                returncode=process.returncode,
                stderr=error_msg,
                stdout=stdout_msg[:500],
                cmd=" ".join(cmd),
            )
            raise RuntimeError(
                f"Claude CLI failed (exit {process.returncode}): "
                f"{error_msg or stdout_msg[:200]}"
            )

        elapsed_ms = int((time.monotonic() - start) * 1000)

        envelope = json.loads(stdout.decode())
        content = envelope.get("result", "")

        logger.info(
            "claude_cli_complete",
            model=model,
            cli_model=cli_model,
            duration_ms=envelope.get("duration_ms", elapsed_ms),
            cost_usd=envelope.get("cost_usd", 0),
        )

        return LLMCallResult(
            content=content,
            model=model,
            input_tokens=0,
            output_tokens=0,
            latency_ms=elapsed_ms,
        )
