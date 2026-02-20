# Claude CLI Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace LiteLLM API calls with `claude -p` subprocess calls using the Strategy Pattern, keeping LiteLLM as a switchable fallback.

**Architecture:** Extract an `LLMBackend` ABC from `LLMClient`, create `LiteLLMBackend` (existing logic) and `ClaudeCLIBackend` (new subprocess-based). `LLMClient` becomes a thin facade that delegates to the injected backend. Config gains `llm_backend` field to select backend at startup.

**Tech Stack:** Python 3.12+, asyncio (create_subprocess_exec), Pydantic, pytest, structlog

---

### Task 1: LLMBackend ABC + LiteLLMBackend

**Files:**
- Create: `orchestrator/src/orchestrator/llm/backend.py`
- Create: `orchestrator/tests/unit/test_backend.py`

**Step 1: Write the failing test for LiteLLMBackend**

```python
# orchestrator/tests/unit/test_backend.py
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.llm.backend import LiteLLMBackend, LLMBackend
from orchestrator.llm.client import LLMCallResult


class TestLiteLLMBackend:
    def test_is_llm_backend(self):
        backend = LiteLLMBackend(api_key="test-key")
        assert isinstance(backend, LLMBackend)

    @pytest.mark.asyncio
    async def test_complete_returns_llm_call_result(self):
        backend = LiteLLMBackend(api_key="test-key")

        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = '{"score": 42}'
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50

        with patch("orchestrator.llm.backend.acompletion", return_value=mock_response):
            result = await backend.complete(
                messages=[{"role": "user", "content": "test"}],
                model="anthropic/claude-sonnet-4-6",
                temperature=0.2,
                max_tokens=2000,
            )

        assert isinstance(result, LLMCallResult)
        assert result.content == '{"score": 42}'
        assert result.model == "anthropic/claude-sonnet-4-6"
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_complete_passes_all_params_to_litellm(self):
        backend = LiteLLMBackend(api_key="my-key")

        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = "ok"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        with patch("orchestrator.llm.backend.acompletion", return_value=mock_response) as mock_call:
            await backend.complete(
                messages=[{"role": "user", "content": "hello"}],
                model="anthropic/claude-opus-4-6",
                temperature=0.7,
                max_tokens=500,
            )
            mock_call.assert_called_once_with(
                model="anthropic/claude-opus-4-6",
                messages=[{"role": "user", "content": "hello"}],
                temperature=0.7,
                max_tokens=500,
                api_key="my-key",
            )

    @pytest.mark.asyncio
    async def test_complete_handles_none_usage(self):
        backend = LiteLLMBackend(api_key="test-key")

        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = "ok"
        mock_response.usage = None

        with patch("orchestrator.llm.backend.acompletion", return_value=mock_response):
            result = await backend.complete(
                messages=[{"role": "user", "content": "test"}],
                model="test-model",
                temperature=0.2,
                max_tokens=2000,
            )

        assert result.input_tokens == 0
        assert result.output_tokens == 0
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/jhlin/playground/sentinel/orchestrator && uv run pytest tests/unit/test_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.llm.backend'`

**Step 3: Write the LLMBackend ABC + LiteLLMBackend implementation**

```python
# orchestrator/src/orchestrator/llm/backend.py
from __future__ import annotations

import time
from abc import ABC, abstractmethod

from litellm import acompletion

from orchestrator.llm.client import LLMCallResult


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
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/jhlin/playground/sentinel/orchestrator && uv run pytest tests/unit/test_backend.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/llm/backend.py orchestrator/tests/unit/test_backend.py
git commit -m "feat: add LLMBackend ABC and LiteLLMBackend implementation"
```

---

### Task 2: Refactor LLMClient to accept LLMBackend

**Files:**
- Modify: `orchestrator/src/orchestrator/llm/client.py`
- Modify: `orchestrator/tests/unit/test_llm_client.py`

**Step 1: Update tests to use backend injection**

Update `test_llm_client.py` to construct `LLMClient` with a backend. The client now delegates to the backend, so we mock the backend instead of `acompletion`.

```python
# orchestrator/tests/unit/test_llm_client.py
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from orchestrator.llm.backend import LLMBackend
from orchestrator.llm.client import LLMCallResult, LLMClient


def _make_backend(content: str = "response", input_tokens: int = 50, output_tokens: int = 25) -> LLMBackend:
    """Create a mock LLMBackend that returns a fixed LLMCallResult."""
    backend = AsyncMock(spec=LLMBackend)
    backend.complete.return_value = LLMCallResult(
        content=content,
        model="test-model",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=10,
    )
    return backend


class TestLLMClient:
    def test_create_client(self):
        backend = AsyncMock(spec=LLMBackend)
        client = LLMClient(backend=backend, model="anthropic/claude-sonnet-4-6")
        assert client.model == "anthropic/claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_call_returns_result(self):
        backend = _make_backend(content='{"sentiment_score": 72}', input_tokens=100, output_tokens=50)
        client = LLMClient(backend=backend, model="anthropic/claude-sonnet-4-6")

        result = await client.call(messages=[{"role": "user", "content": "analyze"}])

        assert isinstance(result, LLMCallResult)
        assert result.content == '{"sentiment_score": 72}'
        assert result.input_tokens == 100
        assert result.output_tokens == 50

    @pytest.mark.asyncio
    async def test_call_delegates_params_to_backend(self):
        backend = _make_backend()
        client = LLMClient(
            backend=backend,
            model="anthropic/claude-sonnet-4-6",
            temperature=0.5,
            max_tokens=500,
        )

        await client.call(messages=[{"role": "user", "content": "test"}])

        backend.complete.assert_called_once_with(
            [{"role": "user", "content": "test"}],
            model="anthropic/claude-sonnet-4-6",
            temperature=0.5,
            max_tokens=500,
        )

    @pytest.mark.asyncio
    async def test_call_with_zero_temperature(self):
        backend = _make_backend()
        client = LLMClient(
            backend=backend,
            model="anthropic/claude-sonnet-4-6",
            temperature=0.5,
        )

        await client.call(
            messages=[{"role": "user", "content": "test"}],
            temperature=0.0,
        )

        call_kwargs = backend.complete.call_args
        assert call_kwargs.kwargs["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_call_with_model_override(self):
        backend = _make_backend()
        client = LLMClient(backend=backend, model="anthropic/claude-sonnet-4-6")

        await client.call(
            messages=[{"role": "user", "content": "test"}],
            model="anthropic/claude-opus-4-6",
        )

        call_kwargs = backend.complete.call_args
        assert call_kwargs.kwargs["model"] == "anthropic/claude-opus-4-6"


class TestLLMCallResult:
    def test_result_is_frozen(self):
        result = LLMCallResult(
            content="test",
            model="test-model",
            input_tokens=10,
            output_tokens=5,
            latency_ms=100,
        )
        with pytest.raises(Exception):
            result.content = "modified"  # type: ignore
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/jhlin/playground/sentinel/orchestrator && uv run pytest tests/unit/test_llm_client.py -v`
Expected: FAIL — `LLMClient.__init__` doesn't accept `backend` param yet.

**Step 3: Update LLMClient implementation**

```python
# orchestrator/src/orchestrator/llm/client.py
from __future__ import annotations

import structlog
from pydantic import BaseModel

from orchestrator.llm.backend import LLMBackend

logger = structlog.get_logger(__name__)


class LLMCallResult(BaseModel, frozen=True):
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int


class LLMClient:
    def __init__(
        self,
        backend: LLMBackend,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> None:
        self._backend = backend
        self.model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def call(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMCallResult:
        effective_model = model or self.model

        result = await self._backend.complete(
            messages,
            model=effective_model,
            temperature=temperature if temperature is not None else self._temperature,
            max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
        )

        logger.info(
            "llm_call_complete",
            model=effective_model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms,
        )

        return result
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/jhlin/playground/sentinel/orchestrator && uv run pytest tests/unit/test_llm_client.py -v`
Expected: All tests PASS

**Step 5: Run full test suite to check nothing broke**

Run: `cd /Users/jhlin/playground/sentinel/orchestrator && uv run pytest tests/ -v`
Expected: Some tests may fail because other test files still construct `LLMClient(model=..., api_key=...)` with the old signature. Note these failures for Task 3.

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/llm/client.py orchestrator/tests/unit/test_llm_client.py
git commit -m "refactor: LLMClient accepts LLMBackend via constructor injection"
```

---

### Task 3: Fix callers — update LLMClient construction sites

**Files:**
- Modify: `orchestrator/src/orchestrator/__main__.py` (lines 82-88)
- Modify: `orchestrator/tests/unit/test_agent_base.py` (LLMClient construction)
- Modify: `orchestrator/tests/unit/test_agent_sentiment.py` (LLMClient construction)
- Modify: `orchestrator/tests/unit/test_agent_market.py` (LLMClient construction)
- Modify: `orchestrator/tests/unit/test_agent_proposer.py` (LLMClient construction)
- Modify: `orchestrator/tests/unit/test_main.py` (create_app_components calls)
- Modify: any other test files that construct `LLMClient` directly

**Step 1: Update `create_app_components` in `__main__.py`**

Replace lines 82-88:

```python
    # Before:
    llm_client = LLMClient(
        model=llm_model,
        api_key=anthropic_api_key,
        temperature=llm_temperature,
        max_tokens=llm_max_tokens,
    )

    # After:
    from orchestrator.llm.backend import LiteLLMBackend

    backend = LiteLLMBackend(api_key=anthropic_api_key)
    llm_client = LLMClient(
        backend=backend,
        model=llm_model,
        temperature=llm_temperature,
        max_tokens=llm_max_tokens,
    )
```

**Step 2: Update all test files that construct `LLMClient` directly**

In every test file, replace:
```python
LLMClient(model="anthropic/claude-sonnet-4-6", api_key="test-key")
```
with:
```python
from unittest.mock import AsyncMock
from orchestrator.llm.backend import LLMBackend

backend = AsyncMock(spec=LLMBackend)
LLMClient(backend=backend, model="anthropic/claude-sonnet-4-6")
```

Search for all construction sites:
```bash
cd /Users/jhlin/playground/sentinel/orchestrator && grep -rn "LLMClient(" tests/
```

Fix each one. For agent tests that mock `acompletion`, keep the mock but patch `orchestrator.llm.backend.acompletion` instead of `orchestrator.llm.client.acompletion`.

**Step 3: Run full test suite**

Run: `cd /Users/jhlin/playground/sentinel/orchestrator && uv run pytest tests/ -v`
Expected: All tests PASS

**Step 4: Run linter**

Run: `cd /Users/jhlin/playground/sentinel/orchestrator && uv run ruff check src/ tests/`
Expected: No errors

**Step 5: Commit**

```bash
git add -u
git commit -m "refactor: update all LLMClient callers to use backend injection"
```

---

### Task 4: ClaudeCLIBackend — message conversion helpers

**Files:**
- Modify: `orchestrator/src/orchestrator/llm/backend.py`
- Modify: `orchestrator/tests/unit/test_backend.py`

**Step 1: Write failing tests for message conversion**

```python
# Add to orchestrator/tests/unit/test_backend.py

from orchestrator.llm.backend import _extract_system_prompt, _flatten_messages


class TestMessageConversion:
    def test_extract_system_prompt_from_messages(self):
        messages = [
            {"role": "system", "content": "You are an analyst."},
            {"role": "user", "content": "Analyze BTC."},
        ]
        system, remaining = _extract_system_prompt(messages)
        assert system == "You are an analyst."
        assert remaining == [{"role": "user", "content": "Analyze BTC."}]

    def test_extract_system_prompt_no_system_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        system, remaining = _extract_system_prompt(messages)
        assert system is None
        assert remaining == [{"role": "user", "content": "Hello"}]

    def test_flatten_single_user_message(self):
        messages = [{"role": "user", "content": "Analyze BTC."}]
        result = _flatten_messages(messages)
        assert result == "Analyze BTC."

    def test_flatten_multi_turn_retry(self):
        messages = [
            {"role": "user", "content": "Analyze BTC."},
            {"role": "assistant", "content": "(invalid output)"},
            {"role": "user", "content": "Your previous response failed validation."},
        ]
        result = _flatten_messages(messages)
        assert "Analyze BTC." in result
        assert "(invalid output)" in result
        assert "Your previous response failed validation." in result
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/jhlin/playground/sentinel/orchestrator && uv run pytest tests/unit/test_backend.py::TestMessageConversion -v`
Expected: FAIL — functions not defined

**Step 3: Implement helper functions**

Add to `orchestrator/src/orchestrator/llm/backend.py`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/jhlin/playground/sentinel/orchestrator && uv run pytest tests/unit/test_backend.py::TestMessageConversion -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/llm/backend.py orchestrator/tests/unit/test_backend.py
git commit -m "feat: add message conversion helpers for CLI backend"
```

---

### Task 5: ClaudeCLIBackend — model name mapping

**Files:**
- Modify: `orchestrator/src/orchestrator/llm/backend.py`
- Modify: `orchestrator/tests/unit/test_backend.py`

**Step 1: Write failing tests**

```python
# Add to orchestrator/tests/unit/test_backend.py

from orchestrator.llm.backend import _map_model_name


class TestModelNameMapping:
    def test_maps_sonnet(self):
        assert _map_model_name("anthropic/claude-sonnet-4-6") == "sonnet"

    def test_maps_opus(self):
        assert _map_model_name("anthropic/claude-opus-4-6") == "opus"

    def test_maps_haiku(self):
        assert _map_model_name("anthropic/claude-haiku-4-5-20251001") == "haiku"

    def test_passes_through_unknown(self):
        assert _map_model_name("custom-model") == "custom-model"

    def test_passes_through_alias(self):
        assert _map_model_name("sonnet") == "sonnet"
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/jhlin/playground/sentinel/orchestrator && uv run pytest tests/unit/test_backend.py::TestModelNameMapping -v`
Expected: FAIL

**Step 3: Implement model name mapping**

```python
# Add to orchestrator/src/orchestrator/llm/backend.py

import re


def _map_model_name(model: str) -> str:
    """Map LiteLLM model names to Claude CLI aliases.

    Examples:
        "anthropic/claude-sonnet-4-6" -> "sonnet"
        "anthropic/claude-opus-4-6" -> "opus"
        "sonnet" -> "sonnet" (pass-through)
    """
    match = re.search(r"claude-(sonnet|opus|haiku)", model)
    if match:
        return match.group(1)
    return model
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/jhlin/playground/sentinel/orchestrator && uv run pytest tests/unit/test_backend.py::TestModelNameMapping -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/llm/backend.py orchestrator/tests/unit/test_backend.py
git commit -m "feat: add model name mapping for CLI backend"
```

---

### Task 6: ClaudeCLIBackend — subprocess implementation

**Files:**
- Modify: `orchestrator/src/orchestrator/llm/backend.py`
- Modify: `orchestrator/tests/unit/test_backend.py`

**Step 1: Write failing tests for ClaudeCLIBackend**

```python
# Add to orchestrator/tests/unit/test_backend.py

import json


class TestClaudeCLIBackend:
    def test_is_llm_backend(self):
        backend = ClaudeCLIBackend()
        assert isinstance(backend, LLMBackend)

    @pytest.mark.asyncio
    async def test_complete_returns_result(self):
        backend = ClaudeCLIBackend(cli_path="/usr/bin/claude", timeout=60)

        cli_output = json.dumps({
            "result": '{"score": 42}',
            "cost_usd": 0.01,
            "duration_ms": 1500,
            "num_turns": 1,
            "is_error": False,
            "session_id": "test-session",
        })

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (cli_output.encode(), b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            result = await backend.complete(
                messages=[
                    {"role": "system", "content": "You are an analyst."},
                    {"role": "user", "content": "Analyze BTC."},
                ],
                model="anthropic/claude-sonnet-4-6",
                temperature=0.2,
                max_tokens=2000,
            )

        assert isinstance(result, LLMCallResult)
        assert result.content == '{"score": 42}'
        assert result.model == "anthropic/claude-sonnet-4-6"
        assert result.latency_ms >= 0

        # Verify CLI was called correctly
        exec_args = mock_exec.call_args
        cmd_args = exec_args[0]
        assert cmd_args[0] == "/usr/bin/claude"
        assert "-p" in cmd_args
        assert "--output-format" in cmd_args
        assert "json" in cmd_args
        assert "--model" in cmd_args
        assert "sonnet" in cmd_args
        assert "--system-prompt" in cmd_args

    @pytest.mark.asyncio
    async def test_complete_strips_claudecode_env(self):
        backend = ClaudeCLIBackend()

        cli_output = json.dumps({
            "result": "ok",
            "is_error": False,
            "session_id": "s",
            "cost_usd": 0,
            "duration_ms": 100,
            "num_turns": 1,
        })

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (cli_output.encode(), b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            with patch.dict("os.environ", {"CLAUDECODE": "1"}):
                await backend.complete(
                    messages=[{"role": "user", "content": "test"}],
                    model="sonnet",
                    temperature=0.2,
                    max_tokens=2000,
                )

        exec_kwargs = mock_exec.call_args[1]
        assert "CLAUDECODE" not in exec_kwargs.get("env", {})

    @pytest.mark.asyncio
    async def test_complete_raises_on_nonzero_exit(self):
        backend = ClaudeCLIBackend()

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"Error: something went wrong")
        mock_process.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with pytest.raises(RuntimeError, match="Claude CLI failed"):
                await backend.complete(
                    messages=[{"role": "user", "content": "test"}],
                    model="sonnet",
                    temperature=0.2,
                    max_tokens=2000,
                )

    @pytest.mark.asyncio
    async def test_complete_raises_on_timeout(self):
        backend = ClaudeCLIBackend(timeout=1)

        mock_process = AsyncMock()
        mock_process.communicate.side_effect = TimeoutError()
        mock_process.kill = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with pytest.raises(TimeoutError):
                await backend.complete(
                    messages=[{"role": "user", "content": "test"}],
                    model="sonnet",
                    temperature=0.2,
                    max_tokens=2000,
                )
```

Update the import at the top of the test file:

```python
from orchestrator.llm.backend import (
    ClaudeCLIBackend,
    LiteLLMBackend,
    LLMBackend,
    _extract_system_prompt,
    _flatten_messages,
    _map_model_name,
)
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/jhlin/playground/sentinel/orchestrator && uv run pytest tests/unit/test_backend.py::TestClaudeCLIBackend -v`
Expected: FAIL — `ClaudeCLIBackend` not defined

**Step 3: Implement ClaudeCLIBackend**

Add to `orchestrator/src/orchestrator/llm/backend.py`:

```python
import asyncio
import json
import os

import structlog

logger = structlog.get_logger(__name__)


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
            "--output-format", "json",
            "--model", cli_model,
        ]
        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])

        # Remove CLAUDECODE env var to avoid nested session error
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

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
            logger.error("claude_cli_failed", returncode=process.returncode, stderr=error_msg)
            raise RuntimeError(f"Claude CLI failed (exit {process.returncode}): {error_msg}")

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
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/jhlin/playground/sentinel/orchestrator && uv run pytest tests/unit/test_backend.py -v`
Expected: All tests PASS

**Step 5: Run linter**

Run: `cd /Users/jhlin/playground/sentinel/orchestrator && uv run ruff check src/ tests/`
Expected: No errors

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/llm/backend.py orchestrator/tests/unit/test_backend.py
git commit -m "feat: add ClaudeCLIBackend with subprocess execution"
```

---

### Task 7: Config changes + backend factory

**Files:**
- Modify: `orchestrator/src/orchestrator/config.py`
- Modify: `orchestrator/src/orchestrator/__main__.py`
- Modify: `orchestrator/tests/unit/test_config.py`
- Modify: `orchestrator/tests/unit/test_main.py`

**Step 1: Write failing tests for new config fields**

```python
# Add to orchestrator/tests/unit/test_config.py

def test_settings_llm_backend_defaults():
    settings = Settings(
        telegram_bot_token="test",
        telegram_admin_chat_ids=[123],
    )
    assert settings.llm_backend == "api"
    assert settings.claude_cli_path == "claude"
    assert settings.claude_cli_timeout == 120


def test_settings_api_key_optional_for_cli():
    """anthropic_api_key should default to empty string for CLI mode."""
    settings = Settings(
        telegram_bot_token="test",
        telegram_admin_chat_ids=[123],
        llm_backend="cli",
    )
    assert settings.anthropic_api_key == ""
    assert settings.llm_backend == "cli"
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/jhlin/playground/sentinel/orchestrator && uv run pytest tests/unit/test_config.py::test_settings_llm_backend_defaults -v`
Expected: FAIL — fields don't exist yet, and `anthropic_api_key` is still required

**Step 3: Update config**

```python
# orchestrator/src/orchestrator/config.py — modify the LLM section

    # LLM
    anthropic_api_key: str = ""                      # Optional for CLI mode
    llm_model: str = "anthropic/claude-sonnet-4-6"
    llm_model_premium: str = "anthropic/claude-opus-4-6"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2000
    llm_max_retries: int = 1

    # LLM Backend
    llm_backend: str = "api"                         # "api" | "cli"
    claude_cli_path: str = "claude"
    claude_cli_timeout: int = 120                    # seconds
```

**Step 4: Fix existing tests that pass `anthropic_api_key` as required**

Existing `test_config.py` tests pass `anthropic_api_key="test-key"` — they should still work since it now has a default. The `test_settings_requires_telegram_token` test should still fail correctly.

**Step 5: Update `create_app_components` to accept backend config**

Add params to `create_app_components`:

```python
    llm_backend: str = "api",
    claude_cli_path: str = "claude",
    claude_cli_timeout: int = 120,
```

Replace the LLMClient construction block:

```python
    # LLM
    if llm_backend == "cli":
        from orchestrator.llm.backend import ClaudeCLIBackend
        backend = ClaudeCLIBackend(cli_path=claude_cli_path, timeout=claude_cli_timeout)
    else:
        from orchestrator.llm.backend import LiteLLMBackend
        backend = LiteLLMBackend(api_key=anthropic_api_key)

    llm_client = LLMClient(
        backend=backend,
        model=llm_model,
        temperature=llm_temperature,
        max_tokens=llm_max_tokens,
    )
```

Also update `_build_components` to pass the new settings:

```python
    llm_backend=settings.llm_backend,
    claude_cli_path=settings.claude_cli_path,
    claude_cli_timeout=settings.claude_cli_timeout,
```

**Step 6: Run full test suite**

Run: `cd /Users/jhlin/playground/sentinel/orchestrator && uv run pytest tests/ -v`
Expected: All tests PASS

**Step 7: Run linter**

Run: `cd /Users/jhlin/playground/sentinel/orchestrator && uv run ruff check src/ tests/`
Expected: No errors

**Step 8: Commit**

```bash
git add -u
git commit -m "feat: add llm_backend config and backend factory in create_app_components"
```

---

### Task 8: Integration smoke test + final verification

**Files:**
- Modify: `orchestrator/tests/unit/test_main.py`

**Step 1: Add test for CLI backend component creation**

```python
# Add to orchestrator/tests/unit/test_main.py

def test_create_app_components_with_cli_backend():
    """Verify components can be constructed with CLI backend."""
    components = create_app_components(
        telegram_bot_token="test-token",
        telegram_admin_chat_ids=[123],
        exchange_id="binance",
        database_url="sqlite:///:memory:",
        llm_backend="cli",
        claude_cli_path="/usr/local/bin/claude",
        claude_cli_timeout=60,
    )
    assert "bot" in components
    assert "runner" in components
```

**Step 2: Run the test**

Run: `cd /Users/jhlin/playground/sentinel/orchestrator && uv run pytest tests/unit/test_main.py::test_create_app_components_with_cli_backend -v`
Expected: PASS

**Step 3: Run full test suite with coverage**

Run: `cd /Users/jhlin/playground/sentinel/orchestrator && uv run pytest tests/ -v --cov=orchestrator`
Expected: All tests PASS, coverage >= 80%

**Step 4: Run linter**

Run: `cd /Users/jhlin/playground/sentinel/orchestrator && uv run ruff check src/ tests/`
Expected: No errors

**Step 5: Commit**

```bash
git add orchestrator/tests/unit/test_main.py
git commit -m "test: add integration test for CLI backend component creation"
```

---

## Summary

| Task | Description | New/Modified Files |
|------|-------------|-------------------|
| 1 | LLMBackend ABC + LiteLLMBackend | `backend.py`, `test_backend.py` |
| 2 | Refactor LLMClient to accept backend | `client.py`, `test_llm_client.py` |
| 3 | Fix all callers (main + tests) | `__main__.py`, all agent test files |
| 4 | Message conversion helpers | `backend.py`, `test_backend.py` |
| 5 | Model name mapping | `backend.py`, `test_backend.py` |
| 6 | ClaudeCLIBackend subprocess impl | `backend.py`, `test_backend.py` |
| 7 | Config + backend factory | `config.py`, `__main__.py`, tests |
| 8 | Integration smoke test | `test_main.py` |
