from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.llm.backend import (
    ClaudeCLIBackend,
    LiteLLMBackend,
    LLMBackend,
    _extract_system_prompt,
    _flatten_messages,
    _map_model_name,
)
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
    async def test_complete_without_system_prompt(self):
        """Skill-based agents send a single user message — no --system-prompt flag."""
        backend = ClaudeCLIBackend()

        cli_output = json.dumps({
            "result": '```json\n{"score": 42}\n```',
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
                    {"role": "user", "content": "Use the sentiment skill.\n\nData: ..."},
                ],
                model="sonnet",
                temperature=0.2,
                max_tokens=2000,
            )

        assert result.content == '```json\n{"score": 42}\n```'

        # Verify --system-prompt is NOT in the command
        exec_args = mock_exec.call_args
        cmd_args = exec_args[0]
        assert "--system-prompt" not in cmd_args

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
