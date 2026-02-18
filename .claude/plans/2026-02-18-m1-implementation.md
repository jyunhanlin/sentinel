# M1: 3-Model Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the 3-LLM agent pipeline that analyzes market data, generates structured trade proposals, and pushes results through Telegram.

**Architecture:** Three LLM agents (Sentiment, Market, Proposer) orchestrated by a pipeline runner. LLM-1 and LLM-2 run in parallel via asyncio.gather, LLM-3 consumes their outputs. Schema validation with retry+degrade on failure. APScheduler for periodic execution + TG `/run` for manual triggers. All LLM calls recorded to SQLite for traceability.

**Tech Stack:** Python 3.12+, uv, LiteLLM, Pydantic v2, CCXT (async), python-telegram-bot, APScheduler, structlog, SQLModel, pytest, pytest-asyncio

**Design doc:** `.claude/plans/2026-02-18-m1-pipeline-design.md`

---

### Task 1: Add LiteLLM & APScheduler Dependencies

**Files:**
- Modify: `orchestrator/pyproject.toml`
- Modify: `orchestrator/src/orchestrator/config.py`

**Step 1: Add dependencies**

```bash
cd orchestrator && uv add litellm apscheduler
```

**Step 2: Add LLM config fields to Settings**

Add to `orchestrator/src/orchestrator/config.py`, inside `class Settings`, after the `anthropic_api_key` field:

```python
    # LLM
    anthropic_api_key: str
    llm_model: str = "anthropic/claude-sonnet-4-6"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2000
    llm_max_retries: int = 1
```

**Step 3: Verify import**

```bash
cd orchestrator && uv run python -c "import litellm; import apscheduler; print('OK')"
```

Expected: `OK`

**Step 4: Commit**

```bash
git add orchestrator/pyproject.toml orchestrator/uv.lock orchestrator/src/orchestrator/config.py
git commit -m "chore: add litellm and apscheduler dependencies, LLM config fields"
```

---

### Task 2: LLM Client Wrapper

**Files:**
- Create: `orchestrator/src/orchestrator/llm/client.py`
- Test: `orchestrator/tests/unit/test_llm_client.py`

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_llm_client.py
import pytest
from unittest.mock import AsyncMock, patch

from orchestrator.llm.client import LLMClient, LLMCallResult


class TestLLMClient:
    def test_create_client(self):
        client = LLMClient(model="anthropic/claude-sonnet-4-6", api_key="test-key")
        assert client.model == "anthropic/claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_call_returns_result(self):
        client = LLMClient(model="anthropic/claude-sonnet-4-6", api_key="test-key")

        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = '{"sentiment_score": 72}'
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50

        with patch("orchestrator.llm.client.acompletion", return_value=mock_response):
            result = await client.call(
                messages=[{"role": "user", "content": "analyze"}],
            )

        assert isinstance(result, LLMCallResult)
        assert result.content == '{"sentiment_score": 72}'
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_call_with_custom_params(self):
        client = LLMClient(
            model="anthropic/claude-sonnet-4-6",
            api_key="test-key",
            temperature=0.5,
            max_tokens=500,
        )

        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = "response"
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 25

        with patch("orchestrator.llm.client.acompletion", return_value=mock_response) as mock_call:
            await client.call(messages=[{"role": "user", "content": "test"}])
            mock_call.assert_called_once()
            call_kwargs = mock_call.call_args[1]
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["max_tokens"] == 500


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

**Step 2: Run test to verify it fails**

```bash
cd orchestrator && uv run pytest tests/unit/test_llm_client.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.llm.client'`

**Step 3: Write minimal implementation**

```python
# orchestrator/src/orchestrator/llm/client.py
from __future__ import annotations

import time

import structlog
from litellm import acompletion
from pydantic import BaseModel

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
        model: str,
        api_key: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def call(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMCallResult:
        start = time.monotonic()

        response = await acompletion(
            model=self.model,
            messages=messages,
            temperature=temperature or self._temperature,
            max_tokens=max_tokens or self._max_tokens,
            api_key=self._api_key,
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        content = response.choices[0].message.content

        result = LLMCallResult(
            content=content,
            model=self.model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            latency_ms=elapsed_ms,
        )

        logger.info(
            "llm_call_complete",
            model=self.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms,
        )

        return result
```

**Step 4: Run test to verify it passes**

```bash
cd orchestrator && uv run pytest tests/unit/test_llm_client.py -v
```

Expected: 4 passed

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/llm/client.py orchestrator/tests/unit/test_llm_client.py
git commit -m "feat: add LiteLLM client wrapper with cost tracking"
```

---

### Task 3: Schema Validator

**Files:**
- Create: `orchestrator/src/orchestrator/llm/schema_validator.py`
- Test: `orchestrator/tests/unit/test_schema_validator.py`

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_schema_validator.py
import pytest

from orchestrator.llm.schema_validator import ValidationSuccess, ValidationFailure, validate_llm_output
from orchestrator.models import SentimentReport


class TestValidateLLMOutput:
    def test_valid_json(self):
        raw = '{"sentiment_score": 72, "key_events": [], "sources": ["news"], "confidence": 0.8}'
        result = validate_llm_output(raw, SentimentReport)
        assert isinstance(result, ValidationSuccess)
        assert result.value.sentiment_score == 72

    def test_json_with_surrounding_text(self):
        raw = 'Here is my analysis:\n{"sentiment_score": 50, "key_events": [], "sources": [], "confidence": 0.5}\nDone.'
        result = validate_llm_output(raw, SentimentReport)
        assert isinstance(result, ValidationSuccess)
        assert result.value.sentiment_score == 50

    def test_invalid_json(self):
        raw = "This is not JSON at all"
        result = validate_llm_output(raw, SentimentReport)
        assert isinstance(result, ValidationFailure)
        assert "JSON" in result.error_message

    def test_schema_violation(self):
        raw = '{"sentiment_score": 200, "key_events": [], "sources": [], "confidence": 0.5}'
        result = validate_llm_output(raw, SentimentReport)
        assert isinstance(result, ValidationFailure)
        assert "sentiment_score" in result.error_message

    def test_missing_required_field(self):
        raw = '{"sentiment_score": 50}'
        result = validate_llm_output(raw, SentimentReport)
        assert isinstance(result, ValidationFailure)
        assert len(result.error_message) > 0

    def test_json_in_markdown_code_block(self):
        raw = '```json\n{"sentiment_score": 60, "key_events": [], "sources": [], "confidence": 0.7}\n```'
        result = validate_llm_output(raw, SentimentReport)
        assert isinstance(result, ValidationSuccess)
        assert result.value.sentiment_score == 60
```

**Step 2: Run test to verify it fails**

```bash
cd orchestrator && uv run pytest tests/unit/test_schema_validator.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# orchestrator/src/orchestrator/llm/schema_validator.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class ValidationSuccess[T]:
    value: T


@dataclass(frozen=True)
class ValidationFailure:
    error_message: str


type ValidationResult[T] = ValidationSuccess[T] | ValidationFailure


def _extract_json(raw: str) -> str | None:
    """Extract JSON object from raw LLM output, handling markdown code blocks and surrounding text."""
    # Try markdown code block first
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if md_match:
        return md_match.group(1).strip()

    # Try to find a JSON object
    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        return brace_match.group(0)

    return None


def validate_llm_output(raw: str, model_class: type[T]) -> ValidationResult[T]:
    """Parse and validate raw LLM string output against a Pydantic model."""
    json_str = _extract_json(raw)
    if json_str is None:
        return ValidationFailure(error_message="Could not extract JSON from LLM output.")

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return ValidationFailure(error_message=f"Invalid JSON: {e}")

    try:
        value = model_class.model_validate(data)
    except ValidationError as e:
        errors = "; ".join(
            f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
            for err in e.errors()
        )
        return ValidationFailure(error_message=f"Schema validation failed: {errors}")

    return ValidationSuccess(value=value)
```

**Step 4: Run test to verify it passes**

```bash
cd orchestrator && uv run pytest tests/unit/test_schema_validator.py -v
```

Expected: 6 passed

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/llm/schema_validator.py orchestrator/tests/unit/test_schema_validator.py
git commit -m "feat: add schema validator for LLM output parsing"
```

---

### Task 4: BaseAgent with Retry + Degrade

**Files:**
- Create: `orchestrator/src/orchestrator/agents/base.py`
- Test: `orchestrator/tests/unit/test_agent_base.py`

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_agent_base.py
import pytest
from unittest.mock import AsyncMock

from orchestrator.agents.base import BaseAgent
from orchestrator.llm.client import LLMCallResult, LLMClient
from orchestrator.llm.schema_validator import ValidationFailure, ValidationSuccess
from orchestrator.models import SentimentReport, KeyEvent


class FakeAgent(BaseAgent[SentimentReport]):
    output_model = SentimentReport

    def _build_messages(self, **kwargs) -> list[dict]:
        return [{"role": "user", "content": "analyze sentiment"}]

    def _get_default_output(self) -> SentimentReport:
        return SentimentReport(
            sentiment_score=50,
            key_events=[],
            sources=[],
            confidence=0.3,
        )

    def _build_retry_messages(self, original_messages: list[dict], error: str) -> list[dict]:
        return original_messages + [
            {"role": "assistant", "content": "invalid"},
            {"role": "user", "content": f"Fix: {error}. Respond with valid JSON only."},
        ]


class TestBaseAgent:
    @pytest.mark.asyncio
    async def test_successful_call(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content='{"sentiment_score": 72, "key_events": [], "sources": ["news"], "confidence": 0.8}',
            model="test",
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        agent = FakeAgent(client=mock_client, max_retries=1)
        result = await agent.analyze()

        assert result.output.sentiment_score == 72
        assert result.degraded is False
        assert len(result.llm_calls) == 1

    @pytest.mark.asyncio
    async def test_retry_then_succeed(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.side_effect = [
            LLMCallResult(
                content='{"sentiment_score": "bad"}',  # invalid
                model="test",
                input_tokens=100,
                output_tokens=50,
                latency_ms=500,
            ),
            LLMCallResult(
                content='{"sentiment_score": 72, "key_events": [], "sources": [], "confidence": 0.8}',
                model="test",
                input_tokens=150,
                output_tokens=60,
                latency_ms=600,
            ),
        ]

        agent = FakeAgent(client=mock_client, max_retries=1)
        result = await agent.analyze()

        assert result.output.sentiment_score == 72
        assert result.degraded is False
        assert len(result.llm_calls) == 2

    @pytest.mark.asyncio
    async def test_degrade_after_all_retries_fail(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content="not json at all",
            model="test",
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        agent = FakeAgent(client=mock_client, max_retries=1)
        result = await agent.analyze()

        assert result.output.sentiment_score == 50  # default
        assert result.output.confidence == 0.3  # default
        assert result.degraded is True
        assert len(result.llm_calls) == 2  # original + 1 retry
```

**Step 2: Run test to verify it fails**

```bash
cd orchestrator && uv run pytest tests/unit/test_agent_base.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# orchestrator/src/orchestrator/agents/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import structlog
from pydantic import BaseModel

from orchestrator.llm.client import LLMCallResult, LLMClient
from orchestrator.llm.schema_validator import ValidationSuccess, validate_llm_output

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class AgentResult(BaseModel, Generic[T]):
    output: T
    degraded: bool = False
    llm_calls: list[LLMCallResult] = []

    model_config = {"arbitrary_types_allowed": True}


class BaseAgent(ABC, Generic[T]):
    output_model: type[T]

    def __init__(self, client: LLMClient, max_retries: int = 1) -> None:
        self._client = client
        self._max_retries = max_retries

    async def analyze(self, **kwargs) -> AgentResult[T]:
        messages = self._build_messages(**kwargs)
        llm_calls: list[LLMCallResult] = []

        for attempt in range(1 + self._max_retries):
            call_result = await self._client.call(messages)
            llm_calls.append(call_result)

            validation = validate_llm_output(call_result.content, self.output_model)

            if isinstance(validation, ValidationSuccess):
                logger.info(
                    "agent_success",
                    agent=self.__class__.__name__,
                    attempt=attempt + 1,
                )
                return AgentResult(
                    output=validation.value,
                    degraded=False,
                    llm_calls=llm_calls,
                )

            # Retry with error feedback
            logger.warning(
                "agent_validation_failed",
                agent=self.__class__.__name__,
                attempt=attempt + 1,
                error=validation.error_message,
            )

            if attempt < self._max_retries:
                messages = self._build_retry_messages(messages, validation.error_message)

        # All retries exhausted — degrade
        logger.warning(
            "agent_degraded",
            agent=self.__class__.__name__,
            total_attempts=1 + self._max_retries,
        )
        return AgentResult(
            output=self._get_default_output(),
            degraded=True,
            llm_calls=llm_calls,
        )

    @abstractmethod
    def _build_messages(self, **kwargs) -> list[dict]:
        ...

    @abstractmethod
    def _get_default_output(self) -> T:
        ...

    def _build_retry_messages(self, original_messages: list[dict], error: str) -> list[dict]:
        return original_messages + [
            {"role": "assistant", "content": "(invalid output)"},
            {
                "role": "user",
                "content": (
                    f"Your previous response failed validation: {error}\n"
                    "Please respond with ONLY a valid JSON object matching the schema."
                ),
            },
        ]
```

**Step 4: Run test to verify it passes**

```bash
cd orchestrator && uv run pytest tests/unit/test_agent_base.py -v
```

Expected: 3 passed

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/agents/base.py orchestrator/tests/unit/test_agent_base.py
git commit -m "feat: add BaseAgent ABC with retry and degrade logic"
```

---

### Task 5: SentimentAgent (LLM-1)

**Files:**
- Create: `orchestrator/src/orchestrator/agents/sentiment.py`
- Test: `orchestrator/tests/unit/test_agent_sentiment.py`

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_agent_sentiment.py
import pytest
from unittest.mock import AsyncMock

from orchestrator.agents.sentiment import SentimentAgent
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.llm.client import LLMCallResult, LLMClient
from orchestrator.models import SentimentReport


def make_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        symbol="BTC/USDT:USDT",
        timeframe="1h",
        current_price=95200.0,
        volume_24h=1_000_000.0,
        funding_rate=0.0001,
        ohlcv=[
            [1700000000000, 94000.0, 95500.0, 93500.0, 95200.0, 1000.0],
            [1700003600000, 95200.0, 96000.0, 95000.0, 95800.0, 800.0],
        ],
    )


class TestSentimentAgent:
    @pytest.mark.asyncio
    async def test_successful_analysis(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=(
                '{"sentiment_score": 72, "key_events": '
                '[{"event": "BTC rally", "impact": "positive", "source": "market"}], '
                '"sources": ["market_data"], "confidence": 0.75}'
            ),
            model="test",
            input_tokens=200,
            output_tokens=100,
            latency_ms=1000,
        )

        agent = SentimentAgent(client=mock_client)
        result = await agent.analyze(snapshot=make_snapshot())

        assert isinstance(result.output, SentimentReport)
        assert result.output.sentiment_score == 72
        assert result.degraded is False

    @pytest.mark.asyncio
    async def test_prompt_contains_market_data(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content='{"sentiment_score": 50, "key_events": [], "sources": [], "confidence": 0.5}',
            model="test",
            input_tokens=200,
            output_tokens=100,
            latency_ms=500,
        )

        agent = SentimentAgent(client=mock_client)
        await agent.analyze(snapshot=make_snapshot())

        call_args = mock_client.call.call_args
        messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]
        user_msg = messages[-1]["content"]
        assert "BTC/USDT:USDT" in user_msg
        assert "95200" in user_msg

    @pytest.mark.asyncio
    async def test_degrade_returns_neutral(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content="broken",
            model="test",
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        agent = SentimentAgent(client=mock_client, max_retries=0)
        result = await agent.analyze(snapshot=make_snapshot())

        assert result.degraded is True
        assert result.output.sentiment_score == 50
        assert result.output.confidence <= 0.3
```

**Step 2: Run test to verify it fails**

```bash
cd orchestrator && uv run pytest tests/unit/test_agent_sentiment.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# orchestrator/src/orchestrator/agents/sentiment.py
from __future__ import annotations

from orchestrator.agents.base import BaseAgent
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.models import SentimentReport


class SentimentAgent(BaseAgent[SentimentReport]):
    output_model = SentimentReport

    def _build_messages(self, **kwargs) -> list[dict]:
        snapshot: MarketSnapshot = kwargs["snapshot"]

        system_prompt = (
            "You are a crypto market sentiment analyst. "
            "Analyze the provided market data and infer the current market sentiment.\n\n"
            "Respond with ONLY a JSON object matching this schema:\n"
            "{\n"
            '  "sentiment_score": <int 0-100, 50=neutral, >50=bullish, <50=bearish>,\n'
            '  "key_events": [{"event": "<description>", "impact": "positive|negative|neutral", "source": "<source>"}],\n'
            '  "sources": ["<list of data sources used>"],\n'
            '  "confidence": <float 0.0-1.0>\n'
            "}"
        )

        ohlcv_summary = self._summarize_ohlcv(snapshot)

        user_prompt = (
            f"Symbol: {snapshot.symbol}\n"
            f"Current Price: {snapshot.current_price}\n"
            f"24h Volume: {snapshot.volume_24h:,.0f}\n"
            f"Funding Rate: {snapshot.funding_rate:.6f}\n"
            f"Timeframe: {snapshot.timeframe}\n\n"
            f"Recent OHLCV ({len(snapshot.ohlcv)} candles):\n{ohlcv_summary}\n\n"
            "Based on this market data and your knowledge of crypto markets, "
            "analyze the current sentiment. Consider price action, volume trends, "
            "and funding rate implications."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _get_default_output(self) -> SentimentReport:
        return SentimentReport(
            sentiment_score=50,
            key_events=[],
            sources=["degraded"],
            confidence=0.1,
        )

    @staticmethod
    def _summarize_ohlcv(snapshot: MarketSnapshot) -> str:
        if not snapshot.ohlcv:
            return "No OHLCV data available"

        lines = []
        for candle in snapshot.ohlcv[-10:]:  # last 10 candles max
            ts, o, h, l, c, v = candle[0], candle[1], candle[2], candle[3], candle[4], candle[5]
            lines.append(f"  O={o:.1f} H={h:.1f} L={l:.1f} C={c:.1f} V={v:.0f}")
        return "\n".join(lines)
```

**Step 4: Run test to verify it passes**

```bash
cd orchestrator && uv run pytest tests/unit/test_agent_sentiment.py -v
```

Expected: 3 passed

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/agents/sentiment.py orchestrator/tests/unit/test_agent_sentiment.py
git commit -m "feat: add SentimentAgent (LLM-1) with market-data-driven prompts"
```

---

### Task 6: MarketAgent (LLM-2)

**Files:**
- Create: `orchestrator/src/orchestrator/agents/market.py`
- Test: `orchestrator/tests/unit/test_agent_market.py`

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_agent_market.py
import pytest
from unittest.mock import AsyncMock

from orchestrator.agents.market import MarketAgent
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.llm.client import LLMCallResult, LLMClient
from orchestrator.models import MarketInterpretation, Trend, VolatilityRegime


def make_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        symbol="BTC/USDT:USDT",
        timeframe="1h",
        current_price=95200.0,
        volume_24h=1_000_000.0,
        funding_rate=0.0001,
        ohlcv=[
            [1700000000000, 94000.0, 95500.0, 93500.0, 95200.0, 1000.0],
            [1700003600000, 95200.0, 96000.0, 95000.0, 95800.0, 800.0],
        ],
    )


class TestMarketAgent:
    @pytest.mark.asyncio
    async def test_successful_analysis(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=(
                '{"trend": "up", "volatility_regime": "medium", '
                '"key_levels": [{"type": "support", "price": 93000}], '
                '"risk_flags": ["funding_elevated"]}'
            ),
            model="test",
            input_tokens=200,
            output_tokens=100,
            latency_ms=1000,
        )

        agent = MarketAgent(client=mock_client)
        result = await agent.analyze(snapshot=make_snapshot())

        assert isinstance(result.output, MarketInterpretation)
        assert result.output.trend == Trend.UP
        assert result.degraded is False

    @pytest.mark.asyncio
    async def test_degrade_returns_neutral(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content="broken",
            model="test",
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        agent = MarketAgent(client=mock_client, max_retries=0)
        result = await agent.analyze(snapshot=make_snapshot())

        assert result.degraded is True
        assert result.output.trend == Trend.RANGE
```

**Step 2: Run test to verify it fails**

```bash
cd orchestrator && uv run pytest tests/unit/test_agent_market.py -v
```

Expected: FAIL

**Step 3: Write minimal implementation**

```python
# orchestrator/src/orchestrator/agents/market.py
from __future__ import annotations

from orchestrator.agents.base import BaseAgent
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.models import MarketInterpretation, Trend, VolatilityRegime


class MarketAgent(BaseAgent[MarketInterpretation]):
    output_model = MarketInterpretation

    def _build_messages(self, **kwargs) -> list[dict]:
        snapshot: MarketSnapshot = kwargs["snapshot"]

        system_prompt = (
            "You are a crypto technical analyst. "
            "Analyze the provided OHLCV data, funding rate, and volume to determine "
            "market structure, trend, volatility regime, key price levels, and risk flags.\n\n"
            "Respond with ONLY a JSON object matching this schema:\n"
            "{\n"
            '  "trend": "up" | "down" | "range",\n'
            '  "volatility_regime": "low" | "medium" | "high",\n'
            '  "key_levels": [{"type": "support|resistance", "price": <number>}],\n'
            '  "risk_flags": ["<flag_name>"]  // e.g. funding_elevated, oi_near_ath, volume_declining\n'
            "}"
        )

        ohlcv_summary = self._summarize_ohlcv(snapshot)

        user_prompt = (
            f"Symbol: {snapshot.symbol}\n"
            f"Current Price: {snapshot.current_price}\n"
            f"24h Volume: {snapshot.volume_24h:,.0f}\n"
            f"Funding Rate: {snapshot.funding_rate:.6f}\n"
            f"Timeframe: {snapshot.timeframe}\n\n"
            f"OHLCV Data ({len(snapshot.ohlcv)} candles):\n{ohlcv_summary}\n\n"
            "Identify the trend direction, volatility regime, key support/resistance levels, "
            "and any risk flags from the data."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _get_default_output(self) -> MarketInterpretation:
        return MarketInterpretation(
            trend=Trend.RANGE,
            volatility_regime=VolatilityRegime.MEDIUM,
            key_levels=[],
            risk_flags=["analysis_degraded"],
        )

    @staticmethod
    def _summarize_ohlcv(snapshot: MarketSnapshot) -> str:
        if not snapshot.ohlcv:
            return "No OHLCV data available"

        lines = []
        for candle in snapshot.ohlcv[-20:]:  # last 20 candles for technical analysis
            ts, o, h, l, c, v = candle[0], candle[1], candle[2], candle[3], candle[4], candle[5]
            lines.append(f"  O={o:.1f} H={h:.1f} L={l:.1f} C={c:.1f} V={v:.0f}")
        return "\n".join(lines)
```

**Step 4: Run test to verify it passes**

```bash
cd orchestrator && uv run pytest tests/unit/test_agent_market.py -v
```

Expected: 2 passed

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/agents/market.py orchestrator/tests/unit/test_agent_market.py
git commit -m "feat: add MarketAgent (LLM-2) for technical analysis"
```

---

### Task 7: ProposerAgent (LLM-3)

**Files:**
- Create: `orchestrator/src/orchestrator/agents/proposer.py`
- Test: `orchestrator/tests/unit/test_agent_proposer.py`

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_agent_proposer.py
import pytest
from unittest.mock import AsyncMock

from orchestrator.agents.proposer import ProposerAgent
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.llm.client import LLMCallResult, LLMClient
from orchestrator.models import (
    KeyLevel,
    MarketInterpretation,
    SentimentReport,
    Side,
    TradeProposal,
    Trend,
    VolatilityRegime,
)


def make_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        symbol="BTC/USDT:USDT",
        timeframe="1h",
        current_price=95200.0,
        volume_24h=1_000_000.0,
        funding_rate=0.0001,
        ohlcv=[[1700000000000, 94000.0, 95500.0, 93500.0, 95200.0, 1000.0]],
    )


def make_sentiment() -> SentimentReport:
    return SentimentReport(
        sentiment_score=72,
        key_events=[],
        sources=["market_data"],
        confidence=0.8,
    )


def make_market() -> MarketInterpretation:
    return MarketInterpretation(
        trend=Trend.UP,
        volatility_regime=VolatilityRegime.MEDIUM,
        key_levels=[KeyLevel(type="support", price=93000.0)],
        risk_flags=[],
    )


class TestProposerAgent:
    @pytest.mark.asyncio
    async def test_successful_proposal(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=(
                '{"symbol": "BTC/USDT:USDT", "side": "long", '
                '"entry": {"type": "market"}, "position_size_risk_pct": 1.5, '
                '"stop_loss": 93000, "take_profit": [97000], '
                '"time_horizon": "4h", "confidence": 0.75, '
                '"invalid_if": [], "rationale": "Bullish momentum"}'
            ),
            model="test",
            input_tokens=300,
            output_tokens=150,
            latency_ms=1500,
        )

        agent = ProposerAgent(client=mock_client)
        result = await agent.analyze(
            snapshot=make_snapshot(),
            sentiment=make_sentiment(),
            market=make_market(),
        )

        assert isinstance(result.output, TradeProposal)
        assert result.output.side == Side.LONG
        assert result.degraded is False

    @pytest.mark.asyncio
    async def test_prompt_contains_all_inputs(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content=(
                '{"symbol": "BTC/USDT:USDT", "side": "flat", '
                '"entry": {"type": "market"}, "position_size_risk_pct": 0, '
                '"stop_loss": null, "take_profit": [], '
                '"time_horizon": "4h", "confidence": 0.5, '
                '"invalid_if": [], "rationale": "No clear signal"}'
            ),
            model="test",
            input_tokens=300,
            output_tokens=150,
            latency_ms=1000,
        )

        agent = ProposerAgent(client=mock_client)
        await agent.analyze(
            snapshot=make_snapshot(),
            sentiment=make_sentiment(),
            market=make_market(),
        )

        call_args = mock_client.call.call_args
        messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]
        user_msg = messages[-1]["content"]
        # Should contain data from all three inputs
        assert "sentiment_score" in user_msg or "72" in user_msg
        assert "up" in user_msg.lower() or "trend" in user_msg.lower()
        assert "95200" in user_msg

    @pytest.mark.asyncio
    async def test_degrade_returns_flat(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.call.return_value = LLMCallResult(
            content="broken",
            model="test",
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        agent = ProposerAgent(client=mock_client, max_retries=0)
        result = await agent.analyze(
            snapshot=make_snapshot(),
            sentiment=make_sentiment(),
            market=make_market(),
        )

        assert result.degraded is True
        assert result.output.side == Side.FLAT
```

**Step 2: Run test to verify it fails**

```bash
cd orchestrator && uv run pytest tests/unit/test_agent_proposer.py -v
```

Expected: FAIL

**Step 3: Write minimal implementation**

```python
# orchestrator/src/orchestrator/agents/proposer.py
from __future__ import annotations

from orchestrator.agents.base import BaseAgent
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.models import (
    EntryOrder,
    MarketInterpretation,
    SentimentReport,
    Side,
    TradeProposal,
)


class ProposerAgent(BaseAgent[TradeProposal]):
    output_model = TradeProposal

    def _build_messages(self, **kwargs) -> list[dict]:
        snapshot: MarketSnapshot = kwargs["snapshot"]
        sentiment: SentimentReport = kwargs["sentiment"]
        market: MarketInterpretation = kwargs["market"]

        system_prompt = (
            "You are a crypto trade proposal generator. "
            "Based on sentiment analysis, technical analysis, and current market data, "
            "generate a structured trade proposal.\n\n"
            "Rules:\n"
            "- If no clear edge, set side='flat' with position_size_risk_pct=0\n"
            "- stop_loss MUST be below entry for long, above entry for short\n"
            "- position_size_risk_pct: 0.5-2.0% typical range\n"
            "- confidence: be conservative, rarely above 0.8\n\n"
            "Respond with ONLY a JSON object matching this schema:\n"
            "{\n"
            '  "symbol": "<symbol>",\n'
            '  "side": "long" | "short" | "flat",\n'
            '  "entry": {"type": "market"} or {"type": "limit", "price": <number>},\n'
            '  "position_size_risk_pct": <float 0.0-2.0>,\n'
            '  "stop_loss": <number or null>,\n'
            '  "take_profit": [<number>, ...],\n'
            '  "time_horizon": "<e.g. 4h, 1d>",\n'
            '  "confidence": <float 0.0-1.0>,\n'
            '  "invalid_if": ["<condition>"],\n'
            '  "rationale": "<1-2 sentence explanation>"\n'
            "}"
        )

        key_levels_str = ", ".join(
            f"{kl.type}={kl.price}" for kl in market.key_levels
        ) or "none identified"

        risk_flags_str = ", ".join(market.risk_flags) or "none"

        user_prompt = (
            f"=== Market Data ===\n"
            f"Symbol: {snapshot.symbol}\n"
            f"Current Price: {snapshot.current_price}\n"
            f"24h Volume: {snapshot.volume_24h:,.0f}\n"
            f"Funding Rate: {snapshot.funding_rate:.6f}\n\n"
            f"=== Sentiment Analysis ===\n"
            f"Sentiment Score: {sentiment.sentiment_score}/100\n"
            f"Confidence: {sentiment.confidence}\n"
            f"Key Events: {', '.join(e.event for e in sentiment.key_events) or 'none'}\n\n"
            f"=== Technical Analysis ===\n"
            f"Trend: {market.trend}\n"
            f"Volatility: {market.volatility_regime}\n"
            f"Key Levels: {key_levels_str}\n"
            f"Risk Flags: {risk_flags_str}\n\n"
            f"Generate a trade proposal for {snapshot.symbol}."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _get_default_output(self) -> TradeProposal:
        return TradeProposal(
            symbol="unknown",
            side=Side.FLAT,
            entry=EntryOrder(type="market"),
            position_size_risk_pct=0.0,
            stop_loss=None,
            take_profit=[],
            time_horizon="4h",
            confidence=0.0,
            invalid_if=[],
            rationale="Analysis degraded — no trade signal generated",
        )
```

**Step 4: Run test to verify it passes**

```bash
cd orchestrator && uv run pytest tests/unit/test_agent_proposer.py -v
```

Expected: 3 passed

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/agents/proposer.py orchestrator/tests/unit/test_agent_proposer.py
git commit -m "feat: add ProposerAgent (LLM-3) for trade proposal generation"
```

---

### Task 8: Aggregator

**Files:**
- Create: `orchestrator/src/orchestrator/pipeline/aggregator.py`
- Test: `orchestrator/tests/unit/test_aggregator.py`

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_aggregator.py
import pytest

from orchestrator.models import EntryOrder, Side, TradeProposal
from orchestrator.pipeline.aggregator import AggregationResult, aggregate_proposal


class TestAggregateProposal:
    def test_valid_long_proposal(self):
        proposal = TradeProposal(
            symbol="BTC/USDT:USDT",
            side=Side.LONG,
            entry=EntryOrder(type="market"),
            position_size_risk_pct=1.5,
            stop_loss=93000.0,
            take_profit=[97000.0],
            time_horizon="4h",
            confidence=0.75,
            invalid_if=[],
            rationale="Bullish",
        )
        result = aggregate_proposal(proposal, current_price=95200.0)
        assert result.valid is True
        assert result.proposal == proposal

    def test_long_with_sl_above_entry_rejected(self):
        proposal = TradeProposal(
            symbol="BTC/USDT:USDT",
            side=Side.LONG,
            entry=EntryOrder(type="market"),
            position_size_risk_pct=1.5,
            stop_loss=97000.0,  # SL above current price for long = wrong
            take_profit=[99000.0],
            time_horizon="4h",
            confidence=0.75,
            invalid_if=[],
            rationale="Bad SL",
        )
        result = aggregate_proposal(proposal, current_price=95200.0)
        assert result.valid is False
        assert "stop_loss" in result.rejection_reason.lower()

    def test_short_with_sl_below_entry_rejected(self):
        proposal = TradeProposal(
            symbol="BTC/USDT:USDT",
            side=Side.SHORT,
            entry=EntryOrder(type="market"),
            position_size_risk_pct=1.5,
            stop_loss=93000.0,  # SL below current price for short = wrong
            take_profit=[91000.0],
            time_horizon="4h",
            confidence=0.75,
            invalid_if=[],
            rationale="Bad SL",
        )
        result = aggregate_proposal(proposal, current_price=95200.0)
        assert result.valid is False
        assert "stop_loss" in result.rejection_reason.lower()

    def test_flat_always_valid(self):
        proposal = TradeProposal(
            symbol="BTC/USDT:USDT",
            side=Side.FLAT,
            entry=EntryOrder(type="market"),
            position_size_risk_pct=0.0,
            stop_loss=None,
            take_profit=[],
            time_horizon="4h",
            confidence=0.5,
            invalid_if=[],
            rationale="No trade",
        )
        result = aggregate_proposal(proposal, current_price=95200.0)
        assert result.valid is True

    def test_long_without_sl_rejected(self):
        proposal = TradeProposal(
            symbol="BTC/USDT:USDT",
            side=Side.LONG,
            entry=EntryOrder(type="market"),
            position_size_risk_pct=1.5,
            stop_loss=None,  # No SL for a directional trade
            take_profit=[97000.0],
            time_horizon="4h",
            confidence=0.75,
            invalid_if=[],
            rationale="Missing SL",
        )
        result = aggregate_proposal(proposal, current_price=95200.0)
        assert result.valid is False
```

**Step 2: Run test to verify it fails**

```bash
cd orchestrator && uv run pytest tests/unit/test_aggregator.py -v
```

Expected: FAIL

**Step 3: Write minimal implementation**

```python
# orchestrator/src/orchestrator/pipeline/aggregator.py
from __future__ import annotations

from pydantic import BaseModel

from orchestrator.models import Side, TradeProposal


class AggregationResult(BaseModel, frozen=True):
    valid: bool
    proposal: TradeProposal
    rejection_reason: str = ""


def aggregate_proposal(proposal: TradeProposal, *, current_price: float) -> AggregationResult:
    """Validate a TradeProposal for sanity before forwarding."""
    if proposal.side == Side.FLAT:
        return AggregationResult(valid=True, proposal=proposal)

    # Directional trades require a stop loss
    if proposal.stop_loss is None:
        return AggregationResult(
            valid=False,
            proposal=proposal,
            rejection_reason="Directional trade (long/short) requires a stop_loss.",
        )

    # SL must be on the correct side of entry
    if proposal.side == Side.LONG and proposal.stop_loss >= current_price:
        return AggregationResult(
            valid=False,
            proposal=proposal,
            rejection_reason=(
                f"Long stop_loss ({proposal.stop_loss}) must be below "
                f"current price ({current_price})."
            ),
        )

    if proposal.side == Side.SHORT and proposal.stop_loss <= current_price:
        return AggregationResult(
            valid=False,
            proposal=proposal,
            rejection_reason=(
                f"Short stop_loss ({proposal.stop_loss}) must be above "
                f"current price ({current_price})."
            ),
        )

    return AggregationResult(valid=True, proposal=proposal)
```

**Step 4: Run test to verify it passes**

```bash
cd orchestrator && uv run pytest tests/unit/test_aggregator.py -v
```

Expected: 5 passed

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/pipeline/aggregator.py orchestrator/tests/unit/test_aggregator.py
git commit -m "feat: add proposal aggregator with sanity checks"
```

---

### Task 9: Storage Repository Extension

**Files:**
- Modify: `orchestrator/src/orchestrator/storage/repository.py`
- Modify: `orchestrator/tests/unit/test_storage.py`

**Step 1: Write the failing test**

Append to `orchestrator/tests/unit/test_storage.py`:

```python
# Add these imports at top:
from orchestrator.storage.models import LLMCallRecord, TradeProposalRecord
from orchestrator.storage.repository import LLMCallRepository, TradeProposalRepository

# Add these test classes:

class TestLLMCallRepository:
    def test_save_and_list_calls(self, session):
        repo = LLMCallRepository(session)
        repo.save_call(
            run_id="run-001",
            agent_type="sentiment",
            prompt="analyze",
            response='{"score": 72}',
            model="test-model",
            latency_ms=500,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
        )
        repo.save_call(
            run_id="run-001",
            agent_type="market",
            prompt="analyze",
            response='{"trend": "up"}',
            model="test-model",
            latency_ms=600,
            input_tokens=120,
            output_tokens=60,
            cost_usd=0.012,
        )
        calls = repo.list_by_run("run-001")
        assert len(calls) == 2


class TestTradeProposalRepository:
    def test_save_and_get_proposal(self, session):
        repo = TradeProposalRepository(session)
        repo.save_proposal(
            proposal_id="prop-001",
            run_id="run-001",
            proposal_json='{"side": "long"}',
            risk_check_result="approved",
        )
        result = repo.get_latest_by_symbol("run-001")
        assert result is not None

    def test_get_latest_proposals(self, session):
        repo = TradeProposalRepository(session)
        repo.save_proposal(
            proposal_id="prop-001",
            run_id="run-001",
            proposal_json='{"symbol": "BTC/USDT:USDT"}',
        )
        repo.save_proposal(
            proposal_id="prop-002",
            run_id="run-002",
            proposal_json='{"symbol": "ETH/USDT:USDT"}',
        )
        results = repo.get_recent(limit=10)
        assert len(results) == 2
```

**Step 2: Run test to verify it fails**

```bash
cd orchestrator && uv run pytest tests/unit/test_storage.py -v
```

Expected: FAIL with `ImportError`

**Step 3: Extend repository.py**

Add to `orchestrator/src/orchestrator/storage/repository.py`:

```python
# Add to imports at top:
from orchestrator.storage.models import LLMCallRecord, PipelineRunRecord, TradeProposalRecord


class LLMCallRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def save_call(
        self,
        *,
        run_id: str,
        agent_type: str,
        prompt: str,
        response: str,
        model: str,
        latency_ms: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> LLMCallRecord:
        record = LLMCallRecord(
            run_id=run_id,
            agent_type=agent_type,
            prompt=prompt,
            response=response,
            model=model,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
        self._session.add(record)
        self._session.commit()
        self._session.refresh(record)
        return record

    def list_by_run(self, run_id: str) -> list[LLMCallRecord]:
        statement = select(LLMCallRecord).where(LLMCallRecord.run_id == run_id)
        return list(self._session.exec(statement).all())


class TradeProposalRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def save_proposal(
        self,
        *,
        proposal_id: str,
        run_id: str,
        proposal_json: str,
        risk_check_result: str = "",
        risk_check_reason: str = "",
    ) -> TradeProposalRecord:
        record = TradeProposalRecord(
            proposal_id=proposal_id,
            run_id=run_id,
            proposal_json=proposal_json,
            risk_check_result=risk_check_result,
            risk_check_reason=risk_check_reason,
        )
        self._session.add(record)
        self._session.commit()
        self._session.refresh(record)
        return record

    def get_latest_by_symbol(self, run_id: str) -> TradeProposalRecord | None:
        statement = (
            select(TradeProposalRecord)
            .where(TradeProposalRecord.run_id == run_id)
            .order_by(TradeProposalRecord.created_at.desc())
        )
        return self._session.exec(statement).first()

    def get_recent(self, *, limit: int = 10) -> list[TradeProposalRecord]:
        statement = (
            select(TradeProposalRecord)
            .order_by(TradeProposalRecord.created_at.desc())
            .limit(limit)
        )
        return list(self._session.exec(statement).all())
```

**Step 4: Run test to verify it passes**

```bash
cd orchestrator && uv run pytest tests/unit/test_storage.py -v
```

Expected: 7 passed (4 old + 3 new)

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/storage/repository.py orchestrator/tests/unit/test_storage.py
git commit -m "feat: extend storage with LLMCall and TradeProposal repositories"
```

---

### Task 10: Pipeline Runner

**Files:**
- Create: `orchestrator/src/orchestrator/pipeline/runner.py`
- Test: `orchestrator/tests/unit/test_runner.py`

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_runner.py
import pytest
from unittest.mock import AsyncMock, MagicMock

from orchestrator.agents.base import AgentResult
from orchestrator.exchange.data_fetcher import MarketSnapshot
from orchestrator.llm.client import LLMCallResult
from orchestrator.models import (
    EntryOrder,
    MarketInterpretation,
    SentimentReport,
    Side,
    TradeProposal,
    Trend,
    VolatilityRegime,
)
from orchestrator.pipeline.runner import PipelineResult, PipelineRunner


def make_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        symbol="BTC/USDT:USDT",
        timeframe="1h",
        current_price=95200.0,
        volume_24h=1_000_000.0,
        funding_rate=0.0001,
        ohlcv=[[1700000000000, 94000.0, 95500.0, 93500.0, 95200.0, 1000.0]],
    )


def make_llm_call() -> LLMCallResult:
    return LLMCallResult(
        content="{}",
        model="test",
        input_tokens=100,
        output_tokens=50,
        latency_ms=500,
    )


class TestPipelineRunner:
    @pytest.mark.asyncio
    async def test_successful_run(self):
        sentiment_result = AgentResult(
            output=SentimentReport(
                sentiment_score=72, key_events=[], sources=["test"], confidence=0.8
            ),
            degraded=False,
            llm_calls=[make_llm_call()],
        )
        market_result = AgentResult(
            output=MarketInterpretation(
                trend=Trend.UP,
                volatility_regime=VolatilityRegime.MEDIUM,
                key_levels=[],
                risk_flags=[],
            ),
            degraded=False,
            llm_calls=[make_llm_call()],
        )
        proposal_result = AgentResult(
            output=TradeProposal(
                symbol="BTC/USDT:USDT",
                side=Side.LONG,
                entry=EntryOrder(type="market"),
                position_size_risk_pct=1.5,
                stop_loss=93000.0,
                take_profit=[97000.0],
                time_horizon="4h",
                confidence=0.75,
                invalid_if=[],
                rationale="Bullish",
            ),
            degraded=False,
            llm_calls=[make_llm_call()],
        )

        mock_sentiment = AsyncMock()
        mock_sentiment.analyze.return_value = sentiment_result
        mock_market = AsyncMock()
        mock_market.analyze.return_value = market_result
        mock_proposer = AsyncMock()
        mock_proposer.analyze.return_value = proposal_result
        mock_fetcher = AsyncMock()
        mock_fetcher.fetch_snapshot.return_value = make_snapshot()
        mock_repo = MagicMock()
        mock_repo.create_run.return_value = MagicMock(run_id="test-run")

        runner = PipelineRunner(
            data_fetcher=mock_fetcher,
            sentiment_agent=mock_sentiment,
            market_agent=mock_market,
            proposer_agent=mock_proposer,
            pipeline_repo=mock_repo,
            llm_call_repo=MagicMock(),
            proposal_repo=MagicMock(),
        )

        result = await runner.execute("BTC/USDT:USDT")

        assert isinstance(result, PipelineResult)
        assert result.status == "completed"
        assert result.proposal is not None
        assert result.proposal.side == Side.LONG

    @pytest.mark.asyncio
    async def test_run_with_invalid_proposal(self):
        sentiment_result = AgentResult(
            output=SentimentReport(
                sentiment_score=72, key_events=[], sources=["test"], confidence=0.8
            ),
            degraded=False,
            llm_calls=[make_llm_call()],
        )
        market_result = AgentResult(
            output=MarketInterpretation(
                trend=Trend.UP,
                volatility_regime=VolatilityRegime.MEDIUM,
                key_levels=[],
                risk_flags=[],
            ),
            degraded=False,
            llm_calls=[make_llm_call()],
        )
        # Proposal with SL on wrong side
        proposal_result = AgentResult(
            output=TradeProposal(
                symbol="BTC/USDT:USDT",
                side=Side.LONG,
                entry=EntryOrder(type="market"),
                position_size_risk_pct=1.5,
                stop_loss=97000.0,  # above price = invalid for long
                take_profit=[99000.0],
                time_horizon="4h",
                confidence=0.75,
                invalid_if=[],
                rationale="Bad proposal",
            ),
            degraded=False,
            llm_calls=[make_llm_call()],
        )

        mock_sentiment = AsyncMock()
        mock_sentiment.analyze.return_value = sentiment_result
        mock_market = AsyncMock()
        mock_market.analyze.return_value = market_result
        mock_proposer = AsyncMock()
        mock_proposer.analyze.return_value = proposal_result
        mock_fetcher = AsyncMock()
        mock_fetcher.fetch_snapshot.return_value = make_snapshot()
        mock_repo = MagicMock()
        mock_repo.create_run.return_value = MagicMock(run_id="test-run")

        runner = PipelineRunner(
            data_fetcher=mock_fetcher,
            sentiment_agent=mock_sentiment,
            market_agent=mock_market,
            proposer_agent=mock_proposer,
            pipeline_repo=mock_repo,
            llm_call_repo=MagicMock(),
            proposal_repo=MagicMock(),
        )

        result = await runner.execute("BTC/USDT:USDT")

        assert result.status == "rejected"
        assert "stop_loss" in result.rejection_reason.lower()
```

**Step 2: Run test to verify it fails**

```bash
cd orchestrator && uv run pytest tests/unit/test_runner.py -v
```

Expected: FAIL

**Step 3: Write minimal implementation**

```python
# orchestrator/src/orchestrator/pipeline/runner.py
from __future__ import annotations

import asyncio
import uuid

import structlog
from pydantic import BaseModel

from orchestrator.agents.base import AgentResult, BaseAgent
from orchestrator.exchange.data_fetcher import DataFetcher
from orchestrator.models import MarketInterpretation, SentimentReport, TradeProposal
from orchestrator.pipeline.aggregator import aggregate_proposal
from orchestrator.storage.repository import (
    LLMCallRepository,
    PipelineRepository,
    TradeProposalRepository,
)

logger = structlog.get_logger(__name__)


class PipelineResult(BaseModel, frozen=True):
    run_id: str
    symbol: str
    status: str  # completed, rejected, failed
    proposal: TradeProposal | None = None
    rejection_reason: str = ""
    sentiment_degraded: bool = False
    market_degraded: bool = False
    proposer_degraded: bool = False


class PipelineRunner:
    def __init__(
        self,
        *,
        data_fetcher: DataFetcher,
        sentiment_agent: BaseAgent,
        market_agent: BaseAgent,
        proposer_agent: BaseAgent,
        pipeline_repo: PipelineRepository,
        llm_call_repo: LLMCallRepository,
        proposal_repo: TradeProposalRepository,
    ) -> None:
        self._data_fetcher = data_fetcher
        self._sentiment_agent = sentiment_agent
        self._market_agent = market_agent
        self._proposer_agent = proposer_agent
        self._pipeline_repo = pipeline_repo
        self._llm_call_repo = llm_call_repo
        self._proposal_repo = proposal_repo

    async def execute(self, symbol: str, *, timeframe: str = "1h") -> PipelineResult:
        run_id = str(uuid.uuid4())
        log = logger.bind(run_id=run_id, symbol=symbol)
        log.info("pipeline_start")

        self._pipeline_repo.create_run(run_id=run_id, symbol=symbol)

        try:
            # Fetch market data
            snapshot = await self._data_fetcher.fetch_snapshot(symbol, timeframe=timeframe)
            log.info("snapshot_fetched", price=snapshot.current_price)

            # Run LLM-1 and LLM-2 in parallel
            sentiment_result, market_result = await asyncio.gather(
                self._sentiment_agent.analyze(snapshot=snapshot),
                self._market_agent.analyze(snapshot=snapshot),
            )

            self._save_llm_calls(run_id, "sentiment", sentiment_result)
            self._save_llm_calls(run_id, "market", market_result)

            # Run LLM-3 (depends on LLM-1 + LLM-2)
            proposer_result = await self._proposer_agent.analyze(
                snapshot=snapshot,
                sentiment=sentiment_result.output,
                market=market_result.output,
            )
            self._save_llm_calls(run_id, "proposer", proposer_result)

            # Validate proposal
            aggregation = aggregate_proposal(
                proposer_result.output, current_price=snapshot.current_price
            )

            if aggregation.valid:
                self._proposal_repo.save_proposal(
                    proposal_id=aggregation.proposal.proposal_id,
                    run_id=run_id,
                    proposal_json=aggregation.proposal.model_dump_json(),
                    risk_check_result="approved",
                )
                self._pipeline_repo.update_run_status(run_id, "completed")
                log.info("pipeline_completed", side=aggregation.proposal.side)

                return PipelineResult(
                    run_id=run_id,
                    symbol=symbol,
                    status="completed",
                    proposal=aggregation.proposal,
                    sentiment_degraded=sentiment_result.degraded,
                    market_degraded=market_result.degraded,
                    proposer_degraded=proposer_result.degraded,
                )
            else:
                self._proposal_repo.save_proposal(
                    proposal_id=aggregation.proposal.proposal_id,
                    run_id=run_id,
                    proposal_json=aggregation.proposal.model_dump_json(),
                    risk_check_result="rejected",
                    risk_check_reason=aggregation.rejection_reason,
                )
                self._pipeline_repo.update_run_status(run_id, "rejected")
                log.warning("pipeline_rejected", reason=aggregation.rejection_reason)

                return PipelineResult(
                    run_id=run_id,
                    symbol=symbol,
                    status="rejected",
                    proposal=aggregation.proposal,
                    rejection_reason=aggregation.rejection_reason,
                    sentiment_degraded=sentiment_result.degraded,
                    market_degraded=market_result.degraded,
                    proposer_degraded=proposer_result.degraded,
                )

        except Exception as e:
            log.error("pipeline_failed", error=str(e))
            self._pipeline_repo.update_run_status(run_id, "failed")
            return PipelineResult(
                run_id=run_id,
                symbol=symbol,
                status="failed",
                rejection_reason=str(e),
            )

    def _save_llm_calls(self, run_id: str, agent_type: str, result: AgentResult) -> None:
        for call in result.llm_calls:
            self._llm_call_repo.save_call(
                run_id=run_id,
                agent_type=agent_type,
                prompt="(see messages)",
                response=call.content,
                model=call.model,
                latency_ms=call.latency_ms,
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
            )
```

**Step 4: Run test to verify it passes**

```bash
cd orchestrator && uv run pytest tests/unit/test_runner.py -v
```

Expected: 2 passed

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/pipeline/runner.py orchestrator/tests/unit/test_runner.py
git commit -m "feat: add pipeline runner orchestrating 3-agent flow"
```

---

### Task 11: Pipeline Scheduler

**Files:**
- Create: `orchestrator/src/orchestrator/pipeline/scheduler.py`
- Test: `orchestrator/tests/unit/test_scheduler.py`

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_scheduler.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.pipeline.scheduler import PipelineScheduler


class TestPipelineScheduler:
    def test_create_scheduler(self):
        mock_runner = AsyncMock()
        scheduler = PipelineScheduler(
            runner=mock_runner,
            symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
            interval_minutes=15,
        )
        assert scheduler.symbols == ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        assert scheduler.interval_minutes == 15

    @pytest.mark.asyncio
    async def test_run_once_all_symbols(self):
        mock_runner = AsyncMock()
        mock_runner.execute.return_value = MagicMock(status="completed")

        scheduler = PipelineScheduler(
            runner=mock_runner,
            symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
            interval_minutes=15,
        )

        results = await scheduler.run_once()

        assert len(results) == 2
        assert mock_runner.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_run_once_single_symbol(self):
        mock_runner = AsyncMock()
        mock_runner.execute.return_value = MagicMock(status="completed")

        scheduler = PipelineScheduler(
            runner=mock_runner,
            symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
            interval_minutes=15,
        )

        results = await scheduler.run_once(symbols=["BTC/USDT:USDT"])

        assert len(results) == 1
        mock_runner.execute.assert_called_once_with("BTC/USDT:USDT")
```

**Step 2: Run test to verify it fails**

```bash
cd orchestrator && uv run pytest tests/unit/test_scheduler.py -v
```

Expected: FAIL

**Step 3: Write minimal implementation**

```python
# orchestrator/src/orchestrator/pipeline/scheduler.py
from __future__ import annotations

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from orchestrator.pipeline.runner import PipelineResult, PipelineRunner

logger = structlog.get_logger(__name__)


class PipelineScheduler:
    def __init__(
        self,
        *,
        runner: PipelineRunner,
        symbols: list[str],
        interval_minutes: int = 15,
    ) -> None:
        self.symbols = symbols
        self.interval_minutes = interval_minutes
        self._runner = runner
        self._scheduler: AsyncIOScheduler | None = None

    async def run_once(
        self, *, symbols: list[str] | None = None
    ) -> list[PipelineResult]:
        target_symbols = symbols or self.symbols
        results = []
        for symbol in target_symbols:
            logger.info("scheduler_running_symbol", symbol=symbol)
            result = await self._runner.execute(symbol)
            results.append(result)
            logger.info(
                "scheduler_symbol_done",
                symbol=symbol,
                status=result.status,
            )
        return results

    def start(self) -> None:
        self._scheduler = AsyncIOScheduler()
        self._scheduler.add_job(
            self.run_once,
            trigger=IntervalTrigger(minutes=self.interval_minutes),
            id="pipeline_scheduler",
            name="Pipeline Scheduler",
            replace_existing=True,
        )
        self._scheduler.start()
        logger.info(
            "scheduler_started",
            interval_minutes=self.interval_minutes,
            symbols=self.symbols,
        )

    def stop(self) -> None:
        if self._scheduler:
            self._scheduler.shutdown(wait=False)
            logger.info("scheduler_stopped")
```

**Step 4: Run test to verify it passes**

```bash
cd orchestrator && uv run pytest tests/unit/test_scheduler.py -v
```

Expected: 3 passed

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/pipeline/scheduler.py orchestrator/tests/unit/test_scheduler.py
git commit -m "feat: add pipeline scheduler with APScheduler"
```

---

### Task 12: Telegram Enhancements (formatters + /run + wire /status /coin)

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/formatters.py`
- Modify: `orchestrator/src/orchestrator/telegram/bot.py`
- Modify: `orchestrator/tests/unit/test_telegram.py`

**Step 1: Write the failing test**

Add to `orchestrator/tests/unit/test_telegram.py`:

```python
# Add import at top:
from orchestrator.telegram.formatters import format_help, format_proposal, format_status, format_welcome
from orchestrator.models import EntryOrder, Side, TradeProposal
from orchestrator.pipeline.runner import PipelineResult

# Add test classes:

class TestFormatProposal:
    def test_format_long_proposal(self):
        result = PipelineResult(
            run_id="test-run",
            symbol="BTC/USDT:USDT",
            status="completed",
            proposal=TradeProposal(
                symbol="BTC/USDT:USDT",
                side=Side.LONG,
                entry=EntryOrder(type="market"),
                position_size_risk_pct=1.5,
                stop_loss=93000.0,
                take_profit=[97000.0],
                time_horizon="4h",
                confidence=0.75,
                invalid_if=[],
                rationale="Bullish momentum with strong support",
            ),
        )
        msg = format_proposal(result)
        assert "LONG" in msg
        assert "BTC/USDT:USDT" in msg
        assert "93000" in msg or "93,000" in msg
        assert "97000" in msg or "97,000" in msg
        assert "Bullish" in msg

    def test_format_flat_proposal(self):
        result = PipelineResult(
            run_id="test-run",
            symbol="BTC/USDT:USDT",
            status="completed",
            proposal=TradeProposal(
                symbol="BTC/USDT:USDT",
                side=Side.FLAT,
                entry=EntryOrder(type="market"),
                position_size_risk_pct=0.0,
                stop_loss=None,
                take_profit=[],
                time_horizon="4h",
                confidence=0.5,
                invalid_if=[],
                rationale="No clear signal",
            ),
        )
        msg = format_proposal(result)
        assert "FLAT" in msg

    def test_format_rejected_proposal(self):
        result = PipelineResult(
            run_id="test-run",
            symbol="BTC/USDT:USDT",
            status="rejected",
            proposal=TradeProposal(
                symbol="BTC/USDT:USDT",
                side=Side.LONG,
                entry=EntryOrder(type="market"),
                position_size_risk_pct=1.5,
                stop_loss=97000.0,
                take_profit=[99000.0],
                time_horizon="4h",
                confidence=0.75,
                invalid_if=[],
                rationale="Bad",
            ),
            rejection_reason="SL on wrong side",
        )
        msg = format_proposal(result)
        assert "REJECTED" in msg or "rejected" in msg


class TestFormatStatus:
    def test_format_status_with_results(self):
        results = [
            PipelineResult(
                run_id="r1",
                symbol="BTC/USDT:USDT",
                status="completed",
                proposal=TradeProposal(
                    symbol="BTC/USDT:USDT",
                    side=Side.LONG,
                    entry=EntryOrder(type="market"),
                    position_size_risk_pct=1.0,
                    stop_loss=93000.0,
                    take_profit=[97000.0],
                    time_horizon="4h",
                    confidence=0.7,
                    invalid_if=[],
                    rationale="test",
                ),
            ),
        ]
        msg = format_status(results)
        assert "BTC" in msg

    def test_format_status_empty(self):
        msg = format_status([])
        assert "No" in msg or "no" in msg
```

**Step 2: Run test to verify it fails**

```bash
cd orchestrator && uv run pytest tests/unit/test_telegram.py -v
```

Expected: FAIL with `ImportError` on `format_proposal`

**Step 3: Update formatters.py**

Replace `orchestrator/src/orchestrator/telegram/formatters.py` entirely:

```python
# orchestrator/src/orchestrator/telegram/formatters.py
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.pipeline.runner import PipelineResult


def format_welcome() -> str:
    return (
        "Welcome to Sentinel Orchestrator!\n\n"
        "I analyze crypto markets using multiple AI models and generate "
        "trade proposals with risk management.\n\n"
        "Use /help to see available commands."
    )


def format_help() -> str:
    return (
        "Available commands:\n\n"
        "/start - Welcome message\n"
        "/status - Account overview & latest proposals\n"
        "/coin <symbol> - Detailed analysis for a symbol (e.g. /coin BTC)\n"
        "/run - Trigger pipeline for all symbols\n"
        "/run <symbol> - Trigger pipeline for specific symbol\n"
        "/history - Recent trade records\n"
        "/help - Show this message"
    )


def format_proposal(result: PipelineResult) -> str:
    if result.proposal is None:
        return f"Pipeline {result.status}: {result.rejection_reason or 'No proposal generated'}"

    p = result.proposal
    status_emoji = {"completed": "NEW", "rejected": "REJECTED", "failed": "FAILED"}.get(
        result.status, result.status.upper()
    )

    lines = [
        f"[{status_emoji}] {p.symbol}",
        f"Side: {p.side.value.upper()}",
    ]

    if p.side.value != "flat":
        lines.append(f"Entry: {p.entry.type}")
        lines.append(f"Risk: {p.position_size_risk_pct}%")
        if p.stop_loss is not None:
            lines.append(f"SL: {p.stop_loss:,.1f}")
        if p.take_profit:
            tp_str = ", ".join(f"{tp:,.1f}" for tp in p.take_profit)
            lines.append(f"TP: {tp_str}")

    lines.append(f"Horizon: {p.time_horizon}")
    lines.append(f"Confidence: {p.confidence:.0%}")
    lines.append(f"Rationale: {p.rationale}")

    if result.status == "rejected":
        lines.append(f"\nREJECTED: {result.rejection_reason}")

    degraded_agents = []
    if result.sentiment_degraded:
        degraded_agents.append("sentiment")
    if result.market_degraded:
        degraded_agents.append("market")
    if result.proposer_degraded:
        degraded_agents.append("proposer")
    if degraded_agents:
        lines.append(f"\nDegraded: {', '.join(degraded_agents)}")

    return "\n".join(lines)


def format_status(results: list[PipelineResult]) -> str:
    if not results:
        return "No pipeline results yet. Use /run to trigger analysis."

    lines = ["Latest pipeline results:\n"]
    for r in results:
        side = r.proposal.side.value.upper() if r.proposal else "N/A"
        conf = f"{r.proposal.confidence:.0%}" if r.proposal else "N/A"
        lines.append(f"  {r.symbol}: {side} (confidence: {conf}) [{r.status}]")

    return "\n".join(lines)
```

**Step 4: Update bot.py — add /run and wire /status /coin**

Modify `orchestrator/src/orchestrator/telegram/bot.py`. The full updated file:

```python
# orchestrator/src/orchestrator/telegram/bot.py
from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from orchestrator.telegram.formatters import format_help, format_proposal, format_status, format_welcome

if TYPE_CHECKING:
    from orchestrator.pipeline.scheduler import PipelineScheduler

logger = structlog.get_logger(__name__)


def is_admin(chat_id: int, *, admin_ids: list[int]) -> bool:
    return chat_id in admin_ids


class SentinelBot:
    def __init__(self, token: str, admin_chat_ids: list[int]) -> None:
        self.token = token
        self.admin_chat_ids = admin_chat_ids
        self._app: Application | None = None
        self._scheduler: PipelineScheduler | None = None
        self._latest_results: dict[str, object] = {}  # symbol → PipelineResult

    def set_scheduler(self, scheduler: PipelineScheduler) -> None:
        self._scheduler = scheduler

    def build(self) -> Application:
        self._app = Application.builder().token(self.token).build()
        self._app.add_handler(CommandHandler("start", self._start_handler))
        self._app.add_handler(CommandHandler("help", self._help_handler))
        self._app.add_handler(CommandHandler("status", self._status_handler))
        self._app.add_handler(CommandHandler("coin", self._coin_handler))
        self._app.add_handler(CommandHandler("run", self._run_handler))
        return self._app

    async def push_proposal(self, chat_id: int, result) -> None:
        """Push a pipeline result to a specific chat."""
        if self._app is None:
            return
        msg = format_proposal(result)
        await self._app.bot.send_message(chat_id=chat_id, text=msg)

    async def push_to_admins(self, result) -> None:
        """Push a pipeline result to all admin chats."""
        for chat_id in self.admin_chat_ids:
            await self.push_proposal(chat_id, result)

    async def _check_admin(self, update: Update) -> bool:
        chat_id = update.effective_chat.id if update.effective_chat else 0
        if not is_admin(chat_id, admin_ids=self.admin_chat_ids):
            logger.warning("unauthorized_access", chat_id=chat_id)
            return False
        return True

    async def _start_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        await update.message.reply_text(format_welcome())

    async def _help_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        await update.message.reply_text(format_help())

    async def _status_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        results = list(self._latest_results.values())
        await update.message.reply_text(format_status(results))

    async def _coin_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        args = context.args
        if not args:
            await update.message.reply_text("Usage: /coin <symbol> (e.g. /coin BTC)")
            return

        query = args[0].upper()
        # Find matching symbol
        matching = [
            r for sym, r in self._latest_results.items() if query in sym.upper()
        ]

        if not matching:
            await update.message.reply_text(
                f"No recent analysis for {query}. Use /run to trigger analysis."
            )
            return

        for result in matching:
            await update.message.reply_text(format_proposal(result))

    async def _run_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_admin(update):
            return
        if self._scheduler is None:
            await update.message.reply_text("Pipeline not configured.")
            return

        args = context.args
        await update.message.reply_text("Running pipeline...")

        if args:
            # Map short name to full symbol
            query = args[0].upper()
            symbols = [
                s for s in self._scheduler.symbols if query in s.upper()
            ]
            if not symbols:
                await update.message.reply_text(f"Unknown symbol: {query}")
                return
            results = await self._scheduler.run_once(symbols=symbols)
        else:
            results = await self._scheduler.run_once()

        for result in results:
            self._latest_results[result.symbol] = result
            await update.message.reply_text(format_proposal(result))
```

**Step 5: Run test to verify it passes**

```bash
cd orchestrator && uv run pytest tests/unit/test_telegram.py -v
```

Expected: All tests pass (6 old + 5 new = 11)

**Step 6: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/ orchestrator/tests/unit/test_telegram.py
git commit -m "feat: enhance Telegram bot with /run, proposal formatting, and wired /status /coin"
```

---

### Task 13: Wire Entrypoint (__main__.py)

**Files:**
- Modify: `orchestrator/src/orchestrator/__main__.py`
- Modify: `orchestrator/tests/unit/test_main.py`

**Step 1: Write the failing test**

Replace `orchestrator/tests/unit/test_main.py`:

```python
# orchestrator/tests/unit/test_main.py
from orchestrator.__main__ import create_app_components


def test_create_app_components():
    """Verify we can construct core components without starting services."""
    components = create_app_components(
        telegram_bot_token="test-token",
        telegram_admin_chat_ids=[123],
        exchange_id="binance",
        database_url="sqlite:///:memory:",
        anthropic_api_key="test-key",
        llm_model="anthropic/claude-sonnet-4-6",
        llm_temperature=0.2,
        llm_max_tokens=2000,
        llm_max_retries=1,
        pipeline_symbols=["BTC/USDT:USDT"],
        pipeline_interval_minutes=15,
    )
    assert "bot" in components
    assert "exchange_client" in components
    assert "db_engine" in components
    assert "scheduler" in components
    assert "runner" in components
```

**Step 2: Run test to verify it fails**

```bash
cd orchestrator && uv run pytest tests/unit/test_main.py -v
```

Expected: FAIL (missing `scheduler` and `runner` keys)

**Step 3: Update __main__.py**

Replace `orchestrator/src/orchestrator/__main__.py`:

```python
# orchestrator/src/orchestrator/__main__.py
from __future__ import annotations

import structlog
from sqlmodel import Session

from orchestrator.agents.market import MarketAgent
from orchestrator.agents.proposer import ProposerAgent
from orchestrator.agents.sentiment import SentimentAgent
from orchestrator.config import Settings
from orchestrator.exchange.client import ExchangeClient
from orchestrator.exchange.data_fetcher import DataFetcher
from orchestrator.llm.client import LLMClient
from orchestrator.logging import setup_logging
from orchestrator.pipeline.runner import PipelineRunner
from orchestrator.pipeline.scheduler import PipelineScheduler
from orchestrator.storage.database import create_db_engine, init_db
from orchestrator.storage.repository import (
    LLMCallRepository,
    PipelineRepository,
    TradeProposalRepository,
)
from orchestrator.telegram.bot import SentinelBot

logger = structlog.get_logger(__name__)


def create_app_components(
    *,
    telegram_bot_token: str,
    telegram_admin_chat_ids: list[int],
    exchange_id: str,
    database_url: str,
    anthropic_api_key: str,
    llm_model: str = "anthropic/claude-sonnet-4-6",
    llm_temperature: float = 0.2,
    llm_max_tokens: int = 2000,
    llm_max_retries: int = 1,
    pipeline_symbols: list[str] | None = None,
    pipeline_interval_minutes: int = 15,
) -> dict:
    # Database
    db_engine = create_db_engine(database_url)
    init_db(db_engine)
    session = Session(db_engine)

    # LLM
    llm_client = LLMClient(
        model=llm_model,
        api_key=anthropic_api_key,
        temperature=llm_temperature,
        max_tokens=llm_max_tokens,
    )

    # Agents
    sentiment_agent = SentimentAgent(client=llm_client, max_retries=llm_max_retries)
    market_agent = MarketAgent(client=llm_client, max_retries=llm_max_retries)
    proposer_agent = ProposerAgent(client=llm_client, max_retries=llm_max_retries)

    # Exchange
    exchange_client = ExchangeClient(exchange_id=exchange_id)
    data_fetcher = DataFetcher(exchange_client)

    # Repositories
    pipeline_repo = PipelineRepository(session)
    llm_call_repo = LLMCallRepository(session)
    proposal_repo = TradeProposalRepository(session)

    # Pipeline
    runner = PipelineRunner(
        data_fetcher=data_fetcher,
        sentiment_agent=sentiment_agent,
        market_agent=market_agent,
        proposer_agent=proposer_agent,
        pipeline_repo=pipeline_repo,
        llm_call_repo=llm_call_repo,
        proposal_repo=proposal_repo,
    )

    symbols = pipeline_symbols or ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    scheduler = PipelineScheduler(
        runner=runner,
        symbols=symbols,
        interval_minutes=pipeline_interval_minutes,
    )

    # Telegram
    bot = SentinelBot(token=telegram_bot_token, admin_chat_ids=telegram_admin_chat_ids)
    bot.set_scheduler(scheduler)

    return {
        "bot": bot,
        "exchange_client": exchange_client,
        "db_engine": db_engine,
        "scheduler": scheduler,
        "runner": runner,
    }


def main() -> None:
    setup_logging(json_output=True)

    settings = Settings()  # type: ignore[call-arg]
    logger.info("starting_sentinel", exchange=settings.exchange_id)

    components = create_app_components(
        telegram_bot_token=settings.telegram_bot_token,
        telegram_admin_chat_ids=settings.telegram_admin_chat_ids,
        exchange_id=settings.exchange_id,
        database_url=settings.database_url,
        anthropic_api_key=settings.anthropic_api_key,
        llm_model=settings.llm_model,
        llm_temperature=settings.llm_temperature,
        llm_max_tokens=settings.llm_max_tokens,
        llm_max_retries=settings.llm_max_retries,
        pipeline_symbols=settings.pipeline_symbols,
        pipeline_interval_minutes=settings.pipeline_interval_minutes,
    )

    # Start scheduler
    components["scheduler"].start()

    # Start bot (blocking)
    app = components["bot"].build()
    logger.info("bot_ready", admin_ids=settings.telegram_admin_chat_ids)
    app.run_polling()


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

```bash
cd orchestrator && uv run pytest tests/unit/test_main.py -v
```

Expected: 1 passed

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/__main__.py orchestrator/tests/unit/test_main.py
git commit -m "feat: wire M1 pipeline components into application entrypoint"
```

---

### Task 14: Final Verification

**Step 1: Run full test suite with coverage**

```bash
cd orchestrator && uv run pytest -v --cov=orchestrator --cov-report=term-missing
```

Expected: All pass, 80%+ coverage

**Step 2: Run linter**

```bash
cd orchestrator && uv run ruff check src/ tests/
```

Expected: No errors (or auto-fix with `--fix`)

**Step 3: Verify all imports work**

```bash
cd orchestrator && uv run python -c "
from orchestrator.llm.client import LLMClient
from orchestrator.llm.schema_validator import validate_llm_output
from orchestrator.agents.sentiment import SentimentAgent
from orchestrator.agents.market import MarketAgent
from orchestrator.agents.proposer import ProposerAgent
from orchestrator.pipeline.runner import PipelineRunner
from orchestrator.pipeline.scheduler import PipelineScheduler
from orchestrator.pipeline.aggregator import aggregate_proposal
print('All M1 imports OK')
"
```

Expected: `All M1 imports OK`

**Step 4: Commit any lint fixes**

```bash
cd orchestrator && uv run ruff check --fix src/ tests/
git add -A && git commit -m "chore: lint fixes for M1" --allow-empty
```
