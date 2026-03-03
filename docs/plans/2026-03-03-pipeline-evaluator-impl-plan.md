# Pipeline Evaluator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a `PipelineEvaluator` that compares pipeline proposals against actual trade results, with Telegram `/evaluate` command and close-report feedback.

**Architecture:** Pure computation module in `stats/evaluator.py`, Pydantic models for report data, formatter functions for Telegram display. Evaluator reads from existing repos — no new DB tables.

**Tech Stack:** Python, Pydantic (frozen models), SQLModel repositories, python-telegram-bot

---

### Task 1: Add `get_by_proposal_id` to `TradeProposalRepository`

**Files:**
- Modify: `orchestrator/src/orchestrator/storage/repository.py:81-116`
- Test: `orchestrator/tests/unit/test_repository_proposal.py`

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_repository_proposal.py
import json

import pytest
from sqlmodel import Session, SQLModel, create_engine

from orchestrator.storage.models import TradeProposalRecord
from orchestrator.storage.repository import TradeProposalRepository


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


class TestTradeProposalRepository:
    def test_get_by_proposal_id_found(self, session: Session):
        repo = TradeProposalRepository(session)
        repo.save_proposal(
            proposal_id="p-123",
            run_id="run-1",
            proposal_json=json.dumps({"symbol": "BTC/USDT:USDT", "side": "long"}),
        )
        result = repo.get_by_proposal_id("p-123")
        assert result is not None
        assert result.proposal_id == "p-123"

    def test_get_by_proposal_id_not_found(self, session: Session):
        repo = TradeProposalRepository(session)
        result = repo.get_by_proposal_id("nonexistent")
        assert result is None
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_repository_proposal.py -v`
Expected: FAIL with `AttributeError: 'TradeProposalRepository' object has no attribute 'get_by_proposal_id'`

**Step 3: Write minimal implementation**

Add to `TradeProposalRepository` in `repository.py` after `get_recent()`:

```python
def get_by_proposal_id(self, proposal_id: str) -> TradeProposalRecord | None:
    statement = select(TradeProposalRecord).where(
        TradeProposalRecord.proposal_id == proposal_id
    )
    return self._session.exec(statement).first()
```

**Step 4: Run test to verify it passes**

Run: `cd orchestrator && uv run pytest tests/unit/test_repository_proposal.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/storage/repository.py orchestrator/tests/unit/test_repository_proposal.py
git commit -m "feat: add get_by_proposal_id to TradeProposalRepository"
```

---

### Task 2: Create evaluation data models

**Files:**
- Create: `orchestrator/src/orchestrator/stats/evaluator.py`
- Test: `orchestrator/tests/unit/test_evaluator.py`

**Step 1: Write the failing test**

```python
# orchestrator/tests/unit/test_evaluator.py
import pytest

from orchestrator.stats.evaluator import (
    ConfidenceBucket,
    EvaluationReport,
    PeriodStats,
    SymbolStats,
    TradeEvaluation,
)


class TestEvaluationModels:
    def test_trade_evaluation_frozen(self):
        ev = TradeEvaluation(
            trade_id="t-1",
            proposal_id="p-1",
            symbol="BTC/USDT:USDT",
            direction_correct=True,
            entry_deviation_pct=0.3,
            close_reason="tp",
            confidence=75,
            pnl=100.0,
        )
        assert ev.direction_correct is True
        with pytest.raises(Exception):
            ev.pnl = 999.0

    def test_evaluation_report_frozen(self):
        report = EvaluationReport(
            total_evaluated=1,
            total_unmatched=0,
            direction_accuracy=1.0,
            avg_entry_deviation_pct=0.3,
            sl_hit_rate=0.0,
            tp_hit_rate=1.0,
            liquidation_rate=0.0,
            by_symbol=[],
            by_confidence=[],
            by_period=[],
            evaluations=[],
        )
        assert report.total_evaluated == 1
        with pytest.raises(Exception):
            report.total_evaluated = 99
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_evaluator.py::TestEvaluationModels -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.stats.evaluator'`

**Step 3: Write minimal implementation**

```python
# orchestrator/src/orchestrator/stats/evaluator.py
from __future__ import annotations

from pydantic import BaseModel


class TradeEvaluation(BaseModel, frozen=True):
    """Single trade vs proposal comparison."""
    trade_id: str
    proposal_id: str
    symbol: str
    direction_correct: bool
    entry_deviation_pct: float | None
    close_reason: str
    confidence: int
    pnl: float


class SymbolStats(BaseModel, frozen=True):
    symbol: str
    total_trades: int
    direction_accuracy: float
    avg_entry_deviation_pct: float
    sl_hit_rate: float
    tp_hit_rate: float
    total_pnl: float


class ConfidenceBucket(BaseModel, frozen=True):
    bucket: str
    total_trades: int
    direction_accuracy: float
    avg_pnl: float


class PeriodStats(BaseModel, frozen=True):
    period: str
    total_trades: int
    direction_accuracy: float
    total_pnl: float


class EvaluationReport(BaseModel, frozen=True):
    total_evaluated: int
    total_unmatched: int
    direction_accuracy: float
    avg_entry_deviation_pct: float
    sl_hit_rate: float
    tp_hit_rate: float
    liquidation_rate: float
    by_symbol: list[SymbolStats]
    by_confidence: list[ConfidenceBucket]
    by_period: list[PeriodStats]
    evaluations: list[TradeEvaluation]
```

**Step 4: Run test to verify it passes**

Run: `cd orchestrator && uv run pytest tests/unit/test_evaluator.py::TestEvaluationModels -v`
Expected: PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/stats/evaluator.py orchestrator/tests/unit/test_evaluator.py
git commit -m "feat: add pipeline evaluation data models"
```

---

### Task 3: Implement `PipelineEvaluator.evaluate_single()`

**Files:**
- Modify: `orchestrator/src/orchestrator/stats/evaluator.py`
- Modify: `orchestrator/tests/unit/test_evaluator.py`

**Step 1: Write the failing tests**

Append to `test_evaluator.py`:

```python
import json
from unittest.mock import MagicMock

from orchestrator.stats.evaluator import PipelineEvaluator, TradeEvaluation


def _make_trade_record(
    *,
    trade_id: str = "t-1",
    proposal_id: str = "p-1",
    symbol: str = "BTC/USDT:USDT",
    side: str = "long",
    entry_price: float = 69000.0,
    pnl: float = 100.0,
    close_reason: str = "tp",
) -> MagicMock:
    rec = MagicMock()
    rec.trade_id = trade_id
    rec.proposal_id = proposal_id
    rec.symbol = symbol
    rec.side = side
    rec.entry_price = entry_price
    rec.pnl = pnl
    rec.close_reason = close_reason
    return rec


def _make_proposal_record(
    *,
    proposal_id: str = "p-1",
    side: str = "long",
    entry_price: float | None = 68800.0,
    confidence: float = 0.75,
) -> MagicMock:
    proposal = {
        "side": side,
        "entry": {"type": "limit", "price": entry_price},
        "confidence": confidence,
    }
    rec = MagicMock()
    rec.proposal_id = proposal_id
    rec.proposal_json = json.dumps(proposal)
    return rec


class TestEvaluateSingle:
    def test_direction_correct_long_profit(self):
        trade_repo = MagicMock()
        proposal_repo = MagicMock()
        trade_repo.get_by_trade_id.return_value = _make_trade_record(pnl=100.0, side="long")
        proposal_repo.get_by_proposal_id.return_value = _make_proposal_record(side="long")

        evaluator = PipelineEvaluator(trade_repo=trade_repo, proposal_repo=proposal_repo)
        result = evaluator.evaluate_single("t-1")

        assert result is not None
        assert result.direction_correct is True
        assert result.confidence == 75

    def test_direction_wrong_long_loss(self):
        trade_repo = MagicMock()
        proposal_repo = MagicMock()
        trade_repo.get_by_trade_id.return_value = _make_trade_record(pnl=-50.0, side="long")
        proposal_repo.get_by_proposal_id.return_value = _make_proposal_record(side="long")

        evaluator = PipelineEvaluator(trade_repo=trade_repo, proposal_repo=proposal_repo)
        result = evaluator.evaluate_single("t-1")

        assert result is not None
        assert result.direction_correct is False

    def test_entry_deviation_calculated(self):
        trade_repo = MagicMock()
        proposal_repo = MagicMock()
        trade_repo.get_by_trade_id.return_value = _make_trade_record(entry_price=69000.0)
        proposal_repo.get_by_proposal_id.return_value = _make_proposal_record(entry_price=68800.0)

        evaluator = PipelineEvaluator(trade_repo=trade_repo, proposal_repo=proposal_repo)
        result = evaluator.evaluate_single("t-1")

        assert result is not None
        # (69000 - 68800) / 68800 * 100 ≈ 0.291%
        assert result.entry_deviation_pct == pytest.approx(0.291, rel=0.01)

    def test_market_order_no_entry_deviation(self):
        trade_repo = MagicMock()
        proposal_repo = MagicMock()
        trade_repo.get_by_trade_id.return_value = _make_trade_record()
        proposal_repo.get_by_proposal_id.return_value = _make_proposal_record(entry_price=None)

        evaluator = PipelineEvaluator(trade_repo=trade_repo, proposal_repo=proposal_repo)
        result = evaluator.evaluate_single("t-1")

        assert result is not None
        assert result.entry_deviation_pct is None

    def test_no_proposal_returns_none(self):
        trade_repo = MagicMock()
        proposal_repo = MagicMock()
        trade_repo.get_by_trade_id.return_value = _make_trade_record()
        proposal_repo.get_by_proposal_id.return_value = None

        evaluator = PipelineEvaluator(trade_repo=trade_repo, proposal_repo=proposal_repo)
        result = evaluator.evaluate_single("t-1")

        assert result is None

    def test_no_trade_returns_none(self):
        trade_repo = MagicMock()
        proposal_repo = MagicMock()
        trade_repo.get_by_trade_id.return_value = None

        evaluator = PipelineEvaluator(trade_repo=trade_repo, proposal_repo=proposal_repo)
        result = evaluator.evaluate_single("t-1")

        assert result is None
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_evaluator.py::TestEvaluateSingle -v`
Expected: FAIL with `ImportError: cannot import name 'PipelineEvaluator'`

**Step 3: Write minimal implementation**

Add to `evaluator.py`:

```python
import json

from orchestrator.storage.repository import PaperTradeRepository, TradeProposalRepository


class PipelineEvaluator:
    def __init__(
        self,
        *,
        trade_repo: PaperTradeRepository,
        proposal_repo: TradeProposalRepository,
    ) -> None:
        self._trade_repo = trade_repo
        self._proposal_repo = proposal_repo

    def evaluate_single(self, trade_id: str) -> TradeEvaluation | None:
        """Evaluate a single closed trade against its proposal."""
        trade = self._trade_repo.get_by_trade_id(trade_id)
        if trade is None:
            return None

        proposal_rec = self._proposal_repo.get_by_proposal_id(trade.proposal_id)
        if proposal_rec is None:
            return None

        proposal = json.loads(proposal_rec.proposal_json)
        proposed_side = proposal.get("side", "")
        proposed_entry_price = proposal.get("entry", {}).get("price")
        raw_confidence = proposal.get("confidence", 0)
        # confidence may be 0-1 float or 1-100 int
        confidence = int(raw_confidence * 100) if raw_confidence <= 1.0 else int(raw_confidence)

        # Direction correct: proposal side matches profitable outcome
        direction_correct = trade.pnl > 0

        # Entry deviation
        entry_deviation_pct: float | None = None
        if proposed_entry_price is not None and proposed_entry_price > 0:
            entry_deviation_pct = (
                (trade.entry_price - proposed_entry_price) / proposed_entry_price * 100
            )

        return TradeEvaluation(
            trade_id=trade.trade_id,
            proposal_id=trade.proposal_id,
            symbol=trade.symbol,
            direction_correct=direction_correct,
            entry_deviation_pct=entry_deviation_pct,
            close_reason=trade.close_reason or "",
            confidence=confidence,
            pnl=trade.pnl,
        )
```

**Step 4: Run test to verify it passes**

Run: `cd orchestrator && uv run pytest tests/unit/test_evaluator.py::TestEvaluateSingle -v`
Expected: PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/stats/evaluator.py orchestrator/tests/unit/test_evaluator.py
git commit -m "feat: implement PipelineEvaluator.evaluate_single()"
```

---

### Task 4: Implement `PipelineEvaluator.evaluate()` with aggregation

**Files:**
- Modify: `orchestrator/src/orchestrator/stats/evaluator.py`
- Modify: `orchestrator/tests/unit/test_evaluator.py`

**Step 1: Write the failing tests**

Append to `test_evaluator.py`:

```python
from orchestrator.stats.evaluator import EvaluationReport


class TestEvaluate:
    def _setup_repos(self, trades, proposals):
        """Setup mock repos with given trades and proposals."""
        trade_repo = MagicMock()
        proposal_repo = MagicMock()
        trade_repo.get_all_closed.return_value = trades
        trade_repo.get_by_trade_id.side_effect = lambda tid: next(
            (t for t in trades if t.trade_id == tid), None
        )
        proposal_repo.get_by_proposal_id.side_effect = lambda pid: next(
            (p for p in proposals if p.proposal_id == pid), None
        )
        return trade_repo, proposal_repo

    def test_evaluate_overall_accuracy(self):
        trades = [
            _make_trade_record(trade_id="t-1", proposal_id="p-1", pnl=100.0, close_reason="tp"),
            _make_trade_record(trade_id="t-2", proposal_id="p-2", pnl=-50.0, close_reason="sl"),
        ]
        proposals = [
            _make_proposal_record(proposal_id="p-1"),
            _make_proposal_record(proposal_id="p-2"),
        ]
        trade_repo, proposal_repo = self._setup_repos(trades, proposals)
        evaluator = PipelineEvaluator(trade_repo=trade_repo, proposal_repo=proposal_repo)

        report = evaluator.evaluate()

        assert report.total_evaluated == 2
        assert report.direction_accuracy == 0.5
        assert report.tp_hit_rate == 0.5
        assert report.sl_hit_rate == 0.5

    def test_evaluate_by_symbol(self):
        trades = [
            _make_trade_record(trade_id="t-1", proposal_id="p-1", symbol="BTC/USDT:USDT", pnl=100.0),
            _make_trade_record(trade_id="t-2", proposal_id="p-2", symbol="ETH/USDT:USDT", pnl=-50.0),
        ]
        proposals = [
            _make_proposal_record(proposal_id="p-1"),
            _make_proposal_record(proposal_id="p-2"),
        ]
        trade_repo, proposal_repo = self._setup_repos(trades, proposals)
        evaluator = PipelineEvaluator(trade_repo=trade_repo, proposal_repo=proposal_repo)

        report = evaluator.evaluate()

        assert len(report.by_symbol) == 2
        btc = next(s for s in report.by_symbol if s.symbol == "BTC/USDT:USDT")
        assert btc.direction_accuracy == 1.0

    def test_evaluate_by_confidence(self):
        trades = [
            _make_trade_record(trade_id="t-1", proposal_id="p-1", pnl=100.0),
            _make_trade_record(trade_id="t-2", proposal_id="p-2", pnl=-50.0),
        ]
        proposals = [
            _make_proposal_record(proposal_id="p-1", confidence=0.80),
            _make_proposal_record(proposal_id="p-2", confidence=0.30),
        ]
        trade_repo, proposal_repo = self._setup_repos(trades, proposals)
        evaluator = PipelineEvaluator(trade_repo=trade_repo, proposal_repo=proposal_repo)

        report = evaluator.evaluate()

        assert len(report.by_confidence) >= 2
        high = next(b for b in report.by_confidence if "high" in b.bucket)
        low = next(b for b in report.by_confidence if "low" in b.bucket)
        assert high.direction_accuracy == 1.0
        assert low.direction_accuracy == 0.0

    def test_evaluate_unmatched_trades(self):
        trades = [
            _make_trade_record(trade_id="t-1", proposal_id="p-1", pnl=100.0),
            _make_trade_record(trade_id="t-2", proposal_id="p-missing", pnl=-50.0),
        ]
        proposals = [
            _make_proposal_record(proposal_id="p-1"),
        ]
        trade_repo, proposal_repo = self._setup_repos(trades, proposals)
        evaluator = PipelineEvaluator(trade_repo=trade_repo, proposal_repo=proposal_repo)

        report = evaluator.evaluate()

        assert report.total_evaluated == 1
        assert report.total_unmatched == 1

    def test_evaluate_empty_trades(self):
        trade_repo, proposal_repo = self._setup_repos([], [])
        evaluator = PipelineEvaluator(trade_repo=trade_repo, proposal_repo=proposal_repo)

        report = evaluator.evaluate()

        assert report.total_evaluated == 0
        assert report.direction_accuracy == 0.0

    def test_evaluate_symbol_filter(self):
        trades = [
            _make_trade_record(trade_id="t-1", proposal_id="p-1", symbol="BTC/USDT:USDT", pnl=100.0),
            _make_trade_record(trade_id="t-2", proposal_id="p-2", symbol="ETH/USDT:USDT", pnl=-50.0),
        ]
        # Add symbol attr for filtering
        trades[0].symbol = "BTC/USDT:USDT"
        trades[1].symbol = "ETH/USDT:USDT"
        proposals = [
            _make_proposal_record(proposal_id="p-1"),
            _make_proposal_record(proposal_id="p-2"),
        ]
        trade_repo, proposal_repo = self._setup_repos(trades, proposals)
        evaluator = PipelineEvaluator(trade_repo=trade_repo, proposal_repo=proposal_repo)

        report = evaluator.evaluate(symbol="BTC/USDT:USDT")

        assert report.total_evaluated == 1
        assert report.direction_accuracy == 1.0
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_evaluator.py::TestEvaluate -v`
Expected: FAIL with `AttributeError: 'PipelineEvaluator' object has no attribute 'evaluate'`

**Step 3: Write minimal implementation**

Add `evaluate()` method to `PipelineEvaluator` in `evaluator.py`:

```python
from collections import defaultdict
from datetime import datetime


class PipelineEvaluator:
    # ... (existing __init__ and evaluate_single) ...

    def evaluate(
        self,
        *,
        symbol: str | None = None,
        since: datetime | None = None,
    ) -> EvaluationReport:
        """Evaluate all closed trades, optionally filtered."""
        closed_trades = self._trade_repo.get_all_closed()

        if symbol:
            closed_trades = [t for t in closed_trades if t.symbol == symbol]
        if since:
            closed_trades = [
                t for t in closed_trades
                if t.closed_at is not None and t.closed_at >= since
            ]

        evaluations: list[TradeEvaluation] = []
        unmatched = 0

        for trade in closed_trades:
            ev = self.evaluate_single(trade.trade_id)
            if ev is None:
                unmatched += 1
            else:
                evaluations.append(ev)

        return self._aggregate(evaluations, unmatched)

    def _aggregate(
        self, evaluations: list[TradeEvaluation], unmatched: int
    ) -> EvaluationReport:
        total = len(evaluations)
        if total == 0:
            return EvaluationReport(
                total_evaluated=0,
                total_unmatched=unmatched,
                direction_accuracy=0.0,
                avg_entry_deviation_pct=0.0,
                sl_hit_rate=0.0,
                tp_hit_rate=0.0,
                liquidation_rate=0.0,
                by_symbol=[],
                by_confidence=[],
                by_period=[],
                evaluations=[],
            )

        direction_correct = sum(1 for e in evaluations if e.direction_correct)
        deviations = [e.entry_deviation_pct for e in evaluations if e.entry_deviation_pct is not None]
        sl_count = sum(1 for e in evaluations if e.close_reason == "sl")
        tp_count = sum(1 for e in evaluations if e.close_reason == "tp")
        liq_count = sum(1 for e in evaluations if e.close_reason == "liquidation")

        return EvaluationReport(
            total_evaluated=total,
            total_unmatched=unmatched,
            direction_accuracy=direction_correct / total,
            avg_entry_deviation_pct=(
                sum(deviations) / len(deviations) if deviations else 0.0
            ),
            sl_hit_rate=sl_count / total,
            tp_hit_rate=tp_count / total,
            liquidation_rate=liq_count / total,
            by_symbol=self._group_by_symbol(evaluations),
            by_confidence=self._group_by_confidence(evaluations),
            by_period=self._group_by_period(evaluations),
            evaluations=evaluations,
        )

    def _group_by_symbol(self, evaluations: list[TradeEvaluation]) -> list[SymbolStats]:
        groups: dict[str, list[TradeEvaluation]] = defaultdict(list)
        for e in evaluations:
            groups[e.symbol].append(e)

        result: list[SymbolStats] = []
        for sym, evs in sorted(groups.items()):
            total = len(evs)
            deviations = [e.entry_deviation_pct for e in evs if e.entry_deviation_pct is not None]
            result.append(SymbolStats(
                symbol=sym,
                total_trades=total,
                direction_accuracy=sum(1 for e in evs if e.direction_correct) / total,
                avg_entry_deviation_pct=sum(deviations) / len(deviations) if deviations else 0.0,
                sl_hit_rate=sum(1 for e in evs if e.close_reason == "sl") / total,
                tp_hit_rate=sum(1 for e in evs if e.close_reason == "tp") / total,
                total_pnl=sum(e.pnl for e in evs),
            ))
        return result

    def _group_by_confidence(self, evaluations: list[TradeEvaluation]) -> list[ConfidenceBucket]:
        buckets: dict[str, list[TradeEvaluation]] = defaultdict(list)
        for e in evaluations:
            if e.confidence <= 33:
                buckets["low (1-33)"].append(e)
            elif e.confidence <= 66:
                buckets["mid (34-66)"].append(e)
            else:
                buckets["high (67-100)"].append(e)

        result: list[ConfidenceBucket] = []
        for bucket_name in ("low (1-33)", "mid (34-66)", "high (67-100)"):
            evs = buckets.get(bucket_name, [])
            if not evs:
                continue
            total = len(evs)
            result.append(ConfidenceBucket(
                bucket=bucket_name,
                total_trades=total,
                direction_accuracy=sum(1 for e in evs if e.direction_correct) / total,
                avg_pnl=sum(e.pnl for e in evs) / total,
            ))
        return result

    def _group_by_period(self, evaluations: list[TradeEvaluation]) -> list[PeriodStats]:
        """Group by ISO week (YYYY-Www)."""
        # Note: we use trade close time from the evaluation;
        # since evaluations don't carry close_at, we group by a simple approach.
        # For now, return empty — will be populated when we have closed_at in TradeEvaluation.
        return []
```

> Note: `by_period` is left empty for now because `TradeEvaluation` doesn't carry `closed_at`. We'll add it in a follow-up if needed, or enhance `evaluate_single()` to include the timestamp.

**Step 4: Run test to verify it passes**

Run: `cd orchestrator && uv run pytest tests/unit/test_evaluator.py::TestEvaluate -v`
Expected: PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/stats/evaluator.py orchestrator/tests/unit/test_evaluator.py
git commit -m "feat: implement PipelineEvaluator.evaluate() with aggregation"
```

---

### Task 5: Add `format_trade_evaluation()` and `format_evaluation_report()` formatters

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/formatters.py`
- Test: `orchestrator/tests/unit/test_formatters_evaluator.py`

**Step 1: Write the failing tests**

```python
# orchestrator/tests/unit/test_formatters_evaluator.py
from orchestrator.stats.evaluator import (
    ConfidenceBucket,
    EvaluationReport,
    SymbolStats,
    TradeEvaluation,
)
from orchestrator.telegram.formatters import (
    format_evaluation_report,
    format_trade_evaluation,
)


class TestFormatTradeEvaluation:
    def test_direction_correct(self):
        ev = TradeEvaluation(
            trade_id="t-1", proposal_id="p-1", symbol="BTC/USDT:USDT",
            direction_correct=True, entry_deviation_pct=0.3,
            close_reason="tp", confidence=75, pnl=100.0,
        )
        result = format_trade_evaluation(ev)
        assert "✅" in result or "Direction correct" in result.lower() or "correct" in result.lower()
        assert "0.3" in result
        assert "75" in result

    def test_direction_wrong(self):
        ev = TradeEvaluation(
            trade_id="t-1", proposal_id="p-1", symbol="BTC/USDT:USDT",
            direction_correct=False, entry_deviation_pct=-1.2,
            close_reason="sl", confidence=40, pnl=-50.0,
        )
        result = format_trade_evaluation(ev)
        assert "❌" in result or "Direction wrong" in result.lower() or "wrong" in result.lower()

    def test_no_entry_deviation(self):
        ev = TradeEvaluation(
            trade_id="t-1", proposal_id="p-1", symbol="BTC/USDT:USDT",
            direction_correct=True, entry_deviation_pct=None,
            close_reason="tp", confidence=75, pnl=100.0,
        )
        result = format_trade_evaluation(ev)
        assert "market" in result.lower() or "N/A" in result or "n/a" in result.lower()


class TestFormatEvaluationReport:
    def test_empty_report(self):
        report = EvaluationReport(
            total_evaluated=0, total_unmatched=0,
            direction_accuracy=0.0, avg_entry_deviation_pct=0.0,
            sl_hit_rate=0.0, tp_hit_rate=0.0, liquidation_rate=0.0,
            by_symbol=[], by_confidence=[], by_period=[], evaluations=[],
        )
        result = format_evaluation_report(report)
        assert "no" in result.lower() or "0" in result

    def test_full_report_contains_sections(self):
        report = EvaluationReport(
            total_evaluated=10, total_unmatched=2,
            direction_accuracy=0.7, avg_entry_deviation_pct=0.5,
            sl_hit_rate=0.3, tp_hit_rate=0.5, liquidation_rate=0.2,
            by_symbol=[
                SymbolStats(
                    symbol="BTC/USDT:USDT", total_trades=10,
                    direction_accuracy=0.7, avg_entry_deviation_pct=0.5,
                    sl_hit_rate=0.3, tp_hit_rate=0.5, total_pnl=500.0,
                ),
            ],
            by_confidence=[
                ConfidenceBucket(
                    bucket="high (67-100)", total_trades=5,
                    direction_accuracy=0.8, avg_pnl=100.0,
                ),
            ],
            by_period=[],
            evaluations=[],
        )
        result = format_evaluation_report(report)
        assert "70" in result  # direction accuracy
        assert "BTC" in result
        assert "high" in result.lower()
```

**Step 2: Run test to verify it fails**

Run: `cd orchestrator && uv run pytest tests/unit/test_formatters_evaluator.py -v`
Expected: FAIL with `ImportError: cannot import name 'format_evaluation_report'`

**Step 3: Write minimal implementation**

Add to `formatters.py` after the existing `format_trade_report()` section:

```python
# ---------------------------------------------------------------------------
# Pipeline evaluation
# ---------------------------------------------------------------------------

def format_trade_evaluation(ev: TradeEvaluation) -> str:
    """Format a single trade evaluation as a one-line pipeline feedback."""
    direction = "\u2705 Direction correct" if ev.direction_correct else "\u274c Direction wrong"
    if ev.entry_deviation_pct is not None:
        dev_str = f"Entry dev {ev.entry_deviation_pct:+.1f}%"
    else:
        dev_str = "Entry dev N/A"
    return f"\U0001f4ca Pipeline: {direction} \u00b7 {dev_str} \u00b7 Confidence {ev.confidence}%"


def format_evaluation_report(report: EvaluationReport) -> str:
    """Format a full evaluation report for Telegram /evaluate command."""
    if report.total_evaluated == 0:
        return "No evaluated trades yet."

    lines = [
        "Pipeline Evaluation",
        "",
        f"\U0001f4ca Trades: {report.total_evaluated}"
        + (f" ({report.total_unmatched} unmatched)" if report.total_unmatched else ""),
        f"\U0001f3af Direction Accuracy: {report.direction_accuracy:.1%}",
        f"\U0001f4cf Avg Entry Dev: {report.avg_entry_deviation_pct:+.2f}%",
        f"\u2705 TP Rate: {report.tp_hit_rate:.1%}"
        f" \u00b7 \u26d4 SL Rate: {report.sl_hit_rate:.1%}"
        f" \u00b7 \U0001f480 Liq: {report.liquidation_rate:.1%}",
    ]

    if report.by_symbol:
        lines.append("")
        lines.append("By Symbol")
        for s in report.by_symbol:
            display = s.symbol.replace(":USDT", "")
            lines.append(
                f"  {display}: {s.direction_accuracy:.0%} acc"
                f" \u00b7 {_pnl_str(s.total_pnl)}"
                f" \u00b7 {s.total_trades} trades"
            )

    if report.by_confidence:
        lines.append("")
        lines.append("By Confidence")
        for b in report.by_confidence:
            lines.append(
                f"  {b.bucket}: {b.direction_accuracy:.0%} acc"
                f" \u00b7 avg {_pnl_str(b.avg_pnl)}"
                f" \u00b7 {b.total_trades} trades"
            )

    return "\n".join(lines)
```

Also add the import at the top of `formatters.py`:

```python
from orchestrator.stats.evaluator import EvaluationReport, TradeEvaluation
```

Note: Use `TYPE_CHECKING` guard to avoid circular imports — these are only used in function signatures, but since `evaluator.py` doesn't import `formatters.py`, direct import is safe here.

**Step 4: Run test to verify it passes**

Run: `cd orchestrator && uv run pytest tests/unit/test_formatters_evaluator.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/formatters.py orchestrator/tests/unit/test_formatters_evaluator.py
git commit -m "feat: add pipeline evaluation formatters"
```

---

### Task 6: Add Telegram `/evaluate` command

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/bot.py`
- Test: Manual verification (Telegram bot commands are integration-tested)

**Step 1: Add the evaluator to bot constructor**

In `bot.py`, the `SentinelBot.__init__()` already receives `trade_repo` and `proposal_repo`. Create the evaluator instance:

```python
# In __init__, after self._proposal_repo = proposal_repo
from orchestrator.stats.evaluator import PipelineEvaluator
self._evaluator: PipelineEvaluator | None = None
if trade_repo is not None and proposal_repo is not None:
    self._evaluator = PipelineEvaluator(
        trade_repo=trade_repo, proposal_repo=proposal_repo,
    )
```

**Step 2: Register the `/evaluate` command**

Add to `_BOT_COMMANDS`:

```python
BotCommand("evaluate", "Pipeline accuracy report"),
```

Add to `format_help()` in `formatters.py`:

```python
"/evaluate — Pipeline accuracy report\n"
```

**Step 3: Add command handler registration**

In the method that registers handlers (find where `CommandHandler("perf", ...)` is registered), add:

```python
CommandHandler("evaluate", self._evaluate_handler),
```

**Step 4: Implement the handler**

```python
async def _evaluate_handler(
    self, update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    if not await self._check_admin(update):
        return
    if self._evaluator is None:
        await self._reply(update, "Evaluator not configured.")
        return

    from orchestrator.telegram.formatters import format_evaluation_report
    report = self._evaluator.evaluate()
    await self._reply(update, format_evaluation_report(report))
```

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/bot.py orchestrator/src/orchestrator/telegram/formatters.py
git commit -m "feat: add /evaluate Telegram command"
```

---

### Task 7: Inject pipeline feedback into close reports

**Files:**
- Modify: `orchestrator/src/orchestrator/telegram/bot.py`

**Step 1: Modify `push_close_report()` (line 317)**

```python
async def push_close_report(self, result: CloseResult) -> None:
    """Push a trade close report to all admin chats."""
    if self._app is None:
        return
    msg = format_trade_report(result)

    # Append pipeline evaluation if available
    if self._evaluator is not None:
        from orchestrator.telegram.formatters import format_trade_evaluation
        ev = self._evaluator.evaluate_single(result.trade_id)
        if ev is not None:
            msg = msg + "\n\n" + format_trade_evaluation(ev)

    for chat_id in self.admin_chat_ids:
        sent = await self._app.bot.send_message(
            chat_id=chat_id, text=msg, reply_markup=_translate_keyboard(),
        )
        self._msg_cache.store(sent.message_id, msg)
```

**Step 2: Modify `_handle_confirm_close()` (line 1693)**

```python
async def _handle_confirm_close(self, query: CallbackQuery, trade_id: str) -> None:
    if self._paper_engine is None or self._data_fetcher is None:
        await query.answer("Not configured")
        return

    try:
        pos = self._paper_engine._find_position(trade_id)
        current_price = await self._data_fetcher.fetch_current_price(pos.symbol)
        result = self._paper_engine.close_position(
            trade_id=trade_id, current_price=current_price,
        )
        text = format_trade_report(result)

        # Append pipeline evaluation
        if self._evaluator is not None:
            from orchestrator.telegram.formatters import format_trade_evaluation
            ev = self._evaluator.evaluate_single(result.trade_id)
            if ev is not None:
                text = text + "\n\n" + format_trade_evaluation(ev)

        await _safe_callback_reply(query, text=text, reply_markup=_translate_keyboard())
        if query.message:
            self._msg_cache.store(query.message.message_id, text)
    except Exception as e:
        await query.answer(f"Error: {e}")
```

**Step 3: Modify `_handle_confirm_reduce()` (line 1787)**

```python
async def _handle_confirm_reduce(
    self, query: CallbackQuery, trade_id: str, pct_str: str, *_args: str,
) -> None:
    pct = float(pct_str)
    if self._paper_engine is None or self._data_fetcher is None:
        await query.answer("Not configured")
        return

    try:
        pos = self._paper_engine._find_position(trade_id)
        current_price = await self._data_fetcher.fetch_current_price(pos.symbol)
        result = self._paper_engine.reduce_position(
            trade_id=trade_id, pct=pct, current_price=current_price,
        )
        text = format_trade_report(result)

        # Append pipeline evaluation
        if self._evaluator is not None:
            from orchestrator.telegram.formatters import format_trade_evaluation
            ev = self._evaluator.evaluate_single(result.trade_id)
            if ev is not None:
                text = text + "\n\n" + format_trade_evaluation(ev)

        await _safe_callback_reply(query, text=text, reply_markup=_translate_keyboard())
        if query.message:
            self._msg_cache.store(query.message.message_id, text)
    except Exception as e:
        await query.answer(f"Error: {e}")
```

**Step 4: Run full test suite**

Run: `cd orchestrator && uv run pytest -v --tb=short`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add orchestrator/src/orchestrator/telegram/bot.py
git commit -m "feat: inject pipeline evaluation into close/reduce reports"
```

---

### Task 8: Run linter and final verification

**Step 1: Run ruff**

Run: `cd orchestrator && uv run ruff check src/ tests/ --fix`
Expected: No errors (or auto-fixed)

**Step 2: Run full test suite with coverage**

Run: `cd orchestrator && uv run pytest -v --cov=orchestrator --cov-report=term-missing`
Expected: All tests PASS, new code covered

**Step 3: Final commit if any fixes needed**

```bash
git add -A
git commit -m "chore: lint fixes for pipeline evaluator"
```
