from __future__ import annotations

from typing import Any

import yaml
from pydantic import BaseModel


class ExpectedRange(BaseModel, frozen=True):
    min: float | None = None
    max: float | None = None


class ExpectedSentiment(BaseModel, frozen=True):
    sentiment_score: ExpectedRange | None = None
    confidence: ExpectedRange | None = None


class ExpectedMarket(BaseModel, frozen=True):
    trend: list[str] | None = None
    volatility_regime: list[str] | None = None


class ExpectedProposal(BaseModel, frozen=True):
    side: list[str] | None = None
    confidence: ExpectedRange | None = None
    has_stop_loss: bool | None = None
    sl_correct_side: bool | None = None


class ExpectedOutputs(BaseModel, frozen=True):
    sentiment: ExpectedSentiment | None = None
    market: ExpectedMarket | None = None
    proposal: ExpectedProposal | None = None


class EvalCase(BaseModel, frozen=True):
    id: str
    description: str
    snapshot: dict[str, Any]
    expected: ExpectedOutputs


def load_dataset(path: str) -> list[EvalCase]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return [EvalCase.model_validate(case) for case in raw]
