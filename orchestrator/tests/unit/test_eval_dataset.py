import os
import tempfile

import pytest
import yaml

from orchestrator.eval.dataset import ExpectedRange, load_dataset


class TestEvalDataset:
    def _write_yaml(self, cases: list[dict]) -> str:
        """Write cases to a temp YAML file and return path."""
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            yaml.dump(cases, f)
        return path

    def test_load_single_case(self):
        cases = [{
            "id": "bull_breakout",
            "description": "Strong uptrend",
            "snapshot": {
                "symbol": "BTC/USDT:USDT",
                "current_price": 95000.0,
                "ohlcv": [[1, 94000, 95500, 93500, 95200, 1000]],
                "funding_rate": 0.01,
                "open_interest": 15000000000,
            },
            "expected": {
                "proposal": {
                    "side": ["long"],
                    "confidence": {"min": 0.5},
                },
            },
        }]
        path = self._write_yaml(cases)
        try:
            dataset = load_dataset(path)
            assert len(dataset) == 1
            assert dataset[0].id == "bull_breakout"
            assert dataset[0].expected.proposal is not None
            assert dataset[0].expected.proposal.side == ["long"]
        finally:
            os.unlink(path)

    def test_load_with_range_constraints(self):
        cases = [{
            "id": "test",
            "description": "test",
            "snapshot": {"symbol": "BTC/USDT:USDT", "current_price": 95000.0, "ohlcv": []},
            "expected": {
                "sentiment": {
                    "sentiment_score": {"min": 60, "max": 90},
                    "confidence": {"min": 0.5},
                },
            },
        }]
        path = self._write_yaml(cases)
        try:
            dataset = load_dataset(path)
            assert dataset[0].expected.sentiment is not None
            assert dataset[0].expected.sentiment.sentiment_score.min == 60
            assert dataset[0].expected.sentiment.sentiment_score.max == 90
        finally:
            os.unlink(path)

    def test_expected_range_is_frozen(self):
        r = ExpectedRange(min=1.0, max=10.0)
        with pytest.raises(Exception):
            r.min = 5.0


class TestGoldenDatasetV1:
    def test_load_golden_v1(self):
        """The golden_v1 dataset should load and have 5 cases."""
        dataset_path = os.path.join(
            os.path.dirname(__file__),
            "../../src/orchestrator/eval/datasets/golden_v1.yaml",
        )
        dataset = load_dataset(dataset_path)
        assert len(dataset) == 5
        ids = {c.id for c in dataset}
        assert "bull_breakout" in ids
        assert "bear_divergence" in ids
        assert "sideways_range" in ids
        assert "high_volatility" in ids
        assert "funding_anomaly" in ids
