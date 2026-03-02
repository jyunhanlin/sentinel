import json

from orchestrator.llm.schema_validator import (
    ValidationFailure,
    ValidationSuccess,
    _extract_json,
    validate_llm_output,
)
from orchestrator.models import TechnicalAnalysis

_VALID = (
    '{"label": "short_term", "trend": "up", "trend_strength": 28.0,'
    ' "volatility_regime": "medium", "volatility_pct": 2.5,'
    ' "momentum": "bullish", "rsi": 62.0, "key_levels": [], "risk_flags": []}'
)


class TestValidateLLMOutput:
    def test_valid_json(self):
        result = validate_llm_output(_VALID, TechnicalAnalysis)
        assert isinstance(result, ValidationSuccess)
        assert result.value.trend.value == "up"

    def test_json_with_surrounding_text(self):
        raw = f"Here is my analysis:\n{_VALID}\nDone."
        result = validate_llm_output(raw, TechnicalAnalysis)
        assert isinstance(result, ValidationSuccess)
        assert result.value.trend_strength == 28.0

    def test_invalid_json(self):
        raw = "This is not JSON at all"
        result = validate_llm_output(raw, TechnicalAnalysis)
        assert isinstance(result, ValidationFailure)
        assert "JSON" in result.error_message

    def test_schema_violation(self):
        raw = (
            '{"label": "short_term", "trend": "invalid_trend", "trend_strength": 28.0,'
            ' "volatility_regime": "medium", "volatility_pct": 2.5,'
            ' "momentum": "bullish", "rsi": 62.0, "key_levels": [], "risk_flags": []}'
        )
        result = validate_llm_output(raw, TechnicalAnalysis)
        assert isinstance(result, ValidationFailure)

    def test_missing_required_field(self):
        raw = '{"label": "short_term", "trend": "up"}'
        result = validate_llm_output(raw, TechnicalAnalysis)
        assert isinstance(result, ValidationFailure)
        assert len(result.error_message) > 0

    def test_json_in_markdown_code_block(self):
        raw = f"```json\n{_VALID}\n```"
        result = validate_llm_output(raw, TechnicalAnalysis)
        assert isinstance(result, ValidationSuccess)
        assert result.value.rsi == 62.0


class TestExtractJsonBlock:
    def test_extracts_from_fenced_json_block(self):
        text = 'Some analysis...\n\n```json\n{"score": 42}\n```\n\nMore text.'
        result = _extract_json(text)
        assert result is not None
        assert '"score": 42' in result

    def test_extracts_from_plain_json_object(self):
        text = '{"score": 42}'
        result = _extract_json(text)
        assert result is not None
        assert '"score": 42' in result

    def test_extracts_json_surrounded_by_analysis(self):
        text = (
            "## Analysis\n"
            "The market looks bullish.\n\n"
            "```json\n"
            '{"trend": "up", "confidence": 0.8}\n'
            "```\n\n"
            "This concludes my analysis."
        )
        result = _extract_json(text)
        assert result is not None
        data = json.loads(result)
        assert data["trend"] == "up"

    def test_returns_none_for_no_json(self):
        text = "Just plain text with no JSON anywhere."
        result = _extract_json(text)
        assert result is None
