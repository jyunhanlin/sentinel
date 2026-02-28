import json

from orchestrator.llm.schema_validator import (
    ValidationFailure,
    ValidationSuccess,
    _extract_json,
    validate_llm_output,
)
from orchestrator.models import SentimentReport


class TestValidateLLMOutput:
    def test_valid_json(self):
        raw = '{"sentiment_score": 72, "key_events": [], "sources": ["news"], "confidence": 0.8}'
        result = validate_llm_output(raw, SentimentReport)
        assert isinstance(result, ValidationSuccess)
        assert result.value.sentiment_score == 72

    def test_json_with_surrounding_text(self):
        raw = (
            'Here is my analysis:\n'
            '{"sentiment_score": 50, "key_events": [], "sources": [], "confidence": 0.5}'
            '\nDone.'
        )
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
        raw = (
            '```json\n'
            '{"sentiment_score": 60, "key_events": [], "sources": [], "confidence": 0.7}'
            '\n```'
        )
        result = validate_llm_output(raw, SentimentReport)
        assert isinstance(result, ValidationSuccess)
        assert result.value.sentiment_score == 60


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
