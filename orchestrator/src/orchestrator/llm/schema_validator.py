from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError


@dataclass(frozen=True)
class ValidationSuccess[T]:
    value: T


@dataclass(frozen=True)
class ValidationFailure:
    error_message: str


type ValidationResult[T] = ValidationSuccess[T] | ValidationFailure


def _extract_json(raw: str) -> str | None:
    """Extract JSON object from raw LLM output.

    Handles markdown code blocks and surrounding text.
    """
    # Try markdown code block first
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if md_match:
        return md_match.group(1).strip()

    # Try to find a JSON object
    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        return brace_match.group(0)

    return None


def validate_llm_output[T: BaseModel](raw: str, model_class: type[T]) -> ValidationResult[T]:
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
