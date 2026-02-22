import pytest

from orchestrator.config import Settings


@pytest.fixture(autouse=True)
def _clean_settings_env(monkeypatch):
    """Remove all env vars that Settings reads, ensuring test isolation from .env and shell."""
    for field_name in Settings.model_fields:
        monkeypatch.delenv(field_name.upper(), raising=False)
