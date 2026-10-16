import pytest

from app.core.config import Settings
from app.llm.base import ProviderConfigurationError
from app.llm.factory import create_provider


def test_factory_returns_ollama_provider() -> None:
    settings = Settings(
        MODEL_PROVIDER="ollama",
        MODEL_NAME="deepseek-r1:1.5b",
        DATABASE_URL="sqlite+pysqlite:///:memory:",
    )

    provider = create_provider(settings)

    assert provider.provider_name == "ollama"


def test_factory_requires_openai_api_key() -> None:
    settings = Settings(
        MODEL_PROVIDER="openai_compatible",
        MODEL_NAME="gpt-4o-mini",
        DATABASE_URL="sqlite+pysqlite:///:memory:",
        OPENAI_API_KEY=None,
    )

    with pytest.raises(ProviderConfigurationError):
        create_provider(settings)

