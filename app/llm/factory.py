from app.core.config import Settings
from app.agent.rag import HashEmbeddingProvider
from app.llm.base import EmbeddingProvider, LLMProvider, ProviderConfigurationError
from app.llm.providers import (
    OllamaProvider,
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleProvider,
)


def create_provider(settings: Settings) -> LLMProvider:
    provider_name = settings.model_provider.strip().lower()
    if provider_name == "openai_compatible":
        return OpenAICompatibleProvider(settings)
    if provider_name == "ollama":
        return OllamaProvider(settings)
    raise ProviderConfigurationError(
        f"Unsupported MODEL_PROVIDER={settings.model_provider!r}. "
        "Expected 'openai_compatible' or 'ollama'."
    )


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    provider_name = settings.embedding_provider.strip().lower()
    if provider_name == "hash":
        return HashEmbeddingProvider(dimensions=settings.embedding_dimensions)
    if provider_name == "openai_compatible":
        return OpenAICompatibleEmbeddingProvider(settings)
    raise ProviderConfigurationError(
        f"Unsupported EMBEDDING_PROVIDER={settings.embedding_provider!r}. "
        "Expected 'hash' or 'openai_compatible'."
    )
