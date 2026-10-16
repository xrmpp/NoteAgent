from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_title: str = Field(default="NoteAgent", alias="APP_TITLE")
    streamlit_server_port: int = Field(default=8501, alias="STREAMLIT_SERVER_PORT")
    model_provider: str = Field(default="openai_compatible", alias="MODEL_PROVIDER")
    model_name: str = Field(default="gpt-4o-mini", alias="MODEL_NAME")
    structured_temperature: float = Field(default=0.0, alias="STRUCTURED_TEMPERATURE")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    embedding_provider: str = Field(default="hash", alias="EMBEDDING_PROVIDER")
    embedding_base_url: str = Field(
        default="https://api.openai.com/v1", alias="EMBEDDING_BASE_URL"
    )
    embedding_api_key: str | None = Field(default=None, alias="EMBEDDING_API_KEY")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=1536, alias="EMBEDDING_DIMENSIONS")
    embedding_batch_size: int = Field(default=32, ge=1, alias="EMBEDDING_BATCH_SIZE")
    embedding_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        alias="EMBEDDING_TIMEOUT_SECONDS",
    )
    rerank_enabled: bool = Field(default=False, alias="RERANK_ENABLED")
    rag_generation_enabled: bool = Field(default=True, alias="RAG_GENERATION_ENABLED")
    rag_generation_max_chunks: int = Field(
        default=5,
        ge=1,
        le=10,
        alias="RAG_GENERATION_MAX_CHUNKS",
    )
    rag_generation_context_chars: int = Field(
        default=4000,
        ge=200,
        le=20000,
        alias="RAG_GENERATION_CONTEXT_CHARS",
    )
    ollama_base_url: str = Field(default="http://host.docker.internal:11434", alias="OLLAMA_BASE_URL")
    database_url: str = Field(
        default="postgresql+psycopg://agent:agent@postgres:5432/ai_agent",
        alias="DATABASE_URL",
    )
    primary_thread_title: str = Field(default="Primary Thread", alias="PRIMARY_THREAD_TITLE")
    auto_ingest_demo_corpus: bool = Field(default=True, alias="AUTO_INGEST_DEMO_CORPUS")
    max_context_messages: int = Field(default=20, alias="MAX_CONTEXT_MESSAGES")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
