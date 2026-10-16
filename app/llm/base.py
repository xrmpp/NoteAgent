from typing import Any, Protocol

from app.domain.models import ChatMessage, ModelResponse


class LLMProvider(Protocol):
    provider_name: str

    def generate_reply(self, messages: list[ChatMessage]) -> ModelResponse:
        ...

    def generate_json(self, messages: list[ChatMessage], schema_name: str) -> dict[str, Any]:
        ...


class ToolCallingProvider(Protocol):
    def generate_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
        *, timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        ...


class EmbeddingProvider(Protocol):
    provider_name: str

    def embed_texts(self, texts: list[str], *, deadline: float | None = None) -> list[list[float]]:
        ...


class ProviderConfigurationError(RuntimeError):
    pass


class ProviderExecutionError(RuntimeError):
    def __init__(self, message: str, *, category: str = "provider_error",
                 http_status: int | None = None, safe_detail: str | None = None) -> None:
        super().__init__(message)
        self.category = category
        self.http_status = http_status
        self.safe_detail = safe_detail


def provider_error_details(exc: Exception) -> dict[str, Any]:
    """Persist caller-authored diagnostics, never arbitrary exception text or response bodies."""
    details: dict[str, Any] = {"exception_type": type(exc).__name__}
    if exc.__cause__ is not None:
        details["cause_type"] = type(exc.__cause__).__name__
    if isinstance(exc, ProviderExecutionError):
        details.update(category=exc.category,
                       message=(exc.safe_detail or "Provider execution failed.")[:300])
        if exc.http_status is not None:
            details["http_status"] = exc.http_status
    elif isinstance(exc, TimeoutError):
        details.update(category="timeout", message="Provider request timed out.")
    elif isinstance(exc, AttributeError):
        details.update(category="provider_implementation_error",
                       message="Provider method or attribute is unavailable.")
    else:
        details.update(category="response_error", message="Model response could not be processed.")
    return details
