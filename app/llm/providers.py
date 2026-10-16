from __future__ import annotations

import json
import time
from typing import Any

import httpx

from app.core.config import Settings
from app.domain.models import ChatMessage, ModelResponse
from app.llm.base import ProviderConfigurationError, ProviderExecutionError


class OpenAICompatibleProvider:
    provider_name = "openai_compatible"

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ProviderConfigurationError(
                "OPENAI_API_KEY is required when MODEL_PROVIDER=openai_compatible."
            )
        self._settings = settings

    def generate_reply(self, messages: list[ChatMessage]) -> ModelResponse:
        payload = {
            "model": self._settings.model_name,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self._settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._settings.openai_base_url.rstrip('/')}/chat/completions"

        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=60.0)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text
            raise ProviderExecutionError(
                f"Model request failed with status {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderExecutionError(f"Model request failed: {exc}") from exc

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderExecutionError(
                f"Unexpected model response shape: {json.dumps(data, ensure_ascii=False)}"
            ) from exc

        if isinstance(content, list):
            content = "".join(
                chunk.get("text", "") for chunk in content if isinstance(chunk, dict)
            ).strip()
        if not isinstance(content, str) or not content.strip():
            raise ProviderExecutionError("Model returned an empty response.")

        return ModelResponse(
            content=content.strip(),
            provider=self.provider_name,
            model=self._settings.model_name,
        )

    def generate_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
        *, timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        """Return the native assistant message; reasoning stays in run-local memory."""
        payload = {
            "model": self._settings.model_name, "messages": messages, "stream": False,
            "tools": tools, "tool_choice": "auto" if tools else "none",
        }
        if not tools:
            payload.pop("tools")
            payload.pop("tool_choice")
        try:
            response = httpx.post(
                f"{self._settings.openai_base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {self._settings.openai_api_key}"},
                json=payload, timeout=timeout_seconds,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise TimeoutError("Model request timed out.") from exc
        except httpx.HTTPStatusError as exc:
            raise ProviderExecutionError(
                f"Model HTTP {exc.response.status_code}.", category="http_error",
                http_status=exc.response.status_code,
                safe_detail="Model endpoint returned a non-success HTTP status.",
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderExecutionError(
                "Model connection failed.", category="connection_error",
                safe_detail="Model connection failed.",
            ) from exc
        try:
            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]
            if message.get("role") != "assistant":
                raise ValueError("Expected assistant role")
            if choice.get("finish_reason") not in {"stop", "tool_calls"}:
                raise ValueError("Incomplete model response")
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                raise ValueError("Invalid tool calls")
            ids = set()
            for call in calls:
                if (call.get("type") != "function" or not isinstance(call.get("id"), str)
                        or not call["id"] or call["id"] in ids
                        or not isinstance(call["function"]["name"], str)
                        or not isinstance(call["function"]["arguments"], str)):
                    raise ValueError("Invalid tool call")
                ids.add(call["id"])
            content = message.get("content")
            if content is not None and not isinstance(content, str):
                raise ValueError("Invalid content")
            if not calls and not content:
                raise ValueError("Empty model response")
            if "reasoning_content" in message and not isinstance(message["reasoning_content"], str):
                raise ValueError("Invalid reasoning content")
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            safe_reasons = {
                "Expected assistant role", "Incomplete model response", "Invalid tool calls",
                "Invalid tool call", "Invalid content", "Empty model response", "Invalid reasoning content",
            }
            detail = str(exc) if type(exc) is ValueError and str(exc) in safe_reasons else {
                json.JSONDecodeError: "Model response is not valid JSON.",
                KeyError: "Required model response field is missing.",
                IndexError: "Model response contains no choices.",
            }.get(type(exc), "Model response has an unexpected value type.")
            raise ProviderExecutionError(
                "Invalid or incomplete tool protocol response.", category="protocol_error",
                http_status=response.status_code, safe_detail=detail,
            ) from exc
        return {
            "message": {key: message[key] for key in
                        ("role", "content", "tool_calls", "reasoning_content") if key in message},
            "usage": data.get("usage") or {},
        }

    def generate_json(self, messages: list[ChatMessage], schema_name: str) -> dict[str, Any]:
        json_messages = [
            ChatMessage(
                role="system",
                content=(
                    "Return only one valid JSON object. Do not include markdown fences. "
                    f"The JSON object must match the requested task: {schema_name}."
                ),
            ),
            *messages,
        ]
        payload = {
            "model": self._settings.model_name,
            "messages": [
                {"role": message.role, "content": message.content} for message in json_messages
            ],
            "stream": False,
            "response_format": {"type": "json_object"},
            "temperature": self._settings.structured_temperature,
        }
        headers = {
            "Authorization": f"Bearer {self._settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._settings.openai_base_url.rstrip('/')}/chat/completions"

        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=60.0)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text
            raise ProviderExecutionError(
                f"Model JSON request failed with status {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderExecutionError(f"Model JSON request failed: {exc}") from exc

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderExecutionError(
                f"Unexpected model JSON response shape: {json.dumps(data, ensure_ascii=False)}"
            ) from exc

        if not isinstance(content, str):
            raise ProviderExecutionError("Model JSON response content is not a string.")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ProviderExecutionError(f"Model did not return valid JSON: {content}") from exc
        if not isinstance(parsed, dict):
            raise ProviderExecutionError("Model JSON response must be an object.")
        return parsed


class OllamaProvider:
    provider_name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def generate_reply(self, messages: list[ChatMessage]) -> ModelResponse:
        raise ProviderExecutionError(
            "Ollama provider is reserved for a later step. "
            f"Configure OLLAMA_BASE_URL={self._settings.ollama_base_url} when wiring it in."
        )

    def generate_json(self, messages: list[ChatMessage], schema_name: str) -> dict[str, Any]:
        raise ProviderExecutionError(
            "Ollama structured JSON generation is reserved for a later step. "
            f"Requested schema={schema_name!r}."
        )


class OpenAICompatibleEmbeddingProvider:
    provider_name = "openai_compatible"

    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport | None = None,
        batch_size: int | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        if not settings.embedding_api_key:
            raise ProviderConfigurationError(
                "EMBEDDING_API_KEY is required when EMBEDDING_PROVIDER=openai_compatible."
            )
        self._settings = settings
        self._transport = transport
        self._batch_size = batch_size or settings.embedding_batch_size
        self._timeout_seconds = timeout_seconds or settings.embedding_timeout_seconds

    @property
    def index_identity(self) -> dict[str, Any]:
        return {"provider": self.provider_name, "model": self._settings.embedding_model,
                "dimensions": self._settings.embedding_dimensions,
                "endpoint": self._settings.embedding_base_url.rstrip("/")}

    def embed_texts(self, texts: list[str], *, deadline: float | None = None) -> list[list[float]]:
        if not texts:
            return []

        embeddings: list[list[float]] = []
        with httpx.Client(transport=self._transport, timeout=self._timeout_seconds) as client:
            for start in range(0, len(texts), self._batch_size):
                batch = texts[start : start + self._batch_size]
                embeddings.extend(self._embed_batch(client, batch, deadline=deadline))
        return embeddings

    def _embed_batch(self, client: httpx.Client, texts: list[str], *, deadline: float | None = None) -> list[list[float]]:
        payload: dict[str, Any] = {
            "model": self._settings.embedding_model,
            "input": texts,
        }
        if self._settings.embedding_model.startswith("text-embedding-3"):
            payload["dimensions"] = self._settings.embedding_dimensions

        headers = {
            "Authorization": f"Bearer {self._settings.embedding_api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._settings.embedding_base_url.rstrip('/')}/embeddings"

        last_error: Exception | None = None
        for _ in range(2):
            try:
                options = {}
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Embedding deadline exceeded")
                    options["timeout"] = min(self._timeout_seconds, remaining)
                response = client.post(url, headers=headers, json=payload, **options)
                response.raise_for_status()
                data = response.json()
                items = data["data"]
                ordered = sorted(items, key=lambda item: int(item.get("index", 0)))
                vectors = [item["embedding"] for item in ordered]
                if len(vectors) != len(texts):
                    raise ProviderExecutionError(
                        f"Embedding response count mismatch: expected {len(texts)}, got {len(vectors)}."
                    )
                return [self._validate_vector(vector) for vector in vectors]
            except httpx.TimeoutException as exc:
                if deadline is not None:
                    raise TimeoutError("Embedding request timed out") from exc
                last_error = exc
            except (httpx.HTTPError, KeyError, TypeError, ValueError, ProviderExecutionError) as exc:
                last_error = exc
        raise ProviderExecutionError(f"Embedding request failed: {last_error}") from last_error

    @staticmethod
    def _validate_vector(vector: Any) -> list[float]:
        if not isinstance(vector, list) or not vector:
            raise ProviderExecutionError("Embedding response must contain non-empty vectors.")
        try:
            return [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise ProviderExecutionError("Embedding vector contains non-numeric values.") from exc
