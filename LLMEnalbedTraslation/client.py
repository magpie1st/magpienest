"""HTTP client utilities for interacting with Ollama on the Jetson board."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import requests
from requests import Response

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    _current_dir = Path(__file__).resolve().parent
    if str(_current_dir) not in sys.path:
        sys.path.insert(0, str(_current_dir))

    from config import JetsonConnectionConfig, DEFAULT_JETSON_CONFIG, normalized_models  # type: ignore
else:
    from .config import JetsonConnectionConfig, DEFAULT_JETSON_CONFIG, normalized_models

DEFAULT_TIMEOUT = 120


class OllamaClientError(RuntimeError):
    """Base exception for Ollama client failures."""


class OllamaClientConnectionError(OllamaClientError):
    """Raised when connecting to the remote Ollama endpoint fails."""


@dataclass
class OllamaGenerateResponse:
    """Container for generate endpoint results."""

    model: str
    response: str
    raw: Dict[str, object]


class OllamaClient:
    """Small helper responsible for HTTP calls to Ollama."""

    def __init__(
        self,
        config: JetsonConnectionConfig | None = None,
        *,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self._config = config or DEFAULT_JETSON_CONFIG
        self._timeout = timeout

    @property
    def config(self) -> JetsonConnectionConfig:
        return self._config

    @property
    def timeout(self) -> int:
        return self._timeout

    def update_config(
        self,
        config: JetsonConnectionConfig | None = None,
        *,
        host: str | None = None,
        port: int | None = None,
        timeout: int | None = None,
    ) -> None:
        """Update the underlying connection settings."""

        if config is not None:
            self._config = config
        else:
            self._config.update(host=host, port=port)
        if timeout is not None and timeout > 0:
            self._timeout = int(timeout)

    # Internal helpers -------------------------------------------------
    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Dict[str, object] | None = None,
        stream: bool = False,
    ) -> Response:
        url = f"{self._config.base_url}{path}"
        try:
            response = requests.request(method, url, json=json_body, timeout=self._timeout, stream=stream)
        except requests.RequestException as exc:  # pragma: no cover - network safety
            raise OllamaClientConnectionError(f"Failed to reach Ollama at {url}: {exc}") from exc

        if response.status_code >= 400:
            raise OllamaClientError(
                f"Ollama request to {path} failed with {response.status_code}: {response.text[:200]}"
            )
        return response

    # Discoverability --------------------------------------------------
    def list_models(self, extra_models: Iterable[str] | None = None) -> List[str]:
        """Return available model names, merging server and static defaults."""

        server_models: List[str] = []
        try:
            response = self._request("GET", "/api/tags")
            payload = response.json()
            models = payload.get("models", []) if isinstance(payload, dict) else []
            for model in models:
                if not isinstance(model, dict):
                    continue
                name = model.get("name")
                if name:
                    server_models.append(str(name))
        except OllamaClientError:
            # Gracefully fall back to static defaults if the Jetson is offline.
            server_models = []
        return normalized_models(list(server_models) + list(extra_models or []))

    # Generation -------------------------------------------------------
    def generate(
        self,
        model: str,
        prompt: str,
        *,
        stream: bool = False,
        options: Optional[Dict[str, object]] = None,
    ) -> OllamaGenerateResponse:
        """Call the Ollama /api/generate endpoint with the supplied prompt."""

        payload: Dict[str, object] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }
        if options:
            payload["options"] = options

        response = self._request("POST", "/api/generate", json_body=payload, stream=stream)

        if stream:
            text_parts: List[str] = []
            raw_chunks: List[Dict[str, object]] = []
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:  # pragma: no cover - malformed chunk
                    continue
                raw_chunks.append(chunk)
                if isinstance(chunk, dict) and chunk.get("error"):
                    response.close()
                    raise OllamaClientError(str(chunk.get("error")))
                piece = chunk.get("response") if isinstance(chunk, dict) else None
                if piece:
                    text_parts.append(str(piece))
                if chunk.get("done") is True:
                    break
            text = "".join(text_parts).strip()
            response.close()
            if not text:
                raise OllamaClientError("Streaming translation did not return any text.")
            return OllamaGenerateResponse(model=model, response=text, raw={"chunks": raw_chunks})

        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise OllamaClientError("Invalid JSON received from Ollama.") from exc

        response.close()

        text = body.get("response") if isinstance(body, dict) else None
        if not text:
            raise OllamaClientError("Ollama response did not include translated text.")
        return OllamaGenerateResponse(model=model, response=str(text), raw=body)
