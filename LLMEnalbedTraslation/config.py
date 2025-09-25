"""Configuration helpers for the LLM-enabled translation app."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

#DEFAULT_HOST = "192.168.55.1"
DEFAULT_HOST = "10.112.164.81"
#DEFAULT_PORT = 11434
DEFAULT_PORT = 11435

# Preserve insertion order while removing any duplicates supplied by configuration.
AVAILABLE_MODELS: List[str] = list(dict.fromkeys([
    "qwen2.5:7b",
    "qwen3:4b",
    "qwen3:8b",
]))


@dataclass
class JetsonConnectionConfig:
    """Connection settings for the remote Jetson Ollama service."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT

    @property
    def base_url(self) -> str:
        """Return the base URL used to contact the Ollama REST API."""

        return f"http://{self.host}:{self.port}"

    def to_dict(self) -> Dict[str, object]:
        """Serialize the configuration to a plain dictionary."""

        return {"host": self.host, "port": self.port}

    @classmethod
    def from_dict(cls, data: Dict[str, object] | None) -> "JetsonConnectionConfig":
        """Create a configuration instance from a dictionary of values."""

        if not data:
            return cls()
        host = str(data.get("host", DEFAULT_HOST))
        try:
            port_val = data.get("port", DEFAULT_PORT)
            port = int(port_val)
        except (TypeError, ValueError):  # pragma: no cover - defensive conversion
            port = DEFAULT_PORT
        return cls(host=host, port=port)

    def update(self, *, host: str | None = None, port: int | None = None) -> None:
        """Mutate the instance with new connection values."""

        if host:
            self.host = host
        if port is not None:
            self.port = port


DEFAULT_JETSON_CONFIG = JetsonConnectionConfig()


def normalized_models(custom_models: Iterable[str] | None = None) -> List[str]:
    """Return the available models merged with optional custom entries."""

    models: Dict[str, None] = dict.fromkeys(AVAILABLE_MODELS)
    if custom_models:
        for name in custom_models:
            if not name:
                continue
            models[str(name)] = None
    return list(models.keys())
