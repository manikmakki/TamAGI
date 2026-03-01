"""Configuration management for TamAGI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    model: str = "llama3.2"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 120
    # Ollama-specific: sets the KV-cache / context window size (num_ctx).
    # Without this, Ollama uses the model's compiled default which can vary
    # and may silently truncate context. Set to match your model's capability.
    # None = leave Ollama to use its default.
    num_ctx: int | None = None


class TamagiIdentity(BaseModel):
    name: str = "Tama"
    personality: str = "curious, helpful, and slightly mischievous"
    initial_energy: int = 80
    initial_happiness: int = 70
    initial_knowledge: int = 10


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 7741
    workers: int = 1


class ChromaDBConfig(BaseModel):
    persist_directory: str = "./data/chromadb"
    collection_name: str = "tamagi_memory"


class ElasticsearchConfig(BaseModel):
    url: str = "http://localhost:9200"
    index: str = "tamagi_memory"


class MemoryConfig(BaseModel):
    backend: str = "chromadb"
    chromadb: ChromaDBConfig = Field(default_factory=ChromaDBConfig)
    elasticsearch: ElasticsearchConfig = Field(default_factory=ElasticsearchConfig)
    retrieval_limit: int = 5
    relevance_threshold: float = 0.5


class GuardrailsConfig(BaseModel):
    allowed_read_paths: list[str] = Field(default_factory=lambda: ["./workspace"])
    allowed_write_paths: list[str] = Field(default_factory=lambda: ["./workspace"])
    exec_allowlist: list[str] = Field(
        default_factory=lambda: [
            "ls", "cat", "head", "tail", "grep", "find",
            "wc", "echo", "date", "python", "pip",
        ]
    )
    max_write_size: int = 10485760  # 10MB
    exec_timeout: int = 30


class WorkspaceConfig(BaseModel):
    path: str = "./workspace"


class HistoryConfig(BaseModel):
    max_conversations: int = 100
    max_messages_per_conversation: int = 200
    persist: bool = True
    persist_path: str = "./data/history"
    # When the total character count of history messages exceeds this threshold,
    # the oldest messages are archived into ChromaDB and removed from the active
    # context. They remain searchable via RAG recall.
    # Set to 0 to disable compression entirely.
    # Rough guide: 32000 chars ≈ 8k tokens at ~4 chars/token.
    context_compress_threshold: int = 32000


class WebSearchConfig(BaseModel):
    """Web search provider configuration."""
    provider: str = "duckduckgo"  # duckduckgo | brave | searxng
    brave_api_key: str = ""
    searxng_url: str = ""  # e.g. http://localhost:8080


class AutonomyConfig(BaseModel):
    """TamAGI's autonomous idle behavior (dream engine)."""
    enabled: bool = True
    interval_minutes: int = 30  # How often TamAGI dreams (0 = disabled)
    active_hours_start: int = 8   # Don't dream before this hour
    active_hours_end: int = 23    # Don't dream after this hour
    # Which activities are enabled: dream, explore, experiment, journal
    activities: list[str] = Field(
        default_factory=lambda: ["dream", "explore", "experiment", "journal"]
    )
    # Relative weights for activity selection (higher = more likely)
    weights: list[int] = Field(
        default_factory=lambda: [30, 25, 25, 20]
    )


class AgentConfig(BaseModel):
    max_tool_rounds: int = 5      # Max tool-call loops per user message
    llm_retry_attempts: int = 1   # Retries on RemoteProtocolError / ConnectError
    llm_retry_delay: float = 2.0  # Seconds between retries


class AuthConfig(BaseModel):
    # Set enabled: true in config.yaml to enforce login.
    # Zero impact when false — all routes remain open as before.
    enabled: bool = False
    # PBKDF2-HMAC-SHA256 hash produced by hash_password() in backend/auth.py,
    # or written by POST /api/auth/setup on first boot.
    # Empty string means no password is set yet (setup endpoint is open).
    password_hash: str = ""


class TamAGIConfig(BaseModel):
    """Root configuration model."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tamagi: TamagiIdentity = Field(default_factory=TamagiIdentity)
    server: ServerConfig = Field(default_factory=ServerConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    history: HistoryConfig = Field(default_factory=HistoryConfig)
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)
    autonomy: AutonomyConfig = Field(default_factory=AutonomyConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)


def load_config(config_path: str | Path | None = None) -> TamAGIConfig:
    """Load configuration from YAML file, with env var overrides."""
    if config_path is None:
        config_path = os.environ.get("TAMAGI_CONFIG", "config.yaml")

    config_path = Path(config_path)

    if config_path.exists():
        with open(config_path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
    else:
        raw = {}

    # Environment variable overrides
    env_mappings = {
        "TAMAGI_LLM_BASE_URL": ("llm", "base_url"),
        "TAMAGI_LLM_API_KEY": ("llm", "api_key"),
        "TAMAGI_LLM_MODEL": ("llm", "model"),
        "TAMAGI_NAME": ("tamagi", "name"),
        "TAMAGI_PORT": ("server", "port"),
        "TAMAGI_BRAVE_API_KEY": ("web_search", "brave_api_key"),
        "TAMAGI_SEARCH_PROVIDER": ("web_search", "provider"),
        "TAMAGI_SEARXNG_URL": ("web_search", "searxng_url"),
        "TAMAGI_AUTONOMY_ENABLED": ("autonomy", "enabled"),
        "TAMAGI_AUTONOMY_INTERVAL": ("autonomy", "interval_minutes"),
    }

    for env_var, path in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            section, key = path
            if section not in raw:
                raw[section] = {}
            # Coerce port/interval to int, enabled to bool
            if key == "port" or key == "interval_minutes":
                value = int(value)
            if key == "enabled":
                value = value.lower() in ("true", "1", "yes")
            raw[section][key] = value

    return TamAGIConfig(**raw)


# Global singleton
_config: TamAGIConfig | None = None


def get_config() -> TamAGIConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: TamAGIConfig) -> None:
    global _config
    _config = config
