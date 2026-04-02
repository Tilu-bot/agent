from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class ModelsConfig(BaseModel):
    fast: str = "llama3.2:3b"
    reasoning: str = "llama3.2:3b"
    code: str = "qwen2.5-coder:3b"
    search: str = "llama3.2:3b"
    math: str = "qwen2.5:3b"
    vision: str = "llava:7b"
    embedding: str = "nomic-embed-text"


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"


class FilesystemToolConfig(BaseModel):
    allow_write: bool = False
    allowed_paths: list[str] = [".agentic/workspace"]


class WebToolConfig(BaseModel):
    timeout_seconds: int = 30
    max_bytes: int = 524288


class ShellToolConfig(BaseModel):
    enabled: bool = True
    docker_image: str = "agentic-sandbox:latest"
    timeout_seconds: int = 30
    memory_limit: str = "256m"
    cpu_quota: int = 50000


class ToolsConfig(BaseModel):
    filesystem: FilesystemToolConfig = FilesystemToolConfig()
    web: WebToolConfig = WebToolConfig()
    shell: ShellToolConfig = ShellToolConfig()


class ArtifactsConfig(BaseModel):
    dir: str = ".agentic/artifacts"


class DatabaseConfig(BaseModel):
    url: str = "sqlite+aiosqlite:///.agentic/agentic.db"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000"]
    max_concurrent_runs: int = 5


class MemoryConfig(BaseModel):
    enabled: bool = True
    recall_limit: int = 3
    max_parallel_tasks: int = 4


class DebateConfig(BaseModel):
    """Multi-candidate planning via LLM debate + voting.

    When enabled, the Orchestrator generates `num_candidates` independent
    plans for the same goal (each with `temperature` > 0) and then asks a
    Voter agent to pick the best one before execution begins.
    """

    enabled: bool = False
    num_candidates: int = 3
    temperature: float = 0.8


class ReflexionConfig(BaseModel):
    """Reflexion loop — retry tasks that fail verification.

    When enabled, any task whose verification confidence falls below
    `min_confidence` (or is explicitly marked unverified) is retried up to
    `max_retries` times.  Before each retry the Reflexion agent analyses the
    failure and produces a corrective prompt that is injected into the next
    ToolOperator call, allowing the model to self-correct.
    """

    enabled: bool = True
    min_confidence: int = 60   # retry if verifier confidence < this
    max_retries: int = 2


class SynthesisConfig(BaseModel):
    """Final answer synthesis.

    When enabled, after all tasks complete the Synthesizer agent reads every
    task result and produces a single coherent final answer that is stored in
    the Run.summary field and emitted as a 'synthesis' event on the SSE
    stream.
    """

    enabled: bool = True


class AppConfig(BaseModel):
    models: ModelsConfig = ModelsConfig()
    ollama: OllamaConfig = OllamaConfig()
    tools: ToolsConfig = ToolsConfig()
    artifacts: ArtifactsConfig = ArtifactsConfig()
    database: DatabaseConfig = DatabaseConfig()
    server: ServerConfig = ServerConfig()
    memory: MemoryConfig = MemoryConfig()
    debate: DebateConfig = DebateConfig()
    reflexion: ReflexionConfig = ReflexionConfig()
    synthesis: SynthesisConfig = SynthesisConfig()


@lru_cache
def load_config(path: str = "config.yaml") -> AppConfig:
    config_path = Path(path)
    if not config_path.is_absolute():
        # Look relative to this file's directory
        config_path = Path(__file__).parent / path
    raw: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open() as f:
            raw = yaml.safe_load(f) or {}
    return AppConfig.model_validate(raw)


def get_config() -> AppConfig:
    return load_config()


# ── Runtime model overrides ────────────────────────────────────────────────────
# These override the YAML config at runtime without restarting the server.
# Keys are model slot names (fast, reasoning, code, search, math, vision, embedding).
_runtime_model_overrides: dict[str, str] = {}


def get_runtime_model_overrides() -> dict[str, str]:
    """Return the current in-memory model slot overrides."""
    return _runtime_model_overrides


def set_runtime_model_override(slot: str, model: str) -> None:
    """Override a model slot at runtime.  Changes take effect immediately."""
    _runtime_model_overrides[slot] = model


def clear_runtime_model_overrides() -> None:
    """Reset all runtime overrides back to the YAML config values."""
    _runtime_model_overrides.clear()
