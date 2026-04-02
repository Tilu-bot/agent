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


class AppConfig(BaseModel):
    models: ModelsConfig = ModelsConfig()
    ollama: OllamaConfig = OllamaConfig()
    tools: ToolsConfig = ToolsConfig()
    artifacts: ArtifactsConfig = ArtifactsConfig()
    database: DatabaseConfig = DatabaseConfig()
    server: ServerConfig = ServerConfig()
    memory: MemoryConfig = MemoryConfig()


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
