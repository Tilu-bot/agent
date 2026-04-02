from __future__ import annotations

from enum import Enum
from typing import Any

from backend.config import get_config, get_runtime_model_overrides
from backend.llm.ollama_client import OllamaClient


class TaskType(str, Enum):
    fast = "fast"
    reasoning = "reasoning"
    code = "code"
    search = "search"
    math = "math"
    vision = "vision"


class ModelRouter:
    """Routes tasks to appropriate models based on task type.

    Priority for selecting a model (highest first):
    1. Per-run override passed to the constructor (e.g. from a run request).
    2. Global runtime override set via ``set_runtime_model_override()``.
    3. YAML config value.
    """

    def __init__(self, run_models: dict[str, str] | None = None):
        self._cfg = get_config()
        self._client = OllamaClient()
        self._run_models: dict[str, str] = run_models or {}

    def select_model(self, task_type: TaskType | str) -> str:
        try:
            task_type = TaskType(task_type) if isinstance(task_type, str) else task_type
        except ValueError:
            task_type = TaskType.fast

        slot = task_type.value

        # 1. Per-run override
        if slot in self._run_models:
            return self._run_models[slot]

        # 2. Global runtime override
        runtime_overrides = get_runtime_model_overrides()
        if slot in runtime_overrides:
            return runtime_overrides[slot]

        # 3. YAML config
        models = self._cfg.models
        mapping = {
            TaskType.fast: models.fast,
            TaskType.reasoning: models.reasoning,
            TaskType.code: models.code,
            TaskType.search: models.search,
            TaskType.math: models.math,
            TaskType.vision: models.vision,
        }
        return mapping.get(task_type, models.fast)

    async def chat(
        self,
        task_type: TaskType | str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
    ) -> str:
        tt = TaskType(task_type) if isinstance(task_type, str) else task_type
        model = self.select_model(tt)
        result = await self._client.chat(
            model=model, messages=messages, options=options, task_type=tt.value
        )
        return result.get("message", {}).get("content", "")

    async def generate(
        self,
        task_type: TaskType | str,
        prompt: str,
        options: dict[str, Any] | None = None,
    ) -> str:
        tt = TaskType(task_type) if isinstance(task_type, str) else task_type
        model = self.select_model(tt)
        result = await self._client.generate(
            model=model, prompt=prompt, options=options, task_type=tt.value
        )
        return result.get("response", "")
