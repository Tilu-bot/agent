from __future__ import annotations

from enum import Enum
from typing import Any

from backend.config import get_config
from backend.llm.ollama_client import OllamaClient


class TaskType(str, Enum):
    fast = "fast"
    reasoning = "reasoning"
    code = "code"


class ModelRouter:
    """Routes tasks to appropriate models based on task type."""

    def __init__(self):
        self._cfg = get_config()
        self._client = OllamaClient()

    def select_model(self, task_type: TaskType | str) -> str:
        models = self._cfg.models
        try:
            task_type = TaskType(task_type) if isinstance(task_type, str) else task_type
        except ValueError:
            task_type = TaskType.fast
        mapping = {
            TaskType.fast: models.fast,
            TaskType.reasoning: models.reasoning,
            TaskType.code: models.code,
        }
        return mapping.get(task_type, models.fast)

    async def chat(
        self,
        task_type: TaskType | str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
    ) -> str:
        model = self.select_model(task_type)
        result = await self._client.chat(model=model, messages=messages, options=options)
        return result.get("message", {}).get("content", "")

    async def generate(
        self,
        task_type: TaskType | str,
        prompt: str,
        options: dict[str, Any] | None = None,
    ) -> str:
        model = self.select_model(task_type)
        result = await self._client.generate(model=model, prompt=prompt, options=options)
        return result.get("response", "")
