from __future__ import annotations

from enum import Enum
from typing import Any, AsyncGenerator

from backend.config import get_config, get_runtime_model_overrides
from backend.llm.hf_client import HFClient
from backend.llm.ollama_client import OllamaClient


def _is_hf_model(model: str) -> bool:
    """Return True when *model* looks like a HuggingFace model ID (contains '/')."""
    return "/" in model


def _llamacpp_client_for(model: str):
    """Return a ``LlamaCppClient`` for *model* if the registry says it's on llama.cpp.

    Returns ``None`` when the model is served by Ollama or the registry hasn't
    been built yet.
    """
    from backend.llm.model_registry import get_registry_sync  # avoid circular import
    from backend.llm.llamacpp_client import LlamaCppClient

    registry = get_registry_sync()
    if registry is None:
        return None
    backend, base_url = registry.backend_for(model)
    if backend == "llamacpp" and base_url:
        return LlamaCppClient(base_url)
    return None


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
    3. YAML config value — validated against the live model registry.

    Backend selection (in priority order)
    --------------------------------------
    1. If the model name contains ``/`` → HuggingFace Inference API.
    2. If the model registry says the model lives on a llama.cpp server
       → :class:`~backend.llm.llamacpp_client.LlamaCppClient`.
    3. Otherwise → local Ollama instance.

    Token tracking
    --------------
    Each ``chat()`` / ``generate()`` call accumulates ``prompt_eval_count``
    and ``eval_count`` from the response.  Call ``token_stats()`` at the end
    of a run to retrieve the totals.
    """

    def __init__(self, run_models: dict[str, str] | None = None):
        self._cfg = get_config()
        self._client = OllamaClient()
        self._hf_client = HFClient()
        self._run_models: dict[str, str] = run_models or {}
        # Accumulated token counts for the lifetime of this router instance.
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0

    def select_model(self, task_type: TaskType | str) -> str:
        """Return the model name to use for *task_type*.

        Selection priority (highest first):

        1. Per-run override (constructor argument).
        2. Global runtime override (``set_runtime_model_override``).
        3. YAML config value — validated against the live Ollama model list
           via the :class:`~backend.llm.model_registry.ModelRegistry`.  If
           the configured model is not installed, the registry automatically
           picks the best available model for that slot's capability.
        """
        try:
            task_type = TaskType(task_type) if isinstance(task_type, str) else task_type
        except ValueError:
            task_type = TaskType.fast

        slot = task_type.value

        # 1. Per-run override — trust caller, no registry check needed
        if slot in self._run_models:
            return self._run_models[slot]

        # 2. Global runtime override — trust operator, no registry check
        runtime_overrides = get_runtime_model_overrides()
        if slot in runtime_overrides:
            return runtime_overrides[slot]

        # 3. YAML config → validate against the live registry (if built)
        models = self._cfg.models
        mapping = {
            TaskType.fast: models.fast,
            TaskType.reasoning: models.reasoning,
            TaskType.code: models.code,
            TaskType.search: models.search,
            TaskType.math: models.math,
            TaskType.vision: models.vision,
        }
        config_model = mapping.get(task_type, models.fast)
        return self._resolve_via_registry(slot, config_model)

    @staticmethod
    def _resolve_via_registry(slot: str, configured_model: str) -> str:
        """Check *configured_model* against the registry and fall back if needed.

        When the registry has not been built yet (``None``) the configured
        value is returned unchanged so the system works in test environments
        without a live Ollama instance.
        """
        from backend.llm.model_registry import get_registry_sync  # avoid circular import

        registry = get_registry_sync()
        if registry is None:
            return configured_model
        return registry.slot_recommendation(slot, configured_model)

    def token_stats(self) -> dict[str, int]:
        """Return accumulated token counts since this router was created."""
        return {
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": self._prompt_tokens + self._completion_tokens,
        }

    async def chat(
        self,
        task_type: TaskType | str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
        format: str | None = None,
    ) -> str:
        tt = TaskType(task_type) if isinstance(task_type, str) else task_type
        model = self.select_model(tt)

        if _is_hf_model(model):
            result = await self._hf_client.chat(model, messages, options=options)
        else:
            llamacpp = _llamacpp_client_for(model)
            if llamacpp is not None:
                result = await llamacpp.chat(model, messages, options=options, format=format)
            else:
                result = await self._client.chat(
                    model=model,
                    messages=messages,
                    options=options,
                    task_type=tt.value,
                    format=format,
                )

        # Accumulate token usage.
        self._prompt_tokens += result.get("prompt_eval_count", 0)
        self._completion_tokens += result.get("eval_count", 0)
        return result.get("message", {}).get("content", "")

    async def chat_stream(
        self,
        task_type: TaskType | str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Async generator that yields content tokens for a streaming chat call."""
        tt = TaskType(task_type) if isinstance(task_type, str) else task_type
        model = self.select_model(tt)

        if _is_hf_model(model):
            async for token in self._hf_client.chat_stream(model, messages, options=options):
                yield token
        else:
            llamacpp = _llamacpp_client_for(model)
            if llamacpp is not None:
                async for token in llamacpp.chat_stream(model, messages, options=options):
                    yield token
            else:
                async for token in self._client.chat_stream(
                    model=model,
                    messages=messages,
                    options=options,
                    task_type=tt.value,
                ):
                    yield token

    async def generate(
        self,
        task_type: TaskType | str,
        prompt: str,
        options: dict[str, Any] | None = None,
        format: str | None = None,
    ) -> str:
        tt = TaskType(task_type) if isinstance(task_type, str) else task_type
        model = self.select_model(tt)

        if _is_hf_model(model):
            # HF client doesn't have a generate() method; wrap as chat.
            result = await self._hf_client.chat(
                model,
                [{"role": "user", "content": prompt}],
                options=options,
            )
            self._prompt_tokens += result.get("prompt_eval_count", 0)
            self._completion_tokens += result.get("eval_count", 0)
            return result.get("message", {}).get("content", "")

        result = await self._client.generate(
            model=model,
            prompt=prompt,
            options=options,
            task_type=tt.value,
            format=format,
        )
        self._prompt_tokens += result.get("prompt_eval_count", 0)
        self._completion_tokens += result.get("eval_count", 0)
        return result.get("response", "")

