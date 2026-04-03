"""Model capability registry.

Discovers which models are available in the local Ollama instance and infers
what each model is expert at, based on well-known naming conventions and
family patterns.

This lets the ModelRouter automatically pick the *best available* model for
any task type, rather than blindly using a YAML-configured name that may not
be installed.

Capability tags align with :class:`~backend.llm.router.TaskType` values::

    fast      — quick, low-latency answers  (small / general models)
    reasoning — deep reasoning, long context (large instruct / chain-of-thought)
    code      — code generation / debugging  (coder-family models)
    math      — numerical / symbolic math    (math-family models)
    vision    — image understanding          (multimodal models)
    embedding — vector embeddings            (embedding-family models)
    search    — web search (tool capability, not model — every model gets it)

The registry is built lazily on first use (one async Ollama ``/api/tags``
call) and cached for the lifetime of the process.  Call
``refresh_registry()`` to re-probe after pulling a new model.
"""

from __future__ import annotations

import asyncio
import re


# ---------------------------------------------------------------------------
# Capability patterns — ordered most-specific first.
# All matching tag-sets are *unioned* together for a given model name so a
# model like "deepseek-math-coder:7b" gets both {"math"} and {"code"}.
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[re.Pattern[str], set[str]]] = [
    # ── Code-specialised ──────────────────────────────────────────────────────
    (
        re.compile(
            r"qwen.*coder|deepseek.*coder|codellama|codegemma|starcoder"
            r"|phi.*code|granite.*code|wizard.*code|yi.*coder",
            re.I,
        ),
        {"code"},
    ),
    # ── Math-specialised ──────────────────────────────────────────────────────
    (
        re.compile(
            r"mathstral|qwen.*math|deepseek.*math|numina|wizard.*math"
            r"|mammoth.*math|metamath",
            re.I,
        ),
        {"math", "reasoning"},
    ),
    # ── Vision / multimodal ───────────────────────────────────────────────────
    (
        re.compile(
            r"llava|bakllava|moondream|minicpm.*v|qwen.*vl|internvl"
            r"|cogvlm|pixtral|phi.*vision|idefics",
            re.I,
        ),
        {"vision"},
    ),
    # ── Embedding-specialised ─────────────────────────────────────────────────
    (
        re.compile(
            r"nomic.*embed|bge.*embed|mxbai.*embed|all-minilm|e5.*embed"
            r"|gte.*embed|jina.*embed|snowflake.*arctic.*embed",
            re.I,
        ),
        {"embedding"},
    ),
    # ── Strong reasoning / large instruct / chain-of-thought ─────────────────
    (
        re.compile(
            r"deepseek.?r1|qwq|marco.?o1|sky.?t1|wizard.*lm"
            r"|llama.*70b|llama.*\d+0b|mixtral|mistral.*\d+x"
            r"|qwen.*72b|qwen.*32b|qwen.*14b|gemma.*27b",
            re.I,
        ),
        {"reasoning"},
    ),
    # ── Small / fast general-purpose ─────────────────────────────────────────
    (
        re.compile(
            r"phi.*mini|smollm|tinyllama"
            r"|qwen.*0\.5b|qwen.*1\.5b|gemma.*2b|llama.*1b",
            re.I,
        ),
        {"fast"},
    ),
    # ── General instruct catch-all ────────────────────────────────────────────
    (
        re.compile(
            r"llama|qwen|gemma|mistral|phi|solar|openchat|nous|hermes"
            r"|dolphin|neural|vicuna|orca|wizard|zephyr|starling|yi"
            r"|falcon|deepseek|command|glm|baichuan",
            re.I,
        ),
        {"reasoning", "fast"},
    ),
]

# Specificity scores — used to rank models when multiple match a capability.
# Higher = more specialised = preferred.
_SPECIFICITY: dict[str, int] = {
    "code": 100,
    "math": 100,
    "vision": 100,
    "embedding": 100,
    "reasoning": 50,
    "fast": 10,
    "search": 0,
}


# ---------------------------------------------------------------------------
# Profile dataclass
# ---------------------------------------------------------------------------

class ModelProfile:
    """Holds the inferred capability set for one Ollama model."""

    __slots__ = ("name", "capabilities")

    def __init__(self, name: str, capabilities: set[str]) -> None:
        self.name = name
        self.capabilities = capabilities

    def score_for(self, capability: str) -> int:
        if capability not in self.capabilities:
            return 0
        return _SPECIFICITY.get(capability, 1)

    def __repr__(self) -> str:  # pragma: no cover
        return f"ModelProfile({self.name!r}, caps={sorted(self.capabilities)})"


def _infer_capabilities(model_name: str) -> set[str]:
    """Return the inferred capability tags for *model_name*.

    At minimum every non-embedding model gets ``{"fast", "search"}``.
    """
    caps: set[str] = set()
    for pattern, tags in _PATTERNS:
        if pattern.search(model_name):
            caps |= tags
    if "embedding" not in caps:
        # All non-embedding models can answer questions (fast) and use web tools (search).
        caps.add("fast")
    caps.add("search")
    return caps


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Discovers and profiles all models available in the local Ollama instance.

    After calling :meth:`refresh` (one async Ollama call) the registry can
    be queried synchronously::

        registry = ModelRegistry()
        await registry.refresh()
        model = registry.best_model_for("code")
        print(registry.capability_summary())
    """

    def __init__(self) -> None:
        self._profiles: list[ModelProfile] = []
        self._available: set[str] = set()
        self.ready: bool = False

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def refresh(self) -> None:
        """Probe Ollama for available models and build capability profiles."""
        from backend.llm.ollama_client import OllamaClient  # avoid circular import

        try:
            names = await OllamaClient().list_models()
        except Exception:
            names = []

        self._available = set(names)
        self._profiles = [
            ModelProfile(n, _infer_capabilities(n)) for n in names
        ]
        self.ready = True

    # ── Synchronous query API (safe to call after refresh()) ──────────────────

    def best_model_for(self, capability: str) -> str | None:
        """Return the name of the best available model for *capability*.

        Returns ``None`` when no models have been discovered yet.
        Ties in specificity score are broken by name length (shorter names
        tend to be the canonical / default variant of a family).
        """
        if not self._profiles:
            return None
        ranked = sorted(
            self._profiles,
            key=lambda p: (-p.score_for(capability), len(p.name)),
        )
        return ranked[0].name

    def is_model_available(self, model: str) -> bool:
        """Return True if *model* appears in Ollama's model list."""
        if not self.ready:
            return True  # optimistic before first refresh
        return model in self._available

    def available_models(self) -> list[str]:
        return sorted(self._available)

    def profiles(self) -> list[ModelProfile]:
        return list(self._profiles)

    def capability_summary(self) -> str:
        """Return a human-readable model ↔ capability table for prompt injection.

        Example output::

            • llama3.2:3b: reasoning (general)
            • qwen2.5-coder:3b: code
            • nomic-embed-text: embedding
        """
        if not self._profiles:
            return "No Ollama models are currently installed."
        lines: list[str] = []
        for p in sorted(self._profiles, key=lambda x: x.name):
            specific = sorted(p.capabilities - {"fast", "search"})
            label = ", ".join(specific) if specific else "general"
            lines.append(f"  • {p.name}: {label}")
        return "\n".join(lines)

    def slot_recommendation(self, slot: str, configured_model: str) -> str:
        """Return the model to actually use for *slot*.

        If *configured_model* is in Ollama, return it unchanged.
        Otherwise return the best available model for *slot*'s capability,
        or *configured_model* itself if nothing is available (so we don't
        silently break existing behaviour).
        """
        if not self.ready or not self._profiles:
            return configured_model
        if self.is_model_available(configured_model):
            return configured_model
        fallback = self.best_model_for(slot)
        return fallback or configured_model

    def slot_recommendation_with_reason(
        self, slot: str, configured_model: str
    ) -> tuple[str, str]:
        """Like :meth:`slot_recommendation` but also returns a reason string."""
        if not self.ready or not self._profiles:
            return configured_model, "registry not yet built"
        if self.is_model_available(configured_model):
            return configured_model, "configured model is available"
        fallback = self.best_model_for(slot)
        if fallback:
            return (
                fallback,
                f"configured model '{configured_model}' not found in Ollama; "
                f"using best available '{fallback}' for '{slot}' tasks",
            )
        return configured_model, f"no models available; falling back to '{configured_model}'"


# ---------------------------------------------------------------------------
# Process-level singleton
# ---------------------------------------------------------------------------

_registry: ModelRegistry | None = None
_lock: asyncio.Lock | None = None


def _get_lock() -> asyncio.Lock:
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock


async def get_registry() -> ModelRegistry:
    """Return the process-wide :class:`ModelRegistry`, refreshing if needed."""
    global _registry
    if _registry is not None and _registry.ready:
        return _registry
    async with _get_lock():
        if _registry is None or not _registry.ready:
            _registry = ModelRegistry()
            await _registry.refresh()
    return _registry


async def refresh_registry() -> ModelRegistry:
    """Force a fresh Ollama probe and return the updated registry."""
    global _registry
    async with _get_lock():
        _registry = ModelRegistry()
        await _registry.refresh()
    return _registry


def get_registry_sync() -> ModelRegistry | None:
    """Return the cached registry without blocking (may be ``None``)."""
    return _registry
