"""Tool result TTL cache.

Wraps ``ToolBus.call`` with an in-memory TTL cache keyed on
``tool_name + sha256(params)``.  Identical tool calls within the same
process lifetime (up to ``ttl_seconds``) return the cached result without
hitting the network or filesystem again.

This prevents the common pattern where two tasks in the same run both call
``web.fetch`` on the same URL, or ``web.search`` with the same query.

Cache is intentionally NOT shared across runs or restarts (purely in-memory).
Cache entries are evicted lazily on read when they have expired.

Tools that should NOT be cached (side-effectful) can be excluded via the
``NEVER_CACHE`` set.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from backend.tools.bus import ToolBus, ToolInput, ToolResult

# Side-effectful tools whose results must never be cached.
NEVER_CACHE: frozenset[str] = frozenset({
    "scratchpad.write",
    "scratchpad.read",
    "scratchpad.list",
    "shell.exec",
    "python.run",
    "filesystem.write",
})


@dataclass
class _Entry:
    result: ToolResult
    expires_at: float


class CachedToolBus(ToolBus):
    """A ToolBus that caches tool results by tool_name + params hash.

    Parameters
    ----------
    ttl_seconds:
        How long (in seconds) a cached result is considered fresh.
        Default 300 s (5 minutes).
    """

    def __init__(self, ttl_seconds: int = 300) -> None:
        super().__init__()
        self._ttl = ttl_seconds
        self._cache: dict[str, _Entry] = {}

    def _cache_key(self, tool_input: ToolInput) -> str:
        # Exclude run_id from the cache key so the same web.fetch URL is
        # cached across different runs within the same process session.
        key_data = {
            "tool": tool_input.tool_name,
            "params": {k: v for k, v in tool_input.params.items() if not k.startswith("_")},
        }
        serialized = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def _is_cacheable(self, tool_name: str) -> bool:
        return tool_name not in NEVER_CACHE

    async def call(self, tool_input: ToolInput) -> ToolResult:
        if not self._is_cacheable(tool_input.tool_name):
            return await super().call(tool_input)

        key = self._cache_key(tool_input)
        now = time.monotonic()

        # Cache hit?
        entry = self._cache.get(key)
        if entry is not None and entry.expires_at > now:
            return entry.result

        # Cache miss — execute and store.
        result = await super().call(tool_input)
        if result.success:
            self._cache[key] = _Entry(
                result=result,
                expires_at=now + self._ttl,
            )
        return result

    def invalidate(self, tool_name: str | None = None) -> None:
        """Evict cache entries.  If tool_name is given, only evict for that tool."""
        if tool_name is None:
            self._cache.clear()
        else:
            keys_to_del = [
                k for k, v in self._cache.items()
                if v.result.tool_name == tool_name
            ]
            for k in keys_to_del:
                del self._cache[k]

    def cache_stats(self) -> dict[str, Any]:
        """Return basic cache statistics for debugging."""
        now = time.monotonic()
        total = len(self._cache)
        live = sum(1 for e in self._cache.values() if e.expires_at > now)
        return {"total_entries": total, "live_entries": live, "ttl_seconds": self._ttl}
