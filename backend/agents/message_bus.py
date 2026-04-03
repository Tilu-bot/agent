"""Agent message bus — per-run pub/sub for cooperative inter-agent communication.

This is the "team chat" that lets agents coordinate rather than working in
isolation.  While a run is executing, any agent can:

* **Broadcast** a message to all listening agents.
* **Request help** from other agents (waits up to a timeout for the first
  useful response).
* **Share knowledge** — shorthand for posting a structured fact message.

Message types
-------------
``help_request``
    An agent is stuck and needs assistance.  Responders can post a
    ``help_response`` to the same channel with their suggestion.
``help_response``
    A reply to a ``help_request``.
``knowledge_share``
    A new fact / finding that all agents should be aware of.
``status``
    An informational progress update.

Design notes
------------
* Each run gets its own ``MessageBus`` instance.  When the run ends the bus
  is discarded — messages are not persisted to the DB.  Important facts
  should be written to the :class:`~backend.agents.knowledge.KnowledgeStore`
  which *is* persisted.
* Help responses are delivered as asyncio events so the requesting agent can
  ``await`` them without busy-looping.
* All posted messages are forwarded to the provided *event_logger* callback
  (async) so they appear in the run's SSE event stream.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AgentMessage:
    """A single message posted to the bus."""

    kind: str                       # help_request | help_response | knowledge_share | status
    from_agent: str                 # name of the sending agent
    content: str                    # human-readable message body
    topic: str = ""                 # optional topic/channel filter
    data: dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.monotonic)

    # For help_request/response threading
    reply_to: str | None = None     # message_id of the request being replied to


# ---------------------------------------------------------------------------
# MessageBus
# ---------------------------------------------------------------------------

# Type alias for the async event logger callback.
_EventLogger = Callable[..., Coroutine[Any, Any, None]]


class MessageBus:
    """Per-run pub/sub bus for agent-to-agent communication.

    Usage::

        bus = MessageBus(event_logger=my_log_fn)

        # Agent A broadcasts a fact it discovered:
        await bus.share_knowledge("agent_a", "missing_lib", "the project uses scipy 1.12")

        # Agent B is stuck and asks for help:
        response = await bus.request_help(
            from_agent="coder",
            topic="missing_library",
            context="ImportError: No module named 'scipy'",
            timeout=10.0,
        )
    """

    def __init__(self, event_logger: _EventLogger | None = None) -> None:
        self._messages: list[AgentMessage] = []
        # Pending help requests: request message_id → asyncio.Future[str]
        self._pending_help: dict[str, asyncio.Future[str]] = {}
        self._logger = event_logger
        self._lock = asyncio.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    async def post(self, message: AgentMessage) -> None:
        """Broadcast a message to all agents (no wait)."""
        async with self._lock:
            self._messages.append(message)
            # Deliver help_response to the corresponding pending future
            if message.kind == "help_response" and message.reply_to:
                fut = self._pending_help.get(message.reply_to)
                if fut and not fut.done():
                    fut.set_result(message.content)

        if self._logger:
            try:
                await self._logger(message)
            except Exception:
                pass

    async def request_help(
        self,
        from_agent: str,
        topic: str,
        context: str,
        timeout: float = 15.0,
    ) -> str | None:
        """Ask all agents for help on *topic*.

        Posts a ``help_request`` message and waits up to *timeout* seconds for
        the first ``help_response`` to arrive.  Returns the response text, or
        ``None`` if no agent responds in time.

        Agents that want to respond should call :meth:`respond_to_help`.
        """
        req = AgentMessage(
            kind="help_request",
            from_agent=from_agent,
            content=context,
            topic=topic,
        )
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[str] = loop.create_future()
        async with self._lock:
            self._pending_help[req.message_id] = fut
        await self.post(req)

        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            async with self._lock:
                self._pending_help.pop(req.message_id, None)

    async def respond_to_help(
        self,
        from_agent: str,
        request_id: str,
        response: str,
    ) -> None:
        """Post a help response targeting a specific request."""
        await self.post(AgentMessage(
            kind="help_response",
            from_agent=from_agent,
            content=response,
            reply_to=request_id,
        ))

    async def share_knowledge(
        self,
        from_agent: str,
        key: str,
        value: str,
    ) -> None:
        """Broadcast a key/value fact to all agents."""
        await self.post(AgentMessage(
            kind="knowledge_share",
            from_agent=from_agent,
            content=f"{key}: {value}",
            topic=key,
            data={"key": key, "value": value},
        ))

    async def broadcast_status(
        self,
        from_agent: str,
        content: str,
    ) -> None:
        """Broadcast a progress/status update."""
        await self.post(AgentMessage(
            kind="status",
            from_agent=from_agent,
            content=content,
        ))

    def all_messages(self) -> list[AgentMessage]:
        """Return all messages posted so far (newest last)."""
        return list(self._messages)

    def pending_help_requests(self) -> list[AgentMessage]:
        """Return open help-requests that have not been answered yet."""
        answered_ids = set()
        for m in self._messages:
            if m.kind == "help_response" and m.reply_to:
                answered_ids.add(m.reply_to)
        return [
            m for m in self._messages
            if m.kind == "help_request" and m.message_id not in answered_ids
        ]

    def knowledge_shares(self) -> list[AgentMessage]:
        """Return all knowledge_share messages posted so far."""
        return [m for m in self._messages if m.kind == "knowledge_share"]
