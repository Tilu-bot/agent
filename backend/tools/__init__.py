from backend.tools.bus import BaseTool, ToolBus, ToolInput, ToolResult  # noqa: F401
from backend.tools.document import DocumentParseTool  # noqa: F401
from backend.tools.filesystem import FilesystemReadTool, FilesystemWriteTool  # noqa: F401
from backend.tools.image import ImageAnalyzeTool  # noqa: F401
from backend.tools.math import MathCalculateTool  # noqa: F401
from backend.tools.scratchpad import ScratchpadListTool, ScratchpadReadTool, ScratchpadWriteTool  # noqa: F401
from backend.tools.shell import ShellExecTool  # noqa: F401
from backend.tools.web import WebFetchTool, WebSearchTool  # noqa: F401


def create_default_tool_bus() -> ToolBus:
    bus = ToolBus()
    # ── Filesystem ─────────────────────────────────────────────────────────────
    bus.register(FilesystemReadTool())
    bus.register(FilesystemWriteTool())
    # ── Web ────────────────────────────────────────────────────────────────────
    bus.register(WebFetchTool())
    bus.register(WebSearchTool())
    # ── Shell (Docker sandbox) ─────────────────────────────────────────────────
    bus.register(ShellExecTool())
    # ── Math ───────────────────────────────────────────────────────────────────
    bus.register(MathCalculateTool())
    # ── Scratchpad (per-run in-memory KV) ─────────────────────────────────────
    bus.register(ScratchpadWriteTool())
    bus.register(ScratchpadReadTool())
    bus.register(ScratchpadListTool())
    # ── Document parsing (PDF, DOCX, text) ────────────────────────────────────
    bus.register(DocumentParseTool())
    # ── Vision / image analysis ───────────────────────────────────────────────
    bus.register(ImageAnalyzeTool())
    return bus
