from backend.tools.bus import BaseTool, ToolBus, ToolInput, ToolResult  # noqa: F401
from backend.tools.filesystem import FilesystemReadTool, FilesystemWriteTool  # noqa: F401
from backend.tools.web import WebFetchTool, WebSearchTool  # noqa: F401
from backend.tools.shell import ShellExecTool  # noqa: F401


def create_default_tool_bus() -> ToolBus:
    bus = ToolBus()
    bus.register(FilesystemReadTool())
    bus.register(FilesystemWriteTool())
    bus.register(WebFetchTool())
    bus.register(WebSearchTool())
    bus.register(ShellExecTool())
    return bus
