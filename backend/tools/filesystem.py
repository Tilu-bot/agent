from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.config import get_config
from backend.tools.bus import BaseTool, ToolResult


class FilesystemReadTool(BaseTool):
    name = "filesystem.read"
    description = "Read a file from the local filesystem. Returns content as text."

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        path_str: str = params.get("path", "")
        if not path_str:
            return ToolResult.from_error(self.name, "Missing 'path' parameter")
        try:
            path = Path(path_str)
            if not path.exists():
                return ToolResult.from_error(self.name, f"File not found: {path_str}")
            if not path.is_file():
                return ToolResult.from_error(self.name, f"Not a file: {path_str}")
            content = path.read_text(encoding="utf-8", errors="replace")
            return ToolResult.from_success(self.name, {"path": str(path), "content": content})
        except Exception as exc:
            return ToolResult.from_error(self.name, str(exc))


class FilesystemWriteTool(BaseTool):
    name = "filesystem.write"
    description = "Write content to a file. Requires allow_write to be enabled in config."

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        cfg = get_config()
        if not cfg.tools.filesystem.allow_write:
            return ToolResult.from_error(
                self.name,
                "filesystem.write is disabled. Set tools.filesystem.allow_write=true in config.",
            )
        path_str: str = params.get("path", "")
        content: str = params.get("content", "")
        if not path_str:
            return ToolResult.from_error(self.name, "Missing 'path' parameter")

        allowed = cfg.tools.filesystem.allowed_paths
        path = Path(path_str).resolve()
        if allowed:
            if not any(str(path).startswith(str(Path(a).resolve())) for a in allowed):
                return ToolResult.from_error(
                    self.name, f"Path '{path_str}' is not in allowed_paths"
                )

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return ToolResult.from_success(self.name, {"path": str(path), "bytes_written": len(content)})
        except Exception as exc:
            return ToolResult.from_error(self.name, str(exc))
