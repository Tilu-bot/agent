"""Document parsing tool.

Extracts readable text from PDF, DOCX, and plain-text files.

Dependencies (all optional — the tool degrades gracefully):
  * pypdf   — PDF parsing (pip install pypdf)
  * python-docx — DOCX parsing (pip install python-docx)

If a dependency is missing the tool returns a clear error asking the user
to install it rather than crashing silently.
"""
from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from backend.config import get_config
from backend.tools.bus import BaseTool, ToolResult

_MAX_CHARS = 50_000  # cap extracted text to avoid huge payloads


def _parse_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("pypdf is required to parse PDFs. Install with: pip install pypdf")
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _parse_docx(path: Path) -> str:
    try:
        import docx  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "python-docx is required to parse .docx files. "
            "Install with: pip install python-docx"
        )
    doc = docx.Document(str(path))
    return "\n".join(para.text for para in doc.paragraphs)


def _parse_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


class DocumentParseTool(BaseTool):
    name = "document.parse"
    description = (
        "Extract readable text from a local file (PDF, DOCX, or plain text). "
        "Params: {\"path\": \"<absolute or relative file path>\", "
        "\"max_chars\": 50000 (optional)}. "
        "Returns {\"path\", \"mime_type\", \"content\", \"truncated\"}."
    )

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        path_str: str = params.get("path", "")
        if not path_str:
            return ToolResult.from_error(self.name, "Missing 'path' parameter")

        path = Path(path_str)
        if not path.exists():
            return ToolResult.from_error(self.name, f"File not found: {path_str}")
        if not path.is_file():
            return ToolResult.from_error(self.name, f"Not a file: {path_str}")

        # Security: only allow files in allowed_paths or the workspace dir
        cfg = get_config()
        allowed = cfg.tools.filesystem.allowed_paths or []
        abs_path = path.resolve()
        if allowed and not any(
            str(abs_path).startswith(str(Path(a).resolve())) for a in allowed
        ):
            return ToolResult.from_error(
                self.name, f"Path '{path_str}' is not in allowed_paths"
            )

        max_chars: int = int(params.get("max_chars", _MAX_CHARS))
        mime_type, _ = mimetypes.guess_type(str(path))
        mime_type = mime_type or "text/plain"

        try:
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                text = _parse_pdf(path)
            elif suffix in (".docx", ".doc"):
                text = _parse_docx(path)
            else:
                text = _parse_text(path)

            truncated = len(text) > max_chars
            content = text[:max_chars]
            return ToolResult.from_success(
                self.name,
                {
                    "path": str(abs_path),
                    "mime_type": mime_type,
                    "content": content,
                    "chars": len(content),
                    "truncated": truncated,
                },
            )
        except ImportError as exc:
            return ToolResult.from_error(self.name, str(exc))
        except Exception as exc:
            return ToolResult.from_error(self.name, f"Parse error: {exc}")
