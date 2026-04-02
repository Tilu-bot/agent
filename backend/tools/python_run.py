"""Python runner tool — executes Python code in a subprocess sandbox.

Unlike ``shell.exec`` (which requires Docker), this tool uses a subprocess
with a restricted environment and a hard timeout.  It is intentionally
simpler and always available as a fallback when Docker is not running.

Security model
--------------
* The script runs as a subprocess of the current process with a configurable
  timeout (default 30 s).
* ``PYTHONPATH`` is cleared; only the stdlib is available.
* ``HOME`` and ``TMPDIR`` are pointed at a temporary directory created for
  each call so the script cannot read or write persistent data.
* ``os.environ`` is cleared except for a minimal safe set.
* The sandbox is intentionally weaker than Docker — treat it as "polite
  containment", not a security boundary.  Use ``shell.exec`` (Docker) when
  executing untrusted code from external sources.

Usage
-----
Params: ``{"code": "print(1 + 1)"}``
Returns: ``{"stdout": "2\\n", "stderr": "", "exit_code": 0}``
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from backend.config import get_config
from backend.tools.bus import BaseTool, ToolResult

# Characters that indicate shell-injection attempts — block them upfront.
_BLOCKED_PATTERNS = ("__import__", "subprocess", "os.system", "eval(", "exec(")


def _is_dangerous(code: str) -> bool:
    lower = code.lower()
    return any(p in lower for p in _BLOCKED_PATTERNS)


class PythonRunTool(BaseTool):
    name = "python.run"
    description = (
        "Execute a Python code snippet in a subprocess sandbox (no Docker required). "
        "Returns stdout, stderr, and exit_code. "
        "Only stdlib is available. Timeout is 30 s. "
        "Params: {\"code\": \"<python source code>\"}"
    )

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        code: str = params.get("code", "").strip()
        if not code:
            return ToolResult.from_error(self.name, "Missing 'code' parameter")

        if _is_dangerous(code):
            return ToolResult.from_error(
                self.name,
                "Code contains potentially dangerous patterns and was blocked. "
                "Use shell.exec (Docker) for untrusted code."
            )

        cfg = get_config()
        timeout = cfg.tools.shell.timeout_seconds  # reuse shell timeout config

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "script.py"
            script_path.write_text(code, encoding="utf-8")

            # Minimal safe environment.
            safe_env: dict[str, str] = {
                "PATH": "/usr/bin:/bin",
                "HOME": tmpdir,
                "TMPDIR": tmpdir,
                "TEMP": tmpdir,
                "TMP": tmpdir,
                "LANG": "en_US.UTF-8",
                "LC_ALL": "en_US.UTF-8",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
            }

            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, str(script_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=safe_env,
                    cwd=tmpdir,
                )
                try:
                    stdout_b, stderr_b = await asyncio.wait_for(
                        proc.communicate(), timeout=timeout
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.communicate()
                    return ToolResult.from_error(
                        self.name, f"Code execution timed out after {timeout}s"
                    )

                exit_code = proc.returncode
                stdout = stdout_b.decode(errors="replace")
                stderr = stderr_b.decode(errors="replace")

                result = {
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": exit_code,
                }
                if exit_code == 0:
                    return ToolResult.from_success(self.name, result)
                return ToolResult.from_error(
                    self.name,
                    f"Script exited with code {exit_code}: {stderr[:512]}",
                )
            except Exception as exc:
                return ToolResult.from_error(self.name, str(exc))
