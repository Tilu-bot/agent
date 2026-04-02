from __future__ import annotations

import asyncio
import shlex
from typing import Any

from backend.config import get_config
from backend.tools.bus import BaseTool, ToolResult


class ShellExecTool(BaseTool):
    name = "shell.exec"
    description = (
        "Execute a shell command inside a Docker sandbox container. "
        "No direct host execution is performed."
    )

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        cfg = get_config()
        if not cfg.tools.shell.enabled:
            return ToolResult.from_error(self.name, "shell.exec is disabled in config")

        command: str = params.get("command", "")
        if not command:
            return ToolResult.from_error(self.name, "Missing 'command' parameter")

        image = cfg.tools.shell.docker_image
        timeout = cfg.tools.shell.timeout_seconds
        memory_limit = cfg.tools.shell.memory_limit
        cpu_quota = cfg.tools.shell.cpu_quota

        docker_cmd = [
            "docker", "run", "--rm",
            "--network=none",
            f"--memory={memory_limit}",
            f"--cpu-quota={cpu_quota}",
            "--read-only",
            "--tmpfs=/tmp:rw,size=64m",
            image,
            "sh", "-c", command,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return ToolResult.from_error(
                    self.name, f"Command timed out after {timeout}s"
                )

            exit_code = proc.returncode
            result = {
                "command": command,
                "exit_code": exit_code,
                "stdout": stdout.decode(errors="replace"),
                "stderr": stderr.decode(errors="replace"),
            }
            if exit_code == 0:
                return ToolResult.from_success(self.name, result)
            return ToolResult.from_error(
                self.name,
                f"Command exited with code {exit_code}: {result['stderr'][:512]}",
            )
        except FileNotFoundError:
            return ToolResult.from_error(
                self.name,
                "Docker not found. Please install Docker Desktop and ensure it is running.",
            )
        except Exception as exc:
            return ToolResult.from_error(self.name, str(exc))
