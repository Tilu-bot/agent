"""Training API — one-click fine-tuning from verified run data.

Endpoints
---------
POST /api/training/start
    Export verified run data and kick off scripts/fine_tune.py as a
    background subprocess.  Returns immediately with a job_id.

GET  /api/training/status
    Return the status of the current (or most recent) training job.

GET  /api/training/jobs
    Return a list of all training jobs with their status and configuration.

The fine-tuning script requires:
    pip install transformers trl datasets peft bitsandbytes

If those packages are not installed the subprocess will fail with a clear
message and the job status will reflect that.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/training", tags=["training"])

# ── In-memory job registry ─────────────────────────────────────────────────────
# Each entry: {id, status, started_at, finished_at, config, output, error}
_jobs: dict[str, dict[str, Any]] = {}
_current_job_id: str | None = None


class StartTrainingRequest(BaseModel):
    model: str = Field(
        "microsoft/Phi-3-mini-4k-instruct",
        description=(
            "HuggingFace model ID to fine-tune. "
            "Defaults to Phi-3-mini-4k-instruct (~3.8 B params, lightweight and "
            "instruction-tuned) which becomes the 'agen-model' after fine-tuning."
        ),
    )
    output_dir: str = Field(
        "./agen-model",
        description="Directory to write the fine-tuned agen-model weights.",
    )
    epochs: int = Field(3, ge=1, le=20, description="Number of training epochs.")
    batch_size: int = Field(4, ge=1, le=32)
    lora_r: int = Field(16, ge=0, description="LoRA rank; set 0 for full fine-tune.")
    min_confidence: int = Field(70, ge=0, le=100)
    format: str = Field("alpaca", description="Training data format: alpaca or sharegpt.")
    api_base: str = Field(
        "http://localhost:8000",
        description="Base URL of this agentic API (used to fetch training data).",
    )


@router.post("/start")
async def start_training(body: StartTrainingRequest) -> dict[str, Any]:
    """Export training data and start the fine-tuning subprocess."""
    global _current_job_id

    # Only allow one training job at a time
    if _current_job_id and _jobs.get(_current_job_id, {}).get("status") == "running":
        raise HTTPException(
            status_code=409,
            detail="A training job is already running. Wait for it to finish.",
        )

    if body.format not in ("alpaca", "sharegpt"):
        raise HTTPException(status_code=400, detail="format must be 'alpaca' or 'sharegpt'")

    job_id = str(uuid.uuid4())
    _current_job_id = job_id

    job: dict[str, Any] = {
        "id": job_id,
        "status": "starting",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "config": body.model_dump(),
        "output": "",
        "error": "",
    }
    _jobs[job_id] = job

    # Launch the training in a background task
    asyncio.create_task(_run_training(job_id, body))

    return {"job_id": job_id, "status": "starting", "message": "Training job enqueued."}


async def _run_training(job_id: str, cfg: StartTrainingRequest) -> None:
    """Run fine_tune.py as a subprocess and capture its output."""
    job = _jobs[job_id]
    job["status"] = "running"

    # Step 1: Download training data from the export endpoint
    import tempfile
    data_path = Path(tempfile.gettempdir()) / f"agentic_training_{job_id}.jsonl"
    export_url = (
        f"{cfg.api_base.rstrip('/')}/api/runs/export/training-data"
        f"?format={cfg.format}&min_confidence={cfg.min_confidence}"
    )

    try:
        import httpx
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(export_url)
            resp.raise_for_status()
            data_path.write_bytes(resp.content)
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = f"Failed to download training data: {exc}"
        job["finished_at"] = datetime.now(timezone.utc).isoformat()
        return

    if data_path.stat().st_size < 10:
        job["status"] = "failed"
        job["error"] = (
            "No training data available. "
            "Run some goals first so the agent can accumulate verified examples."
        )
        job["finished_at"] = datetime.now(timezone.utc).isoformat()
        return

    # Step 2: Locate fine_tune.py
    script = Path(__file__).parent.parent.parent / "scripts" / "fine_tune.py"
    if not script.exists():
        job["status"] = "failed"
        job["error"] = f"Fine-tune script not found at {script}"
        job["finished_at"] = datetime.now(timezone.utc).isoformat()
        return

    # Step 3: Build the command
    cmd = [
        sys.executable, str(script),
        "--data", str(data_path),
        "--model", cfg.model,
        "--output", cfg.output_dir,
        "--epochs", str(cfg.epochs),
        "--batch-size", str(cfg.batch_size),
        "--lora-r", str(cfg.lora_r),
    ]

    # Step 4: Run in executor so we don't block the event loop
    loop = asyncio.get_event_loop()
    try:
        proc_result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2-hour hard limit
            ),
        )
        job["output"] = (proc_result.stdout or "")[-10000:]  # last 10 KB
        job["error"] = (proc_result.stderr or "")[-5000:]
        if proc_result.returncode == 0:
            job["status"] = "completed"
            job["output"] += f"\n\nagen-model saved to: {cfg.output_dir}"
        else:
            job["status"] = "failed"
    except subprocess.TimeoutExpired:
        job["status"] = "failed"
        job["error"] = "Training timed out after 2 hours."
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
    finally:
        job["finished_at"] = datetime.now(timezone.utc).isoformat()
        try:
            data_path.unlink(missing_ok=True)
        except Exception:
            pass


@router.get("/status")
async def training_status() -> dict[str, Any]:
    """Return the status of the most recent training job."""
    if not _current_job_id or _current_job_id not in _jobs:
        return {"status": "idle", "job_id": None}
    job = _jobs[_current_job_id]
    return {
        "job_id": job["id"],
        "status": job["status"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
        "config": job["config"],
        "output_tail": job["output"][-2000:] if job["output"] else "",
        "error": job["error"],
    }


@router.get("/jobs")
async def list_jobs() -> list[dict[str, Any]]:
    """Return all training jobs (most recent first)."""
    jobs = sorted(_jobs.values(), key=lambda j: j["started_at"], reverse=True)
    return [
        {
            "job_id": j["id"],
            "status": j["status"],
            "started_at": j["started_at"],
            "finished_at": j["finished_at"],
            "model": j["config"].get("model"),
        }
        for j in jobs
    ]
