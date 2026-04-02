export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

export interface Run {
  id: string;
  goal: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  summary: string | null;
  created_at: string;
  updated_at: string;
}

export interface Task {
  id: string;
  run_id: string;
  title: string;
  description: string | null;
  agent_role: string | null;
  status: "pending" | "running" | "completed" | "failed" | "skipped";
  depends_on: string[];
  result: string | null;
  artifact_ids: string[];
  created_at: string;
  updated_at: string;
}

export interface AgentEvent {
  id: string;
  run_id: string;
  task_id: string | null;
  kind:
    | "agent_message"
    | "tool_call"
    | "tool_result"
    | "plan_created"
    | "verification"
    | "reflexion"
    | "synthesis";
  agent_role: string | null;
  content: string | null;
  data: unknown;
  created_at: string;
}

export interface Artifact {
  id: string;
  run_id: string;
  task_id: string | null;
  name: string;
  content_type: string;
  sha256: string | null;
  provenance: unknown;
  created_at: string;
}

export interface ModelSlots {
  fast: string;
  reasoning: string;
  code: string;
  search: string;
  math: string;
  vision: string;
  embedding: string;
  [key: string]: string;
}

export interface ModelsResponse {
  available: string[];
  slots: ModelSlots;
  slot_names: string[];
  runtime_overrides: Record<string, string>;
}

export interface TrainingStatus {
  job_id: string | null;
  status: "idle" | "starting" | "running" | "completed" | "failed";
  started_at: string | null;
  finished_at: string | null;
  config: Record<string, unknown> | null;
  output_tail: string;
  error: string;
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${path} failed: ${res.status} ${text}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  // ── Runs ────────────────────────────────────────────────────────────────────
  createRun: (goal: string, models?: Record<string, string>) =>
    apiFetch<Run>("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ goal, models: models ?? null }),
    }),
  listRuns: () => apiFetch<Run[]>("/api/runs"),
  getRun: (id: string) => apiFetch<Run>(`/api/runs/${id}`),
  cancelRun: (id: string) =>
    fetch(`${API_BASE}/api/runs/${id}`, { method: "DELETE" }),
  getTasks: (id: string) => apiFetch<Task[]>(`/api/runs/${id}/tasks`),
  getEvents: (id: string) => apiFetch<AgentEvent[]>(`/api/runs/${id}/events`),
  getArtifacts: (id: string) => apiFetch<Artifact[]>(`/api/runs/${id}/artifacts`),
  getArtifactContent: (runId: string, artifactId: string) =>
    apiFetch<{ content: string }>(`/api/runs/${runId}/artifacts/${artifactId}/content`),
  artifactContentUrl: (runId: string, artifactId: string) =>
    `${API_BASE}/api/runs/${runId}/artifacts/${artifactId}/content?as_download=true`,
  exportTrainingData: (minConfidence = 70, format = "alpaca") =>
    `${API_BASE}/api/runs/export/training-data?min_confidence=${minConfidence}&format=${format}`,
  // ── Models ──────────────────────────────────────────────────────────────────
  getModels: () => apiFetch<ModelsResponse>("/api/models"),
  updateModelSlots: (slots: Partial<ModelSlots>) =>
    apiFetch<{ updated: Record<string, string>; slots: ModelSlots }>("/api/models/slots", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ slots }),
    }),
  resetModelSlots: () =>
    apiFetch<{ reset: boolean; slots: ModelSlots }>("/api/models/slots", {
      method: "DELETE",
    }),
  pullModelStreamUrl: (model: string) =>
    `${API_BASE}/api/models/pull?model=${encodeURIComponent(model)}`,
  // ── Training ────────────────────────────────────────────────────────────────
  startTraining: (config: {
    model?: string;
    output_dir?: string;
    epochs?: number;
    batch_size?: number;
    lora_r?: number;
    min_confidence?: number;
    format?: string;
    api_base?: string;
  }) =>
    apiFetch<{ job_id: string; status: string; message: string }>("/api/training/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    }),
  getTrainingStatus: () => apiFetch<TrainingStatus>("/api/training/status"),
  listTrainingJobs: () =>
    apiFetch<Array<{ job_id: string; status: string; started_at: string; model: string }>>(
      "/api/training/jobs"
    ),
  // ── Health ──────────────────────────────────────────────────────────────────
  health: () => apiFetch<{ status: string; ollama: boolean }>("/api/health"),
  streamUrl: (id: string) => `${API_BASE}/api/runs/${id}/stream`,
};
