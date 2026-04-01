export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

export interface Run {
  id: string;
  goal: string;
  status: "pending" | "running" | "completed" | "failed";
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
  kind: "agent_message" | "tool_call" | "tool_result" | "plan_created" | "verification";
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

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${path} failed: ${res.status} ${text}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  createRun: (goal: string) =>
    apiFetch<Run>("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ goal }),
    }),
  listRuns: () => apiFetch<Run[]>("/api/runs"),
  getRun: (id: string) => apiFetch<Run>(`/api/runs/${id}`),
  getTasks: (id: string) => apiFetch<Task[]>(`/api/runs/${id}/tasks`),
  getEvents: (id: string) => apiFetch<AgentEvent[]>(`/api/runs/${id}/events`),
  getArtifacts: (id: string) => apiFetch<Artifact[]>(`/api/runs/${id}/artifacts`),
  getArtifactContent: (runId: string, artifactId: string) =>
    apiFetch<{ content: string }>(`/api/runs/${runId}/artifacts/${artifactId}/content`),
  health: () => apiFetch<{ status: string; ollama: boolean }>("/api/health"),
};
