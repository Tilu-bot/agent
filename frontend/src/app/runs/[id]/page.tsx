"use client";

import { useState, useEffect, useCallback, use } from "react";
import Link from "next/link";
import { api, Run, Task, AgentEvent, Artifact } from "@/lib/api";
import styles from "./run.module.css";

const KIND_ICON: Record<string, string> = {
  agent_message: "💬",
  tool_call: "🔧",
  tool_result: "📦",
  plan_created: "📋",
  verification: "✅",
};

const ROLE_COLOR: Record<string, string> = {
  orchestrator: "#6c63ff",
  planner: "#42a5f5",
  tool_operator: "#ffb74d",
  verifier: "#4caf7d",
  researcher: "#ab47bc",
  coder: "#26c6da",
};

function StatusBadge({ status }: { status: string }) {
  return <span className={`badge badge-${status}`}>{status}</span>;
}

function TaskCard({ task }: { task: Task }) {
  return (
    <div className={styles.taskCard} data-status={task.status}>
      <div className={styles.taskHeader}>
        <span className={styles.taskTitle}>{task.title}</span>
        <StatusBadge status={task.status} />
      </div>
      {task.description && (
        <p className={styles.taskDesc}>{task.description}</p>
      )}
      {task.agent_role && (
        <span
          className={styles.taskRole}
          style={{ color: ROLE_COLOR[task.agent_role] ?? "#8890b0" }}
        >
          {task.agent_role}
        </span>
      )}
      {task.result && (
        <pre className={styles.taskResult}>{task.result.slice(0, 400)}</pre>
      )}
    </div>
  );
}

function EventRow({ event }: { event: AgentEvent }) {
  const [expanded, setExpanded] = useState(false);
  const icon = KIND_ICON[event.kind] ?? "•";
  const color = ROLE_COLOR[event.agent_role ?? ""] ?? "#8890b0";

  return (
    <div className={styles.eventRow}>
      <div className={styles.eventMeta}>
        <span className={styles.eventIcon}>{icon}</span>
        <span className={styles.eventRole} style={{ color }}>
          {event.agent_role ?? "system"}
        </span>
        <span className={styles.eventTime}>
          {new Date(event.created_at).toLocaleTimeString()}
        </span>
        {Boolean(event.data) && (
          <button
            className={styles.expandBtn}
            onClick={() => setExpanded((v) => !v)}
          >
            {expanded ? "▲" : "▼"}
          </button>
        )}
      </div>
      <div className={styles.eventContent}>{event.content}</div>
      {expanded && Boolean(event.data) && (
        <pre className={styles.eventData}>
          {JSON.stringify(event.data, null, 2)}
        </pre>
      )}
    </div>
  );
}

function ArtifactRow({
  artifact,
  runId,
}: {
  artifact: Artifact;
  runId: string;
}) {
  const [content, setContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function load() {
    if (content !== null) {
      setContent(null);
      return;
    }
    setLoading(true);
    try {
      const data = await api.getArtifactContent(runId, artifact.id);
      setContent(data.content);
    } catch (e) {
      setContent(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={styles.artifactRow}>
      <div className={styles.artifactHeader}>
        <span className={styles.artifactName}>📎 {artifact.name}</span>
        <span className={styles.artifactMeta}>
          {artifact.sha256 ? `sha256:${artifact.sha256.slice(0, 12)}…` : ""}
        </span>
        <button className={styles.expandBtn} onClick={load} disabled={loading}>
          {loading ? "…" : content !== null ? "Hide" : "View"}
        </button>
      </div>
      {content !== null && (
        <pre className={styles.artifactContent}>{content.slice(0, 4000)}</pre>
      )}
    </div>
  );
}

export default function RunPage({ params }: { params: Promise<{ id: string }> }) {
  const { id: runId } = use(params);
  const [run, setRun] = useState<Run | null>(null);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [tab, setTab] = useState<"tasks" | "events" | "artifacts">("events");
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const [r, t, e, a] = await Promise.all([
        api.getRun(runId),
        api.getTasks(runId),
        api.getEvents(runId),
        api.getArtifacts(runId),
      ]);
      setRun(r);
      setTasks(t);
      setEvents(e);
      setArtifacts(a);
    } catch (err) {
      setError(String(err));
    }
  }, [runId]);

  useEffect(() => {
    load();
    const interval = setInterval(() => {
      if (run?.status === "running" || run?.status === "pending") load();
    }, 3000);
    return () => clearInterval(interval);
  }, [load, run?.status]);

  if (error) return <div className={styles.error}>{error}</div>;
  if (!run) return <div className={styles.loading}>Loading…</div>;

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <Link href="/" className={styles.back}>
          ← Runs
        </Link>
        <div className={styles.runInfo}>
          <span className={styles.runGoal}>{run.goal}</span>
          <StatusBadge status={run.status} />
        </div>
      </header>

      <nav className={styles.tabs}>
        {(["events", "tasks", "artifacts"] as const).map((t) => (
          <button
            key={t}
            className={`${styles.tab} ${tab === t ? styles.tabActive : ""}`}
            onClick={() => setTab(t)}
          >
            {t === "events" && `Timeline (${events.length})`}
            {t === "tasks" && `Tasks (${tasks.length})`}
            {t === "artifacts" && `Evidence (${artifacts.length})`}
          </button>
        ))}
      </nav>

      <main className={styles.main}>
        {tab === "tasks" && (
          <div className={styles.taskList}>
            {tasks.length === 0 && (
              <p className={styles.empty}>No tasks yet — plan is being created…</p>
            )}
            {tasks.map((task) => (
              <TaskCard key={task.id} task={task} />
            ))}
          </div>
        )}

        {tab === "events" && (
          <div className={styles.eventList}>
            {events.length === 0 && (
              <p className={styles.empty}>No events yet…</p>
            )}
            {events.map((ev) => (
              <EventRow key={ev.id} event={ev} />
            ))}
          </div>
        )}

        {tab === "artifacts" && (
          <div className={styles.artifactList}>
            {artifacts.length === 0 && (
              <p className={styles.empty}>No artifacts yet.</p>
            )}
            {artifacts.map((a) => (
              <ArtifactRow key={a.id} artifact={a} runId={runId} />
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
