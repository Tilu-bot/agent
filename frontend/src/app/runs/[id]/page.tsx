"use client";

import { useState, useEffect, useCallback, use } from "react";
import Link from "next/link";
import { api, Run, Task, AgentEvent, Artifact } from "@/lib/api";
import styles from "./run.module.css";

const KIND_ICON: Record<string, string> = {
  agent_message: "💬",
  tool_call: "⚡",
  tool_result: "📦",
  plan_created: "📋",
  verification: "✓",
  reflexion: "↻",
  synthesis: "✦",
  thinking: "🧠",
};

const KIND_LABEL: Record<string, string> = {
  tool_call: "tool use",
  tool_result: "result",
  plan_created: "plan",
  verification: "verify",
  reflexion: "reflexion",
  synthesis: "synthesis",
  thinking: "thinking",
};

const ROLE_COLOR: Record<string, string> = {
  orchestrator: "#6c63ff",
  planner: "#42a5f5",
  tool_operator: "#ffb74d",
  verifier: "#4caf7d",
  researcher: "#ab47bc",
  coder: "#26c6da",
  reflexion: "#ff8a65",
  synthesizer: "#ffd54f",
  mathematician: "#80deea",
};

const ARTIFACT_ICON: Record<string, string> = {
  md: "📄",
  txt: "📄",
  json: "📋",
  py: "🐍",
  js: "📜",
  ts: "📜",
  csv: "📊",
  png: "🖼",
  jpg: "🖼",
  pdf: "📕",
};

function artifactIcon(name: string): string {
  const ext = name.split(".").pop()?.toLowerCase() ?? "";
  return ARTIFACT_ICON[ext] ?? "📎";
}

const ACTIVE_STATUSES = new Set(["pending", "running"]);

function StatusBadge({ status }: { status: string }) {
  return <span className={`badge badge-${status}`}>{status}</span>;
}

/** Renders a run summary with basic paragraph / list formatting. */
function SummaryBody({ text }: { text: string }) {
  const blocks = text.split(/\n{2,}/).filter(Boolean);
  return (
    <div className={styles.answerBody}>
      {blocks.map((block, i) => {
        const lines = block.split("\n").filter(Boolean);
        const isList = lines.every((l) => /^[\*\-]\s/.test(l));
        const isNumbered = lines.every((l) => /^\d+\.\s/.test(l));
        if (isList) {
          return (
            <ul key={i}>
              {lines.map((l, j) => (
                <li key={j}>{l.replace(/^[\*\-]\s+/, "")}</li>
              ))}
            </ul>
          );
        }
        if (isNumbered) {
          return (
            <ol key={i}>
              {lines.map((l, j) => (
                <li key={j}>{l.replace(/^\d+\.\s+/, "")}</li>
              ))}
            </ol>
          );
        }
        return <p key={i}>{block}</p>;
      })}
    </div>
  );
}

function TaskCard({ task, index }: { task: Task; index: number }) {
  const [showResult, setShowResult] = useState(false);
  const color = ROLE_COLOR[task.agent_role ?? ""] ?? "#8890b0";

  return (
    <div className={styles.stepCard} data-status={task.status}>
      <div className={styles.stepIndicator}>
        {task.status === "running" ? (
          <div className={styles.spinner} />
        ) : task.status === "completed" ? (
          <span className={styles.stepCheck}>✓</span>
        ) : task.status === "failed" ? (
          <span className={styles.stepFail}>✕</span>
        ) : task.status === "skipped" ? (
          <span className={styles.stepFail}>—</span>
        ) : (
          <span className={styles.stepNum}>{index + 1}</span>
        )}
      </div>
      <div className={styles.stepContent}>
        <div className={styles.stepTitle}>{task.title}</div>
        {task.description && (
          <div className={styles.stepDesc}>{task.description}</div>
        )}
        <div className={styles.stepMeta}>
          {task.agent_role && (
            <span
              className={styles.agentPill}
              style={{ color, borderColor: color }}
            >
              {task.agent_role}
            </span>
          )}
          {task.result && (
            <button
              className={styles.resultToggle}
              onClick={() => setShowResult((v) => !v)}
            >
              {showResult ? "Hide result" : "Show result"}
            </button>
          )}
        </div>
        {showResult && task.result && (
          <pre className={styles.stepResult}>{task.result.slice(0, 600)}</pre>
        )}
      </div>
    </div>
  );
}

function EventRow({ event }: { event: AgentEvent }) {
  const [expanded, setExpanded] = useState(false);
  const icon = KIND_ICON[event.kind] ?? "•";
  const color = ROLE_COLOR[event.agent_role ?? ""] ?? "#8890b0";
  const isToolCall = event.kind === "tool_call";
  const isToolResult = event.kind === "tool_result";
  const isSynthesis = event.kind === "synthesis";
  const isPlan = event.kind === "plan_created";
  const isThinking = event.kind === "thinking";
  const kindLabel = KIND_LABEL[event.kind];

  let contentClass = styles.timelineContent;
  if (isSynthesis) contentClass += " " + styles.synthContent;
  else if (isToolCall) contentClass += " " + styles.toolCallContent;
  else if (isToolResult) contentClass += " " + styles.toolResultContent;
  else if (isThinking) contentClass += " " + styles.thinkingContent;

  return (
    <div className={styles.timelineEntry}>
      <div className={styles.timelineLine} />
      <div className={styles.timelineDot} data-kind={event.kind}>
        {icon}
      </div>
      <div className={styles.timelineBody}>
        <div className={styles.timelineHeader}>
          <span className={styles.timelineRole} style={{ color }}>
            {event.agent_role ?? "system"}
          </span>
          {isToolCall && <span className={styles.toolBadge}>{kindLabel}</span>}
          {isToolResult && (
            <span className={styles.resultBadge}>{kindLabel}</span>
          )}
          {isSynthesis && (
            <span className={styles.synthBadge}>{kindLabel}</span>
          )}
          {isPlan && <span className={styles.planBadge}>{kindLabel}</span>}
          {isThinking && (
            <span className={styles.thinkingBadge}>{kindLabel}</span>
          )}
          <span className={styles.timelineTime}>
            {new Date(event.created_at).toLocaleTimeString()}
          </span>
        </div>

        <div className={contentClass}>{event.content}</div>

        {Boolean(event.data) && (
          <>
            <button
              className={styles.expandBtn}
              onClick={() => setExpanded((v) => !v)}
            >
              {expanded ? "▲ Hide data" : "▼ Show data"}
            </button>
            {expanded && (
              <pre className={styles.eventData}>
                {JSON.stringify(event.data, null, 2)}
              </pre>
            )}
          </>
        )}
      </div>
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

  function download() {
    const url = `${api.artifactContentUrl(runId, artifact.id)}?download=1`;
    const a = document.createElement("a");
    a.href = url;
    a.download = artifact.name;
    a.click();
  }

  return (
    <div className={styles.artifactRow}>
      <div className={styles.artifactHeader}>
        <span className={styles.artifactIcon}>{artifactIcon(artifact.name)}</span>
        <span className={styles.artifactName}>{artifact.name}</span>
        {artifact.sha256 && (
          <span className={styles.artifactMeta}>
            sha256:{artifact.sha256.slice(0, 10)}…
          </span>
        )}
        <div className={styles.artifactBtns}>
          <button
            className={styles.artifactBtn}
            onClick={download}
            title="Download file"
            aria-label="Download artifact"
          >
            ⬇ Download
          </button>
          <button
            className={styles.artifactBtn}
            onClick={load}
            disabled={loading}
          >
            {loading ? "…" : content !== null ? "Hide" : "View"}
          </button>
        </div>
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
  const [cancelling, setCancelling] = useState(false);

  /** Full REST reload — used on mount and as SSE fallback. */
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

    const es = new EventSource(api.streamUrl(runId));

    es.onmessage = (e: MessageEvent) => {
      try {
        const msg = JSON.parse(e.data as string) as {
          type: string;
          payload?: unknown;
        };

        if (msg.type === "run_update") {
          setRun(msg.payload as Run);
        } else if (msg.type === "event") {
          const ev = msg.payload as AgentEvent;
          setEvents((prev) =>
            prev.find((x) => x.id === ev.id) ? prev : [...prev, ev]
          );
        } else if (msg.type === "task_update") {
          const t = msg.payload as Task;
          setTasks((prev) => {
            const idx = prev.findIndex((x) => x.id === t.id);
            if (idx === -1) return [...prev, t];
            const next = [...prev];
            next[idx] = t;
            return next;
          });
        } else if (msg.type === "artifact") {
          const a = msg.payload as Artifact;
          setArtifacts((prev) =>
            prev.find((x) => x.id === a.id) ? prev : [...prev, a]
          );
        } else if (msg.type === "done") {
          es.close();
          load();
        }
      } catch {
        // ignore parse errors
      }
    };

    es.onerror = () => {
      es.close();
      const interval = setInterval(() => {
        load().then(() => {
          setRun((r) => {
            if (r && !ACTIVE_STATUSES.has(r.status)) clearInterval(interval);
            return r;
          });
        });
      }, 3000);
    };

    return () => es.close();
  }, [runId, load]);

  async function handleCancel() {
    if (!run) return;
    setCancelling(true);
    try {
      await api.cancelRun(run.id);
      setRun((r) => (r ? { ...r, status: "cancelled" } : r));
    } catch (err) {
      setError(String(err));
    } finally {
      setCancelling(false);
    }
  }

  if (error) return <div className={styles.error}>{error}</div>;
  if (!run) return <div className={styles.loading}>Loading…</div>;

  const isActive = ACTIVE_STATUSES.has(run.status);
  const completedTasks = tasks.filter((t) => t.status === "completed").length;

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <Link href="/" className={styles.back}>
          ← Runs
        </Link>
        <div className={styles.runInfo}>
          <span className={styles.runGoal}>{run.goal}</span>
          <StatusBadge status={run.status} />
          {isActive && (
            <button
              className={styles.cancelBtn}
              onClick={handleCancel}
              disabled={cancelling}
            >
              {cancelling ? "Cancelling…" : "✕ Cancel"}
            </button>
          )}
        </div>
      </header>

      {isActive && (
        <div className={styles.liveBar}>
          <span className={styles.liveDot} />
          Agent is running
          {tasks.length > 0 && ` · ${completedTasks} of ${tasks.length} tasks done`}
        </div>
      )}

      {/* ── Answer / Summary ──────────────────────────────────────────────── */}
      {run.summary && (
        <div className={styles.answerSection}>
          <div className={styles.answerHeader}>
            <span className={styles.answerIcon}>✦</span>
            <span className={styles.answerTitle}>Answer</span>
          </div>
          <SummaryBody text={run.summary} />
        </div>
      )}

      <nav className={styles.tabs}>
        {(["events", "tasks", "artifacts"] as const).map((t) => (
          <button
            key={t}
            className={`${styles.tab} ${tab === t ? styles.tabActive : ""}`}
            onClick={() => setTab(t)}
          >
            {t === "events" && `Activity (${events.length})`}
            {t === "tasks" && `Tasks (${tasks.length})`}
            {t === "artifacts" && `Files (${artifacts.length})`}
          </button>
        ))}
      </nav>

      <main className={styles.main}>
        {tab === "tasks" && (
          <div className={styles.taskList}>
            {tasks.length > 0 && (
              <div className={styles.progressHeader}>
                <div className={styles.progressLabel}>
                  {completedTasks} of {tasks.length} tasks completed
                </div>
                <div className={styles.progressTrack}>
                  <div
                    className={styles.progressFill}
                    style={{
                      width: `${(completedTasks / tasks.length) * 100}%`,
                    }}
                  />
                </div>
              </div>
            )}
            {tasks.length === 0 && (
              <p className={styles.empty}>No tasks yet — plan is being created…</p>
            )}
            {tasks.map((task, i) => (
              <TaskCard key={task.id} task={task} index={i} />
            ))}
          </div>
        )}

        {tab === "events" && (
          <div className={styles.timelineList}>
            {events.length === 0 && (
              <p className={styles.empty}>
                {isActive ? "Waiting for first event…" : "No events recorded."}
              </p>
            )}
            {events.map((ev) => (
              <EventRow key={ev.id} event={ev} />
            ))}
            {isActive && events.length > 0 && (
              <div className={styles.timelineLive}>
                <span className={styles.timelineLiveDot} />
                Agent is working…
              </div>
            )}
          </div>
        )}

        {tab === "artifacts" && (
          <div className={styles.artifactList}>
            {artifacts.length === 0 && (
              <p className={styles.empty}>No files generated yet.</p>
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

