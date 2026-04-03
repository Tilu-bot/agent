"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { api, Run, ModelsResponse } from "@/lib/api";import styles from "./page.module.css";

const SLOT_LABELS: Record<string, string> = {
  fast: "Fast (quick tasks)",
  reasoning: "Reasoning (planning, debate)",
  code: "Code (coding tasks)",
  search: "Search (web research)",
  math: "Math (calculations)",
  vision: "Vision (image analysis)",
  embedding: "Embedding (memory recall)",
};

function ModelSettings({
  onClose,
}: {
  onClose: () => void;
}) {
  const [modelsData, setModelsData] = useState<ModelsResponse | null>(null);
  const [draft, setDraft] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  // ── Pull a model ──────────────────────────────────────────────────────────
  const [pullModel, setPullModel] = useState("");
  const [pulling, setPulling] = useState(false);
  const [pullStatus, setPullStatus] = useState<string | null>(null);
  const [pullProgress, setPullProgress] = useState<{ completed: number; total: number } | null>(null);

  useEffect(() => {
    api.getModels().then((d) => {
      setModelsData(d);
      setDraft({ ...d.slots });
    }).catch(() => setMsg("Could not load models (is the backend running?)"));
  }, []);

  async function handleSave() {
    if (!modelsData) return;
    setSaving(true);
    setMsg(null);
    try {
      const updated = await api.updateModelSlots(draft);
      setModelsData((prev) => prev ? { ...prev, slots: updated.slots } : null);
      setMsg("✓ Model slots updated.");
    } catch (e) {
      setMsg(`Error: ${e}`);
    } finally {
      setSaving(false);
    }
  }

  async function handleReset() {
    setSaving(true);
    setMsg(null);
    try {
      const result = await api.resetModelSlots();
      setDraft({ ...result.slots });
      setModelsData((prev) => prev ? { ...prev, slots: result.slots } : null);
      setMsg("✓ Reset to config defaults.");
    } catch (e) {
      setMsg(`Error: ${e}`);
    } finally {
      setSaving(false);
    }
  }

  function handlePull() {
    const modelName = pullModel.trim();
    if (!modelName || pulling) return;
    setPulling(true);
    setPullStatus("Connecting…");
    setPullProgress(null);

    const es = new EventSource(api.pullModelStreamUrl(modelName));

    es.onmessage = (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data as string) as {
          status: string;
          completed?: number;
          total?: number;
          error?: string;
        };

        if (data.status === "done") {
          setPullStatus(`✓ "${modelName}" pulled successfully.`);
          setPullProgress(null);
          setPulling(false);
          es.close();
          // Refresh the model list so the new model appears in slot dropdowns.
          api.getModels().then((d) => {
            setModelsData(d);
            setDraft((prev) => ({ ...d.slots, ...prev }));
          }).catch(() => null);
        } else if (data.status === "error") {
          setPullStatus(`✗ Error: ${data.error ?? "unknown"}`);
          setPullProgress(null);
          setPulling(false);
          es.close();
        } else if (data.completed !== undefined && data.total !== undefined && data.total > 0) {
          setPullStatus(`Downloading…`);
          setPullProgress({ completed: data.completed, total: data.total });
        } else {
          setPullStatus(data.status);
          setPullProgress(null);
        }
      } catch {
        // ignore parse errors
      }
    };

    es.onerror = () => {
      setPullStatus("✗ Connection lost. Check Ollama is running.");
      setPulling(false);
      es.close();
    };
  }

  const slots = modelsData?.slot_names ?? [];
  const available = modelsData?.available ?? [];

  return (
    <div className={styles.settingsOverlay} onClick={onClose}>
      <div className={styles.settingsPanel} onClick={(e) => e.stopPropagation()}>
        <div className={styles.settingsHeader}>
          <h2>⚙ Model Settings</h2>
          <button className={styles.closeBtn} onClick={onClose}>✕</button>
        </div>

        {available.length > 0 && (
          <p className={styles.settingsNote}>
            {available.length} model{available.length !== 1 ? "s" : ""} available in Ollama.
          </p>
        )}
        {available.length === 0 && (
          <p className={styles.settingsNote} style={{ color: "var(--error)" }}>
            Ollama has no models pulled. Run <code>ollama pull llama3.2:3b</code> to get started.
          </p>
        )}

        <div className={styles.slotList}>
          {slots.map((slot) => (
            <div key={slot} className={styles.slotRow}>
              <label className={styles.slotLabel}>{SLOT_LABELS[slot] ?? slot}</label>
              {available.length > 0 ? (
                <select
                  className={styles.slotSelect}
                  value={draft[slot] ?? ""}
                  onChange={(e) => setDraft((d) => ({ ...d, [slot]: e.target.value }))}
                >
                  {available.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                  {!available.includes(draft[slot]) && draft[slot] && (
                    <option value={draft[slot]}>{draft[slot]} (not pulled)</option>
                  )}
                </select>
              ) : (
                <input
                  className={styles.slotSelect}
                  value={draft[slot] ?? ""}
                  onChange={(e) => setDraft((d) => ({ ...d, [slot]: e.target.value }))}
                  placeholder="e.g. llama3.2:3b"
                />
              )}
            </div>
          ))}
        </div>

        {msg && <div className={styles.settingsMsg}>{msg}</div>}

        <div className={styles.settingsDivider} />

        {/* ── Pull a model ──────────────────────────────────────────────── */}
        <div className={styles.pullSection}>
          <label className={styles.slotLabel}>Pull a model from Ollama</label>
          <div className={styles.pullRow}>
            <input
              className={styles.pullInput}
              placeholder="e.g. llama3.2:3b"
              value={pullModel}
              onChange={(e) => setPullModel(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handlePull()}
              disabled={pulling}
            />
            <button
              className={styles.pullBtn}
              onClick={handlePull}
              disabled={pulling || !pullModel.trim()}
            >
              {pulling ? "Pulling…" : "⬇ Pull"}
            </button>
          </div>
          {pullProgress && (
            <div className={styles.pullProgressWrap}>
              {(() => {
                const pct = Math.min(100, Math.round((pullProgress.completed / pullProgress.total) * 100));
                return (
                  <>
                    <div className={styles.pullProgressBar} style={{ width: `${pct}%` }} />
                    <span className={styles.pullProgressLabel}>
                      {pct}%
                      &nbsp;·&nbsp;
                      {(pullProgress.completed / 1e9).toFixed(2)} / {(pullProgress.total / 1e9).toFixed(2)} GB
                    </span>
                  </>
                );
              })()}
            </div>
          )}
          {pullStatus && (
            <div className={`${styles.pullStatus} ${pullStatus.startsWith("✓") ? styles.pullOk : pullStatus.startsWith("✗") ? styles.pullErr : ""}`}>
              {pullStatus}
            </div>
          )}
        </div>

        <div className={styles.settingsActions}>
          <button onClick={handleReset} disabled={saving} className={styles.resetBtn}>
            Reset to defaults
          </button>
          <button onClick={handleSave} disabled={saving} className={styles.saveBtn}>
            {saving ? "Saving…" : "Save changes"}
          </button>
        </div>

        <div className={styles.settingsFooter}>
          <p>
            Changes apply immediately to new runs.{" "}
            <a
              href="/api/training/status"
              target="_blank"
              rel="noopener noreferrer"
            >
              Training status ↗
            </a>
            {" · "}
            <a href="/api/models" target="_blank" rel="noopener noreferrer">
              Models API ↗
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}

export default function HomePage() {
  const [goal, setGoal] = useState("");
  const [runs, setRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ollamaOk, setOllamaOk] = useState<boolean | null>(null);
  const [showSettings, setShowSettings] = useState(false);

  const EXAMPLE_GOALS = [
    "Research latest AI breakthroughs",
    "Explain quantum computing simply",
    "Compare top Python web frameworks",
    "Summarize recent climate science",
  ];

  useEffect(() => {
    loadRuns();
    checkHealth();
    const interval = setInterval(loadRuns, 5000);
    return () => clearInterval(interval);
  }, []);

  async function loadRuns() {
    try {
      const data = await api.listRuns();
      setRuns(data);
    } catch {
      // ignore polling errors
    }
  }

  async function checkHealth() {
    try {
      const h = await api.health();
      setOllamaOk(h.ollama);
    } catch {
      setOllamaOk(false);
    }
  }

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    if (!goal.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const run = await api.createRun(goal.trim());
      setGoal("");
      setRuns((prev) => [run, ...prev]);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div className={styles.logo}>⬡ agentic</div>
        <div className={styles.headerRight}>
          {ollamaOk === null ? null : (
            <span className={ollamaOk ? styles.statusOk : styles.statusErr}>
              {ollamaOk ? "● Ollama connected" : "● Ollama offline"}
            </span>
          )}
          <Link href="/chat" className={styles.chatNavBtn}>
            💬 Chat
          </Link>
          <button
            className={styles.settingsBtn}
            onClick={() => setShowSettings(true)}
            title="Model settings"
          >
            ⚙ Models
          </button>
        </div>
      </header>

      {showSettings && (
        <ModelSettings onClose={() => setShowSettings(false)} />
      )}

      <main className={styles.main}>
        <section className={styles.hero}>
          <h1>Local-first AI Agent Runtime</h1>
          <p className={styles.subtitle}>
            Multi-agent, multi-model. Runs entirely on your machine.
          </p>
        </section>

        <section className={styles.createSection}>
          <form onSubmit={handleCreate} className={styles.createForm}>
            <textarea
              placeholder="What do you want the agent to research or do? e.g. 'Research the latest developments in quantum computing and summarize the key findings'"
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              rows={4}
              className={styles.goalInput}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                  handleCreate(e);
                }
              }}
            />
            {error && <div className={styles.error}>{error}</div>}
            <div className={styles.formFooter}>
              <div className={styles.suggestions}>
                {EXAMPLE_GOALS.map((eg) => (
                  <button
                    key={eg}
                    type="button"
                    className={styles.suggestion}
                    onClick={() => setGoal(eg)}
                  >
                    {eg}
                  </button>
                ))}
              </div>
              <span className={styles.kbHint}>⌘↵</span>
              <button
                type="submit"
                disabled={loading || !goal.trim()}
                className={styles.runBtn}
                title="Run agent (Ctrl+Enter / ⌘+Enter)"
              >
                {loading ? "Creating…" : "▶ Run"}
              </button>
            </div>
          </form>
        </section>

        <section className={styles.runsSection}>
          <h2>Recent Runs</h2>
          {runs.length === 0 && (
            <p className={styles.empty}>No runs yet. Enter a goal above to get started.</p>
          )}
          <div className={styles.runsList}>
            {runs.map((run) => (
              <Link key={run.id} href={`/runs/${run.id}`} className={styles.runCard}>
                <div className={styles.runCardTop}>
                  <div className={styles.runGoal}>{run.goal}</div>
                  <span className={`badge badge-${run.status}`}>{run.status}</span>
                </div>
                {run.summary && (
                  <div className={styles.runSummary}>{run.summary}</div>
                )}
                <div className={styles.runMeta}>
                  <span className={styles.runDate}>
                    {new Date(run.created_at).toLocaleString()}
                  </span>
                </div>
              </Link>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}
