"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { api, Run, ModelsResponse } from "@/lib/api";
import styles from "./page.module.css";

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
              placeholder="Describe your goal… e.g. 'Research the latest news about AI and summarize it'"
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              rows={3}
              className={styles.goalInput}
            />
            {error && <div className={styles.error}>{error}</div>}
            <button type="submit" disabled={loading || !goal.trim()}>
              {loading ? "Creating…" : "▶ Run Agent"}
            </button>
          </form>
        </section>

        <section className={styles.runsSection}>
          <h2>Runs</h2>
          {runs.length === 0 && (
            <p className={styles.empty}>No runs yet. Create one above.</p>
          )}
          <div className={styles.runsList}>
            {runs.map((run) => (
              <Link key={run.id} href={`/runs/${run.id}`} className={styles.runCard}>
                <div className={styles.runGoal}>{run.goal}</div>
                {run.summary && (
                  <div className={styles.runSummary}>{run.summary.slice(0, 160)}…</div>
                )}
                <div className={styles.runMeta}>
                  <span className={`badge badge-${run.status}`}>{run.status}</span>
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
