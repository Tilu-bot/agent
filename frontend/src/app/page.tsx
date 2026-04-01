"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { api, Run } from "@/lib/api";
import styles from "./page.module.css";

export default function HomePage() {
  const [goal, setGoal] = useState("");
  const [runs, setRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ollamaOk, setOllamaOk] = useState<boolean | null>(null);

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
        <div className={styles.status}>
          {ollamaOk === null ? null : (
            <span className={ollamaOk ? styles.statusOk : styles.statusErr}>
              {ollamaOk ? "● Ollama connected" : "● Ollama offline"}
            </span>
          )}
        </div>
      </header>

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
              placeholder="Describe your goal… e.g. 'Research the latest news about AI and summarise it'"
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
