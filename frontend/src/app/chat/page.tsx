"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import Link from "next/link";
import { api, ChatMessage, Run } from "@/lib/api";
import styles from "./chat.module.css";

const EXAMPLES = [
  "Explain how transformers work in AI",
  "Write a Python script to sort a list",
  "What is QLoRA fine-tuning?",
  "Compare GPT-4 and Claude",
];

const DEFAULT_SYSTEM =
  "You are a helpful AI assistant built into the Agentic runtime. " +
  "Answer clearly and concisely.";

// ── Inline markdown helpers ───────────────────────────────────────────────────

/** Render inline formatting: **bold**, *italic*, `code` */
function renderInline(text: string): React.ReactNode {
  const parts = text.split(/(\*\*[^*\n]+\*\*|\*[^*\n]+\*|`[^`\n]+`)/g);
  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith("**") && part.endsWith("**") && part.length > 4)
          return <strong key={i}>{part.slice(2, -2)}</strong>;
        if (part.startsWith("*") && part.endsWith("*") && part.length > 2)
          return <em key={i}>{part.slice(1, -1)}</em>;
        if (part.startsWith("`") && part.endsWith("`") && part.length > 2)
          return <code key={i} className={styles.inlineCode}>{part.slice(1, -1)}</code>;
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}

/** Render a GitHub-Flavored Markdown pipe table, or null if not a table. */
function renderTable(block: string): React.ReactNode | null {
  const lines = block.split("\n").filter(Boolean);
  if (lines.length < 3 || !lines[0].includes("|")) return null;
  const isSep = (l: string) => /^[\|\s\-:]+$/.test(l);
  if (!isSep(lines[1])) return null;
  const parseCells = (l: string) =>
    l.replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
  const headers = parseCells(lines[0]);
  const rows = lines.slice(2).map(parseCells);
  return (
    <div className={styles.tableWrap}>
      <table className={styles.mdTable}>
        <thead>
          <tr>{headers.map((h, i) => <th key={i}>{renderInline(h)}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri}>
              {row.map((cell, ci) => <td key={ci}>{renderInline(cell)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Fenced code block with language label and copy button. */
function CodeBlock({ lang, code }: { lang: string; code: string }) {
  const [copied, setCopied] = useState(false);
  function handleCopy() {
    navigator.clipboard?.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }
  return (
    <div className={styles.codeBlock}>
      <div className={styles.codeHeader}>
        {lang && <span className={styles.codeLang}>{lang}</span>}
        <button className={styles.copyBtn} onClick={handleCopy}>
          {copied ? "✓ copied" : "copy"}
        </button>
      </div>
      <pre className={styles.codePre}><code>{code}</code></pre>
    </div>
  );
}

/** Render a single non-code block (paragraph, list, or table). */
function renderBlock(block: string, key: string | number): React.ReactNode {
  const tableEl = renderTable(block);
  if (tableEl) return <div key={key}>{tableEl}</div>;

  const lines = block.split("\n").filter(Boolean);
  const isBullet = lines.every((l) => /^[\*\-]\s/.test(l));
  const isNumbered = lines.every((l) => /^\d+\.\s/.test(l));

  if (isBullet)
    return (
      <ul key={key}>
        {lines.map((l, j) => <li key={j}>{renderInline(l.replace(/^[\*\-]\s+/, ""))}</li>)}
      </ul>
    );
  if (isNumbered)
    return (
      <ol key={key}>
        {lines.map((l, j) => <li key={j}>{renderInline(l.replace(/^\d+\.\s+/, ""))}</li>)}
      </ol>
    );
  return <p key={key}>{renderInline(block)}</p>;
}

/** Full markdown renderer for chat bubbles. */
function BubbleContent({ text }: { text: string }) {
  const segments = text.split(/(```[\s\S]*?```)/g);
  return (
    <div className={styles.bubbleContent}>
      {segments.map((seg, i) => {
        if (seg.startsWith("```")) {
          const firstNl = seg.indexOf("\n");
          const lang = firstNl >= 0 ? seg.slice(3, firstNl).trim() : "";
          const body =
            firstNl >= 0
              ? seg.slice(firstNl + 1).replace(/```$/, "")
              : seg.replace(/^```[^\n]*\n?/, "").replace(/```$/, "");
          return <CodeBlock key={i} lang={lang} code={body} />;
        }
        const blocks = seg.split(/\n{2,}/).filter(Boolean);
        return blocks.map((block, j) => renderBlock(block, `${i}-${j}`));
      })}
    </div>
  );
}

// ── Thinking panel ───────────────────────────────────────────────────────────

const MAX_THINKING_STEP_LENGTH = 200;

let _thinkEvtId = 0;
function nextThinkId() {
  return String(++_thinkEvtId);
}

const KIND_ICONS: Record<string, string> = {
  thinking: "💭",
  agent_message: "🤖",
  tool_call: "🔧",
  tool_result: "✓",
  plan_created: "📋",
  verification: "✅",
  reflexion: "🔄",
  synthesis: "✨",
  critic: "🎯",
  error: "❌",
  task_started: "▶",
  task_completed: "✅",
};

interface ThinkingEvent {
  id: string;
  kind: string;
  agent: string;
  content: string;
}

function ThinkingPanel({
  events,
  isActive,
}: {
  events: ThinkingEvent[];
  isActive: boolean;
}) {
  const [expanded, setExpanded] = useState(true);

  useEffect(() => {
    if (isActive) setExpanded(true);
  }, [isActive]);

  if (events.length === 0 && !isActive) return null;

  return (
    <div className={styles.thinkingPanel}>
      <button
        className={styles.thinkingToggle}
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
      >
        <span className={styles.thinkingToggleIcon}>
          {isActive ? <span className={styles.thinkingSpinner} /> : "💭"}
        </span>
        <span className={styles.thinkingToggleLabel}>
          {isActive
            ? `Thinking… (${events.length} step${events.length !== 1 ? "s" : ""})`
            : `Reasoning (${events.length} step${events.length !== 1 ? "s" : ""})`}
        </span>
        <span className={styles.thinkingChevron}>{expanded ? "▲" : "▼"}</span>
      </button>
      {expanded && (
        <div className={styles.thinkingSteps}>
          {events.map((ev) => (
            <div key={ev.id} className={styles.thinkingStep}>
              <span className={styles.thinkingStepIcon}>
                {KIND_ICONS[ev.kind] ?? "•"}
              </span>
              <div className={styles.thinkingStepBody}>
                <span className={styles.thinkingStepAgent}>{ev.agent || ev.kind}</span>
                {ev.content && (
                  <span className={styles.thinkingStepText}>
                    {ev.content.length > MAX_THINKING_STEP_LENGTH
                      ? ev.content.slice(0, MAX_THINKING_STEP_LENGTH) + "…"
                      : ev.content}
                  </span>
                )}
              </div>
            </div>
          ))}
          {isActive && (
            <div className={`${styles.thinkingStep} ${styles.thinkingStepLive}`}>
              <span className={styles.thinkingStepIcon}>
                <span className={styles.liveDot} />
              </span>
              <span className={styles.thinkingStepText}>Processing…</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Agent run inline bubble ───────────────────────────────────────────────────

interface AgentRunState {
  runId: string;
  status: string;
  synthesis: string;
  events: ThinkingEvent[];
}

function AgentRunBubble({ state }: { state: AgentRunState }) {
  const isActive = state.status === "pending" || state.status === "running";
  return (
    <div className={styles.agentRunBubble}>
      <div className={styles.agentRunHeader}>
        <span className={styles.agentRunIcon}>🤖</span>
        <span className={styles.agentRunLabel}>Agent task</span>
        <span className={`badge badge-${state.status}`}>{state.status}</span>
        <Link href={`/runs/${state.runId}`} className={styles.agentRunLink}>
          View details →
        </Link>
      </div>
      <ThinkingPanel events={state.events} isActive={isActive} />
      {state.synthesis ? (
        <div className={styles.agentSynthesis}>
          <BubbleContent text={state.synthesis} />
        </div>
      ) : isActive && state.events.length === 0 ? (
        <div className={styles.agentRunThinking}>
          <div className={styles.dot} />
          <div className={styles.dot} />
          <div className={styles.dot} />
          <span>Working…</span>
        </div>
      ) : null}
    </div>
  );
}

// ── Message types ─────────────────────────────────────────────────────────────

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
  agentRun?: AgentRunState;
  modelInfo?: { model: string; taskType: string };
}

let _msgId = 0;
function nextId() {
  return String(++_msgId);
}

// ── Chat page ─────────────────────────────────────────────────────────────────

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [system, setSystem] = useState(DEFAULT_SYSTEM);
  const [model, setModel] = useState("");
  const [sending, setSending] = useState(false);
  const [routeMode, setRouteMode] = useState<"direct" | "agentic" | null>(null);
  const [routeReason, setRouteReason] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-resize textarea
  function resizeInput() {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }

  // Subscribe to an agent run's SSE stream and update the inline bubble.
  const trackRun = useCallback(
    (runId: string, assistantId: string) => {
      const es = new EventSource(api.streamUrl(runId));

      es.onmessage = (e: MessageEvent) => {
        try {
          const msg = JSON.parse(e.data as string) as {
            type: string;
            payload: Record<string, unknown>;
          };
          if (msg.type === "run_update") {
            const status = String(msg.payload.status ?? "");
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId && m.agentRun
                  ? { ...m, agentRun: { ...m.agentRun, status } }
                  : m
              )
            );
          } else if (msg.type === "event") {
            const payload = msg.payload as {
              id?: string;
              kind: string;
              agent_role?: string | null;
              content: string | null;
            };
            // Add every non-synthesis event to the thinking panel.
            if (payload.kind !== "synthesis") {
              const ev: ThinkingEvent = {
                id: payload.id ?? nextThinkId(),
                kind: payload.kind,
                // agent_role from the API event maps to the display label in ThinkingEvent.agent
                agent: payload.agent_role ?? payload.kind,
                content: payload.content ?? "",
              };
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId && m.agentRun
                    ? {
                        ...m,
                        agentRun: {
                          ...m.agentRun,
                          events: [...m.agentRun.events, ev],
                        },
                      }
                    : m
                )
              );
            }
            // Capture synthesis as the final answer.
            if (payload.kind === "synthesis" && payload.content) {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId && m.agentRun
                    ? {
                        ...m,
                        agentRun: {
                          ...m.agentRun,
                          synthesis: payload.content ?? "",
                        },
                      }
                    : m
                )
              );
            }
          } else if (msg.type === "done") {
            es.close();
            setSending(false);
          }
        } catch {
          /* ignore parse errors */
        }
      };

      es.onerror = () => {
        es.close();
        setSending(false);
      };
    },
    []
  );

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || sending) return;

    setInput("");
    setSending(true);
    setError(null);
    setRouteMode(null);
    setRouteReason("");
    if (inputRef.current) inputRef.current.style.height = "auto";

    // Build the full message list for the API
    const userMsg: Message = { id: nextId(), role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);

    // Build API messages — include system prompt + full conversation history.
    // Agent run messages contribute their synthesis so follow-up turns have context.
    const apiMessages: ChatMessage[] = [];
    if (system.trim()) {
      apiMessages.push({ role: "system", content: system.trim() });
    }
    for (const m of messages) {
      if (m.agentRun) {
        if (m.agentRun.synthesis) {
          apiMessages.push({ role: m.role, content: m.agentRun.synthesis });
        }
      } else if (m.content) {
        apiMessages.push({ role: m.role, content: m.content });
      }
    }
    apiMessages.push({ role: "user", content: text });

    // Classify the query
    let mode: "direct" | "agentic" = "direct";
    try {
      const cls = await api.classifyChat(apiMessages);
      mode = cls.mode;
      setRouteReason(cls.reason ?? "");
    } catch {
      // Classify failed → fall back to direct
    }
    setRouteMode(mode);

    const assistantId = nextId();

    if (mode === "agentic") {
      // Create an agent run and track it inline
      try {
        const run: Run = await api.createRun(text, model.trim() ? { reasoning: model.trim(), fast: model.trim() } : undefined);
        setMessages((prev) => [
          ...prev,
          {
            id: assistantId,
            role: "assistant",
            content: "",
            agentRun: {
              runId: run.id,
              status: run.status,
              synthesis: "",
              events: [],
            },
          },
        ]);
        trackRun(run.id, assistantId);
      } catch (e) {
        setError(String(e));
        setSending(false);
      }
      return;
    }

    // Direct streaming chat
    const controller = new AbortController();
    abortRef.current = controller;

    setMessages((prev) => [
      ...prev,
      { id: assistantId, role: "assistant", content: "", streaming: true },
    ]);

    let accumulated = "";

    await api.chatStream(
      apiMessages,
      (token) => {
        accumulated += token;
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, content: accumulated } : m
          )
        );
      },
      () => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, streaming: false } : m
          )
        );
        setSending(false);
        abortRef.current = null;
      },
      (err) => {
        setError(err);
        setMessages((prev) => prev.filter((m) => m.id !== assistantId));
        setSending(false);
        abortRef.current = null;
      },
      model.trim() || undefined,
      "fast",
      controller.signal,
      (modelName, taskType) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, modelInfo: { model: modelName, taskType } }
              : m
          )
        );
      }
    );
  }, [input, sending, system, messages, model, trackRun]);

  function handleCancel() {
    abortRef.current?.abort();
    abortRef.current = null;
    setSending(false);
    // Remove the incomplete streaming bubble
    setMessages((prev) => prev.filter((m) => !m.streaming));
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function handleExample(ex: string) {
    setInput(ex);
    inputRef.current?.focus();
  }

  function handleNewChat() {
    handleCancel();
    setMessages([]);
    setError(null);
    setInput("");
    setRouteMode(null);
    setRouteReason("");
    inputRef.current?.focus();
  }

  return (
    <div className={styles.page}>
      {/* ── Top bar ──────────────────────────────────────────────────────── */}
      <div className={styles.topbar}>
        <Link href="/" className={styles.logo}>⬡ agentic</Link>
        <div className={styles.divider} />
        <span className={styles.topbarTitle}>Chat</span>
        <div className={styles.topbarSpacer} />
        <div className={styles.modelSelector}>
          <span className={styles.modelLabel}>Model</span>
          <input
            className={styles.modelInput}
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder="auto (fast slot) or org/model-name"
            title="Ollama model (e.g. llama3.2:3b) or HuggingFace ID (e.g. Qwen/Qwen2.5-3B-Instruct)"
          />
        </div>
        <button className={styles.newChatBtn} onClick={handleNewChat}>
          ✦ New chat
        </button>
      </div>

      {/* ── System prompt ─────────────────────────────────────────────────── */}
      <div className={styles.systemBar}>
        <span className={styles.systemLabel}>System</span>
        <textarea
          className={styles.systemInput}
          value={system}
          onChange={(e) => setSystem(e.target.value)}
          placeholder="Optional system prompt…"
          rows={1}
        />
      </div>

      {/* ── Message thread ────────────────────────────────────────────────── */}
      <div className={styles.messageArea}>
        {messages.length === 0 && (
          <div className={styles.emptyState}>
            <span className={styles.emptyIcon}>✦</span>
            <div className={styles.emptyTitle}>Start a conversation</div>
            <div className={styles.emptyDesc}>
              Ask anything — simple questions get instant answers, complex tasks
              automatically run as agent jobs.
            </div>
            <div className={styles.exampleGrid}>
              {EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  className={styles.exampleChip}
                  onClick={() => handleExample(ex)}
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`${styles.messageRow} ${msg.role === "user" ? styles.user : ""}`}
          >
            <div
              className={`${styles.avatar} ${
                msg.role === "assistant" ? styles.avatarAssistant : styles.avatarUser
              }`}
            >
              {msg.role === "assistant" ? "A" : "U"}
            </div>
            <div
              className={`${styles.bubble} ${
                msg.role === "assistant" ? styles.bubbleAssistant : styles.bubbleUser
              }`}
            >
              {msg.role === "assistant" ? (
                <>
                  {msg.modelInfo && !msg.agentRun && (
                    <div className={styles.modelInfoTag}>
                      ✦ {msg.modelInfo.model}
                      <span className={styles.modelInfoSlot}>{msg.modelInfo.taskType}</span>
                    </div>
                  )}
                  {msg.agentRun ? (
                    <AgentRunBubble state={msg.agentRun} />
                  ) : msg.streaming && msg.content === "" ? (
                    <div className={styles.typingDots}>
                      <div className={styles.dot} />
                      <div className={styles.dot} />
                      <div className={styles.dot} />
                    </div>
                  ) : (
                    <BubbleContent text={msg.content} />
                  )}
                </>
              ) : (
                <BubbleContent text={msg.content} />
              )}
            </div>
          </div>
        ))}

        <div ref={bottomRef} />
      </div>

      {/* ── Input bar ─────────────────────────────────────────────────────── */}
      <div className={styles.inputBar}>
        {error && (
          <div className={styles.errorBanner}>
            {error}
            <button className={styles.errorClose} onClick={() => setError(null)}>
              ✕
            </button>
          </div>
        )}
        <div className={styles.inputWrap}>
          <textarea
            ref={inputRef}
            className={styles.chatInput}
            value={input}
            onChange={(e) => { setInput(e.target.value); resizeInput(); }}
            onKeyDown={handleKeyDown}
            placeholder="Message… (Enter to send, Shift+Enter for new line)"
            rows={1}
            disabled={sending}
            aria-keyshortcuts="Enter"
          />
          {sending ? (
            <button
              className={styles.cancelBtn}
              onClick={handleCancel}
              title="Stop generation"
              aria-label="Stop generation"
            >
              ■
            </button>
          ) : (
            <button
              className={styles.sendBtn}
              onClick={handleSend}
              disabled={!input.trim()}
              title="Send message"
              aria-label="Send message"
            >
              ▶
            </button>
          )}
        </div>
        <div className={styles.inputHint}>
          {routeMode && (
            <span
              className={`${styles.routeTag} ${styles[`route_${routeMode}`]}`}
              title={routeReason}
            >
              {routeMode === "agentic" ? "🤖 agent task" : "💬 direct answer"}
            </span>
          )}
          {routeReason && (
            <span className={styles.routeReason}>{routeReason}</span>
          )}
          {!routeReason && (
            <>
              {model && model.includes("/")
                ? `Using HuggingFace: ${model}`
                : model
                ? `Using Ollama: ${model}`
                : "Auto model from config"}{" "}
              · Enter ↵ to send
            </>
          )}
        </div>
      </div>
    </div>
  );
}

