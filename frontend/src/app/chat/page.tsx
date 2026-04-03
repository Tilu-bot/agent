"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import Link from "next/link";
import { api, ChatMessage } from "@/lib/api";
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

/** Very lightweight markdown renderer for chat bubbles. */
function BubbleContent({ text }: { text: string }) {
  // Split on code fences first, then paragraph-level formatting
  const segments = text.split(/(```[\s\S]*?```)/g);
  return (
    <div className={styles.bubbleContent}>
      {segments.map((seg, i) => {
        if (seg.startsWith("```")) {
          const body = seg.replace(/^```[^\n]*\n?/, "").replace(/```$/, "");
          return <pre key={i}><code>{body}</code></pre>;
        }
        const blocks = seg.split(/\n{2,}/).filter(Boolean);
        return blocks.map((block, j) => {
          const lines = block.split("\n").filter(Boolean);
          const isBullet = lines.every((l) => /^[\*\-]\s/.test(l));
          const isNumbered = lines.every((l) => /^\d+\.\s/.test(l));
          if (isBullet) {
            return (
              <ul key={`${i}-${j}`}>
                {lines.map((l, k) => <li key={k}>{l.replace(/^[\*\-]\s+/, "")}</li>)}
              </ul>
            );
          }
          if (isNumbered) {
            return (
              <ol key={`${i}-${j}`}>
                {lines.map((l, k) => <li key={k}>{l.replace(/^\d+\.\s+/, "")}</li>)}
              </ol>
            );
          }
          return <p key={`${i}-${j}`}>{block}</p>;
        });
      })}
    </div>
  );
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
}

let _msgId = 0;
function nextId() {
  return String(++_msgId);
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [system, setSystem] = useState(DEFAULT_SYSTEM);
  const [model, setModel] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

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

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || sending) return;

    setInput("");
    setSending(true);
    setError(null);
    if (inputRef.current) inputRef.current.style.height = "auto";

    // Build the full message list for the API
    const userMsg: Message = { id: nextId(), role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);

    // Placeholder streaming message
    const assistantId = nextId();
    setMessages((prev) => [
      ...prev,
      { id: assistantId, role: "assistant", content: "", streaming: true },
    ]);

    // Build API messages (include system prompt if set)
    const apiMessages: ChatMessage[] = [];
    if (system.trim()) {
      apiMessages.push({ role: "system", content: system.trim() });
    }
    // Include full conversation history
    for (const m of messages) {
      apiMessages.push({ role: m.role, content: m.content });
    }
    apiMessages.push({ role: "user", content: text });

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
        // Done
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, streaming: false } : m
          )
        );
        setSending(false);
      },
      (err) => {
        setError(err);
        setMessages((prev) => prev.filter((m) => m.id !== assistantId));
        setSending(false);
      },
      model.trim() || undefined
    );
  }, [input, sending, system, messages, model]);

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
    setMessages([]);
    setError(null);
    setInput("");
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
              Ask anything — this chat talks directly to the model you select
              above. For multi-step tasks, use the{" "}
              <Link href="/">goal runner</Link> instead.
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
                  {msg.streaming && msg.content === "" ? (
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
          />
          <button
            className={styles.sendBtn}
            onClick={handleSend}
            disabled={sending || !input.trim()}
            title="Send message"
            aria-label="Send message"
          >
            ▶
          </button>
        </div>
        <div className={styles.inputHint}>
          Enter ↵ to send · Shift+Enter for new line ·{" "}
          {model && model.includes("/")
            ? `Using HuggingFace model: ${model}`
            : model
            ? `Using Ollama model: ${model}`
            : "Auto model from config"}
        </div>
      </div>
    </div>
  );
}
