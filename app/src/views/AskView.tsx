import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { api } from "../api";
import type { LlmOption } from "../api";
import type { Book, ChatSummary, Citation, ThreadState, ThreadTurn, ThreadUsage } from "../types";
import CitationCard, { citationLabel } from "../components/CitationCard";

const EXAMPLES = [
  "Explain the virtue of intentions in one's deeds",
  "What does the Prophet say about mercy to others?",
  "Find hadiths about feeding the poor",
  "Summarize the chapter on intentions",
];

const NOT_FOUND_PREFIX = "I could not find this in the loaded books.";
const RING = 2 * Math.PI * 12;

function ChatIcon() {
  return (
    <svg className="chat-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M4.8 4.5h14.4a1.8 1.8 0 0 1 1.8 1.8v9a1.8 1.8 0 0 1-1.8 1.8H10l-4.4 3.2v-3.2H4.8A1.8 1.8 0 0 1 3 15.3v-9a1.8 1.8 0 0 1 1.8-1.8Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function formatTokens(n: number): string {
  if (n >= 10000) return `${Math.round(n / 1000)}k`;
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

interface Props {
  books: Book[];
  initialScope: number[] | null;
  onScopeChange: (ids: number[] | null) => void;
  onOpenCitation: (citation: Citation) => void;
  onBookmark?: (citation: Citation, description: string) => Promise<void>;
}

export default function AskView({ books, initialScope, onScopeChange, onOpenCitation, onBookmark }: Props) {
  const [query, setQuery] = useState("");
  const [scope, setScope] = useState<number[] | null>(initialScope);
  const [llms, setLlms] = useState<LlmOption[]>([]);
  const [modelPath, setModelPath] = useState("");
  const [modelBusy, setModelBusy] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [streamed, setStreamed] = useState("");
  const [thinking, setThinking] = useState("");
  // The conversation, owned by the sidecar: every answer returns it whole.
  const [turns, setTurns] = useState<ThreadTurn[]>([]);
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [summary, setSummary] = useState("");
  const [usage, setUsage] = useState<ThreadUsage | null>(null);
  const [pending, setPending] = useState<string | null>(null);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [suggestBusy, setSuggestBusy] = useState(false);
  const [compacting, setCompacting] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [showCtx, setShowCtx] = useState(false);
  // The sidebar renders after this view mounts, so the portal target is looked
  // up once and re-rendered into when the tab is revisited.
  const [sidebarSlot, setSidebarSlot] = useState<HTMLElement | null>(null);
  const [menu, setMenu] = useState<{ id: string; x: number; y: number } | null>(null);
  const [scopeOpen, setScopeOpen] = useState(false);
  const scopeRef = useRef<HTMLDivElement>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const renameInputRef = useRef<HTMLInputElement>(null);
  const renameCancelled = useRef(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const ctxRef = useRef<HTMLDivElement>(null);
  const askIdRef = useRef<string | null>(null);

  useEffect(() => setScope(initialScope), [initialScope]);

  useEffect(() => {
    setSidebarSlot(document.getElementById("chat-slot"));
  }, []);

  const loadModels = useCallback(async () => {
    try {
      const list = await api.listLlms();
      setLlms(list.llms);
      setModelPath(list.llms.find((m) => m.selected)?.path ?? list.active);
    } catch {
      /* the model picker is optional: answering still works without it */
    }
  }, []);

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  const pickModel = async (option: LlmOption) => {
    setModelBusy(true);
    setError(null);
    try {
      await api.setLlm(option.default ? null : option.path);
      await loadModels();
    } catch (e) {
      setError(String(e));
      await loadModels();
    } finally {
      setModelBusy(false);
    }
  };

  // Live tokens from the sidecar: the answer appears as it is written instead
  // of after the whole (often multi-minute) CPU generation finishes.
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    api.onProgress((p) => {
      if (p.op === "token") {
        if (!askIdRef.current || p.ask_id !== askIdRef.current) return;
        if (p.phase === "thinking") setThinking((cur) => cur + p.text);
        else setStreamed((cur) => cur + p.text);
        return;
      }
      if (p.op === "thread") {
        if (p.stage === "compacting") {
          setCompacting(true);
          return;
        }
        setCompacting(false);
        if (p.stage === "compacted") {
          const n = p.dropped ?? 0;
          setNotice(`Summarised ${n} earlier ${n === 1 ? "turn" : "turns"} to free context.`);
        }
      }
    }).then((fn) => (unlisten = fn));
    return () => unlisten?.();
  }, []);

  // The context menu closes on any click, scroll or Escape.
  useEffect(() => {
    if (!menu) return;
    const close = () => setMenu(null);
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    window.addEventListener("mousedown", close);
    window.addEventListener("scroll", close, true);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", close);
      window.removeEventListener("scroll", close, true);
      window.removeEventListener("keydown", onKey);
    };
  }, [menu]);

  // The book picker closes on an outside click or Escape.
  useEffect(() => {
    if (!scopeOpen) return;
    const onDown = (e: MouseEvent) => {
      if (!scopeRef.current?.contains(e.target as Node)) setScopeOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setScopeOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [scopeOpen]);

  // The context panel closes when you click anywhere else.
  useEffect(() => {
    if (!showCtx) return;
    const onDown = (e: MouseEvent) => {
      if (!ctxRef.current?.contains(e.target as Node)) setShowCtx(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [showCtx]);

  const noneSelected = Array.isArray(scope) && scope.length === 0 && books.length > 0;

  const scopeLabel = useMemo(() => {
    if (!scope || scope.length === books.length) return "All books";
    if (scope.length === 0) return "No books";
    return scope.length === 1
      ? books.find((b) => b.id === scope[0])?.title ?? "1 book"
      : `${scope.length} of ${books.length}`;
  }, [scope, books]);

  const toggleScope = (id: number) => {
    setScope((prev) => {
      const base = prev ?? books.map((b) => b.id);
      const next = base.includes(id) ? base.filter((x) => x !== id) : [...base, id];
      const normalized = next.length === books.length ? null : next;
      onScopeChange(normalized);
      return normalized;
    });
  };

  const selectAll = () => {
    setScope(null);
    onScopeChange(null);
  };

  const applyThread = useCallback((thread: ThreadState | undefined) => {
    if (!thread) return;
    setTurns(thread.turns);
    setSummary(thread.summary);
    setUsage(thread.usage);
  }, []);

  // The conversation outlives this view: coming back from another tab shows
  // the same thread instead of an empty page.
  useEffect(() => {
    let alive = true;
    api
      .threadState()
      .then((res) => {
        if (!alive) return;
        setChats(res.chats);
        if (res.thread.turns.length > 0) applyThread(res.thread);
      })
      .catch(() => {
        /* a fresh sidecar simply has nothing to restore */
      });
    return () => {
      alive = false;
    };
  }, [applyThread]);

  const scrollToEnd = () => requestAnimationFrame(() => endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }));

  const loadSuggestions = useCallback(async () => {
    setSuggestBusy(true);
    try {
      setSuggestions(await api.suggestFollowups(3));
    } catch {
      setSuggestions([]);
    } finally {
      setSuggestBusy(false);
    }
  }, []);

  const submit = async (e?: FormEvent, preset?: string) => {
    e?.preventDefault();
    const q = (preset ?? query).trim();
    if (q.length < 2 || busy || noneSelected) return;
    const askId =
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID()
        : `ask-${Date.now()}`;
    askIdRef.current = askId;
    setBusy(true);
    setError(null);
    setPending(q);
    setStreamed("");
    setThinking("");
    setSuggestions([]);
    setNotice(null);
    setQuery("");
    scrollToEnd();
    try {
      // new_thread stays false: this question joins the conversation.
      const res = await api.ask(q, scope, askId, false);
      applyThread(res.thread);
      if (res.chats) setChats(res.chats);
      setPending(null);
      scrollToEnd();
      void loadSuggestions();
    } catch (err) {
      setError(String(err));
      setPending(null);
      setQuery(q);
    } finally {
      askIdRef.current = null;
      setBusy(false);
      setStreamed("");
      setThinking("");
    }
  };

  /** Start a new conversation; an untouched one is reused by the sidecar. */
  const startNewChat = async () => {
    if (busy) return;
    setError(null);
    setNotice(null);
    setSuggestions([]);
    setPending(null);
    setStreamed("");
    setThinking("");
    setQuery("");
    setShowCtx(false);
    try {
      const res = await api.newChat();
      applyThread(res.thread);
      setChats(res.chats);
    } catch (e) {
      setError(String(e));
    }
    inputRef.current?.focus();
  };

  const openChatItem = async (id: string) => {
    if (busy || id === activeChatId) return;
    setError(null);
    setNotice(null);
    setSuggestions([]);
    setPending(null);
    setStreamed("");
    setThinking("");
    setQuery("");
    setShowCtx(false);
    try {
      const res = await api.openChat(id);
      applyThread(res.thread);
      setChats(res.chats);
    } catch (e) {
      setError(String(e));
    }
  };

  const startRename = (id: string) => {
    const chat = chats.find((c) => c.id === id);
    renameCancelled.current = false;
    setMenu(null);
    setRenamingId(id);
    setRenameValue(chat?.title ?? "");
  };

  // Once the field is on screen, put the caret in it with the name selected,
  // so typing replaces the old title.
  useEffect(() => {
    if (renamingId) renameInputRef.current?.select();
  }, [renamingId]);

  const commitRename = async () => {
    const id = renamingId;
    if (!id) return;
    const title = renameValue.trim();
    setRenamingId(null);
    if (!title) return; // an empty name keeps the old one
    try {
      const res = await api.renameChat(id, title);
      setChats(res.chats);
    } catch (e) {
      setError(String(e));
    }
  };

  const removeChat = async (id: string) => {
    if (busy) return;
    try {
      const res = await api.deleteChat(id);
      setChats(res.chats);
      applyThread(res.thread);
      setSuggestions([]);
      setNotice(null);
    } catch (e) {
      setError(String(e));
    }
  };

  const compactNow = async () => {
    setCompacting(true);
    setError(null);
    try {
      const res = await api.compactThread();
      applyThread(res.thread);
      setNotice(
        res.compacted
          ? `Summarised ${res.dropped} earlier ${res.dropped === 1 ? "turn" : "turns"} to free context.`
          : "Nothing older to summarise yet — compaction keeps your two most recent turns.",
      );
    } catch (e) {
      setError(String(e));
    } finally {
      setCompacting(false);
    }
  };

  const renderedAnswer = (answer: string, citations: Citation[], prefix: string) => {
    const parts = answer.split(/(\[S\d+\])/g);
    return parts.map((part, i) => {
      const m = part.match(/^\[S(\d+)\]$/);
      if (m) {
        const num = parseInt(m[1], 10);
        const cite = citations.find((c) => c.source_index === num);
        if (cite) {
          return (
            <button
              key={i}
              className="cite-badge"
              title={citationLabel(cite)}
              onClick={() => {
                document.getElementById(`${prefix}-${num}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
              }}
            >
              S{num}
            </button>
          );
        }
        return <span key={i} className="cite-badge">{part}</span>;
      }
      return <span key={i}>{part}</span>;
    });
  };

  const pct = usage?.pct ?? 0;
  const level = pct >= 0.9 ? "hot" : pct >= 0.7 ? "warm" : "";
  const activeChatId = chats.find((c) => c.active)?.id;

  return (
    <div className="page ask-page">
      <div className="ask-head">
        <div>
          <h1 className="page-title">Ask the books</h1>
          <p className="page-sub">
            Question the library in plain language, then follow up — the conversation keeps its place
            and every claim stays tied to a highlighted passage.
          </p>
        </div>
      </div>

      <div className="thread">
        {summary && (
          <div className="thread-summary">
            <div className="thread-summary-label">Earlier turns, summarised</div>
            <p>{summary}</p>
          </div>
        )}

        {turns.map((turn, index) => {
          const notFound = turn.answer.startsWith(NOT_FOUND_PREFIX);
          const prefix = `citation-t${turn.id}`;
          const latest = index === turns.length - 1;
          return (
            <div className="turn" key={turn.id}>
              <div className="turn-q" dir="auto">{turn.question}</div>
              <div className={`card answer ${notFound ? "notfound" : ""}`} dir="auto">
                <div className="answer-label">Answer</div>
                <p>{renderedAnswer(turn.answer, turn.citations, prefix)}</p>
              </div>
              {turn.unverified && !notFound && (
                <div className="error-banner">
                  The model answered without citing a passage. Treat this answer as unsourced until you
                  find it in the text yourself.
                </div>
              )}
              {turn.citations.length > 0 && (
                <div className="sources-block">
                  <div className="sources-head">
                    Sources <span className="sources-count">{turn.citations.length}</span>
                  </div>
                  <div className="citations">
                    {turn.citations.map((c) => (
                      <CitationCard
                        key={c.chunk_id}
                        idPrefix={prefix}
                        citation={c}
                        onHighlight={onOpenCitation}
                        onBookmark={onBookmark}
                      />
                    ))}
                  </div>
                </div>
              )}

              {latest && (suggestBusy || suggestions.length > 0) && (
                <div className="suggest-row">
                  <span className="suggest-label">
                    {suggestBusy ? "Looking for gaps to fill…" : "Keep digging"}
                  </span>
                  {!suggestBusy &&
                    suggestions.map((s) => (
                      <button
                        key={s}
                        className="suggest-chip"
                        onClick={() => {
                          setQuery(s);
                          inputRef.current?.focus();
                        }}
                        title="Put this question in the box"
                      >
                        {s}
                      </button>
                    ))}
                </div>
              )}
            </div>
          );
        })}

        {busy && pending && (
          <div className="turn">
            <div className="turn-q" dir="auto">{pending}</div>
            <div className="card answer streaming" dir="auto">
              <div className="answer-label">Answer</div>
              {thinking && (
                <details className="thinking-box" open={!streamed}>
                  <summary>{streamed ? "Reasoning" : "Thinking…"}</summary>
                  <p className="thinking-text">
                    {thinking}
                    {!streamed && <span className="cursor">▍</span>}
                  </p>
                </details>
              )}
              {streamed ? (
                <p>
                  {streamed}
                  <span className="cursor">▍</span>
                </p>
              ) : (
                !thinking && (
                  <p className="stream-wait">
                    <span className="spinner" /> Searching the books and writing…
                  </p>
                )
              )}
              <div className="stream-note">Still writing — sources appear when the answer is finished.</div>
            </div>
          </div>
        )}

        {(compacting || notice) && (
          <div className={`thread-note ${compacting ? "busy" : ""}`}>
            {compacting ? (
              <>
                <span className="spinner" /> Compacting older turns…
              </>
            ) : (
              <>
                {notice}
                <button className="thread-note-close" onClick={() => setNotice(null)} aria-label="Dismiss">
                  ×
                </button>
              </>
            )}
          </div>
        )}

        {turns.length === 0 && !busy && books.length === 0 && (
          <div className="empty-state">
            <div style={{ fontSize: 44, color: "var(--gold)" }}>☾</div>
            <h2>Your library is empty</h2>
            <p>Import a text-based hadith PDF from the Library tab — everything stays on this device.</p>
          </div>
        )}

        {turns.length === 0 && !busy && books.length > 0 && (
          <div className="ask-intro">
            <h2>Ask anything about your books</h2>
            <p>
              Every answer stays tied to the passage it came from — open a source to see it lit up in the
              page it was printed on.
            </p>
            <div className="examples">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  className="example-chip"
                  onClick={() => {
                    setQuery(ex);
                    inputRef.current?.focus();
                  }}
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}

        {noneSelected && turns.length === 0 && (
          <div className="error-banner">Select at least one book to search.</div>
        )}
        {error && <div className="error-banner">{error}</div>}
        <div ref={endRef} />
      </div>

      <div className="thread-composer">
        <form className="card ask-form" onSubmit={submit}>
          <textarea
            ref={inputRef}
            className="ask-input"
            dir="auto"
            placeholder={
              turns.length > 0
                ? "Ask a follow-up — e.g. which collection grades this narration?"
                : "e.g. Explain what the Prophet ﷺ taught about mercy…"
            }
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submit();
            }}
          />
          <div className="ask-row">
            {llms.length > 0 && (
              <label className="model-pick" title="The local model that writes the answer">
                <span>Model</span>
                <select
                  className="model-select"
                  value={modelPath}
                  disabled={busy || modelBusy}
                  onChange={(e) => {
                    const next = llms.find((m) => m.path === e.target.value);
                    if (next) pickModel(next);
                  }}
                >
                  {llms.map((m) => (
                    <option key={m.path} value={m.path} disabled={!m.installed}>
                      {m.name}
                      {m.installed ? "" : " (not downloaded)"}
                    </option>
                  ))}
                </select>
              </label>
            )}
            {books.length > 0 && (
              <div className="scope-wrap" ref={scopeRef}>
                <button
                  type="button"
                  className="scope-chip scope-pick"
                  onClick={() => setScopeOpen((v) => !v)}
                  disabled={busy}
                  aria-expanded={scopeOpen}
                  aria-haspopup="true"
                  title="Which books to search"
                >
                  <strong>{scopeLabel}</strong>
                </button>
                {scopeOpen && (
                  <div className="scope-menu" role="menu" aria-label="Books to search">
                    <button
                      type="button"
                      role="menuitemradio"
                      aria-checked={scope === null}
                      className={`scope-opt ${scope === null ? "on" : ""}`}
                      onClick={() => {
                        selectAll();
                        setScopeOpen(false);
                      }}
                    >
                      <span className="scope-tick">{scope === null ? "✓" : ""}</span> All loaded books
                    </button>
                    {books.map((b) => {
                      const on = scope === null || scope.includes(b.id);
                      return (
                        <button
                          key={b.id}
                          type="button"
                          role="menuitemcheckbox"
                          aria-checked={on}
                          className={`scope-opt ${on ? "on" : ""}`}
                          onClick={() => toggleScope(b.id)}
                          title={b.title}
                        >
                          <span className="scope-tick">{on ? "✓" : ""}</span>
                          <span className="scope-opt-name">{b.title}</span>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
            <span className="ask-hint"><span className="kbd">Ctrl</span> + <span className="kbd">Enter</span> to ask</span>
            <div className="spacer" />

            <div className="ctx-wrap" ref={ctxRef}>
              <button
                type="button"
                className={`ctx-btn ${level}`}
                onClick={() => setShowCtx((v) => !v)}
                title="Context window"
                aria-expanded={showCtx}
              >
                <svg width="22" height="22" viewBox="0 0 30 30" aria-hidden="true">
                  <circle cx="15" cy="15" r="12" className="ctx-ring-bg" />
                  <circle
                    cx="15"
                    cy="15"
                    r="12"
                    className="ctx-ring-fill"
                    strokeDasharray={`${RING * Math.min(pct, 1)} ${RING}`}
                  />
                </svg>
                <span className="ctx-pct">{Math.round(pct * 100)}%</span>
              </button>

              {showCtx && (
                <div className="ctx-pop" role="dialog" aria-label="Context window">
                  {usage ? (
                    <>
                      <div className="ctx-line ctx-title">
                        <span>Context</span>
                        <strong>{formatTokens(usage.used)}</strong>
                      </div>
                      <div className="ctx-line">
                        <span>Compacts at <strong>{formatTokens(usage.compact_at)}</strong></span>
                        <button
                          className="ctx-compact"
                          onClick={compactNow}
                          disabled={compacting || busy || usage.turns <= 2}
                          title={
                            usage.turns <= 2
                              ? "Compact keeps your two most recent turns; there is nothing older yet."
                              : "Summarise older turns now"
                          }
                        >
                          {compacting ? "Compacting…" : "Compact"}
                        </button>
                      </div>
                      <div className="ctx-divider" />
                      <div className="ctx-line ctx-title">
                        <span>Thread usage</span>
                        <strong>{usage.turns} {usage.turns === 1 ? "turn" : "turns"}</strong>
                      </div>
                      <div className="ctx-line"><span>Prompt</span><span>{formatTokens(usage.used)}</span></div>
                      <div className="ctx-line"><span>Excerpts</span><span>{formatTokens(usage.excerpt_tokens)}</span></div>
                      <div className="ctx-line"><span>Conversation</span><span>{formatTokens(usage.history_tokens)}</span></div>
                      <div className="ctx-line"><span>Window</span><span>{formatTokens(usage.ctx)}</span></div>
                      <div className="ctx-line"><span>Summaries</span><span>{usage.compactions}</span></div>
                      <p className="ctx-note">
                        Older turns are summarised automatically at {Math.round(usage.compact_at / usage.ctx * 100)}% of
                        the window. The window size is set in Settings → Answer model.
                      </p>
                    </>
                  ) : (
                    <p className="ctx-note">Ask a question to start filling the context window.</p>
                  )}
                </div>
              )}
            </div>

            <button
              className="btn btn-primary"
              type="submit"
              disabled={busy || query.trim().length < 2 || books.length === 0 || noneSelected}
            >
              {busy ? (<><span className="spinner" /> Thinking…</>) : turns.length > 0 ? "Follow up →" : "Ask →"}
            </button>
          </div>
        </form>
      </div>

      {sidebarSlot &&
        createPortal(
          <>
            <div className="chat-slot-head">
              <span className="chat-slot-title">Conversations</span>
              <button
                className="chat-new"
                onClick={startNewChat}
                disabled={busy}
                title="New conversation"
                aria-label="New conversation"
              >
                +
              </button>
            </div>
            <div className="chat-list">
              {chats.length === 0 && (
                <div className="chat-empty">No conversations yet. Ask a question to start one.</div>
              )}
              {chats.map((chat) => (
                <div
                  key={chat.id}
                  className={`chat-item ${chat.active ? "active" : ""} ${renamingId === chat.id ? "renaming" : ""}`}
                  onContextMenu={(e) => {
                    e.preventDefault();
                    setMenu({ id: chat.id, x: e.clientX, y: e.clientY });
                  }}
                >
                  {renamingId === chat.id ? (
                    <>
                      <ChatIcon />
                      <input
                        ref={renameInputRef}
                        className="chat-rename"
                        value={renameValue}
                        onChange={(e) => setRenameValue(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") {
                            e.preventDefault();
                            commitRename();
                          } else if (e.key === "Escape") {
                            e.preventDefault();
                            renameCancelled.current = true;
                            setRenamingId(null);
                          }
                        }}
                        onBlur={() => {
                          const cancelled = renameCancelled.current;
                          renameCancelled.current = false;
                          if (!cancelled) commitRename();
                        }}
                        aria-label="Conversation name"
                      />
                    </>
                  ) : (
                    <button
                      className="chat-open"
                      onClick={() => openChatItem(chat.id)}
                      disabled={busy || chat.active}
                      title={chat.title || "New conversation"}
                    >
                      <ChatIcon />
                      <span className="chat-name">{chat.title || "New conversation"}</span>
                    </button>
                  )}
                </div>
              ))}
            </div>
          </>,
          sidebarSlot,
        )}

      {menu &&
        createPortal(
          <div
            className="chat-menu"
            role="menu"
            style={{
              left: Math.min(menu.x, window.innerWidth - 172),
              top: Math.min(menu.y, window.innerHeight - 96),
            }}
            onMouseDown={(e) => e.stopPropagation()}
            onContextMenu={(e) => e.preventDefault()}
          >
            <button role="menuitem" onClick={() => startRename(menu.id)}>
              Rename
            </button>
            <button
              role="menuitem"
              className="danger"
              onClick={() => {
                const id = menu.id;
                setMenu(null);
                void removeChat(id);
              }}
            >
              Delete
            </button>
          </div>,
          document.body,
        )}
    </div>
  );
}
