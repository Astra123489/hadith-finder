import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import type { LlmOption } from "../api";
import type { AskResult, Book, Citation } from "../types";
import CitationCard, { citationLabel } from "../components/CitationCard";

const EXAMPLES = [
  "Explain the virtue of intentions in one's deeds",
  "What does the Prophet say about mercy to others?",
  "Find hadiths about feeding the poor",
  "Summarize the chapter on intentions",
];

const NOT_FOUND_PREFIX = "I could not find this in the loaded books.";

interface Props {
  books: Book[];
  initialScope: number[] | null;
  onScopeChange: (ids: number[] | null) => void;
  onOpenCitation: (citation: Citation) => void;
}

export default function AskView({ books, initialScope, onScopeChange, onOpenCitation }: Props) {
  const [query, setQuery] = useState("");
  const [scope, setScope] = useState<number[] | null>(initialScope);
  const [llms, setLlms] = useState<LlmOption[]>([]);
  const [modelPath, setModelPath] = useState("");
  const [modelBusy, setModelBusy] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<AskResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [streamed, setStreamed] = useState("");
  const [thinking, setThinking] = useState("");
  const answerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const askIdRef = useRef<string | null>(null);

  useEffect(() => setScope(initialScope), [initialScope]);

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
      if (p.op !== "token") return;
      if (!askIdRef.current || p.ask_id !== askIdRef.current) return;
      if (p.phase === "thinking") setThinking((cur) => cur + p.text);
      else setStreamed((cur) => cur + p.text);
    }).then((fn) => (unlisten = fn));
    return () => unlisten?.();
  }, []);

  const noneSelected = Array.isArray(scope) && scope.length === 0 && books.length > 0;

  const scopeLabel = useMemo(() => {
    if (!scope || scope.length === books.length) return "all loaded books";
    if (scope.length === 0) return "no books selected";
    return scope.length === 1
      ? books.find((b) => b.id === scope[0])?.title ?? "1 book"
      : `${scope.length} books selected`;
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

  const submit = async (e?: FormEvent) => {
    e?.preventDefault();
    const q = query.trim();
    if (q.length < 2 || busy || noneSelected) return;
    const askId =
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID()
        : `ask-${Date.now()}`;
    askIdRef.current = askId;
    setBusy(true);
    setError(null);
    setResult(null);
    setStreamed("");
    setThinking("");
    try {
      const res = await api.ask(q, scope, askId);
      setResult(res);
      requestAnimationFrame(() => answerRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }));
    } catch (err) {
      setError(String(err));
      setResult(null);
    } finally {
      askIdRef.current = null;
      setBusy(false);
      setStreamed("");
      setThinking("");
    }
  };

  const isNotFound = result?.answer.startsWith(NOT_FOUND_PREFIX) ?? false;

  const renderedAnswer = useMemo(() => {
    if (!result) return null;
    const parts = result.answer.split(/(\[S\d+\])/g);
    return parts.map((part, i) => {
      const m = part.match(/^\[S(\d+)\]$/);
      if (m) {
        const num = parseInt(m[1], 10);
        const cite = result.citations.find((c) => c.source_index === num);
        if (cite) {
          return (
            <button
              key={i}
              className="cite-badge"
              title={citationLabel(cite)}
              onClick={() => {
                document.getElementById(`citation-${num}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
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
  }, [result]);

  return (
    <div className="page">
      <h1 className="page-title">Ask the books</h1>
      <p className="page-sub">
        Question the library in plain language — the answer is grounded only in the sources you
        loaded, with every claim tied to a highlighted passage.
      </p>

      <form className="card ask-form" onSubmit={submit}>
        <textarea
          ref={inputRef}
          className="ask-input"
          dir="auto"
          placeholder="e.g. Explain what the Prophet ﷺ taught about mercy…"
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
          <span className="ask-hint"><span className="kbd">Ctrl</span> + <span className="kbd">Enter</span> to ask</span>
          <div className="spacer" />
          <button
            className="btn btn-primary"
            type="submit"
            disabled={busy || query.trim().length < 2 || books.length === 0 || noneSelected}
          >
            {busy ? (<><span className="spinner" /> Thinking…</>) : "Ask →"}
          </button>
        </div>
        <div className="scope-row">
          <span>Searching: <strong style={{ color: "var(--text)" }}>{scopeLabel}</strong></span>
          {books.length > 0 && (
            <>
              <button className="scope-chip" type="button" onClick={selectAll}>All</button>
              {books.map((b) => {
                const on = scope === null || scope.includes(b.id);
                return (
                  <button key={b.id} type="button" className={`scope-chip ${on ? "on" : ""}`} onClick={() => toggleScope(b.id)} title={b.title}>
                    {b.title}
                  </button>
                );
              })}
            </>
          )}
        </div>
      </form>

      {books.length === 0 && !result && (
        <div className="empty-state">
          <div style={{ fontSize: 44, color: "var(--gold)" }}>☾</div>
          <h2>Your library is empty</h2>
          <p>Import a text-based hadith PDF from the Library tab — everything stays on this device.</p>
        </div>
      )}

      {noneSelected && !result && (
        <div className="error-banner">Select at least one book to search.</div>
      )}

      {error && <div className="error-banner">{error}</div>}

      {busy && (streamed || thinking) && (
        <div className="card answer streaming" dir="auto">
          {thinking && (
            <details className="thinking-box" open={!streamed}>
              <summary>{streamed ? "Reasoning" : "Thinking…"}</summary>
              <p className="thinking-text">
                {thinking}
                {!streamed && <span className="cursor">▍</span>}
              </p>
            </details>
          )}
          {streamed && (
            <p>
              {streamed}
              <span className="cursor">▍</span>
            </p>
          )}
          <div className="stream-note">Still writing — sources appear when the answer is finished.</div>
        </div>
      )}

      {result && (
        <div ref={answerRef}>
          <div className={`card answer ${isNotFound ? "notfound" : ""}`} dir="auto">
            <p>{renderedAnswer}</p>
          </div>
          {result.unverified && !isNotFound && (
            <div className="error-banner">
              The model answered without citing a passage. Treat this answer as unsourced until you
              find it in the text yourself.
            </div>
          )}
          {result.citations.length > 0 && (
            <div className="citations">
              {result.citations.map((c) => (
                <CitationCard
                  key={c.chunk_id}
                  citation={c}
                  onHighlight={(cite) => onOpenCitation(cite)}
                />
              ))}
            </div>
          )}
        </div>
      )}

      {!result && books.length > 0 && (
        <div className="examples">
          {EXAMPLES.map((ex) => (
            <button key={ex} className="example-chip" onClick={() => { setQuery(ex); inputRef.current?.focus(); }}>
              {ex}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
