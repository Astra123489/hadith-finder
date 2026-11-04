import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
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
  onOpenCitation: (citation: Citation, tag: string) => void;
}

export default function AskView({ books, initialScope, onScopeChange, onOpenCitation }: Props) {
  const [query, setQuery] = useState("");
  const [scope, setScope] = useState<number[] | null>(initialScope);
  const [careful, setCareful] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<AskResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const answerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => setScope(initialScope), [initialScope]);

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
    if (q.length < 2 || busy) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.ask(q, scope, careful);
      setResult(res);
      requestAnimationFrame(() => answerRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }));
    } catch (err) {
      setError(String(err));
      setResult(null);
    } finally {
      setBusy(false);
    }
  };

  const isNotFound = result?.answer.startsWith(NOT_FOUND_PREFIX) ?? false;

  const renderedAnswer = useMemo(() => {
    if (!result) return null;
    const parts = result.answer.split(/(\[S\d+\])/g);
    return parts.map((part, i) => {
      const m = part.match(/^\[S(\d+)\]$/);
      if (m) {
        const idx = parseInt(m[1], 10) - 1;
        if (idx < result.citations.length) {
          return (
            <button
              key={i}
              className="cite-badge"
              title={citationLabel(result.citations[idx])}
              onClick={() => {
                document.getElementById(`citation-${idx + 1}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
              }}
            >
              S{idx + 1}
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
          <button
            type="button"
            className={`toggle-pill ${careful ? "on" : ""}`}
            onClick={() => setCareful((v) => !v)}
            title="Careful mode: the model reasons step-by-step before answering. Slower, better for explanation-type questions."
          >
            {careful ? "✦ Careful (deep reasoning)" : "⚡ Fast"}
          </button>
          <span className="ask-hint"><span className="kbd">Ctrl</span> + <span className="kbd">Enter</span> to ask</span>
          <div className="spacer" />
          <button className="btn btn-primary" type="submit" disabled={busy || query.trim().length < 2 || books.length === 0}>
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

      {error && <div className="error-banner">{error}</div>}

      {result && (
        <div ref={answerRef}>
          <div className={`card answer ${isNotFound ? "notfound" : ""}`} dir="auto">
            <p>{renderedAnswer}</p>
          </div>
          {result.citations.length > 0 && (
            <div className="citations">
              {result.citations.map((c, i) => (
                <CitationCard
                  key={c.chunk_id}
                  index={i}
                  citation={c}
                  onHighlight={(cite) => onOpenCitation(cite, `S${i + 1}`)}
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
