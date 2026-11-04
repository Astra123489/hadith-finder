import { useEffect, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { api } from "../api";
import type { Book } from "../types";

interface Props {
  books: Book[];
  onChanged: () => Promise<void>;
}

interface IndexingState {
  bookId: number;
  title: string;
  done: number;
  total: number;
}

export default function LibraryView({ books, onChanged }: Props) {
  const [importing, setImporting] = useState(false);
  const [indexing, setIndexing] = useState<IndexingState | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<number | null>(null);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    api.onProgress((p) => {
      if (p.op !== "reindex") return;
      setIndexing((cur) =>
        cur && cur.bookId === p.book_id ? { ...cur, done: p.done, total: p.total } : cur,
      );
    }).then((fn) => (unlisten = fn));
    return () => unlisten?.();
  }, []);

  const importPdf = async () => {
    setError(null);
    setNotice(null);
    const selected = await open({
      multiple: false,
      directory: false,
      filters: [{ name: "PDF books", extensions: ["pdf"] }],
    });
    if (typeof selected !== "string") return;

    setImporting(true);
    try {
      const result = await api.importBook(selected);
      if (result.already_exists) {
        setNotice(`“${result.title}” is already in your library.`);
      } else {
        setNotice(`“${result.title}” added — ${result.chunks} passages extracted from ${result.page_count} pages. Now building the search index…`);
        setIndexing({ bookId: result.book_id, title: result.title, done: 0, total: result.chunks });
        try {
          await api.reindexBook(result.book_id);
          setNotice(`“${result.title}” is ready to search.`);
        } catch (indexErr) {
          setError(`Indexing did not finish: ${String(indexErr)}. Text search still works, and you can retry from Settings later.`);
        } finally {
          setIndexing(null);
        }
      }
      await onChanged();
    } catch (err) {
      setError(String(err));
    } finally {
      setImporting(false);
    }
  };

  const deleteBook = async (id: number) => {
    try {
      await api.deleteBook(id);
      setConfirmDelete(null);
      await onChanged();
    } catch (err) {
      setError(String(err));
    }
  };

  const progressPct = indexing && indexing.total > 0 ? Math.round((indexing.done / indexing.total) * 100) : 0;

  return (
    <div className="page">
      <h1 className="page-title">Your library</h1>
      <p className="page-sub">
        Add your own text-based PDF books. Everything is processed locally — no file ever leaves
        this device, and we never redistribute anyone's book.
      </p>

      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 18 }}>
        <button className="btn btn-primary" onClick={importPdf} disabled={importing || indexing !== null}>
          {importing ? (<><span className="spinner" /> Reading PDF…</>) : "+ Import PDF book"}
        </button>
        <span className="ask-hint">Arabic and English books, including bilingual editions with both side by side</span>
      </div>

      {notice && <div className="error-banner" style={{ borderColor: "rgba(53,194,110,0.4)", background: "rgba(53,194,110,0.07)", color: "#b8e6c9" }}>{notice}</div>}
      {error && <div className="error-banner">{error}</div>}

      {books.length === 0 ? (
        <div className="empty-state">
          <div style={{ fontSize: 44, color: "var(--gold)" }}>▤</div>
          <h2>No books yet</h2>
          <p>
            Import a text-based PDF — for example, an Arabic edition of Ṣaḥīḥ al-Bukhārī, an English
            translation, or a bilingual volume. Scanned (image-only) PDFs cannot be read yet.
          </p>
        </div>
      ) : (
        <div className="library-list">
          {books.map((b) => {
            const isIndexing = indexing?.bookId === b.id;
            const ratio = b.chunk_count > 0 ? Math.round((b.embedded_count / b.chunk_count) * 100) : 0;
            return (
              <div key={b.id} className="card book-row">
                <div className="book-cover">
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                    <path d="M4 19V5a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v14" stroke="#d9a441" strokeWidth="1.6" strokeLinecap="round"/>
                    <path d="M4 19a2 2 0 0 0 2 2h12" stroke="#d9a441" strokeWidth="1.6" strokeLinecap="round"/>
                    <path d="M9 7h6M9 11h6" stroke="#e7dcc3" strokeWidth="1.4" strokeLinecap="round"/>
                  </svg>
                </div>
                <div className="book-meta">
                  <div className="book-title" title={b.title}>{b.title}</div>
                  <div className="book-sub">
                    {b.page_count} pages · {b.chunk_count} passages · {b.language === "mixed" ? "Arabic + English" : b.language === "ar" ? "Arabic" : b.language} · index {ratio}%
                  </div>
                  {isIndexing && (
                    <div className="progress-wrap" style={{ marginTop: 8 }}>
                      <div className="progress-bar">
                        <div className="progress-fill" style={{ width: `${progressPct}%` }} />
                      </div>
                      <div className="progress-label">Indexing {indexing.done}/{indexing.total} passages…</div>
                    </div>
                  )}
                </div>
                <div className="book-actions">
                  {confirmDelete === b.id ? (
                    <>
                      <span className="ask-hint">Remove?</span>
                      <button className="btn btn-danger" onClick={() => deleteBook(b.id)}>Yes</button>
                      <button className="btn" onClick={() => setConfirmDelete(null)}>No</button>
                    </>
                  ) : (
                    <button className="btn btn-ghost" onClick={() => setConfirmDelete(b.id)} title="Remove from library">
                      Delete
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
