import { useEffect, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { api } from "../api";
import type { Book } from "../types";

interface Props {
  books: Book[];
  onChanged: () => Promise<void>;
}

interface WorkState {
  bookId: number;
  stage: "reading" | "indexing" | "rebuilding";
  done: number;
  total: number;
}

export default function LibraryView({ books, onChanged }: Props) {
  const [importing, setImporting] = useState(false);
  const [work, setWork] = useState<WorkState | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<number | null>(null);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    api.onProgress((p) => {
      if (p.op === "reindex") {
        setWork((cur) =>
          cur && cur.bookId === p.book_id
            ? { ...cur, stage: "indexing", done: p.done, total: p.total }
            : cur,
        );
      } else if (p.op === "import") {
        setWork((cur) =>
          cur
            ? {
                ...cur,
                stage: p.stage === "extract" ? "reading" : "indexing",
                done: p.done ?? p.page ?? 0,
                total: p.total ?? p.pages ?? 0,
              }
            : cur,
        );
      }
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
        const ocrNote = result.ocr_pages ? ` (${result.ocr_pages} page(s) re-read with OCR)` : "";
        setNotice(`“${result.title}” added — ${result.chunks} narrations from ${result.page_count} pages${ocrNote}. Now building the search index…`);
        setWork({ bookId: result.book_id, stage: "indexing", done: 0, total: result.chunks });
        try {
          await api.reindexBook(result.book_id);
          setNotice(`“${result.title}” is ready to search.`);
        } catch (indexErr) {
          setError(`Indexing did not finish: ${String(indexErr)}. Text search still works — press “Finish index” on the book to retry.`);
        } finally {
          setWork(null);
        }
      }
      await onChanged();
    } catch (err) {
      setError(String(err));
    } finally {
      setImporting(false);
    }
  };

  const finishIndex = async (book: Book) => {
    setError(null);
    setWork({ bookId: book.id, stage: "indexing", done: book.embedded_count, total: book.chunk_count });
    try {
      const res = await api.reindexBook(book.id);
      setNotice(`“${book.title}” index is complete (${res.embedded} passages embedded).`);
    } catch (err) {
      setError(String(err));
    } finally {
      setWork(null);
      await onChanged();
    }
  };

  const rebuild = async (book: Book) => {
    setError(null);
    setNotice(null);
    setWork({ bookId: book.id, stage: "rebuilding", done: 0, total: book.page_count });
    try {
      const res = await api.rebuildBook(book.id);
      setNotice(`“${book.title}” re-read: ${res.chunks} narrations from ${res.page_count} pages${res.ocr_pages ? ` (${res.ocr_pages} OCR page(s))` : ""}. Rebuilding the index…`);
      setWork({ bookId: book.id, stage: "indexing", done: 0, total: res.chunks });
      await api.reindexBook(book.id);
      setNotice(`“${book.title}” is rebuilt and searchable.`);
    } catch (err) {
      setError(`Rebuild failed: ${String(err)}`);
    } finally {
      setWork(null);
      await onChanged();
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

  const pct = work && work.total > 0 ? Math.round((work.done / work.total) * 100) : 0;
  const busy = importing || work !== null;

  return (
    <div className="page">
      <h1 className="page-title">Your library</h1>
      <p className="page-sub">
        Add your own text-based PDF books. Everything is processed locally — no file ever leaves
        this device, and we never redistribute anyone's book.
      </p>

      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 18 }}>
        <button className="btn btn-primary" onClick={importPdf} disabled={busy}>
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
            translation, or a bilingual volume. Scanned pages are re-read with OCR when it is
            installed.
          </p>
        </div>
      ) : (
        <div className="library-list">
          {books.map((b) => {
            const isWorking = work?.bookId === b.id;
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
                    {b.page_count} pages · {b.chunk_count} narrations · {b.language === "mixed" ? "Arabic + English" : b.language === "ar" ? "Arabic" : b.language} · index {ratio}%
                  </div>
                  {isWorking && (
                    <div className="progress-wrap" style={{ marginTop: 8 }}>
                      <div className="progress-bar">
                        <div className="progress-fill" style={{ width: `${pct}%` }} />
                      </div>
                      <div className="progress-label">
                        {work.stage === "reading" || work.stage === "rebuilding"
                          ? `Reading pages ${work.done}/${work.total}…`
                          : `Indexing ${work.done}/${work.total} narrations…`}
                      </div>
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
                    <>
                      {b.embedded_count < b.chunk_count && (
                        <button className="btn" onClick={() => finishIndex(b)} disabled={busy} title="Embed the passages that are missing from the index">
                          Finish index
                        </button>
                      )}
                      <button className="btn btn-ghost" onClick={() => rebuild(b)} disabled={busy} title="Re-read the PDF with the current extractor and rebuild passages, then reindex">
                        Rebuild
                      </button>
                      <button className="btn btn-ghost" onClick={() => setConfirmDelete(b.id)} disabled={busy} title="Remove from library">
                        Delete
                      </button>
                    </>
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
