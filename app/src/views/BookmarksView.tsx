import { useState } from "react";
import { api } from "../api";
import type { Bookmark } from "../types";

interface Props {
  bookmarks: Bookmark[];
  onChanged: (list: Bookmark[]) => void;
  onOpen: (bookmark: Bookmark) => void;
  onOpenAsk: () => void;
}

function savedOn(value: string): string {
  const iso = value.includes("T") ? value : value.replace(" ", "T") + "Z";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

export default function BookmarksView({ bookmarks, onChanged, onOpen, onOpenAsk }: Props) {
  const [editing, setEditing] = useState<number | null>(null);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const remove = async (bookmark: Bookmark) => {
    setBusy(bookmark.id);
    setError(null);
    try {
      const res = await api.deleteBookmark(bookmark.id);
      onChanged(res.bookmarks);
      if (editing === bookmark.id) setEditing(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(null);
    }
  };

  const saveNote = async (bookmark: Bookmark) => {
    setBusy(bookmark.id);
    setError(null);
    try {
      const res = await api.updateBookmark(bookmark.id, draft.trim() || null);
      onChanged(res.bookmarks);
      setEditing(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="page">
      <h1 className="page-title">Bookmarks</h1>
      <p className="page-sub">
        Narrations you kept while reading, with the note you wrote on why — so the one you need is easy to
        find again.
      </p>

      {error && <div className="error-banner">{error}</div>}

      {bookmarks.length === 0 ? (
        <div className="empty-state">
          <div style={{ fontSize: 44, color: "var(--gold)" }}>★</div>
          <h2>Nothing bookmarked yet</h2>
          <p>
            Ask a question, then press <strong>Bookmark</strong> on any source card to keep that narration
            and describe it in your own words. Today's hadith can be saved too.
          </p>
          <button className="btn btn-primary" style={{ marginTop: 16 }} onClick={onOpenAsk}>
            Ask the books →
          </button>
        </div>
      ) : (
        <>
          <div className="bookmarks-summary">
            {bookmarks.length} saved narration{bookmarks.length === 1 ? "" : "s"}
          </div>
          <div className="bookmark-list">
            {bookmarks.map((b) => {
              const canOpen = b.kind === "library" && b.book_id != null;
              const gradings = Array.isArray(b.gradings) ? b.gradings : [];
              return (
                <div key={b.id} className="card bookmark-card">
                  <div className="bookmark-head">
                    <span className="bookmark-title">{b.title}</span>
                    {b.reference && <span className="bookmark-ref">{b.reference}</span>}
                    {b.chapter && <span className="bookmark-chapter">{b.chapter}</span>}
                  </div>

                  <p className="bookmark-text" dir="auto">
                    {b.excerpt}
                  </p>

                  {gradings.length > 0 && (
                    <div className="grading-chips">
                      {gradings.map((g) => (
                        <span key={g} className="grading g-ungraded">{g}</span>
                      ))}
                    </div>
                  )}

                  {editing === b.id ? (
                    <div className="bookmark-form">
                      <textarea
                        className="bookmark-note"
                        rows={3}
                        value={draft}
                        onChange={(e) => setDraft(e.target.value)}
                        placeholder="What should this remind you of?"
                        aria-label="Bookmark description"
                        autoFocus
                      />
                      <div className="bookmark-form-row">
                        <button className="btn btn-primary" onClick={() => saveNote(b)} disabled={busy === b.id}>
                          {busy === b.id ? "Saving…" : "Save note"}
                        </button>
                        <button className="btn btn-ghost" onClick={() => setEditing(null)} disabled={busy === b.id}>
                          Cancel
                        </button>
                      </div>
                    </div>
                  ) : b.description ? (
                    <div className="bookmark-desc">
                      <span className="bookmark-desc-label">Your note</span>
                      {b.description}
                    </div>
                  ) : (
                    <div className="bookmark-desc empty">No note yet.</div>
                  )}

                  <div className="bookmark-actions">
                    {canOpen && (
                      <button className="btn" onClick={() => onOpen(b)}>
                        ✎ Open in book · p. {(b.page_start ?? 0) + 1}
                      </button>
                    )}
                    <button
                      className="btn btn-ghost"
                      onClick={() => {
                        setEditing(b.id);
                        setDraft(b.description ?? "");
                      }}
                    >
                      {b.description ? "Edit note" : "Add note"}
                    </button>
                    <button className="btn btn-danger" onClick={() => remove(b)} disabled={busy === b.id}>
                      Remove
                    </button>
                    <span className="bookmark-date">Saved {savedOn(b.created_at)}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
