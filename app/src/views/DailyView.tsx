import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import type { Bookmark, DailyHadith } from "../types";

interface Props {
  bookmarks: Bookmark[];
  onBookmarks: (list: Bookmark[]) => void;
}

function localDay(date = new Date()): string {
  const shifted = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
  return shifted.toISOString().slice(0, 10);
}

function shiftDay(day: string, delta: number): string {
  const date = new Date(`${day}T12:00:00`);
  date.setDate(date.getDate() + delta);
  return localDay(date);
}

function prettyDay(day: string): string {
  const date = new Date(`${day}T12:00:00`);
  return date.toLocaleDateString(undefined, { weekday: "long", day: "numeric", month: "long", year: "numeric" });
}

export default function DailyView({ bookmarks, onBookmarks }: Props) {
  const today = localDay();
  const [day, setDay] = useState(today);
  const [hadith, setHadith] = useState<DailyHadith | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  const load = useCallback((target: string) => {
    setLoading(true);
    setError(null);
    setSaved(false);
    api
      .dailyHadith(target)
      .then((h) => setHadith(h))
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load(day);
  }, [day, load]);

  const alreadySaved =
    hadith != null &&
    bookmarks.some((b) => b.kind === "canonical" && b.collection === hadith.collection && b.number === hadith.number);

  const save = async () => {
    if (!hadith) return;
    setSaving(true);
    setError(null);
    try {
      const res = await api.addBookmark({
        kind: "canonical",
        title: hadith.collection_name,
        chapter: hadith.book,
        reference: `Hadith ${hadith.number}`,
        collection: hadith.collection,
        number: hadith.number,
        page_start: null,
        page_end: null,
        excerpt: hadith.text,
        gradings: [],
        spans: {},
        book_id: null,
        chunk_id: null,
        description: note.trim() || null,
      });
      onBookmarks(res.bookmarks);
      setSaved(true);
      setNote("");
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="page">
      <h1 className="page-title">Hadith of the day</h1>
      <p className="page-sub">
        One narration a day from the five canonical collections — the same one all day, a new one tomorrow.
      </p>

      <div className="daily-nav">
        <button className="icon-btn" onClick={() => setDay(shiftDay(day, -1))} aria-label="Previous day">
          ‹
        </button>
        <div className="daily-date">
          {day === today ? "Today" : prettyDay(day)}
          <span className="daily-iso">{day}</span>
        </div>
        <button
          className="icon-btn"
          onClick={() => setDay(shiftDay(day, 1))}
          aria-label="Next day"
          disabled={day >= today}
        >
          ›
        </button>
        {day !== today && (
          <button className="btn btn-ghost" onClick={() => setDay(today)}>
            Today
          </button>
        )}
      </div>

      {error && <div className="error-banner">{error}</div>}

      {loading && !hadith && (
        <div className="daily-card card">
          <span className="spinner" /> Loading…
        </div>
      )}

      {hadith && (
        <div className="card daily-card">
          <div className="daily-meta">
            <span className="coll-badge">{hadith.short}</span>
            <span className="daily-ref">{hadith.collection_name} · Hadith {hadith.number}</span>
          </div>
          {hadith.book && <div className="daily-chapter">{hadith.book}</div>}
          <blockquote className="daily-text" dir="auto">
            {hadith.text}
          </blockquote>

          {hadith.also_in.length > 0 && (
            <div className="parallels">
              <div className="parallels-title">Same narration in other collections</div>
              {hadith.also_in.map((p) => (
                <div key={`${p.collection}-${p.number}`} className="parallel-row">
                  <div className="parallel-head">
                    <span className="parallel-label">{p.label}</span>
                    {p.book ? <span className="parallel-book">{p.book}</span> : null}
                    <span className="parallel-score">{Math.round((p.score ?? 0) * 100)}% match</span>
                  </div>
                  <div className="parallel-text">{p.snippet}</div>
                </div>
              ))}
            </div>
          )}

          <div className="daily-actions">
            <textarea
              className="bookmark-note daily-note"
              rows={2}
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="Add a note for yourself (optional)"
              aria-label="Daily hadith bookmark description"
            />
            <div className="bookmark-form-row">
              <button
                className={`btn ${saved || alreadySaved ? "btn-saved" : "btn-primary"}`}
                onClick={save}
                disabled={saving || saved || alreadySaved}
              >
                {saved || alreadySaved ? "★ Bookmarked" : saving ? "Saving…" : "★ Bookmark this hadith"}
              </button>
            </div>
          </div>

          <p className="daily-foot">
            Chosen deterministically from {hadith.collection_name}, so everyone on this date sees the same
            narration. Translations differ between printings — verify before relying on it.
          </p>
        </div>
      )}
    </div>
  );
}
