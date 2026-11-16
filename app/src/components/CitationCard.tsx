import { useState } from "react";
import type { Citation, Parallel } from "../types";

const GRADING_CLASS: Record<string, string> = {
  sahih: "g-sahih",
  hasan: "g-hasan",
  daif: "g-daif",
  weak: "g-weak",
  mawdu: "g-mawdu",
  munkar: "g-munkar",
  mursal: "g-ungraded",
  mawquf: "g-ungraded",
};

export function citationLabel(c: Citation): string {
  const parts = [c.book_title];
  if (c.chapter) parts.push(c.chapter);
  if (c.hadith_ref) parts.push(`Hadith ${c.hadith_ref}`);
  parts.push(
    c.page_start === c.page_end
      ? `p. ${c.page_start + 1}`
      : `pp. ${c.page_start + 1}–${c.page_end + 1}`,
  );
  return parts.join(" · ");
}

interface Props {
  citation: Citation;
  onHighlight: (c: Citation) => void;
  onBookmark?: (c: Citation, description: string) => Promise<void>;
  /** Keeps anchor ids unique when several answers share a page. */
  idPrefix?: string;
}

function ParallelRow({ p }: { p: Parallel }) {
  return (
    <div className="parallel-row">
      <div className="parallel-head">
        <span className="parallel-label">{p.label}</span>
        {p.book ? <span className="parallel-book">{p.book}</span> : null}
        <span className="parallel-score">{Math.round(p.score * 100)}% match</span>
      </div>
      <div className="parallel-text">{p.snippet}</div>
    </div>
  );
}

export default function CitationCard({ citation: c, onHighlight, onBookmark, idPrefix = "citation" }: Props) {
  const n = c.source_index;
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [showForm, setShowForm] = useState(false);
  const [note, setNote] = useState("");
  const [error, setError] = useState<string | null>(null);

  const save = async () => {
    if (!onBookmark) return;
    setSaving(true);
    setError(null);
    try {
      await onBookmark(c, note);
      setSaved(true);
      setShowForm(false);
      setNote("");
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="card citation" id={`${idPrefix}-${n}`}>
      <div className="citation-id">Source S{n}</div>
      <div className="citation-loc">{citationLabel(c)}</div>
      <div className="citation-excerpt">{c.excerpt}</div>
      <div className="grading-chips">
        {c.gradings.length === 0 ? (
          <span className="grading g-ungraded">no grading printed in this source</span>
        ) : (
          c.gradings.map((g) => (
            <span key={g} className={`grading ${GRADING_CLASS[g] ?? "g-ungraded"}`}>
              {g}
            </span>
          ))
        )}
      </div>
      {c.also_in.length > 0 && (
        <div className="parallels">
          <div className="parallels-title">Same narration in other collections</div>
          {c.also_in.map((p) => (
            <ParallelRow key={`${p.collection}-${p.number}`} p={p} />
          ))}
        </div>
      )}

      {showForm && (
        <div className="bookmark-form">
          <textarea
            className="bookmark-note"
            rows={2}
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="Why keep this one? (optional)"
            aria-label="Bookmark description"
          />
          <div className="bookmark-form-row">
            <button className="btn btn-primary" onClick={save} disabled={saving}>
              {saving ? "Saving…" : "Save bookmark"}
            </button>
            <button className="btn btn-ghost" onClick={() => setShowForm(false)} disabled={saving}>
              Cancel
            </button>
          </div>
        </div>
      )}
      {error && <div className="error-banner">{error}</div>}

      <div className="citation-actions">
        <button className="btn" onClick={() => onHighlight(c)}>
          ✎ Highlight in book · p. {c.page_start + 1}
        </button>
        {onBookmark && (
          <button
            className={`btn ${saved ? "btn-saved" : "btn-ghost"}`}
            onClick={() => setShowForm((v) => !v)}
            disabled={saved}
          >
            {saved ? "★ Bookmarked" : "★ Bookmark"}
          </button>
        )}
      </div>
    </div>
  );
}
