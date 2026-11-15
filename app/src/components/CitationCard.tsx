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

export default function CitationCard({ citation: c, onHighlight }: Props) {
  const n = c.source_index;
  return (
    <div className="card citation" id={`citation-${n}`}>
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
      <div>
        <button className="btn" onClick={() => onHighlight(c)}>
          ✎ Highlight in book · p. {c.page_start + 1}
        </button>
      </div>
    </div>
  );
}
