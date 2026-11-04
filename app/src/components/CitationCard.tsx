import type { Citation } from "../types";

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
  index: number; // 0-based; displayed as S1, S2…
  citation: Citation;
  onHighlight: (c: Citation) => void;
}

export default function CitationCard({ index, citation: c, onHighlight }: Props) {
  return (
    <div className="card citation" id={`citation-${index + 1}`}>
      <div className="citation-id">Source S{index + 1}</div>
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
      <div>
        <button className="btn" onClick={() => onHighlight(c)}>
          ✎ Highlight in book · p. {c.page_start + 1}
        </button>
      </div>
    </div>
  );
}
