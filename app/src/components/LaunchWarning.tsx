import { useEffect, useRef } from "react";

/**
 * Shown on every launch, deliberately with no "don't show again".
 *
 * The app is built to keep answers tied to passages, but a confident answer is
 * still only a draft of what the books say — so the first thing every session
 * sees is how to check it.
 */
export default function LaunchWarning({ onClose }: { onClose: () => void }) {
  // Focus the dialog itself rather than the button: the button would otherwise
  // open wearing a focus ring, which reads as a heavy border on every launch.
  const cardRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    cardRef.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      className="warning-overlay"
      role="dialog"
      aria-modal="true"
      aria-labelledby="launch-warning-title"
      onClick={onClose}
    >
      <div
        className="warning-card"
        ref={cardRef}
        tabIndex={-1}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="warning-head">
          <div className="warning-mark" aria-hidden="true">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
              <path
                d="M12 8.6v5.2M12 17.2v.2"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
              />
              <path
                d="M10.6 3.9 2.9 17.4a1.6 1.6 0 0 0 1.4 2.4h15.4a1.6 1.6 0 0 0 1.4-2.4L13.4 3.9a1.6 1.6 0 0 0-2.8 0Z"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinejoin="round"
              />
            </svg>
          </div>
          <h2 className="warning-title" id="launch-warning-title">
            AI can be wrong. The book is the proof.
          </h2>
        </div>
        <p className="warning-lead">
          Hadith Finder writes its answers with an AI model. It can be mistaken — fluently, and in
          detail — so an answer is a starting point, never a verdict.
        </p>
        <ul className="warning-list">
          <li>
            Every claim is tied to a passage. Open <strong>Highlight in book</strong> on the source
            card and read that page yourself before you rely on it.
          </li>
          <li>
            A grading (<em>sahih</em>, <em>hasan</em>, <em>da‘if</em>) appears only when the printed
            book states it. No grade means the app has no verdict — not that the narration is sound.
          </li>
          <li>
            “Same narration in other collections” is a similarity match, not a scholarly
            cross-reference. Verify it against the printed text.
          </li>
          <li>
            For anything touching belief or practice, ask a qualified scholar.
          </li>
        </ul>
        <button className="btn btn-primary" onClick={onClose}>
          I’ll check the sources
        </button>
      </div>
    </div>
  );
}
