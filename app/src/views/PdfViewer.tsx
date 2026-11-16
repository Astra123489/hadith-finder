import { useCallback, useEffect, useRef, useState } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { getDocument, GlobalWorkerOptions, type PDFDocumentProxy } from "pdfjs-dist";
import { isTauri } from "../api";
import type { SpanMap } from "../types";

GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).toString();

interface Props {
  filePath: string;
  title: string;
  initialPage: number; // 0-based
  spans: SpanMap;
  sourceTag?: string;
  onClose: () => void;
}

export default function PdfViewer({ filePath, title, initialPage, spans, sourceTag, onClose }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const [pdf, setPdf] = useState<PDFDocumentProxy | null>(null);
  const [page, setPage] = useState(initialPage);
  const [scale, setScale] = useState(1.35);
  const [numPages, setNumPages] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const url = isTauri() ? convertFileSrc(filePath) : "/demo-fixture.pdf";
    let cancelled = false;
    const task = getDocument({ url });
    task.promise
      .then((doc) => {
        if (cancelled) return;
        setError(null);
        setPdf(doc);
        setNumPages(doc.numPages);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(String(e?.message ?? e));
      });
    return () => {
      cancelled = true;
      task.destroy();
    };
  }, [filePath]);

  useEffect(() => {
    if (!pdf || !canvasRef.current) return;
    let cancelled = false;
    const renderPage = async () => {
      const pageObj = await pdf.getPage(page + 1);
      if (cancelled) return;
      const viewport = pageObj.getViewport({ scale });
      const canvas = canvasRef.current!;
      const ctx = canvas.getContext("2d")!;
      canvas.width = Math.floor(viewport.width);
      canvas.height = Math.floor(viewport.height);
      canvas.style.width = `${viewport.width}px`;
      canvas.style.height = `${viewport.height}px`;
      await pageObj.render({ canvasContext: ctx, viewport }).promise;
      if (cancelled) return;
      if (stageRef.current && overlayRef.current) {
        stageRef.current.style.width = `${viewport.width}px`;
        stageRef.current.style.height = `${viewport.height}px`;
        overlayRef.current.style.width = `${viewport.width}px`;
        overlayRef.current.style.height = `${viewport.height}px`;
      }
      // paint highlights for this page — PyMuPDF rects are top-left origin,
      // same as CSS, so no y-flip is needed (only scale).
      const layer = overlayRef.current;
      if (!layer) return;
      layer.innerHTML = "";
      const rects = spans[String(page)] ?? [];
      for (const [x0, y0, x1, y1] of rects) {
        const el = document.createElement("div");
        el.className = "hl";
        el.style.left = `${x0 * scale}px`;
        el.style.top = `${y0 * scale}px`;
        el.style.width = `${Math.max(4, (x1 - x0) * scale)}px`;
        el.style.height = `${Math.max(4, (y1 - y0) * scale)}px`;
        layer.appendChild(el);
      }
    };
    renderPage().catch((e) => setError(String(e?.message ?? e)));
    return () => {
      cancelled = true;
    };
  }, [pdf, page, scale, spans]);

  const go = useCallback((delta: number) => {
    setPage((p) => Math.min(Math.max(0, p + delta), Math.max(0, numPages - 1)));
  }, [numPages]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
        return;
      }
      // Arrows belong to the page field while it has focus.
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) return;
      if (e.key === "ArrowRight") go(1);
      else if (e.key === "ArrowLeft") go(-1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go, onClose]);

  const zoom = (delta: number) =>
    setScale((s) => Math.min(3, Math.max(0.7, +(s + delta).toFixed(2))));

  // A citation can span pages; show where the other highlighted pages are so
  // the reader does not have to page through the book hunting for them.
  const highlightedPages = Object.keys(spans)
    .map((k) => parseInt(k, 10))
    .filter((n) => Number.isFinite(n) && n !== page)
    .sort((a, b) => a - b);

  return (
    <div className="viewer-overlay" role="dialog" aria-modal="true" aria-label={title}>
      <div className="viewer-toolbar">
        <button
          className="viewer-close"
          onClick={onClose}
          title="Close (Esc)"
          aria-label="Close the book"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M6 6l12 12M18 6L6 18" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
          </svg>
        </button>

        <div className="viewer-heading">
          <div className="viewer-title" title={title}>
            {title}
          </div>
          {sourceTag && <span className="viewer-tag">source {sourceTag}</span>}
          {highlightedPages.map((p) => (
            <button
              key={p}
              className="viewer-jump"
              onClick={() => setPage(p)}
              title={`This citation continues on page ${p + 1}`}
            >
              also p. {p + 1}
            </button>
          ))}
        </div>

        <div className="spacer" />

        <div className="viewer-seg" role="group" aria-label="Page navigation">
          <button
            className="seg-btn"
            onClick={() => go(-1)}
            title="Previous page (←)"
            aria-label="Previous page"
            disabled={page <= 0}
          >
            ‹
          </button>
          <div className="seg-pager">
            <input
              className="viewer-page"
              type="number"
              min={1}
              max={numPages}
              value={page + 1}
              aria-label="Page number"
              onChange={(e) => {
                const n = parseInt(e.target.value, 10);
                if (!Number.isNaN(n)) setPage(Math.min(Math.max(0, n - 1), numPages - 1));
              }}
            />
            <span className="viewer-total">/ {numPages || "…"}</span>
          </div>
          <button
            className="seg-btn"
            onClick={() => go(1)}
            title="Next page (→)"
            aria-label="Next page"
            disabled={numPages > 0 && page >= numPages - 1}
          >
            ›
          </button>
        </div>

        <span className="viewer-sep" aria-hidden="true" />

        <div className="viewer-seg" role="group" aria-label="Zoom">
          <button className="seg-btn" onClick={() => zoom(-0.2)} title="Zoom out" aria-label="Zoom out">
            −
          </button>
          <span className="seg-zoom">{Math.round(scale * 100)}%</span>
          <button className="seg-btn" onClick={() => zoom(0.2)} title="Zoom in" aria-label="Zoom in">
            +
          </button>
        </div>
      </div>
      <div className="viewer-body">
        {error ? (
          <div className="viewer-error">
            <strong>This page could not be opened.</strong>
            <p>
              The PDF is read straight from disk — it may have moved, or the file may be locked by another
              program.
            </p>
            <code>{error}</code>
          </div>
        ) : !pdf ? (
          <div className="viewer-loading">
            <span className="spinner" /> Opening the book…
          </div>
        ) : (
          <div className="viewer-stage" ref={stageRef}>
            <canvas ref={canvasRef} />
            <div className="highlight-layer" ref={overlayRef} />
          </div>
        )}
      </div>
    </div>
  );
}
