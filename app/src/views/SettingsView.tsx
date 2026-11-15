import { useEffect, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { api, formatMB, type DownloadProgress, type ModelsStatus } from "../api";

type ModelKey = "llm" | "embedder" | "reranker";

export default function SettingsView() {
  const [info, setInfo] = useState<{ version: string; library: string; ocr?: { mode: string; available: boolean } } | null>(null);
  const [models, setModels] = useState<ModelsStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<ModelKey | null>(null);
  const [llmBusy, setLlmBusy] = useState(false);
  const [progress, setProgress] = useState<Record<ModelKey, DownloadProgress | null>>({
    llm: null, embedder: null, reranker: null,
  });

  const refresh = () => api.modelsStatus().then(setModels).catch((e) => setError(String(e)));

  useEffect(() => {
    api.ping().then(setInfo).catch(() => setInfo(null));
    refresh();
    let unlisten: (() => void) | undefined;
    api.onProgress((p) => {
      if (p.op !== "download") return;
      setProgress((cur) => ({ ...cur, [p.model]: p }));
    }).then((fn) => (unlisten = fn));
    return () => unlisten?.();
  }, []);

  const chooseGguf = async () => {
    const picked = await open({
      multiple: false,
      directory: false,
      filters: [{ name: "GGUF models", extensions: ["gguf"] }],
    });
    if (typeof picked !== "string") return;
    setLlmBusy(true);
    setError(null);
    try {
      await api.setLlm(picked);
      await refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setLlmBusy(false);
    }
  };

  const resetLlm = async () => {
    setLlmBusy(true);
    setError(null);
    try {
      await api.setLlm(null);
      await refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setLlmBusy(false);
    }
  };

  const startDownload = async (model: ModelKey) => {
    setBusy(model);
    setError(null);
    try {
      await api.downloadModel(model);
      await refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(null);
      setProgress((cur) => ({ ...cur, [model]: null }));
    }
  };

  const progressPct = (p: DownloadProgress | null) =>
    p ? Math.min(100, Math.round(((p.file_index * p.total + p.downloaded) / (p.total * p.file_total)) * 100)) : 0;

  const modelKeys: ModelKey[] = ["llm", "embedder", "reranker"];

  return (
    <div className="page">
      <h1 className="page-title">Settings</h1>
      <p className="page-sub">Everything runs on this device. No accounts, no servers, no telemetry.</p>

      {error && <div className="error-banner">{error}</div>}

      <div className="card settings-section">
        <h2>Answer model</h2>
        {models?.llm_active ? (
          <div className="book-row" style={{ border: "1px solid var(--line)", borderRadius: 12 }}>
            <div className="book-cover" style={{ height: 46, width: 46, borderRadius: 10 }}>
              <span style={{ color: models.llm_active.exists ? "var(--green)" : "var(--red)", fontSize: 20 }}>
                {models.llm_active.exists ? "✓" : "!"}
              </span>
            </div>
            <div className="book-meta">
              <div className="book-title" style={{ fontSize: 15 }}>{models.llm_active.name}</div>
              <div className="book-sub" style={{ whiteSpace: "normal" }}>
                {models.llm_active.custom ? "your own GGUF · " : "downloaded default · "}
                {formatMB(models.llm_active.size)}
                {!models.llm_active.exists && " · file is missing"}
              </div>
              <div className="book-sub" style={{ whiteSpace: "normal", wordBreak: "break-all" }}>
                {models.llm_active.path}
              </div>
            </div>
            <div className="book-actions">
              <button className="btn btn-primary" onClick={chooseGguf} disabled={llmBusy}>
                {llmBusy ? "Loading…" : "Use a local .gguf…"}
              </button>
              {models.llm_active.custom && (
                <button className="btn btn-ghost" onClick={resetLlm} disabled={llmBusy}>
                  Use default
                </button>
              )}
            </div>
          </div>
        ) : (
          <p>…</p>
        )}
        <p style={{ marginTop: 12, fontSize: 12, color: "var(--text-faint)" }}>
          Any GGUF file works — including one you already have in LM Studio or Ollama. The model is
          loaded on this device; switching models takes effect on the next question and does not
          require re-indexing (answers only; embeddings are separate).
        </p>
      </div>

      <div className="card settings-section">
        <h2>AI models — one-time download ({models ? formatMB(models.total_bytes) : "…"})</h2>
        <p>
          Hadith Finder is powered by three small open models that run fully locally. Downloaded once
          into your data folder (resumable if interrupted) — they are never shared with anyone, and
          they are free: Qwen (Alibaba, Apache-2.0) and BGE models (BAAI, MIT).
        </p>
        <div className="library-list" style={{ marginTop: 14 }}>
          {models && modelKeys.map((key) => {
            const m = models[key];
            const p = progress[key];
            const pct = progressPct(p);
            const active = busy === key;
            return (
              <div key={key} className="book-row" style={{ border: "1px solid var(--line)", borderRadius: 12 }}>
                <div className="book-cover" style={{ height: 46, width: 46, borderRadius: 10 }}>
                  <span style={{ color: m.ready ? "var(--green)" : "var(--gold)", fontSize: 20 }}>{m.ready ? "✓" : "↓"}</span>
                </div>
                <div className="book-meta">
                  <div className="book-title" style={{ fontSize: 15 }}>{m.name}</div>
                  <div className="book-sub" style={{ whiteSpace: "normal" }}>{m.purpose} · {m.license} · {formatMB(m.total_bytes)}</div>
                  {active && p && (
                    <div className="progress-wrap" style={{ marginTop: 8, minWidth: 260 }}>
                      <div className="progress-bar"><div className="progress-fill" style={{ width: `${pct}%` }} /></div>
                      <div className="progress-label">
                        {p.file} — {formatMB(p.downloaded)} / {formatMB(p.total)} ({pct}%)
                      </div>
                    </div>
                  )}
                </div>
                <div className="book-actions">
                  {m.ready ? (
                    <span className="grading g-sahih">installed</span>
                  ) : (
                    <button className="btn btn-primary" disabled={busy !== null} onClick={() => startDownload(key)}>
                      {active ? (<><span className="spinner" /> Downloading…</>) : "Download"}
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
        <p style={{ marginTop: 12, fontSize: 12, color: "var(--text-faint)" }}>
          Needs roughly 6.2 GB of free disk. When all three show “installed”, asking and highlighting
          are fully unlocked — offline, forever.
        </p>
      </div>

      <div className="card settings-section">
        <h2>Careful vs. Fast answering</h2>
        <p>
          <strong>Fast</strong> answers directly (best for direct lookups). <strong>Careful</strong> lets the model
          reason step-by-step first — slower, noticeably stronger for “explain…” and comparison questions.
          Toggle it in the Ask tab; it applies to the next question only.
        </p>
      </div>

      <div className="card settings-section">
        <h2>Your library</h2>
        <p>Stored locally at:</p>
        <p><code>{info?.library ?? "…"}</code></p>
        <p>
          The library is a single portable file (<code>library.haddb</code>, SQLite) containing your books'
          extracted passages and search index — never the PDFs themselves, which stay where they are.
          Back it up by copying that file. Models live next to it in <code>models/</code>.
        </p>
        <p>
          If a book was imported before an extractor improvement, open <strong>Library → Rebuild</strong> on it:
          the PDF is re-read on this device and the passages and index are regenerated.
        </p>
      </div>

      <div className="card settings-section">
        <h2>Text extraction (OCR)</h2>
        <p>
          Scanned pages, and pages whose text layer was damaged by the publisher, are re-read with
          RapidOCR on this device. Status:{" "}
          <strong>{info?.ocr ? (info.ocr.available ? `RapidOCR ready (mode: ${info.ocr.mode})` : "not installed — text-layer only") : "…"}</strong>
        </p>
        <p style={{ color: "var(--text-faint)", fontSize: 12 }}>
          The shipped OCR model reads Latin script; Arabic pages that contain no text layer at all
          still need an Arabic OCR pass before importing.
        </p>
      </div>

      <div className="card settings-section">
        <h2>Honest limitations</h2>
        <ul>
          <li>Scanned pages are OCR'd when RapidOCR is available; the shipped model reads Latin script only.</li>
          <li>Multi-column layouts and heavy footnotes can interleave text; citations still point to the right page.</li>
          <li>“Same narration in other collections” compares against a bundled corpus of the five canonical collections. It is a similarity match, not a scholarly cross-reference — verify before relying on it.</li>
          <li>Authenticity gradings are shown <em>only when the printed book states them</em> — the app never grades narrations itself.</li>
          <li>AI answers can be wrong. Always verify against the highlighted source passage before acting on a religious matter.</li>
        </ul>
      </div>

      <div className="card settings-section">
        <h2>License & credits</h2>
        <p>
          Hadith Finder itself is open source. It ships without any book content — you import your own
          PDFs. Models belong to their authors (Qwen team at Alibaba, BAAI) under their respective free
          licenses. Hadith texts themselves are classical works; translations may be copyrighted by their
          publishers — use copies you're entitled to keep.
        </p>
        <p style={{ color: "var(--text-faint)" }}>Version 0.1.0 · Not affiliated with any religious authority.</p>
      </div>
    </div>
  );
}
