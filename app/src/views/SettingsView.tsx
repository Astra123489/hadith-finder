import { useEffect, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import {
  api,
  formatMB,
  type DailyNotify,
  type DownloadProgress,
  type ModelsStatus,
} from "../api";
import type { ActiveModel, KvCache, LlmDevice } from "../api";

type ModelKey = "llm" | "embedder" | "reranker";

export default function SettingsView() {
  const [info, setInfo] = useState<{ version: string; library: string; ocr?: { mode: string; available: boolean } } | null>(null);
  const [models, setModels] = useState<ModelsStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<ModelKey | null>(null);
  const [llmBusy, setLlmBusy] = useState(false);
  const [ctxValue, setCtxValue] = useState("6144");
  const [ctxBusy, setCtxBusy] = useState(false);
  const [optBusy, setOptBusy] = useState(false);
  const [notify, setNotify] = useState<DailyNotify | null>(null);
  const [notifyBusy, setNotifyBusy] = useState(false);
  const [llmStatus, setLlmStatus] = useState<{
    stage: "loading" | "ready" | "error";
    message?: string;
    gpu_layers?: number;
    kv?: string;
    kv_applied?: boolean;
  } | null>(null);
  const [progress, setProgress] = useState<Record<ModelKey, DownloadProgress | null>>({
    llm: null, embedder: null, reranker: null,
  });

  const refresh = () => api.modelsStatus().then(setModels).catch((e) => setError(String(e)));

  useEffect(() => {
    api.ping().then(setInfo).catch(() => setInfo(null));
    api.dailyNotify().then(setNotify).catch(() => setNotify(null));
    refresh();
    let unlisten: (() => void) | undefined;
    api.onProgress((p) => {
      if (p.op === "llm") {
        setLlmStatus(p);
        if (p.stage !== "loading") refresh();
        return;
      }
      if (p.op !== "download") return;
      setProgress((cur) => ({ ...cur, [p.model]: p }));
    }).then((fn) => (unlisten = fn));
    return () => unlisten?.();
  }, []);

  useEffect(() => {
    const ctx = models?.llm_active?.ctx;
    if (ctx) setCtxValue(String(ctx));
  }, [models?.llm_active?.ctx]);

  /** Toggle or reschedule the desktop reminder; the sidecar owns the clock. */
  const saveNotify = async (opts: { enabled?: boolean; time?: string }) => {
    setNotifyBusy(true);
    setError(null);
    try {
      setNotify(await api.setDailyNotify(opts));
    } catch (e) {
      setError(String(e));
      api.dailyNotify().then(setNotify).catch(() => setNotify(null));
    } finally {
      setNotifyBusy(false);
    }
  };

  const saveCtx = async (value?: number) => {
    const n = value ?? Number(ctxValue);
    if (!Number.isFinite(n) || n <= 0) {
      setError("Context length must be a whole number of tokens.");
      return;
    }
    setCtxBusy(true);
    setError(null);
    try {
      const res = await api.setLlmCtx(Math.round(n));
      setCtxValue(String(res.llm_active.ctx));
      await refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setCtxBusy(false);
    }
  };

  const saveLlmOptions = async (opts: { device?: LlmDevice; kv?: KvCache; auto_load?: boolean }) => {
    setOptBusy(true);
    setError(null);
    try {
      await api.setLlmOptions(opts);
      await refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setOptBusy(false);
    }
  };

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
          <div className="model-row">
            <div className={`model-badge ${models.llm_active.exists ? "ok" : "bad"}`}>
              {models.llm_active.exists ? "✓" : "!"}
            </div>
            <div className="model-text">
              <div className="model-name">{models.llm_active.name}</div>
              <div className="model-meta">
                {models.llm_active.custom ? "your own GGUF · " : "downloaded default · "}
                {formatMB(models.llm_active.size)}
                {!models.llm_active.exists && " · file is missing"}
              </div>
              <div className="model-path">{models.llm_active.path}</div>
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
        {models?.llm_active?.gpu && (
          <div className="gpu-line">
            {models.llm_active.gpu.available ? (
              <>
                GPU acceleration: <strong>{models.llm_active.gpu.device}</strong>
                {models.llm_active.gpu.vram_free_mb
                  ? ` · ${(models.llm_active.gpu.vram_free_mb / 1024).toFixed(1)} GB free`
                  : ""}{" "}
                — answers run on the graphics card.
              </>
            ) : (
              <>GPU acceleration: unavailable in this build — answers run on the CPU (slower).</>
            )}
            {models.onnx && (
              <>
                <br />
                {models.onnx.providers[0] !== "CPUExecutionProvider" ? (
                  <>Retrieval models (embeddings + reranking): <strong>GPU (DirectML)</strong>.</>
                ) : (
                  <>Retrieval models (embeddings + reranking): CPU.</>
                )}
              </>
            )}
          </div>
        )}
        {models?.llm_active && (() => {
          const active = models.llm_active as ActiveModel;
          const gpuReady = Boolean(active.gpu?.available);
          const deviceLabels: Record<string, string> = { auto: "Auto", gpu: "GPU", cpu: "CPU" };
          const deviceHints: Record<string, string> = {
            auto: "Use the graphics card when this build supports it, otherwise the CPU.",
            gpu: "Offload every layer to the GPU — fastest, needs the model to fit in VRAM.",
            cpu: "Keep the weights in system RAM — slower, but works on any machine.",
          };
          const runtime = active.runtime;
          return (
            <div className="setting-block">
              <div className="setting-label">Where the model runs</div>
              <div className="chip-row">
                {(active.device_options ?? ["auto", "gpu", "cpu"]).map((device) => (
                  <button
                    key={device}
                    className={`scope-chip ${active.device === device ? "on" : ""}`}
                    onClick={() => saveLlmOptions({ device })}
                    disabled={optBusy || (device === "gpu" && !gpuReady)}
                    title={deviceHints[device]}
                  >
                    {deviceLabels[device] ?? device}
                  </button>
                ))}
                {!gpuReady && (
                  <span className="ask-hint">this build has no GPU backend — CPU only</span>
                )}
              </div>
              <p className="setting-note">
                {deviceHints[active.device] ?? ""} Changing this reloads the model on your next question
                {active.auto_load ? " (or at launch)" : ""}.
              </p>

              <div className="setting-label">KV cache precision</div>
              <div className="chip-row">
                {(active.kv_options ?? ["f16", "q8_0", "q4_0"]).map((kv) => (
                  <button
                    key={kv}
                    className={`scope-chip ${active.kv === kv ? "on" : ""}`}
                    onClick={() => saveLlmOptions({ kv })}
                    disabled={optBusy}
                    title={
                      kv === "f16"
                        ? "Full precision K/V cache — the default, no quality loss."
                        : `Quantize the K/V cache to ${kv.toUpperCase()} — frees VRAM and RAM for a longer context, at a small precision cost.`
                    }
                  >
                    {kv === "f16" ? "F16 (default)" : kv.toUpperCase()}
                  </button>
                ))}
              </div>
              <p className="setting-note">
                Quantizing the K/V cache shrinks the memory the context needs — a Q8_0 or Q4_0 cache fits a
                longer window in the same VRAM. It needs flash attention; if this llama.cpp build does not
                support it, the app quietly falls back to F16.
                {runtime?.loaded && (
                  <>
                    {" "}Currently loaded with <strong>{runtime.kv.toUpperCase()}</strong>
                    {!runtime.kv_applied && active.kv !== "f16" ? " (fallback — not supported by this build)" : ""}
                    {runtime.gpu_layers !== 0 ? ` · ${runtime.gpu_layers === -1 ? "all" : runtime.gpu_layers} GPU layers` : " · CPU"}.
                  </>
                )}
              </p>

              <label className="setting-toggle">
                <input
                  type="checkbox"
                  checked={Boolean(active.auto_load)}
                  disabled={optBusy}
                  onChange={(e) => saveLlmOptions({ auto_load: e.target.checked })}
                />
                <span>
                  <strong>Load the answer model when Hadith Finder starts</strong>
                  <span className="setting-toggle-sub">
                    The first question is then instant, at the cost of RAM from launch. Off by default —
                    the model loads on first use instead.
                  </span>
                </span>
              </label>

              {llmStatus && (
                <div className={`llm-status ${llmStatus.stage}`}>
                  {llmStatus.stage === "loading" && (<><span className="spinner" /> Loading the model in the background…</>)}
                  {llmStatus.stage === "ready" && (
                    <>Model ready{runtime?.gpu_layers ? " on the GPU" : " on the CPU"} — answers will start immediately.</>
                  )}
                  {llmStatus.stage === "error" && (<>Could not load the model: {llmStatus.message}</>)}
                </div>
              )}
            </div>
          );
        })()}
        {models?.llm_active && (
          <div className="setting-block">
            <div className="setting-label">Context window</div>
            <div className="chip-row">
              {[2048, 4096, 6144, 8192, 16384, 32768].map((n) => (
                <button
                  key={n}
                  className={`scope-chip ${Number(ctxValue) === n ? "on" : ""}`}
                  onClick={() => saveCtx(n)}
                  disabled={ctxBusy}
                >
                  {n}
                </button>
              ))}
              <input
                className="page-input ctx-input"
                type="number"
                min={models.llm_active.ctx_min}
                max={models.llm_active.ctx_max}
                step={256}
                value={ctxValue}
                onChange={(e) => setCtxValue(e.target.value)}
                aria-label="Context length in tokens"
              />
              <button className="btn" onClick={() => saveCtx()} disabled={ctxBusy}>
                {ctxBusy ? "Saving…" : "Save"}
              </button>
              <span className="ask-hint">tokens · default {models.llm_active.ctx_default}</span>
            </div>
            <p className="setting-note">
              The context window is the room for the retrieved excerpts plus the answer. Larger
              windows fit longer passages but use more RAM and make the first token slower; the
              model supports up to {models.llm_active.ctx_max.toLocaleString()} tokens. Takes effect
              on your next question.
            </p>
          </div>
        )}
        <p className="setting-note">
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
          they are free: Gemma 4 (Google, Apache-2.0) and BGE models (BAAI, MIT).
        </p>
        <div className="library-list models-list">
          {models && modelKeys.map((key) => {
            const m = models[key];
            const p = progress[key];
            const pct = progressPct(p);
            const active = busy === key;
            return (
              <div key={key} className="model-row">
                <div className={`model-badge ${m.ready ? "ok" : "idle"}`}>{m.ready ? "✓" : "↓"}</div>
                <div className="model-text">
                  <div className="model-name">{m.name}</div>
                  <div className="model-meta">{m.purpose} · {m.license} · {formatMB(m.total_bytes)}</div>
                  {active && p && (
                    <div className="progress-wrap model-progress">
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
        <p className="setting-note">
          Needs roughly 6.2 GB of free disk. When all three show “installed”, asking and highlighting
          are fully unlocked — offline, forever.
        </p>
      </div>

      <div className="card settings-section">
        <h2>How answers are written</h2>
        <p>
          Thinking is always on: the model reasons before it answers, which is what makes “explain…”
          and comparison questions reliable. There is no fast/careful switch to tune — instead, pick
          which installed model writes the answer from the <strong>Model</strong> selector in the Ask tab
          (the installed models are managed in the card above).
        </p>
      </div>

      <div className="card settings-section">
        <h2>Hadith of the day on your desktop</h2>
        <p>
          The day's narration can arrive as a desktop notification, outside the app window — so you
          can read it without opening anything.
        </p>
        <div className="notify-row">
          <label className="notify-toggle">
            <input
              type="checkbox"
              checked={notify?.enabled ?? false}
              disabled={notifyBusy}
              onChange={(e) => saveNotify({ enabled: e.target.checked })}
            />
            <span>Show it as a desktop notification</span>
          </label>
          <label className="notify-time">
            <span>at</span>
            <input
              type="time"
              value={notify?.time ?? "07:00"}
              disabled={notifyBusy || !notify?.enabled}
              onChange={(e) =>
                setNotify((cur) => (cur ? { ...cur, time: e.target.value } : cur))
              }
              onBlur={(e) => {
                if (e.target.value && e.target.value !== notify?.time) {
                  saveNotify({ time: e.target.value });
                }
              }}
            />
          </label>
        </div>
        {notify?.schedule_error && (
          <p className="setting-error">{notify.schedule_error}</p>
        )}
        <p className="setting-note">
          Windows raises it from Task Scheduler, so Hadith Finder does not need to be open and
          nothing keeps running in the background: a short-lived process shows the notification and
          exits. If the computer was off at that time, Windows runs it as soon as it is back. Each
          day reaches the desktop once.
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
        <p className="setting-note">
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
          PDFs. Models belong to their authors (Google DeepMind — Gemma; the abliterated Gemma build
          is by TrevorJS; embeddings and reranking by BAAI) under their respective free licenses. Hadith texts themselves are classical works; translations may be copyrighted by their
          publishers — use copies you're entitled to keep.
        </p>
        <p className="setting-note">Version 0.2.0 · Not affiliated with any religious authority.</p>
      </div>
    </div>
  );
}
