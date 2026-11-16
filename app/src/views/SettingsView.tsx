import { useEffect, useRef, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { findUpdate, restartApp, useAvailableUpdate, type Update } from "../update";
import {
  api,
  formatMB,
  isTauri,
  type ApiConfig,
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
  const [endpoint, setEndpoint] = useState<ApiConfig | null>(null);
  const [apiBase, setApiBase] = useState("");
  const [apiModel, setApiModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [apiBusy, setApiBusy] = useState(false);
  const [apiNote, setApiNote] = useState<{ ok: boolean; text: string } | null>(null);
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
  const [update, setUpdate] = useState<{
    stage: "idle" | "checking" | "none" | "available" | "downloading" | "error";
    version?: string;
    notes?: string;
    received?: number;
    total?: number;
  }>({ stage: "idle" });
  const pendingUpdate = useRef<Update | null>(null);

  /** Ask GitHub Releases for a newer signed build. */
  const checkUpdates = async () => {
    if (!isTauri()) {
      setUpdate({ stage: "error", notes: "Updates apply to the installed desktop app." });
      return;
    }
    setUpdate({ stage: "checking" });
    try {
      const found = await findUpdate();
      if (!found) {
        setUpdate({ stage: "none" });
        return;
      }
      pendingUpdate.current = found;
      setUpdate({ stage: "available", version: found.version, notes: found.body ?? "" });
    } catch (e) {
      setUpdate({ stage: "error", notes: String(e) });
    }
  };

  /** Download and apply, then restart into the new version. */
  const installUpdate = async () => {
    const found = pendingUpdate.current;
    if (!found) return;
    setUpdate((u) => ({ ...u, stage: "downloading", received: 0, total: 0 }));
    try {
      await found.downloadAndInstall((event) => {
        if (event.event === "Started") {
          setUpdate((u) => ({ ...u, total: event.data.contentLength ?? 0 }));
        } else if (event.event === "Progress") {
          setUpdate((u) => ({ ...u, received: (u.received ?? 0) + event.data.chunkLength }));
        }
      });
      await restartApp();
    } catch (e) {
      setUpdate({ stage: "error", notes: String(e) });
    }
  };

  const updatePct = update.total
    ? Math.min(100, Math.round(((update.received ?? 0) / update.total) * 100))
    : 0;

  // The launch check may already know about a release; fetch its details here
  // rather than making anyone press the button a second time.
  const knownUpdate = useAvailableUpdate();
  const autoChecked = useRef(false);
  useEffect(() => {
    if (!knownUpdate || autoChecked.current) return;
    autoChecked.current = true;
    checkUpdates();
    // checkUpdates is stable enough for a one-shot effect; the ref guards it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [knownUpdate]);

  const refresh = () => api.modelsStatus().then(setModels).catch((e) => setError(String(e)));

  useEffect(() => {
    api.ping().then(setInfo).catch(() => setInfo(null));
    api.dailyNotify().then(setNotify).catch(() => setNotify(null));
    api
      .apiConfig()
      .then(({ api: cfg }) => {
        setEndpoint(cfg);
        setApiBase(cfg.base_url);
        setApiModel(cfg.model);
      })
      .catch(() => setEndpoint(null));
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

  /** Save the endpoint fields and the key in one go, then switch over. */
  const saveEndpoint = async (enabled: boolean) => {
    setApiBusy(true);
    setApiNote(null);
    setError(null);
    try {
      if (apiKey.trim()) {
        await api.setApiKey(apiKey.trim());
        setApiKey("");
      }
      const { api: cfg } = await api.setApiConfig({
        enabled,
        base_url: apiBase,
        model: apiModel,
      });
      setEndpoint(cfg);
      setApiNote({
        ok: true,
        text: cfg.ready
          ? "Saved — answers will come from your endpoint."
          : "Saved. Add whatever is missing above to switch answers over.",
      });
    } catch (e) {
      setError(String(e));
    } finally {
      setApiBusy(false);
    }
  };

  const testEndpoint = async () => {
    setApiBusy(true);
    setApiNote(null);
    try {
      const { test } = await api.testApi({
        base_url: apiBase,
        model: apiModel,
        key: apiKey.trim() || undefined,
      });
      setApiNote({
        ok: true,
        text: `Answered in ${test.latency_ms} ms as ${test.model}${
          test.reply ? ` (“${test.reply}”)` : ""
        }.`,
      });
    } catch (e) {
      setApiNote({ ok: false, text: String(e) });
    } finally {
      setApiBusy(false);
    }
  };

  const removeApiKey = async () => {
    setApiBusy(true);
    setApiNote(null);
    try {
      const { api: cfg } = await api.clearApiKey();
      setEndpoint(cfg);
      setApiNote({ ok: true, text: "Key removed from this machine." });
    } catch (e) {
      setError(String(e));
    } finally {
      setApiBusy(false);
    }
  };

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
  /** Which writer is answering right now decides what this card is about. */
  const usingEndpoint = Boolean(endpoint?.enabled);

  return (
    <div className="page">
      <h1 className="page-title">Settings</h1>
      <p className="page-sub">Everything runs on this device. No accounts, no servers, no telemetry.</p>

      {error && <div className="error-banner">{error}</div>}

      <div className="card settings-section">
        <h2>Answer model</h2>
        {usingEndpoint && endpoint && (
          <>
            <div className="model-row writer-banner">
              <div className={`model-badge ${endpoint.ready ? "ok" : "bad"}`}>
                {endpoint.ready ? "✓" : "!"}
              </div>
              <div className="model-text">
                <div className="model-name">Your endpoint · {endpoint.model || "no model set"}</div>
                <div className="model-meta">{endpoint.base_url || "no endpoint set"}</div>
                <div className="model-path">
                  Answers are written by this endpoint. Search, citations and highlighting still run
                  on this device.
                </div>
              </div>
              <div className="book-actions">
                <button
                  className="btn"
                  onClick={() => saveEndpoint(false)}
                  disabled={apiBusy}
                  title="Go back to answering with the local GGUF"
                >
                  Use the local model
                </button>
              </div>
            </div>
            {!endpoint.ready && (
              <p className="setting-error">
                This endpoint is switched on but incomplete — questions will fail until the
                endpoint, model and key are all set in the card below.
              </p>
            )}
            <p className="setting-note">
              The local model settings are hidden while the endpoint answers. Switch back above, or
              from the model picker on the Ask screen, to bring them back.
            </p>
          </>
        )}
        <div className={`local-writer${usingEndpoint ? " is-hidden" : ""}`}>
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

              </>
            ) : (
              <>GPU acceleration: unavailable in this build — running on the CPU.</>
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
                {active.device === "cpu"
                  ? "Weights stay in system RAM."
                  : active.device === "gpu"
                    ? "Every layer on the GPU — needs the model to fit in VRAM."
                    : "GPU when this build supports it, CPU otherwise."}{" "}
                Reloads on your next question.
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
                A smaller cache fits a longer context in the same memory. Needs flash attention;
                otherwise F16.
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
                    An instant first question, at the cost of RAM from launch. Off by default.
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
              Up to {models.llm_active.ctx_max.toLocaleString()} tokens. Larger fits longer passages,
              at the cost of memory and a slower first token.
            </p>
          </div>
        )}
        <p className="setting-note">
          Any GGUF works, including files from LM Studio or Ollama; switching applies to your next
          question.
        </p>
        </div>
      </div>

      <div className="card settings-section">
        <h2>Bring your own model endpoint</h2>
        <p>
          Write answers with a model you already have — OpenAI, OpenRouter, Groq, or any local server
          speaking the same API. Search and citations stay on this device.
        </p>
        <div className="api-grid">
          <label className="api-field">
            <span>Endpoint</span>
            <input
              value={apiBase}
              spellCheck={false}
              placeholder="https://api.openai.com/v1"
              onChange={(e) => setApiBase(e.target.value)}
            />
          </label>
          <label className="api-field">
            <span>Model</span>
            <input
              value={apiModel}
              spellCheck={false}
              placeholder="gpt-4o-mini"
              onChange={(e) => setApiModel(e.target.value)}
            />
          </label>
          <label className="api-field api-key">
            <span>API key</span>
            <input
              type="password"
              value={apiKey}
              spellCheck={false}
              autoComplete="off"
              placeholder={endpoint?.has_key ? "saved — paste a new key to replace" : "sk-…"}
              onChange={(e) => setApiKey(e.target.value)}
            />
          </label>
        </div>
        <div className="api-actions">
          <label className="notify-toggle">
            <input
              type="checkbox"
              checked={endpoint?.enabled ?? false}
              disabled={apiBusy}
              onChange={(e) => saveEndpoint(e.target.checked)}
            />
            <span>Answer with this endpoint</span>
          </label>
          <button className="btn" disabled={apiBusy} onClick={() => saveEndpoint(endpoint?.enabled ?? false)}>
            {apiBusy ? "Working…" : "Save"}
          </button>
          <button className="btn" disabled={apiBusy} onClick={testEndpoint}>
            Test key
          </button>
          {endpoint?.has_key && (
            <button className="btn btn-quiet" disabled={apiBusy} onClick={removeApiKey}>
              Remove key
            </button>
          )}
        </div>
        {apiNote && (
          <p className={apiNote.ok ? "setting-ok" : "setting-error"}>{apiNote.text}</p>
        )}
        <p className="setting-note">
          The key is encrypted for your Windows account and kept beside the library, not inside it.
          It is sent only to the endpoint above.
        </p>
      </div>

      <div className="card settings-section">
        <h2>AI models</h2>
        <p>Three small open models, downloaded once and run entirely on this device.</p>
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
          About 6.2 GB of disk, resumable if interrupted. All three installed means asking and
          highlighting work offline.
        </p>
      </div>

      <div className="card settings-section">
        <h2>Hadith of the day on your desktop</h2>
        <p>
          The day's narration can arrive as a desktop notification, without opening the app.
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
          Windows raises it from Task Scheduler, so the app can stay closed. A day missed while the
          computer was off arrives when it next starts.
        </p>
      </div>

      <div className="card settings-section">
        <h2>Version &amp; updates</h2>
        <p>
          Hadith Finder {info?.version ?? "…"}. Updates come from this project's GitHub Releases and are
          signed, so a tampered download is refused before it reaches the app.
        </p>
        <div className="update-row">
          <button
            className="btn"
            onClick={checkUpdates}
            disabled={update.stage === "checking" || update.stage === "downloading"}
          >
            {update.stage === "checking" ? "Checking…" : "Check for updates"}
          </button>
          {update.stage === "none" && <span className="setting-ok">You're on the latest version.</span>}
          {update.stage === "available" && (
            <span className="ask-hint">Version {update.version} is available.</span>
          )}
        </div>
        {update.stage === "available" && (
          <>
            {update.notes && <p className="setting-note update-notes">{update.notes}</p>}
            <div className="update-row">
              <button className="btn btn-primary" onClick={installUpdate}>
                Download &amp; install
              </button>
              <span className="ask-hint">The app restarts when the update is applied.</span>
            </div>
          </>
        )}
        {update.stage === "downloading" && (
          <div className="progress-wrap model-progress">
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${updatePct}%` }} />
            </div>
            <span className="progress-label">Downloading the update… {updatePct}%</span>
          </div>
        )}
        {update.stage === "error" && <p className="setting-error">{update.notes}</p>}
      </div>

      <div className="card settings-section">
        <h2>Your library</h2>
        <p>
          <code>{info?.library ?? "…"}</code>
        </p>
        <p>
          One portable file (<code>library.haddb</code>) holding passages and the search index — never
          the PDFs, which stay where they are. Copy it to back it up; models sit beside it in{" "}
          <code>models/</code>.
        </p>
        <p className="setting-note">
          Scanned pages are re-read with RapidOCR ({""}
          {info?.ocr ? (info.ocr.available ? `ready, ${info.ocr.mode}` : "not installed") : "…"}), Latin
          script only. Books imported before an extractor change can be rebuilt from <strong>Library</strong>.
        </p>
      </div>

      <div className="card settings-section">
        <h2>Honest limitations</h2>
        <ul>
          <li>Multi-column pages and heavy footnotes can interleave text; citations still point to the right page.</li>
          <li>“Same narration in other collections” is a similarity match, not a scholarly cross-reference.</li>
          <li>Gradings appear <em>only when the printed book states them</em> — the app never grades narrations.</li>
          <li>Answers can be wrong. Verify against the highlighted passage before acting on a religious matter.</li>
        </ul>
      </div>

      <div className="card settings-section">
        <h2>License & credits</h2>
        <p>
          AGPL-3.0, and no bundled book content — you import your own PDFs. Models belong to their
          authors (Gemma: Google DeepMind and TrevorJS; BGE: BAAI); translations may be copyrighted,
          so use copies you're entitled to keep.
        </p>
        <p className="setting-note">Version 0.3.0 · Not affiliated with any religious authority.</p>
      </div>
    </div>
  );
}
