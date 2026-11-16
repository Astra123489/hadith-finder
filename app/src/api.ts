import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type {
  AskResult,
  Book,
  Bookmark,
  ChatSummary,
  DailyHadith,
  ImportResult,
  Narration,
  NewBookmark,
  ThreadState,
  ThreadTurn,
  ThreadUsage,
} from "./types";

export interface ReindexProgress {
  type: "progress";
  op: "reindex";
  book_id: number;
  done: number;
  total: number;
}

export interface ImportProgress {
  type: "progress";
  op: "import";
  stage: "extract" | "chunks";
  page?: number;
  pages?: number;
  done?: number;
  total?: number;
}

export interface TokenProgress {
  type: "progress";
  op: "token";
  ask_id: string;
  text: string;
  phase?: "thinking" | "answer";
}

export interface DownloadProgress {
  type: "progress";
  op: "download";
  model: "llm" | "embedder" | "reranker";
  file: string;
  downloaded: number;
  total: number;
  file_index: number;
  file_total: number;
}

export interface CrossrefProgress {
  type: "progress";
  op: "crossref";
  stage: string;
  done: number;
  total: number;
}

/** The conversation being folded away to free context. */
export interface ThreadProgress {
  type: "progress";
  op: "thread";
  stage: "compacting" | "compacted";
  dropped?: number;
  compactions?: number;
  summary?: string;
}

/** The answer model loading in the background (auto-load at launch). */
export interface LlmProgress {
  type: "progress";
  op: "llm";
  stage: "loading" | "ready" | "error";
  model?: string;
  device?: string;
  kv?: string;
  kv_applied?: boolean;
  gpu_layers?: number;
  message?: string;
}

export type Progress =
  | ReindexProgress
  | ImportProgress
  | TokenProgress
  | DownloadProgress
  | CrossrefProgress
  | LlmProgress
  | ThreadProgress;

export interface ModelInfo {
  name: string;
  purpose: string;
  license: string;
  dir: string;
  ready: boolean;
  bytes_installed: number;
  total_bytes: number;
  files: { name: string; size: number; installed: boolean }[];
}

export interface LlmOption {
  path: string;
  name: string;
  size: number;
  installed: boolean;
  default: boolean;
  selected: boolean;
}

export interface GpuStatus {
  available: boolean;
  device: string | null;
  vram_total_mb: number | null;
  vram_free_mb: number | null;
  layers: number | null;
}

export type LlmDevice = "auto" | "gpu" | "cpu";
export type KvCache = "f16" | "q8_0" | "q4_0";

export interface ActiveModel {
  name: string;
  path: string;
  exists: boolean;
  size: number;
  custom: boolean;
  ctx: number;
  ctx_default: number;
  ctx_min: number;
  ctx_max: number;
  /** Where the weights are offloaded: auto, GPU only, or CPU only. */
  device: LlmDevice;
  device_options: LlmDevice[];
  /** K/V cache precision; quantizing it frees VRAM at a small quality cost. */
  kv: KvCache;
  kv_options: KvCache[];
  auto_load: boolean;
  gpu: GpuStatus;
  runtime?: {
    loaded: boolean;
    model: string;
    gpu_layers: number;
    device: string;
    kv: string;
    kv_applied: boolean;
    flash_attn: boolean;
  };
}

export type ModelsStatus = Record<"llm" | "embedder" | "reranker", ModelInfo> & {
  all_ready: boolean;
  total_bytes: number;
  ocr?: { mode: string; available: boolean };
  onnx?: { providers: string[]; available: string[] };
  llm_active?: ActiveModel;
};

export interface RebuildResult {
  book_id: number;
  title: string;
  page_count: number;
  chunks: number;
  language: string;
  ocr_pages: number;
}

export const isTauri = () =>
  typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

const DEMO_BOOK: Book = {
  id: 999,
  title: "Demo — Book of Intentions",
  author: "Hadith Finder demo",
  language: "mixed",
  file_path: "demo-fixture.pdf",
  page_count: 3,
  created_at: new Date(1735689600000).toISOString(),
  chunk_count: 14,
  embedded_count: 14,
};

const DEMO_RESULT: AskResult = {
  answer:
    "Deeds are judged by the intention behind them [S1], and the reward follows what a person actually intended [S2]. The compiler grades this transmission sahih [S1].",
  citations: [
    {
      chunk_id: 11,
      book_id: 999,
      book_title: DEMO_BOOK.title,
      chapter: "Book of Intentions",
      hadith_ref: "1",
      page_start: 0,
      page_end: 0,
      spans: { "0": [[72, 96, 520, 118]] },
      excerpt:
        "Narrated 'Umar: I heard the Messenger of Allah say: Actions are judged by intentions, and each person will have what they intended.",
      gradings: ["sahih"],
      source_index: 1,
      also_in: [
        {
          collection: "bukhari",
          collection_name: "Sahih al-Bukhari",
          number: 1,
          book: "Revelation",
          label: "Bukhari 1",
          snippet:
            "The reward of deeds depends upon the intentions and every person will get the reward according to what he has intended.",
          score: 0.72,
          same_collection: false,
        },
        {
          collection: "muslim",
          collection_name: "Sahih Muslim",
          number: 1907,
          book: "Leadership",
          label: "Muslim 1907",
          snippet:
            "Actions are judged by intentions, so each man will have what he intended.",
          score: 0.68,
          same_collection: false,
        },
      ],
    },
    {
      chunk_id: 12,
      book_id: 999,
      book_title: DEMO_BOOK.title,
      chapter: "Book of Intentions",
      hadith_ref: "2",
      page_start: 0,
      page_end: 0,
      spans: { "0": [[72, 140, 520, 162]] },
      excerpt:
        "Whoever emigrated for a worldly gain or a woman to marry, his emigration was for what he emigrated for.",
      gradings: ["sahih"],
      source_index: 2,
      also_in: [],
    },
  ],
  retrieved: 9,
  unverified: false,
  scoped_empty: false,
};

const demoModel: ModelInfo = {
  name: "Ready",
  purpose: "Preview mode uses a canned demo answer — run the desktop app for real inference.",
  license: "Apache-2.0",
  dir: "demo",
  ready: true,
  bytes_installed: 1,
  total_bytes: 1,
  files: [{ name: "demo", size: 1, installed: true }],
};

const demoGpu: GpuStatus = {
  available: true,
  device: "Demo GPU",
  vram_total_mb: 8192,
  vram_free_mb: 7168,
  layers: -1,
};

const demoActive: ActiveModel = {
  name: "gemma-4-E2B-it-uncensored-Q8_0",
  path: "(preview)",
  exists: true,
  size: 4967490080,
  custom: true,
  ctx: 6144,
  ctx_default: 6144,
  ctx_min: 1024,
  ctx_max: 131072,
  device: "auto",
  device_options: ["auto", "gpu", "cpu"],
  kv: "f16",
  kv_options: ["f16", "q8_0", "q4_0"],
  auto_load: false,
  gpu: demoGpu,
  runtime: {
    loaded: false,
    model: "",
    gpu_layers: 0,
    device: "auto",
    kv: "f16",
    kv_applied: false,
    flash_attn: false,
  },
};

// The browser preview keeps settings in memory so the UI behaves like the
// sidecar-backed app (save → refresh → still saved).
let demoActiveState: ActiveModel = { ...demoActive };

const demoBookmarks: Bookmark[] = [
  {
    id: 1,
    kind: "canonical",
    title: "Sahih al-Bukhari",
    chapter: "Revelation",
    reference: "Hadith 1",
    collection: "bukhari",
    number: 1,
    page_start: null,
    page_end: null,
    excerpt:
      "The reward of deeds depends upon the intentions and every person will get the reward according to what he has intended.",
    gradings: [],
    spans: {},
    book_id: null,
    chunk_id: null,
    description: "The hadith that starts the Sahih — worth memorising.",
    created_at: new Date(1735689600000).toISOString(),
  },
];

// The preview's conversations: the same shape the sidecar keeps, in memory, so
// follow-ups, compaction, the context meter and the chat switcher all behave.
const DEMO_CTX = 6144;

interface DemoChat {
  id: string;
  title: string;
  turns: ThreadTurn[];
  summary: string;
  compactions: number;
  nextId: number;
  createdAt: string;
  updatedAt: string;
}

function newDemoChat(): DemoChat {
  const now = new Date().toISOString();
  return {
    id: `c${Math.random().toString(36).slice(2, 10)}`,
    title: "",
    turns: [],
    summary: "",
    compactions: 0,
    nextId: 1,
    createdAt: now,
    updatedAt: now,
  };
}

let demoChats: DemoChat[] = [newDemoChat()];
let demoActiveId = demoChats[0].id;

function activeDemoChat(): DemoChat {
  return demoChats.find((c) => c.id === demoActiveId) ?? demoChats[0];
}

function startDemoChat(): DemoChat {
  const chat = newDemoChat();
  demoChats = [chat, ...demoChats];
  demoActiveId = chat.id;
  return chat;
}

function demoChatList(): ChatSummary[] {
  return [...demoChats]
    .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
    .map((c) => ({
      id: c.id,
      title: c.title,
      turns: c.turns.length,
      created_at: c.createdAt,
      updated_at: c.updatedAt,
      active: c.id === demoActiveId,
    }));
}

function demoTitleFor(question: string): string {
  const text = question.replace(/[?.!]+$/, "").trim();
  if (text.length <= 58) return text;
  return `${text.slice(0, 58).replace(/\s+\S*$/, "")}…`;
}

function demoUsage(chat: DemoChat): ThreadUsage {
  const prose = [chat.summary, ...chat.turns.map((t) => `${t.question} ${t.answer}`)].join(" ");
  const history = Math.round(prose.length / 4);
  const excerpts = chat.turns.length ? 900 : 0;
  const used = Math.min(DEMO_CTX, 260 + history + excerpts);
  return {
    used,
    ctx: DEMO_CTX,
    compact_at: Math.round(DEMO_CTX * 0.7),
    pct: Math.round((used / DEMO_CTX) * 10000) / 10000,
    history_tokens: history,
    excerpt_tokens: excerpts,
    turns: chat.turns.length,
    compactions: chat.compactions,
  };
}

function demoThreadState(chat: DemoChat = activeDemoChat()): ThreadState {
  return {
    id: chat.id,
    title: chat.title,
    turns: chat.turns,
    summary: chat.summary,
    compactions: chat.compactions,
    usage: demoUsage(chat),
  };
}

const demoApi = {
  ping: () =>
    Promise.resolve({
      pong: true,
      version: "0.2.0 (browser preview)",
      library: "demo library (in-browser preview)",
      ocr: { mode: "auto", available: false },
    }),
  listBooks: () => Promise.resolve([DEMO_BOOK]),
  importBook: () =>
    Promise.reject(new Error("The browser preview cannot import PDFs — use the desktop app.")),
  rebuildBook: () => Promise.reject(new Error("The browser preview cannot rebuild a book.")),
  reindexBook: () => Promise.resolve({ book_id: 999, embedded: 14, total: 14 }),
  deleteBook: () => Promise.resolve({ deleted: false }),
  ask: (query: string, _bookIds: number[] | null, _askId?: string, newThread?: boolean) => {
    if (newThread) startDemoChat();
    const chat = activeDemoChat();
    const now = new Date().toISOString();
    chat.turns = [
      ...chat.turns,
      {
        id: chat.nextId++,
        question: query,
        answer: DEMO_RESULT.answer,
        citations: DEMO_RESULT.citations,
        unverified: false,
        created_at: now,
      },
    ];
    chat.updatedAt = now;
    if (!chat.title) chat.title = demoTitleFor(query);
    return Promise.resolve({ ...DEMO_RESULT, thread: demoThreadState(chat), chats: demoChatList() });
  },
  listChats: () => Promise.resolve({ chats: demoChatList(), active: demoActiveId }),
  threadState: () => Promise.resolve({ thread: demoThreadState(), chats: demoChatList() }),
  newChat: () => {
    const current = activeDemoChat();
    const chat = current.turns.length ? startDemoChat() : current;
    return Promise.resolve({ thread: demoThreadState(chat), chats: demoChatList() });
  },
  openChat: (id: string) => {
    const found = demoChats.find((c) => c.id === id);
    if (found) demoActiveId = found.id;
    return Promise.resolve({ thread: demoThreadState(), chats: demoChatList() });
  },
  renameChat: (id: string, title: string) => {
    const chat = demoChats.find((c) => c.id === id);
    if (chat) {
      chat.title = title;
      chat.updatedAt = new Date().toISOString();
    }
    return Promise.resolve({ chats: demoChatList() });
  },
  readNarration: (collection: string, number: number) => {
    const found = DEMO_RESULT.citations
      .flatMap((c) => c.also_in)
      .find((p) => p.collection === collection && p.number === number);
    if (!found) return Promise.reject(new Error(`no such narration: ${collection} ${number}`));
    return Promise.resolve({
      collection: found.collection,
      collection_name: found.collection_name,
      short: found.label.split(" ")[0],
      number: found.number,
      book: found.book,
      text: found.snippet,
    });
  },
  rewindThread: (turnId: number) => {
    const chat = activeDemoChat();
    const index = chat.turns.findIndex((t) => t.id === turnId);
    if (index < 0) return Promise.reject(new Error(`no such turn: ${turnId}`));
    const dropped = chat.turns.slice(index);
    chat.turns = chat.turns.slice(0, index);
    if (!chat.turns.length) chat.summary = "";
    chat.updatedAt = new Date().toISOString();
    return Promise.resolve({
      dropped: dropped.length,
      question: dropped[0].question,
      thread: demoThreadState(chat),
      chats: demoChatList(),
    });
  },
  deleteChat: (id: string) => {
    const before = demoChats.length;
    demoChats = demoChats.filter((c) => c.id !== id);
    if (!demoChats.length) {
      startDemoChat();
    } else if (!demoChats.some((c) => c.id === demoActiveId)) {
      demoActiveId = demoChats[0].id;
    }
    return Promise.resolve({
      deleted: demoChats.length !== before,
      thread: demoThreadState(),
      chats: demoChatList(),
    });
  },
  resetThread: () => {
    const chat = startDemoChat();
    return Promise.resolve({ reset: true, thread: demoThreadState(chat), chats: demoChatList() });
  },
  compactThread: () => {
    const chat = activeDemoChat();
    const dropped = chat.turns.slice(0, Math.max(0, chat.turns.length - 2));
    if (!dropped.length) {
      return Promise.resolve({ compacted: false, dropped: 0, thread: demoThreadState(chat) });
    }
    const asked = dropped.map((t) => t.question.replace(/[?.!]+$/, "")).join("; ");
    chat.summary = `Earlier questions: ${asked}. Sources cited: ${DEMO_BOOK.title}.`;
    chat.turns = chat.turns.slice(-2);
    chat.compactions += 1;
    chat.updatedAt = new Date().toISOString();
    return Promise.resolve({ compacted: true, dropped: dropped.length, thread: demoThreadState(chat) });
  },
  suggestFollowups: (count?: number) =>
    Promise.resolve(
      [
        "What does “the reward follows what a person actually intended” mean in practice?",
        "Are there conditions or exceptions to this?",
        "How would that look in an ordinary day?",
      ].slice(0, count ?? 3),
    ),
  listLlms: () =>
    Promise.resolve({
      active: "(preview)",
      llms: [
        {
          path: "(preview)",
          name: "Gemma 4 E2B Instruct (Q8_0)",
          size: 4967490080,
          installed: true,
          default: true,
          selected: true,
        },
      ],
    }),
  stats: () => Promise.resolve({ books: 1, chunks: 14, embedded: 14 }),
  modelsStatus: () =>
    Promise.resolve({
      llm: { ...demoModel, name: "Qwen3-8B-Instruct (preview)" },
      embedder: { ...demoModel, name: "BGE-M3 (preview)" },
      reranker: { ...demoModel, name: "bge-reranker-v2-m3 (preview)" },
      all_ready: true,
      total_bytes: 0,
      ocr: { mode: "auto", available: false },
      llm_active: demoActiveState,
    }),
  downloadModel: (model: "llm" | "embedder" | "reranker") =>
    Promise.resolve({ model, ready: true, dir: "demo" }),
  setLlm: (_path: string | null) => Promise.resolve({ llm_active: demoActiveState }),
  setLlmCtx: (_ctx: number) => {
    demoActiveState = { ...demoActiveState, ctx: _ctx };
    return Promise.resolve({ llm_active: demoActiveState });
  },
  setLlmOptions: (opts: { device?: LlmDevice; kv?: KvCache; auto_load?: boolean }) => {
    demoActiveState = { ...demoActiveState, ...opts };
    return Promise.resolve({ llm_active: demoActiveState });
  },
  listBookmarks: () => Promise.resolve(demoBookmarks),
  addBookmark: (bookmark: NewBookmark) =>
    Promise.resolve({
      id: Date.now(),
      bookmarks: [
        { ...bookmark, id: Date.now(), created_at: new Date().toISOString() },
        ...demoBookmarks,
      ],
    }),
  deleteBookmark: (id: number) =>
    Promise.resolve({ deleted: true, bookmarks: demoBookmarks.filter((b) => b.id !== id) }),
  updateBookmark: (id: number, description: string | null) =>
    Promise.resolve({
      updated: true,
      bookmarks: demoBookmarks.map((b) => (b.id === id ? { ...b, description } : b)),
    }),
  dailyHadith: (date?: string) =>
    Promise.resolve({
      date: date ?? new Date().toISOString().slice(0, 10),
      collection: "bukhari",
      collection_name: "Sahih al-Bukhari",
      short: "Bukhari",
      number: 1,
      book: "Revelation",
      text:
        "I heard Allah's Messenger (ﷺ) saying, \"The reward of deeds depends upon the intentions and every person will get the reward according to what he has intended.\"",
      also_in: [],
    } satisfies DailyHadith),
  onProgress: () => Promise.resolve(() => {}),
};

const realApi = {
  ping: () =>
    invoke<{ pong: boolean; version: string; library: string; ocr?: { mode: string; available: boolean } }>(
      "sidecar_ping",
    ),

  listBooks: () => invoke<{ books: Book[] }>("list_books").then((r) => r.books),

  importBook: (path: string, title?: string) =>
    invoke<ImportResult>("import_book", { path, title: title ?? null }),

  rebuildBook: (bookId: number) => invoke<RebuildResult>("rebuild_book", { bookId }),

  reindexBook: (bookId: number) =>
    invoke<{ book_id: number; embedded: number; total: number }>("reindex_book", { bookId }),

  deleteBook: (bookId: number) => invoke<{ deleted: boolean }>("delete_book", { bookId }),

  ask: (query: string, bookIds: number[] | null, askId?: string, newThread?: boolean) =>
    invoke<AskResult>("ask", { query, bookIds, askId: askId ?? null, newThread: newThread ?? false }),

  listChats: () => invoke<{ chats: ChatSummary[]; active: string | null }>("list_chats"),

  threadState: () => invoke<{ thread: ThreadState; chats: ChatSummary[] }>("thread_state"),

  newChat: () => invoke<{ thread: ThreadState; chats: ChatSummary[] }>("new_chat"),

  openChat: (id: string) =>
    invoke<{ thread: ThreadState; chats: ChatSummary[] }>("open_chat", { id }),

  renameChat: (id: string, title: string) =>
    invoke<{ chats: ChatSummary[] }>("rename_chat", { id, title }),

  deleteChat: (id: string) =>
    invoke<{ deleted: boolean; thread: ThreadState; chats: ChatSummary[] }>("delete_chat", { id }),

  resetThread: () =>
    invoke<{ reset: boolean; thread: ThreadState; chats: ChatSummary[] }>("reset_thread"),

  compactThread: () =>
    invoke<{ compacted: boolean; dropped: number; thread: ThreadState }>("compact_thread"),

  rewindThread: (turnId: number) =>
    invoke<{ dropped: number; question: string; thread: ThreadState; chats: ChatSummary[] }>(
      "rewind_thread",
      { turnId },
    ),

  readNarration: (collection: string, number: number) =>
    invoke<{ narration: Narration }>("read_narration", { collection, number }).then(
      (r) => r.narration as Narration,
    ),

  suggestFollowups: (count?: number) =>
    invoke<{ suggestions: string[] }>("suggest_followups", { count: count ?? 3 }).then(
      (r) => r.suggestions,
    ),

  listLlms: () => invoke<{ llms: LlmOption[]; active: string }>("list_llms"),

  stats: () =>
    invoke<{ books: number; chunks: number; embedded: number }>("stats"),

  modelsStatus: () => invoke<ModelsStatus>("models_status"),

  downloadModel: (model: "llm" | "embedder" | "reranker") =>
    invoke<{ model: string; ready: boolean; dir: string }>("download_model", { model }),

  setLlm: (path: string | null) =>
    invoke<{ llm_active: ActiveModel }>("set_llm", { path }),

  setLlmCtx: (ctx: number) =>
    invoke<{ llm_active: ActiveModel }>("set_llm_ctx", { ctx }),

  setLlmOptions: (opts: { device?: LlmDevice; kv?: KvCache; auto_load?: boolean }) =>
    invoke<{ llm_active: ActiveModel }>("set_llm_options", {
      device: opts.device ?? null,
      kv: opts.kv ?? null,
      autoLoad: opts.auto_load ?? null,
    }),

  listBookmarks: () =>
    invoke<{ bookmarks: Bookmark[] }>("list_bookmarks").then((r) => r.bookmarks),

  addBookmark: (bookmark: NewBookmark) =>
    invoke<{ id: number; bookmarks: Bookmark[] }>("add_bookmark", { bookmark }),

  deleteBookmark: (id: number) =>
    invoke<{ deleted: boolean; bookmarks: Bookmark[] }>("delete_bookmark", { id }),

  updateBookmark: (id: number, description: string | null) =>
    invoke<{ updated: boolean; bookmarks: Bookmark[] }>("update_bookmark", { id, description }),

  dailyHadith: (date?: string) =>
    invoke<{ hadith: DailyHadith }>("daily_hadith", { date: date ?? null }).then((r) => r.hadith),

  onProgress: (cb: (p: Progress) => void): Promise<UnlistenFn> =>
    listen<Progress>("sidecar-progress", (e) => cb(e.payload)),
};

export const api = isTauri() ? realApi : demoApi;

export function formatMB(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  return `${bytes} B`;
}
