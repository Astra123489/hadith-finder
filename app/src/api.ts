import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type { AskResult, Book, ImportResult } from "./types";

export interface ReindexProgress {
  type: "progress";
  op: "reindex";
  book_id: number;
  done: number;
  total: number;
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

export type ModelsStatus = Record<"llm" | "embedder" | "reranker", ModelInfo> & {
  all_ready: boolean;
  total_bytes: number;
};

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
    "The demo book records the Prophet ﷺ teaching that deeds are judged by their intentions, and that each person shall have only what they intended [S1]. A parallel narration states the same principle twice, with the compiler grading the transmission sahih [S2][S3]. Taken together, the narrations agree on a single rule: the written reward of an act turns on the intention behind it, not on its outward shape — and even a good intention denied its act is still recorded for the servant.",
  citations: [
    {
      chunk_id: 1001,
      book_id: 999,
      book_title: DEMO_BOOK.title,
      chapter: "Chapter 1 — Book of Intentions",
      hadith_ref: "1",
      page_start: 0,
      page_end: 0,
      spans: { "0": [[55.0, 116.94, 539.79, 129.22], [55.0, 130.15, 193.98, 142.42]] },
      excerpt:
        "1. Narrated Tester: The Prophet ﷺ said, 'Deeds are judged by intentions, and every person shall have only what he intended.'",
      gradings: [],
    },
    {
      chunk_id: 1005,
      book_id: 999,
      book_title: DEMO_BOOK.title,
      chapter: "Chapter 2 — Book of Sincerity",
      hadith_ref: "5",
      page_start: 1,
      page_end: 1,
      spans: { "1": [[54.62, 54.94, 343.26, 67.22]] },
      excerpt:
        "5. قال رسول الله ﷺ: إنما الأعمال بالنيات، وإنما لكل امرئ ما نوى. حديث صحيح.",
      gradings: ["sahih"],
    },
    {
      chunk_id: 1006,
      book_id: 999,
      book_title: DEMO_BOOK.title,
      chapter: "Chapter 2 — Book of Sincerity",
      hadith_ref: "5",
      page_start: 1,
      page_end: 1,
      spans: { "1": [[55.0, 75.15, 539.71, 87.42], [55.0, 88.35, 202.51, 100.62]] },
      excerpt:
        "5. The Messenger of Allah ﷺ said, 'Actions are by intentions, and each person shall have what they intended.' This hadith is sahih.",
      gradings: ["sahih"],
    },
  ],
  retrieved: 9,
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

const demoApi = {
  ping: () => Promise.resolve({ pong: true, version: "0.1.0 (browser preview)", library: "demo library (in-browser preview)" }),
  listBooks: () => Promise.resolve([DEMO_BOOK]),
  importBook: () => Promise.reject(new Error("The browser preview cannot import PDFs — use the desktop app.")),
  reindexBook: () => Promise.resolve({ book_id: 999, embedded: 14 }),
  deleteBook: () => Promise.resolve({ deleted: false }),
  ask: () => Promise.resolve(DEMO_RESULT),
  stats: () => Promise.resolve({ books: 1, chunks: 14, embedded: 14 }),
  modelsStatus: () =>
    Promise.resolve({
      llm: { ...demoModel, name: "Qwen3-8B-Instruct (preview)" },
      embedder: { ...demoModel, name: "BGE-M3 (preview)" },
      reranker: { ...demoModel, name: "bge-reranker-v2-m3 (preview)" },
      all_ready: true,
      total_bytes: 0,
    }),
  downloadModel: (model: "llm" | "embedder" | "reranker") =>
    Promise.resolve({ model, ready: true, dir: "demo" }),
  onProgress: () => Promise.resolve(() => {}),
};

const realApi = {
  ping: () => invoke<{ pong: boolean; version: string; library: string }>("sidecar_ping"),

  listBooks: () => invoke<{ books: Book[] }>("list_books").then((r) => r.books),

  importBook: (path: string, title?: string) =>
    invoke<ImportResult>("import_book", { path, title: title ?? null }),

  reindexBook: (bookId: number) =>
    invoke<{ book_id: number; embedded: number }>("reindex_book", { bookId }),

  deleteBook: (bookId: number) => invoke<{ deleted: boolean }>("delete_book", { bookId }),

  ask: (query: string, bookIds: number[] | null, careful: boolean) =>
    invoke<AskResult>("ask", { query, bookIds, careful }),

  stats: () =>
    invoke<{ books: number; chunks: number; embedded: number }>("stats"),

  modelsStatus: () => invoke<ModelsStatus>("models_status"),

  downloadModel: (model: "llm" | "embedder" | "reranker") =>
    invoke<{ model: string; ready: boolean; dir: string }>("download_model", { model }),

  onProgress: (cb: (p: ReindexProgress | DownloadProgress) => void): Promise<UnlistenFn> =>
    listen<ReindexProgress | DownloadProgress>("sidecar-progress", (e) => cb(e.payload)),
};

export const api = isTauri() ? realApi : demoApi;

export function formatMB(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  return `${bytes} B`;
}
