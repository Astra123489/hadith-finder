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

export type Progress =
  | ReindexProgress
  | ImportProgress
  | TokenProgress
  | DownloadProgress
  | CrossrefProgress;

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

export interface ActiveModel {
  name: string;
  path: string;
  exists: boolean;
  size: number;
  custom: boolean;
}

export type ModelsStatus = Record<"llm" | "embedder" | "reranker", ModelInfo> & {
  all_ready: boolean;
  total_bytes: number;
  ocr?: { mode: string; available: boolean };
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

const demoApi = {
  ping: () =>
    Promise.resolve({
      pong: true,
      version: "0.1.0 (browser preview)",
      library: "demo library (in-browser preview)",
      ocr: { mode: "auto", available: false },
    }),
  listBooks: () => Promise.resolve([DEMO_BOOK]),
  importBook: () =>
    Promise.reject(new Error("The browser preview cannot import PDFs — use the desktop app.")),
  rebuildBook: () => Promise.reject(new Error("The browser preview cannot rebuild a book.")),
  reindexBook: () => Promise.resolve({ book_id: 999, embedded: 14, total: 14 }),
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
      ocr: { mode: "auto", available: false },
    }),
  downloadModel: (model: "llm" | "embedder" | "reranker") =>
    Promise.resolve({ model, ready: true, dir: "demo" }),
  setLlm: (_path: string | null) =>
    Promise.resolve({
      llm_active: {
        name: "gemma-4-E2B-it-uncensored-Q8_0",
        path: "(preview)",
        exists: true,
        size: 4967490080,
        custom: true,
      },
    }),
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

  ask: (query: string, bookIds: number[] | null, careful: boolean, askId?: string) =>
    invoke<AskResult>("ask", { query, bookIds, careful, askId: askId ?? null }),

  stats: () =>
    invoke<{ books: number; chunks: number; embedded: number }>("stats"),

  modelsStatus: () => invoke<ModelsStatus>("models_status"),

  downloadModel: (model: "llm" | "embedder" | "reranker") =>
    invoke<{ model: string; ready: boolean; dir: string }>("download_model", { model }),

  setLlm: (path: string | null) =>
    invoke<{ llm_active: ActiveModel }>("set_llm", { path }),

  onProgress: (cb: (p: Progress) => void): Promise<UnlistenFn> =>
    listen<Progress>("sidecar-progress", (e) => cb(e.payload)),
};

export const api = isTauri() ? realApi : demoApi;

export function formatMB(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  return `${bytes} B`;
}
