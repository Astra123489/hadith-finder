export interface Book {
  id: number;
  title: string;
  author: string | null;
  language: string;
  file_path: string;
  page_count: number;
  created_at: string;
  chunk_count: number;
  embedded_count: number;
}

export type SpanMap = Record<string, number[][]>;

/** The same narration, as printed in another collection. */
export interface Parallel {
  collection: string;
  collection_name: string;
  number: number;
  book: string;
  label: string;
  snippet: string;
  score: number;
  same_collection: boolean;
}

/** A cross-reference resolved to a page in the reader's own copy of the book. */
export interface NarrationPlace {
  found: boolean;
  collection: string;
  collection_name: string;
  short: string;
  number: number;
  label: string;
  book_id?: number;
  book_title?: string;
  page_start?: number;
  page_end?: number;
  spans?: SpanMap;
  score?: number;
}

export interface Citation {
  chunk_id: number;
  book_id: number;
  book_title: string;
  chapter: string | null;
  hadith_ref: string | null;
  page_start: number;
  page_end: number;
  spans: SpanMap;
  excerpt: string;
  gradings: string[];
  /** Excerpt number in the answer (`[S3]` → 3), so badges and cards agree. */
  source_index: number;
  also_in: Parallel[];
}

/** What the question asked for, as the sidecar routed it. */
export interface TurnIntent {
  kind: string;
  count: number | null;
  chapter: string | null;
  novel: boolean;
  top_k: number;
  previous: boolean;
}

/** The source a reader pointed at with "Ask about this". */
export interface TurnReference {
  chunk_id: number;
  book_title: string;
  chapter: string | null;
  hadith_ref: string | null;
  page_start: number;
  page_end: number;
}

export interface AskResult {
  answer: string;
  citations: Citation[];
  retrieved: number;
  /** The model answered without citing an excerpt. */
  unverified: boolean;
  scoped_empty: boolean;
  /** The token limit stopped the model mid-answer. */
  truncated?: boolean;
  /** "stop" for a finished answer, "length" when the cap cut it off. */
  finish_reason?: string | null;
  /** Present when the answer concerned a ruling; shown as-is. */
  notice?: string | null;
  /** The source this question was asked about, when one was picked. */
  reference?: TurnReference | null;
  intent?: TurnIntent;
  /** The whole conversation after this answer, when the sidecar returns it. */
  thread?: ThreadState;
  chats?: ChatSummary[];
}

/** The result of finishing an answer the token limit cut off. */
export interface ContinueResult {
  turn: ThreadTurn;
  thread: ThreadState;
  chats?: ChatSummary[];
  /** The continuation was cut off again, so it can be continued again. */
  truncated: boolean;
  finish_reason?: string | null;
}

/** A saved conversation in the chat switcher. */
/** A named drawer in the conversation rail. */
export interface ChatFolder {
  id: string;
  name: string;
  created_at: string;
}

export interface ChatSummary {
  id: string;
  title: string;
  turns: number;
  created_at: string;
  updated_at: string;
  active: boolean;
  folder_id?: string | null;
}

/** How full the model's context window is, and where compaction kicks in. */
export interface ThreadUsage {
  used: number;
  ctx: number;
  /** Token count at which older turns are summarised (70% of the window). */
  compact_at: number;
  /** 0–1 share of the window in use. */
  pct: number;
  history_tokens: number;
  excerpt_tokens: number;
  turns: number;
  compactions: number;
}

/** One question and its grounded answer, kept for the conversation view. */
export interface ThreadTurn {
  id: number;
  question: string;
  answer: string;
  citations: Citation[];
  unverified: boolean;
  /** The model stopped at its token limit, so the answer is incomplete. */
  truncated?: boolean;
  /** "This is a reading aid, not a fatwa" on a ruling answer. */
  notice?: string | null;
  /** The source the reader clicked "Ask about this" on, if any. */
  reference?: TurnReference | null;
  intent?: TurnIntent | null;
  created_at: string;
}

export interface ThreadState {
  id: string;
  /** Named after the question that started the chat. */
  title: string;
  turns: ThreadTurn[];
  /** Replaces turns that were folded away to free context. */
  summary: string;
  compactions: number;
  usage: ThreadUsage;
}

export interface Bookmark {
  id: number;
  /** "library" = a passage in an imported PDF; "canonical" = a bundled collection entry. */
  kind: "library" | "canonical";
  title: string;
  chapter: string | null;
  reference: string | null;
  collection: string | null;
  number: number | null;
  page_start: number | null;
  page_end: number | null;
  excerpt: string;
  gradings: string[];
  spans: SpanMap;
  book_id: number | null;
  chunk_id: number | null;
  /** The user's own note, so a saved hadith can say why it was kept. */
  description: string | null;
  created_at: string;
}

export type NewBookmark = Omit<Bookmark, "id" | "created_at">;

/** The narration of the day from the bundled canonical collections. */
export interface DailyHadith {
  date: string;
  collection: string;
  collection_name: string;
  short: string;
  number: number;
  book: string;
  text: string;
  also_in: {
    collection: string;
    collection_name: string;
    short: string;
    number: number;
    book: string;
    snippet: string;
    label: string;
    score: number;
  }[];
}

export interface ImportResult {
  book_id: number;
  title: string;
  page_count: number;
  chunks: number;
  language?: string;
  ocr_pages?: number;
  already_exists: boolean;
}
