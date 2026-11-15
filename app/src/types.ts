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

export interface AskResult {
  answer: string;
  citations: Citation[];
  retrieved: number;
  /** The model answered without citing an excerpt. */
  unverified: boolean;
  scoped_empty: boolean;
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
