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
}

export interface AskResult {
  answer: string;
  citations: Citation[];
  retrieved: number;
}

export interface ImportResult {
  book_id: number;
  title: string;
  page_count: number;
  chunks: number;
  language?: string;
  already_exists: boolean;
}
