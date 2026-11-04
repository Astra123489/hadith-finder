# Hadith Finder — Architecture

A free, fully-local desktop app (Tauri 2, Windows first) that lets anyone load their
own digital books (text-based PDFs, Arabic and/or English) and ask questions in plain
language. The app reasons over the book and points to the exact passage — with page
numbers, hadith-style references, and the text highlighted in the PDF viewer.

Copyright note: the app is distributed **without any book content** ("bring your own
PDF"). Only the app, models, and code ship from this repository.

## Component map

```
app/                  Tauri 2 shell — React + TypeScript frontend
  src/                Library UI, Ask UI, PDF viewer with highlight overlay
  src-tauri/          Rust core: app state, IPC, sidecar lifecycle, SQLite glue
sidecar/              Python AI backend (bundled as one exe via PyInstaller)
  hadith_sidecar/
    __main__.py       stdio newline-delimited JSON-RPC loop
    protocol.py       request/response framing, error codes
    db.py             SQLite schema (books, chunks, embeddings, FTS5, gradings)
    normalize.py      Arabic text normalization (diacritics, alef/ya forms)
    chunking.py       hadith-aware boundary detection (numbering, حَدَّثَنَا)
    ingest.py         PyMuPDF: text + word bounding boxes per page → chunks + spans
    embeddings.py     provider interface: ONNX BGE-M3 (default) | hash (tests)
    retrieval.py      hybrid search: FTS5 keyword + dense vectors, RRF fusion
    rerank.py         provider interface: bge-reranker-v2-m3 ONNX | identity
    gradings.py       extract authenticity rulings (صحيح/حسن/ضعيف ...) when printed
    qa.py             orchestrates: normalize → retrieve → rerank → prompt → answer
    llm.py            provider interface: llama.cpp (production) | echo (tests)
  tests/              unit + end-to-end tests on synthetic bilingual PDFs
docs/                 this file + model selection rationale
```

## Why this split

- **Tauri/Rust**: small binary, native installer, secure by default, owns files,
  DB, and window management. No Python needed for the UI.
- **Python sidecar** (not pure Rust): PyMuPDF is the only widely-used open tool that
  reliably extracts text *and glyph-ordered Arabic* from text PDFs, plus word
  bounding boxes (needed for highlighting). Rust PDF crates mangle Arabic glyph
  order/ligatures. The sidecar keeps that complexity contained.
- Bundled with PyInstaller → single `hadith-sidecar.exe`; end users never see Python.
  Rust spawns it on app start, talks over stdio (no ports, no firewall prompts).

## Storage

One SQLite file per library (`library.haddb`):

- `books` — id, title, author, file path, SHA-256, page count, imported at
- `chunks` — book_id, page range, `hadith_ref` (e.g. "1/42"), chapter, `arabic`,
  `english`, raw text, normalized text, token count, `spans_json` (per page:
  merged word rects for highlight overlays)
- `chunks_fts` — FTS5 virtual table over normalized Arabic+English text (keyword arm)
- `embeddings` — float32 vectors (BLOB), cosine via in-memory NumPy (a full
  multi-book library is ≲ 50k chunks ≈ 200 MB — trivial to scan)
- `gradings` — extracted rulings per chunk (when the book prints them), else the
  UI says so explicitly
- `meta` — schema version, model info

## Accuracy pipeline (the point of the project)

1. **Hadith-aware chunking.** Numbered boundaries (`[123]`, `123.`, Arabic numerals
   line-start) or, failing that, `حَدَّثَنَا` at line starts. Tiny fragments merge,
   oversized hadiths split at sentence boundaries. Chapter headings
   (كتاب/باب/Book/Chapter) attach to following chunks as metadata.
2. **Arabic normalization** (search index only; originals preserved for display):
   strip tashkeel, unify alef variants → ا, ى→ي, drop tatweel, collapse whitespace.
3. **Hybrid retrieval.** FTS5 (BM25) + dense cosine, fused with Reciprocal Rank
   Fusion (k=60). Catches both conceptual matches and exact names/terms.
4. **Cross-encoder rerank** (bge-reranker-v2-m3, multilingual): top-30 → top-5.
   Largest single precision win.
5. **Grounded answering.** Strict prompt: answer ONLY from supplied excerpts, cite
   inline as [S1..Sn] (each S# carries book/chapter/hadith/page), and if nothing
   suffices, reply exactly "I could not find this in the loaded books." — no
   invented hadiths.
6. **Summarization** = map-reduce over chapter slices, same citation rules.
7. **Gradings**: regex/keyword extraction of صحيح/حسن/ضعيف/موضوع + English
   "sahih/hasan/da'if/mawdu" when present in the printed text.

## LLM choice (16 GB RAM / 4–8 GB VRAM / integrated GPU)

| Role | Model | License | Size Q4_K_M |
|------|-------|---------|-------------|
| Default | Qwen3-8B-Instruct (thinking toggle: off = Fast, on = Careful) | Apache-2.0 | ~5.2 GB |
| Arabic-first option (settings) | Falcon-H1-Arabic-7B | Apache-2.0 | ~4.7 GB |
| Fast mode for weak CPUs | Qwen3-4B | Apache-2.0 | ~2.5 GB |
| Embeddings | BGE-M3 (dense arm) | MIT | ~1.1 GB (int8 ONNX) |
| Reranker | bge-reranker-v2-m3 | MIT | ~1.1 GB (int8 ONNX) |

- Models are **downloaded on first run** (HuggingFace, SHA-256 verified, resumable),
  not bundled — installer stays ~40 MB.
- 8B on pure CPU ≈ 2–5 tok/s (usable), faster with partial GPU offload. Careful
  mode trades latency for reasoning on "explain the meaning" questions.
- Llama excluded (weaker Arabic, restrictive license). 14B+ excluded (too slow here).

## Highlighting (all three layers)

1. **Page citations** — every chunk stores `page_start`/`page_end`; the viewer opens
   PDF.js at that page.
2. **Hadith-style references** — parsed from the book's own numbering; cross-book
   canonical refs only when the printed text contains them.
3. **Highlighted passage** — word rects captured at ingest, merged per line,
   rendered as yellow overlay divs in the PDF viewer (PDF.js viewport-aware).

## Build order

1. ✅ Scaffold (app/, sidecar/, this doc)
2. ✅ Sidecar ingest: PDF → hadith chunks + spans → SQLite (tested)
3. ✅ Hybrid retrieval + rerank wiring (tested)
4. ✅ Ask flow end-to-end + citations
5. ✅ Frontend: library, import, ask, PDF viewer + highlight overlay (verified)
6. ✅ Model downloader (Settings → AI Models, resumable, size-verified)
7. 📦 PyInstaller sidecar + Tauri NSIS installer (in progress)

## Verified with real models (on-device, demo bilingual book)

- BGE-M3 int8 ONNX embeddings: dim 1024, 14 passages embedded < 1 s (CPU).
- Cross-lingual Arabic→English retrieval: query "نية" → intent hadiths surface
  (top hits: refs 5 and 2). Reranker query "weak narration" → da'if narration
  ref 7 scores **+3.9** while everything else scores negative.
- Highlight overlay positions exactly over cited passages (English + shaped-RTL
  Arabic), verified visually on the demo book.
