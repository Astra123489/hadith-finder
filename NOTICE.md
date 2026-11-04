# NOTICE — third-party software and models

Hadith Finder itself is AGPL-3.0 (see LICENSE). It bundles or downloads the following
third-party works, each under its own license:

## Bundled / required libraries

| Component | License | Notes |
|---|---|---|
| PyMuPDF (MuPDF) | AGPL-3.0 | PDF text + word-position extraction |
| SQLite (via Python stdlib) | Public domain | Local storage |
| NumPy | BSD-3-Clause | Vector math |
| Tauri | MIT / Apache-2.0 | Desktop shell |
| React | MIT | UI |
| pdf.js (pdfjs-dist) | Apache-2.0 | In-app PDF rendering |
| llama-cpp-python (when bundled) | MIT | LLM inference |
| ONNX Runtime (when bundled) | MIT | Embedding / reranker inference |
| tokenizers (HuggingFace) | Apache-2.0 | Tokenizer |

## Models (downloaded on first run, not shipped with the installer)

| Model | Author | License |
|---|---|---|
| Qwen3-8B-Instruct (Q4_K_M GGUF) | Alibaba / Qwen team | Apache-2.0 |
| BGE-M3 | BAAI | MIT |
| bge-reranker-v2-m3 | BAAI | MIT |

Model weights remain the property of their authors. Thanks to the Qwen team, BAAI, the
PyMuPDF team (Artifex), z.ai's llama.cpp project, and the entire open-source community.

## Books

Hadith Finder contains **no** hadith books, translations, or religious texts of any kind.
Classical hadith texts are classical works; their translations may be copyrighted. Users
must obtain their own copies. Nothing in this project endorses or is endorsed by any
religious authority.

## Attribution

If you redistribute or fork Hadith Finder, please keep this NOTICE, the LICENSE, and the
“You must verify religious content against the sources” disclaimer intact.
