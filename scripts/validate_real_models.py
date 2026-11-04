"""Downloads the real open models into a throwaway test dir and validates
the actual accuracy pipeline on the demo book. Not shipped; dev tool only."""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sidecar"))

TEST_DIR = os.path.join(os.environ.get("TEMP", "."), "hf_models_test")
os.environ["HADITH_MODELS_DIR"] = os.path.join(TEST_DIR, "models")
TEST_LIB = os.path.join(TEST_DIR, "library.haddb")
os.makedirs(TEST_DIR, exist_ok=True)

STAGE = sys.argv[1] if len(sys.argv) > 1 else "embed-rerank"

from hadith_sidecar import db, protocol
from hadith_sidecar.downloader import download_model, status
from hadith_sidecar.embeddings import OnnxM3Embedder
from hadith_sidecar.rerank import OnnxReranker
from hadith_sidecar.ingest import ingest_pdf
from hadith_sidecar.retrieval import hybrid_search
from hadith_sidecar.qa import ask


def notify(**kw):
    op = kw.get("op")
    if op == "download":
        mb = kw["downloaded"] / 1e6
        tot = kw["total"] / 1e6
        print(f"\r  {kw['model']}/{kw['file']}: {mb:.0f}/{tot:.0f} MB", end="", flush=True)
    else:
        print(kw)


if STAGE == "download":
    for model in ["embedder", "reranker", "llm"]:
        print(f"=== downloading {model} ===")
        download_model(model, notify)
        print()
    print("all models present")
    sys.exit(0)

print("models status:", status()["all_ready"])

conn = db.connect(TEST_LIB)
demo = os.path.join(os.path.dirname(__file__), "..", "demo", "demo-book-of-intentions.pdf")
ingest_pdf(conn, demo, title="Demo Book of Intentions")
embedder = OnnxM3Embedder(os.environ["HADITH_MODELS_DIR"] + "/bge-m3")

rows = list(db.iter_book_chunks(conn, 1))
print(f"reindexing {len(rows)} chunks with real BGE-M3…")
t0 = time.time()
vecs = embedder.embed([r["norm_text"] for r in rows])
with conn:
    for r, v in zip(rows, vecs):
        db.set_embedding(conn, r["id"], v)
print(f"embedded in {time.time() - t0:.1f}s, dim={vecs.shape[1]}")

reranker = OnnxReranker(os.environ["HADITH_MODELS_DIR"] + "/bge-reranker-v2-m3")

for query in ["mercy towards others", "نية", "weak narration"]:
    hits = hybrid_search(conn, query, embedder, pool=30)
    chunks = db.get_chunks(conn, [h.chunk_id for h in hits[:8]])
    scores = reranker.rerank(query, [c["text"] for c in chunks])
    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    print(f"\nquery: {query}")
    for i in order[:3]:
        c = chunks[i]
        print(f"  score={scores[i]:.3f} ref={c['hadith_ref']} :: {c['text'][:80].encode('unicode_escape').decode()[:80]}")

if STAGE == "ask":
    from hadith_sidecar.llm import LlamaCppProvider
    print("\n=== real LLM answer (may take a few minutes on CPU) ===")
    llm = LlamaCppProvider()
    result = ask(conn, "What does the Prophet say about mercy?", book_ids=None,
                 embedder=embedder, llm=llm, reranker=reranker, careful=False)
    print("answer:", result["answer"][:800])
    print("citations:", [(c["hadith_ref"], c["page_start"] + 1, c["gradings"]) for c in result["citations"]])
