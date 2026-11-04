import json
import os
import pymupdf
import subprocess
import sys
from pathlib import Path

exe = Path(sys.argv[1])
library = Path(sys.argv[2])

# build a sample pdf with no spaces in the path
sample = Path(os.environ.get("TEMP", ".")) / "hf_exe_smoke_book.pdf"
doc = pymupdf.open()
page = doc.new_page()
page.insert_text((72, 72), "1. Narrated Tester: The Prophet said intentions raise every deed with care and attention.", fontsize=11)
page.insert_text((72, 90), "2. Narrated Tester: The Prophet said whoever shows mercy will be shown mercy.", fontsize=11)
doc.save(str(sample))
doc.close()
print("sample:", sample, sample.exists())

env = dict(os.environ)
env.update({
    "HADITH_EMBEDDER": "hash",
    "HADITH_LLM": "echo",
    "HADITH_RERANKER": "identity",
    "HADITH_LIBRARY": str(library),
    "HADITH_MODELS_DIR": str(library.parent / "models"),
})

proc = subprocess.Popen(
    [str(exe), str(library)],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True, encoding="utf-8", errors="replace", env=env,
)
requests = [
    {"id": 1, "method": "ping", "params": {}},
    {"id": 2, "method": "import_book", "params": {"path": str(sample), "title": "Smoke Book"}},
    {"id": 3, "method": "reindex_book", "params": {"book_id": 1, "batch": 4}},
    {"id": 4, "method": "ask", "params": {"query": "intentions"}},
    {"id": 5, "method": "models_status", "params": {}},
    {"id": 6, "method": "shutdown", "params": {}},
]
for req in requests:
    proc.stdin.write(json.dumps(req) + "\n")
    proc.stdin.flush()
proc.stdin.close()

for raw in proc.stdout:
    msg = json.loads(raw)
    if "id" not in msg:
        print("progress:", msg)
        continue
    if msg["id"] == 1:
        print("ping:", msg["result"])
    elif msg["id"] == 2:
        print("import:", msg)
    elif msg["id"] == 4 and msg.get("ok"):
        print("ask ok — citations:", [(c["hadith_ref"], c["page_start"]) for c in msg["result"]["citations"]])
    elif msg["id"] == 5 and msg.get("ok"):
        print("models_status all_ready:", msg["result"]["all_ready"])
    else:
        print(f"id={msg['id']} ok={msg.get('ok')}", msg.get("error") or "")
proc.wait(timeout=120)
print("exit code:", proc.returncode)
