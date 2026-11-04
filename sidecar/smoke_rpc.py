import json
import subprocess
import sys
from pathlib import Path

env_lib = sys.argv[1]
sample_pdf = sys.argv[2]

proc = subprocess.Popen(
    [sys.executable, "-m", "hadith_sidecar", env_lib],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True, encoding="utf-8", cwd=str(Path(__file__).resolve().parent),
)

requests = [
    {"id": 1, "method": "ping", "params": {}},
    {"id": 2, "method": "import_book", "params": {"path": sample_pdf, "title": "Smoke Book"}},
    {"id": 3, "method": "reindex_book", "params": {"book_id": 1, "batch": 8}},
    {"id": 4, "method": "list_books", "params": {}},
    {"id": 5, "method": "ask", "params": {"query": "mercy", "top_k": 3}},
    {"id": 6, "method": "stats", "params": {}},
    {"id": 7, "method": "shutdown", "params": {}},
]
# The sample PDF is created by conftest in a pytest tmp dir; regenerate equivalent here.
import pymupdf, tempfile

lines = [
    "Chapter 1 Book of Mercy",
    "1. Narrated Tester: The Prophet said, 'Whoever does not show mercy will",
    "   not be shown mercy.' This hadith is sahih.",
    "2. Narrated Tester: The Prophet said, 'Every act of kindness is charity.'",
]
doc = pymupdf.open()
page = doc.new_page(width=595, height=842)
y = 72.0
for l in lines:
    page.insert_text((72, y), l, fontsize=11, fontname="helv")
    y += 18
with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
    tmp_pdf = f.name
doc.save(tmp_pdf)
doc.close()

for req in requests:
    if req["method"] == "import_book":
        req["params"]["path"] = tmp_pdf
    proc.stdin.write(json.dumps(req) + "\n")
    proc.stdin.flush()

proc.stdin.close()
for raw in proc.stdout:
    msg = json.loads(raw)
    if "id" in msg:
        ok = msg.get("ok")
        if msg["method"] if False else True:
            pass
        label = {1: "ping", 2: "import", 3: "reindex", 4: "list", 5: "ask", 6: "stats", 7: "shutdown"}.get(msg["id"])
        print(f"--- {label} ok={ok}")
        if not ok:
            print(json.dumps(msg, ensure_ascii=False)[:500])
        elif msg["id"] == 5:
            print("answer:", msg["result"]["answer"][:120])
            print("citations:", [(c["hadith_ref"], c["page_start"]) for c in msg["result"]["citations"]])
    else:
        print("progress:", msg)
proc.wait(timeout=30)
print("exit code:", proc.returncode)
