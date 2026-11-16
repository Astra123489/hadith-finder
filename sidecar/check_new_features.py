"""Drive the sidecar over its stdio protocol and check the new settings,
bookmarks and daily-hadith methods."""

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(tempfile.gettempdir(), "hf_settings_test.haddb")
for suffix in ("", "-wal", "-shm"):
    try:
        os.remove(LIB + suffix)
    except OSError:
        pass

proc = subprocess.Popen(
    [sys.executable, "-m", "hadith_sidecar", LIB],
    cwd=HERE,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
    encoding="utf-8",
)
_next_id = [0]


def call(method, params=None):
    _next_id[0] += 1
    rid = _next_id[0]
    proc.stdin.write(json.dumps({"id": rid, "method": method, "params": params or {}}) + "\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if not line:
            raise SystemExit(f"sidecar closed while waiting for {method}")
        msg = json.loads(line)
        if msg.get("id") == rid:
            if not msg.get("ok"):
                raise SystemExit(f"{method} failed: {msg.get('error')}")
            return msg["result"]
        print("  notify:", json.dumps(msg)[:160])


failures = []


def check(label, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {label} {detail}")
    if not condition:
        failures.append(label)


ping = call("ping")
check("ping", ping.get("pong") is True, ping.get("version"))

status = call("models_status")
active = status["llm_active"]
check("device default auto", active["device"] == "auto", active["device"])
check("kv default f16", active["kv"] == "f16", active["kv"])
check("auto_load default off", active["auto_load"] is False)
check("kv options", active["kv_options"] == ["f16", "q8_0", "q4_0"], active["kv_options"])
check("runtime not loaded", active["runtime"]["loaded"] is False)

after = call("set_llm", {"device": "cpu", "kv": "q8_0", "auto_load": True})["llm_active"]
check("device saved", after["device"] == "cpu", after["device"])
check("kv saved", after["kv"] == "q8_0", after["kv"])
check("auto_load saved", after["auto_load"] is True)
check("gpu info present", "available" in after["gpu"], after["gpu"])

try:
    call("set_llm", {"device": "tpu"})
    check("rejects bad device", False)
except SystemExit as exc:
    check("rejects bad device", "device must be one of" in str(exc))

try:
    call("set_llm", {"kv": "q2_k"})
    check("rejects bad kv", False)
except SystemExit as exc:
    check("rejects bad kv", "kv cache must be one of" in str(exc))

day = call("daily_hadith", {"date": "2026-11-15"})["hadith"]
check("daily hadith has text", bool(day and len(day["text"]) > 80), day["collection_name"] if day else "")
check("daily hadith stable", call("daily_hadith", {"date": "2026-11-15"})["hadith"]["number"] == day["number"])
other = call("daily_hadith", {"date": "2026-11-16"})["hadith"]
check("next day differs", other["number"] != day["number"] or other["collection"] != day["collection"])

try:
    call("daily_hadith", {"date": "15/11/2026"})
    check("rejects bad date", False)
except SystemExit as exc:
    check("rejects bad date", "ISO date" in str(exc))

added = call(
    "add_bookmark",
    {
        "bookmark": {
            "kind": "canonical",
            "title": day["collection_name"],
            "chapter": day["book"],
            "reference": f"Hadith {day['number']}",
            "collection": day["collection"],
            "number": day["number"],
            "excerpt": day["text"],
            "description": "Read again on the trip",
        }
    },
)
check("bookmark added", added["id"] > 0 and len(added["bookmarks"]) == 1)
bookmark = added["bookmarks"][0]
check("bookmark fields", bookmark["collection"] == day["collection"] and bookmark["number"] == day["number"])
check("bookmark description", bookmark["description"] == "Read again on the trip")
check("bookmark gradings list", bookmark["gradings"] == [])

library = call(
    "add_bookmark",
    {
        "bookmark": {
            "kind": "library",
            "title": "My Book",
            "chapter": "Chapter 2",
            "reference": "p. 42",
            "page_start": 41,
            "page_end": 41,
            "excerpt": "A narration from an imported book.",
            "gradings": ["sahih"],
            "spans_json": {"41": [[10, 20, 100, 120]]},
            "book_id": 7,
            "chunk_id": 99,
        }
    },
)
check("two bookmarks", len(library["bookmarks"]) == 2)
lib = library["bookmarks"][0]
check("spans parsed", lib["spans"] == {"41": [[10, 20, 100, 120]]}, lib["spans"])
check("gradings parsed", lib["gradings"] == ["sahih"])

updated = call("update_bookmark", {"id": bookmark["id"], "description": "  Updated note  "})
check("description updated", updated["bookmarks"][-1]["description"] == "Updated note")

try:
    call("add_bookmark", {"bookmark": {"title": "No text"}})
    check("rejects empty excerpt", False)
except SystemExit as exc:
    check("rejects empty excerpt", "needs an excerpt" in str(exc))

removed = call("delete_bookmark", {"id": lib["id"]})
check("bookmark deleted", removed["deleted"] is True and len(removed["bookmarks"]) == 1)

# shutdown exits without replying by design.
try:
    proc.stdin.write(json.dumps({"id": 999, "method": "shutdown", "params": {}}) + "\n")
    proc.stdin.flush()
except (BrokenPipeError, ValueError):
    pass
proc.wait(timeout=10)

# --- auto-load at launch (llm_auto_load is now 1 in this library) ---
print("\nlaunch with auto-load enabled:")
proc2 = subprocess.Popen(
    [sys.executable, "-m", "hadith_sidecar", LIB],
    cwd=HERE,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
    encoding="utf-8",
)
first = json.loads(proc2.stdout.readline())
check(
    "preload announced without blocking",
    first.get("type") == "progress" and first.get("op") == "llm",
    f"stage={first.get('stage')}",
)
proc2.stdin.write(json.dumps({"id": 1, "method": "ping", "params": {}}) + "\n")
proc2.stdin.flush()
while True:
    msg = json.loads(proc2.stdout.readline())
    if msg.get("id") == 1:
        check("app usable while model loads", msg.get("ok") is True)
        break
proc2.stdin.close()
proc2.wait(timeout=20)

print("\nfailures:", failures or "none")
sys.exit(1 if failures else 0)
