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

narration = call(
    "read_narration", {"collection": day["collection"], "number": day["number"]}
)["narration"]
check("read_narration returns the full matn", len(narration["text"]) >= len(day["text"]), len(narration["text"]))
check("read_narration names the collection", bool(narration["collection_name"]), narration["collection_name"])
try:
    call("read_narration", {"collection": day["collection"], "number": 999999})
    check("read_narration rejects an unknown number", False)
except SystemExit as exc:
    check("read_narration rejects an unknown number", "no such narration" in str(exc))

place = call(
    "locate_narration", {"collection": day["collection"], "number": day["number"]}
)["place"]
check("locate_narration names the collection", place["label"] == f"{place['short']} {day['number']}", place["label"])
check("locate_narration reports an absent book", place["found"] is False)
notify = call("daily_notify")
check("notify off by default", notify["enabled"] is False, notify["time"])
check("notify time defaults to 7am", notify["time"] == "07:00")
check("notify not due while off", notify["due"] is False)

armed = call("set_daily_notify", {"enabled": True, "time": "00:00"})
check("notify enabled and rescheduled", armed["enabled"] is True and armed["time"] == "00:00")
check("notify due once its time has passed", armed["due"] is True)
check(
    "due notify carries the day's narration",
    bool(armed["hadith"] and len(armed["hadith"]["text"]) > 40),
    f'{armed["hadith"]["short"]} {armed["hadith"]["number"]}' if armed["hadith"] else "",
)

try:
    call("set_daily_notify", {"time": "25:00"})
    check("notify rejects a bad time", False)
except SystemExit as exc:
    check("notify rejects a bad time", "HH:MM" in str(exc))

shown = call("mark_daily_notify", {"date": armed["date"]})
check("notify shown once a day", shown["due"] is False and shown["last_shown"] == armed["date"])

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class _StubEndpoint(BaseHTTPRequestHandler):
    """A minimal OpenAI-compatible endpoint, streaming and plain."""

    seen_auth = None

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).seen_auth = self.headers.get("Authorization")
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for piece in ("Grounded ", "answer ", "[S1]."):
                event = {"choices": [{"delta": {"content": piece}}]}
                self.wfile.write(b"data: " + json.dumps(event).encode() + bytes((13, 10, 13, 10)))
            self.wfile.write(b"data: [DONE]" + bytes((13, 10, 13, 10)))
            return
        payload = json.dumps(
            {"model": body.get("model"), "choices": [{"message": {"content": "ready"}}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


endpoint = HTTPServer(("127.0.0.1", 0), _StubEndpoint)
threading.Thread(target=endpoint.serve_forever, daemon=True).start()
api_base = f"http://127.0.0.1:{endpoint.server_port}/v1"

cfg = call("api_config")["api"]
check("endpoint off by default", cfg["enabled"] is False and cfg["has_key"] is False)

try:
    call("set_api_config", {"base_url": "ftp://example.com"})
    check("endpoint rejects a bad url", False)
except SystemExit as exc:
    check("endpoint rejects a bad url", "http" in str(exc))

stored = call("set_api_key", {"key": "sk-test-123"})["api"]
check("key stored", stored["has_key"] is True)
check("key never echoed back", "sk-test-123" not in json.dumps(stored))
key_file = LIB + ".keys.json"
check(
    "key file holds ciphertext only",
    os.path.isfile(key_file) and "sk-test-123" not in open(key_file, encoding="utf-8").read(),
)

armed = call(
    "set_api_config", {"base_url": api_base, "model": "test-model", "enabled": True}
)["api"]
check("endpoint ready", armed["ready"] is True and armed["base_url"] == api_base, armed["model"])

test = call("test_api", {})["test"]
check("endpoint answers a test call", test["ok"] is True and test["model"] == "test-model", test["reply"])
check("key sent as a bearer token", _StubEndpoint.seen_auth == "Bearer sk-test-123")

listed = call("list_llms")
check(
    "model picker offers the endpoint",
    listed["llms"][0]["path"] == "api:openai" and listed["active"] == "api:openai",
    listed["llms"][0]["name"],
)

sys.path.insert(0, HERE)
from hadith_sidecar.llm import OpenAIProvider

provider = OpenAIProvider(api_base, "test-model", "sk-test-123")
tokens = []
answer = provider.generate("system", "user", on_token=lambda piece, phase: tokens.append(piece))
check("provider streams the answer", answer == "Grounded answer [S1].", answer)
check("provider emits each delta", len(tokens) == 3, len(tokens))

cleared = call("clear_api_key")["api"]
check("key cleared", cleared["has_key"] is False and cleared["ready"] is False)
check("key file removed", not os.path.isfile(key_file))
endpoint.shutdown()

import sqlite3

# The collection really imported, with one of its narrations inside: the
# cross-reference should land on that page.
seed = sqlite3.connect(LIB)
matn = seed.execute("SELECT text FROM xref WHERE collection='bukhari' AND number=1").fetchone()[0]
seed.execute(
    "INSERT INTO books(id, title, author, language, file_path, sha256, page_count)"
    " VALUES (7, 'Sahih al-Bukhari', 'Test', 'en', 'bukhari.pdf', 'seed-7', 9)"
)
seed.execute(
    "INSERT INTO chunks(book_id, chunk_index, page_start, page_end, hadith_ref, chapter,"
    " text, norm_text, spans_json) VALUES (7, 0, 41, 41, NULL, NULL, ?, ?, ?)",
    (matn, matn.lower(), '{"41": [[10, 20, 100, 40]]}'),
)
seed.commit()
seed.close()
landed = call("locate_narration", {"collection": "bukhari", "number": 1})["place"]
check("locate_narration finds the imported book", landed["found"] is True and landed["book_id"] == 7)
check("locate_narration lands on the page", landed.get("page_start") == 41, landed.get("page_start"))
check("locate_narration carries the highlight", landed.get("spans") == {"41": [[10, 20, 100, 40]]})

try:
    call("locate_narration", {"collection": day["collection"], "number": 999999})
    check("locate_narration rejects an unknown number", False)
except SystemExit as exc:
    check("locate_narration rejects an unknown number", "no such narration" in str(exc))

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
