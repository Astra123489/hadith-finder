"""Check the conversation layer: history, compaction, usage and follow-ups.

Two passes:
1. thread.py directly with a stub model (deterministic, no files).
2. the same behaviour over the stdio protocol with the test providers
   (echo answers, hash embeddings), which proves the wiring in __main__.

Run:  python check_thread.py
"""

import json
import os
import re
import subprocess
import sys
import tempfile

try:  # the Arabic in the suggestion templates must not crash a cp1252 console
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hadith_sidecar import thread as T  # noqa: E402

failures = []


def check(label, condition, detail=""):
    print(f"[{'ok  ' if condition else 'FAIL'}] {label} {detail}")
    if not condition:
        failures.append(label)


class StubLLM:
    """Fixed size tokenisation and a scripted summary."""

    context_size = 4096

    def __init__(self, *, summary="Summary: the reader asked about intentions; Bukhari 1 and Muslim 1907 were cited as parallels.",
                 suggestions="1. What does “the reward follows what he intended” mean in practice?\n2. Are there acts where the intention does not change the reward?",
                 fail=False):
        self.summary = summary
        self.suggestions = suggestions
        self.fail = fail
        self.calls = []

    def count_tokens(self, text):
        return max(1, len(text) // 4)

    def generate(self, system, user, *, max_tokens=512, temperature=0.2, on_token=None):
        self.calls.append(user)
        if self.fail:
            raise RuntimeError("model unavailable")
        if "compress a research conversation" in system:
            return self.summary
        if "follow-up questions" in system:
            return self.suggestions
        return "An answer [S1]."


def citation(**over):
    base = {
        "book_title": "Demo — Book of Intentions",
        "chapter": "Book of Intentions",
        "hadith_ref": "1",
        "gradings": [],
        "also_in": [
            {"collection": "bukhari", "collection_name": "Sahih al-Bukhari", "number": 1}
        ],
    }
    base.update(over)
    return base


def turn(conv, question, citations=None):
    return T.add_turn(conv, question, {
        "answer": "Deeds are judged by intention [S1], so the reward follows it [S2].",
        "citations": citations if citations is not None else [citation()],
        "unverified": False,
    })


# ---------------------------------------------------------------- direct tests

llm = StubLLM()
conv = T.new_conversation()
check("empty conversation has no turns", conv["turns"] == [])
check("empty history is blank", T.history_block(conv) == "")
check("empty conversation has no suggestions", T.suggest(conv, llm) == [])

turn(conv, "What is the virtue of intentions?")
check("turn appended with an id", conv["turns"][0]["id"] == 1, conv["turns"][0]["id"])
check("second turn gets a new id", turn(conv, "And what about emigration?")["id"] == 2)

history = T.history_block(conv)
check("history carries the question", "virtue of intentions" in history)
check("history strips citation tags", "[S1]" not in history, history[:90])
check("history keeps both turns", history.count("User:") == 2)

info = T.usage(llm, conv, 4096, excerpt_tokens=800)
check("usage counts the window", info["ctx"] == 4096 and 0 < info["used"] <= 4096, info["used"])
check("usage reports a percentage", 0 < info["pct"] < 1, info["pct"])
check("usage reports the compact point", info["compact_at"] == int(4096 * T.COMPACT_AT), info["compact_at"])
check("usage reports turns", info["turns"] == 2)
check("excerpt budget shrinks with history", T.excerpt_chars(llm, conv, 4096, 2600) <= 2600)

# Compaction keeps the newest turns verbatim and folds the rest.
for i in range(5):
    turn(conv, f"Follow-up question {i} about intention and deed?")
events = []
before = len(conv["turns"])
result = T.compact(conv, llm, notify=lambda **kw: events.append(kw["stage"]))
check("compaction reports what it dropped", result["compacted"] and result["dropped"] == before - 2, result)
check("compaction keeps two turns", len(conv["turns"]) == 2, len(conv["turns"]))
check("compaction kept only the newest", "Follow-up question 4" in conv["turns"][-1]["question"])
check("compaction counts up", conv["compactions"] == 1)
check("compaction notifies both stages", events == ["compacting", "compacted"], events)
check("summary replaces the dropped turns", "Summary:" in conv["summary"], conv["summary"][:50])
check("history includes the summary", "Summary of earlier turns" in T.history_block(conv))
check("history no longer has dropped turns", "Follow-up question 0" not in T.history_block(conv))
check("compacting again is a no-op", T.compact(conv, llm)["compacted"] is False)

# A model that cannot summarise still leaves a usable thread.
broken = StubLLM(fail=True)
conv2 = T.new_conversation()
for i in range(4):
    turn(conv2, f"Question {i} about feeding the poor?")
out = T.compact(conv2, broken, notify=None)
check("failed summary still compacts", out["compacted"] is True)
check("fallback summary lists the questions", "Earlier questions:" in conv2["summary"], conv2["summary"][:60])
check("fallback summary keeps citing books", "Demo — Book of Intentions" in conv2["summary"])

# Suggestions: model output parsed, numbering and duplicates removed.
conv3 = T.new_conversation()
turn(conv3, "What does the Prophet say about mercy?")
sugs = T.suggest(conv3, StubLLM(), 3)
check("suggestions parsed from the model", sugs[0].startswith("What does"), sugs[:1])
check("suggestions are questions", all(s.endswith("?") for s in sugs), sugs)
check("suggestions filled up from the answer", len(sugs) == 3, sugs)

fallback = T.suggest(conv3, StubLLM(fail=True), 3)
check("suggestions survive a dead model", len(fallback) >= 2, fallback)
check("fallback quotes the answer", any("Deeds are judged by intention" in s for s in fallback), fallback)
check(
    "fallback stays off the source cards",
    not any(re.search(r"graded|narrat|collection|chapter", s, re.I) for s in fallback),
    fallback,
)
check(
    "source-only suggestions are dropped",
    T._parse_suggestions(
        "1. How does Sahih al-Bukhari 1 narrate this?\n2. Is this narration graded authentic?",
        "What is the virtue of intentions?",
    )
    == [],
)

empty_cites = T.new_conversation()
turn(empty_cites, "What is said about patience?", citations=[])
gaps = T.suggest(empty_cites, StubLLM(fail=True), 2)
check("ungrounded answer still gets questions", len(gaps) == 2, gaps)

named = T.new_conversation("c-1")
check("conversation carries an id", named["id"] == "c-1")
check("conversation starts unnamed", named["title"] == "")
check("title comes from the question", T.title_for("What is the virtue of intentions?") == "What is the virtue of intentions")
check("long titles are shortened", T.title_for("mercy " * 40).endswith("…"))
turn(named, "What is the virtue of intentions?")
check("the first turn names the chat", named["title"].startswith("What is the virtue"), named["title"])
check("a later turn keeps the name", turn(named, "And something else?") and named["title"].startswith("What is the virtue"))
check("snapshot exposes the chat id", T.snapshot(named, info)["id"] == "c-1")

snap = T.snapshot(conv, info)
check("snapshot carries the transcript", len(snap["turns"]) == len(conv["turns"]))
check("snapshot carries usage", snap["usage"]["ctx"] == 4096)
check("snapshot is json serialisable", json.loads(json.dumps(snap)) is not None)

# ------------------------------------------------------------ protocol tests

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(tempfile.gettempdir(), "hf_thread_test.haddb")
for suffix in ("", "-wal", "-shm"):
    try:
        os.remove(LIB + suffix)
    except OSError:
        pass

def session(path, fresh=True, **env_extra):
    """A sidecar over stdio, with the test providers.

    fresh=False keeps the library file, which is how a restart is tested.
    """
    if fresh:
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(path + suffix)
            except OSError:
                pass
    env = dict(os.environ, HADITH_LLM="echo", HADITH_EMBEDDER="hash",
               HADITH_RERANKER="identity", **env_extra)
    proc = subprocess.Popen(
        [sys.executable, "-m", "hadith_sidecar", path],
        cwd=HERE, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True, encoding="utf-8",
    )
    counter = [0]

    def call(method, params=None):
        counter[0] += 1
        rid = counter[0]
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

    return proc, call


proc, call = session(LIB)
try:
    first = call("ask", {"query": "What is the virtue of intentions?", "ask_id": "a1"})
    check("ask returns a thread", "thread" in first)
    check("first ask has one turn", len(first["thread"]["turns"]) == 1, len(first["thread"]["turns"]))
    check("thread carries usage", first["thread"]["usage"]["ctx"] > 0, first["thread"]["usage"])
    check("thread does not leak the prompt", "prompt" not in first)

    second = call("ask", {"query": "Tell me more about that", "ask_id": "a2"})
    check("follow-up appends a turn", len(second["thread"]["turns"]) == 2)
    check("follow-up keeps the first question", second["thread"]["turns"][0]["question"].startswith("What is the virtue"))
    check("chat is named after the first question", second["thread"]["title"].startswith("What is the virtue"), second["thread"]["title"])
    check("chat list counts the turns", second["chats"][0]["turns"] == 2, second["chats"])
    first_id = second["thread"]["id"]

    opened = call("new_chat", {})
    check("new_chat opens an empty chat", opened["thread"]["turns"] == [], opened["thread"]["turns"])
    check("new_chat is a different chat", opened["thread"]["id"] != first_id)
    check("new_chat adds a row", len(opened["chats"]) == 2, len(opened["chats"]))
    check("new_chat becomes active", next(c for c in opened["chats"] if c["active"])["id"] == opened["thread"]["id"])

    again = call("new_chat", {})
    check(
        "an untouched chat is reused",
        again["thread"]["id"] == opened["thread"]["id"] and len(again["chats"]) == 2,
        len(again["chats"]),
    )

    call("ask", {"query": "What about patience?"})
    check("chats keep separate transcripts", len(call("thread_state")["thread"]["turns"]) == 1)

    reopened = call("open_chat", {"id": first_id})
    check("open_chat restores the earlier transcript", len(reopened["thread"]["turns"]) == 2)
    check("open_chat marks it active", next(c for c in reopened["chats"] if c["active"])["id"] == first_id)

    renamed = call("rename_chat", {"id": first_id, "title": "Intentions, revisited"})
    check(
        "rename_chat changes the name",
        any(c["id"] == first_id and c["title"] == "Intentions, revisited" for c in renamed["chats"]),
        [c["title"] for c in renamed["chats"]],
    )
    try:
        call("rename_chat", {"id": first_id, "title": "   "})
        check("rename rejects an empty name", False)
    except SystemExit as exc:
        check("rename rejects an empty name", "needs a name" in str(exc))

    other_id = next(c["id"] for c in reopened["chats"] if c["id"] != first_id)
    removed = call("delete_chat", {"id": other_id})
    check("delete_chat removes the row", removed["deleted"] is True and len(removed["chats"]) == 1, len(removed["chats"]))

    call("ask", {"query": "One more question"})
    compacted = call("compact_thread", {})
    check("compact collapses the thread", compacted["thread"]["compactions"] == 1, compacted)
    check("compact reports a summary", bool(compacted["thread"]["summary"]), compacted["thread"]["summary"][:60])

    suggestions = call("suggest_followups", {"count": 3})["suggestions"]
    check("suggestions over the wire", isinstance(suggestions, list) and len(suggestions) >= 1, suggestions)

    state = call("thread_state")["thread"]
    check("thread_state returns the conversation", len(state["turns"]) >= 1, len(state["turns"]))
    check("thread_state reports usage without a model", state["usage"]["ctx"] > 0, state["usage"])

    reset = call("reset_thread", {})
    check("reset_thread", reset["reset"] is True and reset["thread"]["turns"] == [])
    check("no suggestions without a thread", call("suggest_followups", {})["suggestions"] == [])
finally:
    try:
        proc.stdin.close()
        proc.wait(timeout=20)
    except Exception:
        proc.kill()

# Chats live in the library file, so opening the app again finds them.
after_proc, call_after = session(LIB, fresh=False)
try:
    chats = call_after("list_chats")["chats"]
    check("chats survive a restart", len(chats) >= 1, len(chats))
    restored = call_after("thread_state")["thread"]
    check("restart restores the open chat", restored["id"] == next(c["id"] for c in chats if c["active"]), restored["id"])
    check("restart keeps chat titles", any(c["title"] for c in chats), [c["title"] for c in chats])
    check("renames survive a restart", any(c["title"] == "Intentions, revisited" for c in chats), [c["title"] for c in chats])
finally:
    try:
        after_proc.stdin.close()
        after_proc.wait(timeout=20)
    except Exception:
        after_proc.kill()

# A small window forces the automatic compaction path: the thread grows, the
# window does not, so __main__ must fold turns away on its own.
small, call_small = session(os.path.join(tempfile.gettempdir(), "hf_thread_small.haddb"), HADITH_CTX="1024")
try:
    turns_seen = 0
    compactions = 0
    for i in range(6):
        res = call_small("ask", {"query": f"Question {i} about patience and gratitude in the sunnah"})
        turns_seen = len(res["thread"]["turns"])
        compactions = res["thread"]["compactions"]
    check("small window compacts automatically", compactions >= 1, compactions)
    check("compacted thread stays small", turns_seen <= 3, turns_seen)
    check("thread keeps a summary", bool(res["thread"]["summary"]))
    check("compaction is announced", res["thread"]["usage"]["compactions"] >= 1)
finally:
    try:
        small.stdin.close()
        small.wait(timeout=20)
    except Exception:
        small.kill()

print()
if failures:
    print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
    raise SystemExit(1)
print("all conversation checks passed")
