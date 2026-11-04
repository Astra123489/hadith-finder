"""stdio newline-delimited JSON-RPC.

Rust (or tests) writes one JSON object per line to stdin, e.g.
  {"id": 1, "method": "ping", "params": {}}
The sidecar answers one line per request:
  {"id": 1, "ok": true, "result": {...}}
  {"id": 1, "ok": false, "error": {"code": "...", "message": "..."}}
Progress notifications (no "id") may be interleaved:
  {"type": "progress", "op": "reindex", "done": 120, "total": 4000}
"""

from __future__ import annotations

import json
import sys
import traceback

ERROR_METHOD = "method_not_found"
ERROR_INVALID = "invalid_params"
ERROR_INTERNAL = "internal"


def write_line(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def respond(req_id, result) -> None:
    write_line({"id": req_id, "ok": True, "result": result})


def fail(req_id, code: str, message: str) -> None:
    write_line({"id": req_id, "ok": False, "error": {"code": code, "message": message}})


def notify(**fields) -> None:
    write_line({"type": "progress", **fields})


def run(dispatch) -> int:
    # Windows consoles default to cp1252 — force UTF-8 for the wire.
    try:
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        sys.stdout.reconfigure(encoding="utf-8", errors="strict")
    except (AttributeError, ValueError):
        pass

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            fail(None, ERROR_INVALID, f"malformed JSON: {e}")
            continue
        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params") or {}
        if not isinstance(method, str):
            fail(req_id, ERROR_INVALID, "missing method")
            continue
        try:
            result = dispatch(method, params)
            respond(req_id, result)
        except KeyError as e:
            fail(req_id, ERROR_METHOD, f"unknown method: {e.args[0]!r}")
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            fail(req_id, ERROR_INVALID, str(e))
        except Exception:
            traceback.print_exc(file=sys.stderr)
            fail(req_id, ERROR_INTERNAL, "internal sidecar error (see stderr)")
    return 0
