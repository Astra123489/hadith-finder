//! Bridge to the Python AI sidecar (stdio newline-delimited JSON-RPC).
//!
//! Spawn order:
//!   1. `HADITH_SIDECAR_EXE` env var (explicit override, used in packaging),
//!   2. `hadith-sidecar(.exe)` next to the app executable (bundled via PyInstaller),
//!   3. dev fallback: `py -3.13 -m hadith_sidecar <library>` with cwd = `sidecar/`.
//!
//! A dedicated reader thread owns stdout. Responses are filed under their
//! request id and waiters are woken by a condvar, so:
//!   * timeouts are real (the old blocking `read_until` could hang forever),
//!   * progress notifications are re-emitted as `sidecar-progress` while a
//!     long import/indexing call is still running,
//!   * no global lock is held for the duration of a call.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use tauri::{AppHandle, Emitter};

struct Shared {
    responses: HashMap<u64, Value>,
    dead: bool,
}

pub struct Sidecar {
    child: Mutex<Child>,
    stdin: Mutex<ChildStdin>,
    shared: Arc<(Mutex<Shared>, Condvar)>,
    next_id: AtomicU64,
}

fn sidecar_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("sidecar")
}

fn bundled_sidecar() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    let name = if cfg!(windows) { "hadith-sidecar.exe" } else { "hadith-sidecar" };
    let candidate = dir.join(name);
    candidate.exists().then_some(candidate)
}

impl Sidecar {
    pub fn spawn(library: &Path, app: Option<AppHandle>) -> Result<Self, String> {
        let mut cmd = if let Ok(explicit) = std::env::var("HADITH_SIDECAR_EXE") {
            Command::new(explicit)
        } else if let Some(bundled) = bundled_sidecar() {
            Command::new(bundled)
        } else {
            let root = sidecar_root();
            if !root.join("hadith_sidecar").is_dir() {
                return Err(format!(
                    "Sidecar not found: expected package at {}. \
                     Run from the repo or bundle hadith-sidecar(.exe).",
                    root.display()
                ));
            }
            let py = if cfg!(windows) { "py" } else { "python3" };
            let mut c = Command::new(py);
            if cfg!(windows) {
                c.args(["-3.13", "-m", "hadith_sidecar"]);
            } else {
                c.args(["-3", "-m", "hadith_sidecar"]);
            }
            c.current_dir(&root);
            c
        };

        cmd.arg(library.as_os_str())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()); // sidecar logs go to stderr of host

        let mut child = cmd.spawn().map_err(|e| format!("spawn sidecar: {e}"))?;
        let stdin = child.stdin.take().ok_or("no stdin")?;
        let stdout = child.stdout.take().ok_or("no stdout")?;

        let shared = Arc::new((
            Mutex::new(Shared { responses: HashMap::new(), dead: false }),
            Condvar::new(),
        ));

        let reader_shared = Arc::clone(&shared);
        std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                let Ok(line) = line else { break };
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let Ok(msg) = serde_json::from_str::<Value>(trimmed) else {
                    eprintln!("sidecar: unparseable line: {trimmed}");
                    continue;
                };
                if msg.get("type").and_then(Value::as_str) == Some("progress") {
                    if let Some(app) = app.as_ref() {
                        let _ = app.emit("sidecar-progress", msg);
                    }
                    continue;
                }
                let Some(id) = msg.get("id").and_then(Value::as_u64) else {
                    continue;
                };
                let (lock, cv) = &*reader_shared;
                if let Ok(mut state) = lock.lock() {
                    state.responses.insert(id, msg);
                    cv.notify_all();
                }
            }
            let (lock, cv) = &*reader_shared;
            if let Ok(mut state) = lock.lock() {
                state.dead = true;
                cv.notify_all();
            }
        });

        Ok(Self {
            child: Mutex::new(child),
            stdin: Mutex::new(stdin),
            shared,
            next_id: AtomicU64::new(1),
        })
    }

    pub fn kill(&self) {
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    pub fn call(&self, method: &str, params: Value, timeout: Duration) -> Result<Value, String> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let req = json!({ "id": id, "method": method, "params": params });
        {
            let mut stdin = self.stdin.lock().map_err(|e| e.to_string())?;
            writeln!(stdin, "{req}").map_err(|e| format!("write sidecar stdin: {e}"))?;
            stdin.flush().map_err(|e| format!("flush sidecar stdin: {e}"))?;
        }

        let deadline = Instant::now() + timeout;
        let (lock, cv) = &*self.shared;
        let mut state = lock.lock().map_err(|e| e.to_string())?;
        loop {
            if let Some(msg) = state.responses.remove(&id) {
                if msg.get("ok").and_then(Value::as_bool) == Some(true) {
                    return Ok(msg.get("result").cloned().unwrap_or(Value::Null));
                }
                let err = msg.get("error").cloned().unwrap_or(json!({ "message": "unknown error" }));
                return Err(err
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("sidecar error")
                    .to_string());
            }
            if state.dead {
                return Err("sidecar exited unexpectedly".into());
            }
            let now = Instant::now();
            if now >= deadline {
                return Err(format!("sidecar call '{method}' timed out"));
            }
            let (guard, _) = cv
                .wait_timeout(state, deadline - now)
                .map_err(|e| e.to_string())?;
            state = guard;
        }
    }
}
