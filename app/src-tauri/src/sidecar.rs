//! Bridge to the Python AI sidecar (stdio newline-delimited JSON-RPC).
//!
//! Spawn order:
//!   1. `HADITH_SIDECAR_EXE` env var (explicit override, used in packaging),
//!   2. `hadith-sidecar(.exe)` next to the app executable (bundled via PyInstaller),
//!   3. dev fallback: `py -3.13 -m hadith_sidecar <library>` with cwd = `sidecar/`.
//!
//! Calls are serialized (one outstanding request at a time) and blocking; async
//! commands offload them to the blocking thread pool so the UI never stalls.
//! Progress notifications from the sidecar are re-emitted as `sidecar-progress`
//! (payload keeps `op`: "reindex" | "download").

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use tauri::{AppHandle, Emitter};

pub struct Sidecar {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
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
    pub fn spawn(library: &Path) -> Result<Self, String> {
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
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 1,
        })
    }

    pub fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    pub fn call(
        &mut self,
        method: &str,
        params: Value,
        app: Option<&AppHandle>,
        timeout: Duration,
    ) -> Result<Value, String> {
        let id = self.next_id;
        self.next_id += 1;
        let req = json!({ "id": id, "method": method, "params": params });
        writeln!(self.stdin, "{req}").map_err(|e| format!("write sidecar stdin: {e}"))?;
        self.stdin.flush().map_err(|e| format!("flush sidecar stdin: {e}"))?;

        let deadline = Instant::now() + timeout;
        let mut line = String::new();
        loop {
            if Instant::now() > deadline {
                return Err(format!("sidecar call '{method}' timed out"));
            }
            line.clear();
            let n = self
                .stdout
                .read_line(&mut line)
                .map_err(|e| format!("read sidecar stdout: {e}"))?;
            if n == 0 {
                return Err("sidecar exited unexpectedly".into());
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let msg: Value = serde_json::from_str(trimmed)
                .map_err(|e| format!("bad sidecar JSON: {e}: {trimmed}"))?;
            if msg.get("type").and_then(Value::as_str) == Some("progress") {
                if let Some(app) = app {
                    let _ = app.emit("sidecar-progress", msg);
                }
                continue;
            }
            if msg.get("id").and_then(Value::as_u64) != Some(id) {
                eprintln!("sidecar: ignoring out-of-order message: {trimmed}");
                continue;
            }
            if msg.get("ok").and_then(Value::as_bool) == Some(true) {
                return Ok(msg.get("result").cloned().unwrap_or(Value::Null));
            }
            let err = msg.get("error").cloned().unwrap_or(json!({"message": "unknown error"}));
            return Err(err
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("sidecar error")
                .to_string());
        }
    }
}
