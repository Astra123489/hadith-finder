mod sidecar;

use serde_json::{json, Value};
use tauri::{AppHandle, Manager, RunEvent};
use tauri_plugin_notification::NotificationExt;

pub struct SidecarState(sidecar::Sidecar);

fn call_blocking(
    app: &AppHandle,
    method: &'static str,
    params: Value,
    timeout_secs: u64,
) -> Result<Value, String> {
    let state = app.state::<SidecarState>();
    state
        .0
        .call(method, params, std::time::Duration::from_secs(timeout_secs))
}

#[tauri::command]
async fn sidecar_ping(app: AppHandle) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || call_blocking(&app, "ping", json!({}), 30))
        .await
        .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn import_book(app: AppHandle, path: String, title: Option<String>) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(
            &app,
            "import_book",
            json!({ "path": path, "title": title }),
            6 * 60 * 60,
        )
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn rebuild_book(app: AppHandle, book_id: i64) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(&app, "rebuild_book", json!({ "book_id": book_id }), 6 * 60 * 60)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn reindex_book(app: AppHandle, book_id: i64) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(&app, "reindex_book", json!({ "book_id": book_id }), 6 * 60 * 60)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn list_books(app: AppHandle) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || call_blocking(&app, "list_books", json!({}), 60))
        .await
        .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn delete_book(app: AppHandle, book_id: i64) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(&app, "delete_book", json!({ "book_id": book_id }), 60)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn ask(
    app: AppHandle,
    query: String,
    book_ids: Option<Vec<i64>>,
    ask_id: Option<String>,
    new_thread: Option<bool>,
    pinned_chunk_ids: Option<Vec<i64>>,
) -> Result<Value, String> {
    let trimmed = query.trim().to_string();
    if trimmed.len() < 2 {
        return Err("please ask a complete question".into());
    }
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(
            &app,
            "ask",
            json!({
                "query": trimmed,
                "book_ids": book_ids,
                "ask_id": ask_id,
                "new_thread": new_thread.unwrap_or(false),
                "pinned_chunk_ids": pinned_chunk_ids,
            }),
            // CPU inference on an 8B model plus reranking: generous, but finite.
            30 * 60,
        )
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn list_chats(app: AppHandle) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || call_blocking(&app, "list_chats", json!({}), 30))
        .await
        .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn new_chat(app: AppHandle, folder_id: Option<String>) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(&app, "new_chat", json!({ "folder_id": folder_id }), 30)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn open_chat(app: AppHandle, id: String) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(&app, "open_chat", json!({ "id": id }), 30)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn rename_chat(app: AppHandle, id: String, title: String) -> Result<Value, String> {
    let name = title.trim().to_string();
    if name.is_empty() {
        return Err("a conversation needs a name".into());
    }
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(&app, "rename_chat", json!({ "id": id, "title": name }), 30)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn delete_chat(app: AppHandle, id: String) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(&app, "delete_chat", json!({ "id": id }), 30)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn thread_state(app: AppHandle) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || call_blocking(&app, "thread_state", json!({}), 30))
        .await
        .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn reset_thread(app: AppHandle) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(&app, "reset_thread", json!({}), 30)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn compact_thread(app: AppHandle) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        // Summarising a long thread is one more (short) generation.
        call_blocking(&app, "compact_thread", json!({}), 15 * 60)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn read_narration(app: AppHandle, collection: String, number: i64) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(
            &app,
            "read_narration",
            json!({ "collection": collection, "number": number }),
            30,
        )
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn locate_narration(app: AppHandle, collection: String, number: i64) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(
            &app,
            "locate_narration",
            json!({ "collection": collection, "number": number }),
            30,
        )
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn rewind_thread(app: AppHandle, turn_id: i64) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(&app, "rewind_thread", json!({ "turn_id": turn_id }), 30)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn suggest_followups(app: AppHandle, count: Option<i64>) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(
            &app,
            "suggest_followups",
            json!({ "count": count.unwrap_or(3) }),
            15 * 60,
        )
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn list_llms(app: AppHandle) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(&app, "list_llms", json!({}), 60)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn models_status(app: AppHandle) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(&app, "models_status", json!({}), 60)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn set_llm(app: AppHandle, path: Option<String>) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(&app, "set_llm", json!({ "path": path }), 120)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn set_llm_ctx(app: AppHandle, ctx: Option<i64>) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(&app, "set_llm", json!({ "ctx": ctx }), 120)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn download_model(app: AppHandle, model: String) -> Result<Value, String> {
    if !matches!(model.as_str(), "llm" | "embedder" | "reranker") {
        return Err("model must be one of: llm, embedder, reranker".into());
    }
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(&app, "download_model", json!({ "model": model }), 6 * 60 * 60)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn set_llm_options(
    app: AppHandle,
    device: Option<String>,
    kv: Option<String>,
    auto_load: Option<bool>,
) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        // Only send the keys that changed: the sidecar treats a present key as
        // "set this", so sending nulls would erase unrelated settings.
        let mut params = serde_json::Map::new();
        if let Some(device) = device {
            params.insert("device".into(), json!(device));
        }
        if let Some(kv) = kv {
            params.insert("kv".into(), json!(kv));
        }
        if let Some(auto_load) = auto_load {
            params.insert("auto_load".into(), json!(auto_load));
        }
        call_blocking(&app, "set_llm", Value::Object(params), 120)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn list_bookmarks(app: AppHandle) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(&app, "list_bookmarks", json!({}), 60)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn add_bookmark(app: AppHandle, bookmark: Value) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(&app, "add_bookmark", json!({ "bookmark": bookmark }), 60)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn delete_bookmark(app: AppHandle, id: i64) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(&app, "delete_bookmark", json!({ "id": id }), 60)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn update_bookmark(
    app: AppHandle,
    id: i64,
    description: Option<String>,
) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(
            &app,
            "update_bookmark",
            json!({ "id": id, "description": description }),
            60,
        )
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn daily_hadith(app: AppHandle, date: Option<String>) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(&app, "daily_hadith", json!({ "date": date }), 120)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn add_folder(app: AppHandle, name: String) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(&app, "add_folder", json!({ "name": name }), 60)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn rename_folder(app: AppHandle, id: String, name: String) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(&app, "rename_folder", json!({ "id": id, "name": name }), 60)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn delete_folder(app: AppHandle, id: String) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(&app, "delete_folder", json!({ "id": id }), 60)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn move_chat(
    app: AppHandle,
    id: String,
    folder_id: Option<String>,
) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(
            &app,
            "move_chat",
            json!({ "id": id, "folder_id": folder_id }),
            60,
        )
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn api_config(app: AppHandle) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || call_blocking(&app, "api_config", json!({}), 60))
        .await
        .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn set_api_config(
    app: AppHandle,
    enabled: Option<bool>,
    base_url: Option<String>,
    model: Option<String>,
) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let mut params = serde_json::Map::new();
        if let Some(enabled) = enabled {
            params.insert("enabled".into(), json!(enabled));
        }
        if let Some(base_url) = base_url {
            params.insert("base_url".into(), json!(base_url));
        }
        if let Some(model) = model {
            params.insert("model".into(), json!(model));
        }
        call_blocking(&app, "set_api_config", Value::Object(params), 60)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn set_api_key(app: AppHandle, key: String) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(&app, "set_api_key", json!({ "key": key }), 60)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn clear_api_key(app: AppHandle) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || call_blocking(&app, "clear_api_key", json!({}), 60))
        .await
        .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn test_api(
    app: AppHandle,
    base_url: Option<String>,
    model: Option<String>,
    key: Option<String>,
) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let mut params = serde_json::Map::new();
        if let Some(base_url) = base_url {
            params.insert("base_url".into(), json!(base_url));
        }
        if let Some(model) = model {
            params.insert("model".into(), json!(model));
        }
        if let Some(key) = key {
            params.insert("key".into(), json!(key));
        }
        call_blocking(&app, "test_api", Value::Object(params), 120)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn daily_notify(app: AppHandle) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || call_blocking(&app, "daily_notify", json!({}), 60))
        .await
        .map_err(|e| e.to_string())?
}

const TASK_NAME: &str = "Hadith Finder Daily Hadith";

/// True when Windows Task Scheduler started us just to raise the reminder.
fn notify_mode() -> bool {
    std::env::args().skip(1).any(|arg| arg == "--notify")
}

/// Show today's reminder, then leave. Called by the scheduled task.
fn run_notify_run(app: AppHandle) {
    std::thread::spawn(move || {
        for _ in 0..160 {
            if call_blocking(&app, "ping", json!({}), 10).is_ok() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(250));
        }
        notify_tick(&app);
        app.exit(0);
    });
}

/// Windows only shows a notification from a running process, so the schedule
/// starts this executable with `--notify`: it hides its window, raises the
/// toast and quits, leaving nothing in the background. `-StartWhenAvailable`
/// is what delivers a day whose time passed while the machine was off.
#[cfg(windows)]
fn sync_task(enabled: bool, time: &str) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let exe = exe.to_string_lossy().replace('\'', "''");
    let script = if enabled {
        format!(
            "$ErrorActionPreference = 'Stop'\n\
             $action = New-ScheduledTaskAction -Execute '{exe}' -Argument '--notify'\n\
             $trigger = New-ScheduledTaskTrigger -Daily -At ([datetime]::ParseExact('{time}', 'HH:mm', $null))\n\
             $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 5)\n\
             Register-ScheduledTask -TaskName '{TASK_NAME}' -Action $action -Trigger $trigger -Settings $settings -Description \"Shows the day's hadith as a desktop notification.\" -Force | Out-Null\n"
        )
    } else {
        format!(
            "Unregister-ScheduledTask -TaskName '{TASK_NAME}' -Confirm:$false -ErrorAction SilentlyContinue\n"
        )
    };
    let path = std::env::temp_dir().join("hadith-finder-notify.ps1");
    std::fs::write(&path, script).map_err(|e| e.to_string())?;
    let output = std::process::Command::new("powershell")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
        ])
        .arg(&path)
        .output()
        .map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&path);
    if output.status.success() {
        return Ok(());
    }
    let detail = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let detail = if detail.is_empty() {
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    } else {
        detail
    };
    Err(format!("could not update the Windows task: {detail}"))
}

#[cfg(not(windows))]
fn sync_task(_enabled: bool, _time: &str) -> Result<(), String> {
    Ok(())
}

#[tauri::command]
async fn set_daily_notify(
    app: AppHandle,
    enabled: Option<bool>,
    time: Option<String>,
) -> Result<Value, String> {
    let state = tauri::async_runtime::spawn_blocking(move || {
        let mut params = serde_json::Map::new();
        if let Some(enabled) = enabled {
            params.insert("enabled".into(), json!(enabled));
        }
        if let Some(time) = time {
            params.insert("time".into(), json!(time));
        }
        call_blocking(&app, "set_daily_notify", Value::Object(params), 60)
    })
    .await
    .map_err(|e| e.to_string())??;

    let armed = state.get("enabled").and_then(Value::as_bool).unwrap_or(false);
    let at = state
        .get("time")
        .and_then(Value::as_str)
        .unwrap_or("07:00")
        .to_string();
    let schedule = tauri::async_runtime::spawn_blocking(move || sync_task(armed, &at))
        .await
        .map_err(|e| e.to_string())?;

    let mut out = state.as_object().cloned().unwrap_or_default();
    match schedule {
        Ok(()) => {
            out.insert("scheduled".into(), json!(armed));
            out.insert("schedule_error".into(), Value::Null);
        }
        Err(e) => {
            out.insert("scheduled".into(), json!(false));
            out.insert("schedule_error".into(), json!(e));
        }
    }
    Ok(Value::Object(out))
}

/// Windows toasts clip long bodies, so the reminder carries the opening lines.
fn shorten(text: &str, limit: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= limit {
        return flat;
    }
    let cut: String = flat.chars().take(limit).collect();
    match cut.rfind(' ') {
        Some(at) => format!("{}…", &cut[..at]),
        None => format!("{cut}…"),
    }
}

/// Show the day's narration on the desktop once its time has passed.
///
/// Polling rather than one long sleep means a machine that was off at the
/// scheduled time still gets the reminder shortly after it next starts.
fn notify_tick(app: &AppHandle) {
    let status = match call_blocking(app, "daily_notify", json!({}), 60) {
        Ok(status) => status,
        Err(_) => return,
    };
    if !status.get("due").and_then(Value::as_bool).unwrap_or(false) {
        return;
    }
    let hadith = status.get("hadith").cloned().unwrap_or(Value::Null);
    let text = hadith.get("text").and_then(Value::as_str).unwrap_or("").trim();
    if text.is_empty() {
        return;
    }
    let short = hadith.get("short").and_then(Value::as_str).unwrap_or("Hadith");
    let number = hadith.get("number").and_then(Value::as_i64).unwrap_or(0);
    let shown = app
        .notification()
        .builder()
        .title(format!("Hadith of the day — {short} {number}"))
        .body(shorten(text, 220))
        .show();
    // Only record the day once it has actually reached the desktop, so a
    // refused or failed notification is retried instead of lost.
    if shown.is_err() {
        return;
    }
    let date = status.get("date").cloned().unwrap_or(Value::Null);
    let _ = call_blocking(app, "mark_daily_notify", json!({ "date": date }), 60);
}

fn start_daily_notify(app: AppHandle) {
    std::thread::spawn(move || loop {
        notify_tick(&app);
        std::thread::sleep(std::time::Duration::from_secs(30));
    });
}

#[tauri::command]
async fn stats(app: AppHandle) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || call_blocking(&app, "stats", json!({}), 60))
        .await
        .map_err(|e| e.to_string())?
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .setup(|app| {
            let lib_dir = app
                .path()
                .app_data_dir()
                .expect("no app data dir");
            std::fs::create_dir_all(&lib_dir)?;
            let library = lib_dir.join("library.haddb");
            let sc = sidecar::Sidecar::spawn(&library, Some(app.handle().clone()))
                .map_err(|e| e.to_string())?;
            app.manage(SidecarState(sc));
            if notify_mode() {
                // Started by the scheduler: hide the window it came with, raise
                // the one toast and leave. Nothing stays running.
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.hide();
                }
                run_notify_run(app.handle().clone());
            } else {
                start_daily_notify(app.handle().clone());
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            sidecar_ping,
            import_book,
            rebuild_book,
            reindex_book,
            list_books,
            delete_book,
            ask,
            thread_state,
            list_chats,
            new_chat,
            open_chat,
            rename_chat,
            delete_chat,
            reset_thread,
            compact_thread,
            rewind_thread,
            read_narration,
            locate_narration,
            suggest_followups,
            models_status,
            list_llms,
            set_llm,
            set_llm_ctx,
            set_llm_options,
            download_model,
            list_bookmarks,
            add_bookmark,
            delete_bookmark,
            update_bookmark,
            daily_hadith,
            add_folder,
            rename_folder,
            delete_folder,
            move_chat,
            api_config,
            set_api_config,
            set_api_key,
            clear_api_key,
            test_api,
            daily_notify,
            set_daily_notify,
            stats,
        ])
        .build(tauri::generate_context!())
        .expect("error while building Hadith Finder");

    app.run(|app_handle, event| {
        if let RunEvent::Exit = event {
            if let Some(state) = app_handle.try_state::<SidecarState>() {
                state.0.kill();
            }
        }
    });
}
