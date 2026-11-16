mod sidecar;

use serde_json::{json, Value};
use tauri::{AppHandle, Manager, RunEvent};

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
async fn new_chat(app: AppHandle) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || call_blocking(&app, "new_chat", json!({}), 30))
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
