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
    careful: bool,
    ask_id: Option<String>,
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
                "careful": careful,
                "ask_id": ask_id,
            }),
            // CPU inference on an 8B model plus reranking: generous, but finite.
            30 * 60,
        )
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
            models_status,
            set_llm,
            download_model,
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
