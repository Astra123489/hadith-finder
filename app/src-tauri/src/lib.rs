mod sidecar;

use std::sync::Mutex;

use serde_json::{json, Value};
use tauri::{AppHandle, Manager, RunEvent};

pub struct SidecarState(Mutex<sidecar::Sidecar>);

fn call_blocking(
    app: &AppHandle,
    method: &'static str,
    params: Value,
    with_progress: bool,
) -> Result<Value, String> {
    let state = app.state::<SidecarState>();
    let mut sc = state.0.lock().map_err(|e| e.to_string())?;
    let app_ref = with_progress.then_some(app);
    // Indexing whole books can take a long time on CPU; give it room.
    let timeout = if with_progress {
        std::time::Duration::from_secs(6 * 60 * 60)
    } else {
        std::time::Duration::from_secs(600)
    };
    sc.call(method, params, app_ref, timeout)
}

#[tauri::command]
async fn sidecar_ping(app: AppHandle) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || call_blocking(&app, "ping", json!({}), false))
        .await
        .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn import_book(app: AppHandle, path: String, title: Option<String>) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(&app, "import_book", json!({ "path": path, "title": title }), false)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn reindex_book(app: AppHandle, book_id: i64) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(&app, "reindex_book", json!({ "book_id": book_id }), true)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn list_books(app: AppHandle) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || call_blocking(&app, "list_books", json!({}), false))
        .await
        .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn delete_book(app: AppHandle, book_id: i64) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(&app, "delete_book", json!({ "book_id": book_id }), false)
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
) -> Result<Value, String> {
    let trimmed = query.trim().to_string();
    if trimmed.len() < 2 {
        return Err("please ask a complete question".into());
    }
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(
            &app,
            "ask",
            json!({ "query": trimmed, "book_ids": book_ids, "careful": careful }),
            false,
        )
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn models_status(app: AppHandle) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        call_blocking(&app, "models_status", json!({}), false)
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
        call_blocking(&app, "download_model", json!({ "model": model }), true)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn stats(app: AppHandle) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || call_blocking(&app, "stats", json!({}), false))
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
            let sc = sidecar::Sidecar::spawn(&library).map_err(|e| e.to_string())?;
            app.manage(SidecarState(Mutex::new(sc)));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            sidecar_ping,
            import_book,
            reindex_book,
            list_books,
            delete_book,
            ask,
            models_status,
            download_model,
            stats,
        ])
        .build(tauri::generate_context!())
        .expect("error while building Hadith Finder");

    app.run(|app_handle, event| {
        if let RunEvent::Exit = event {
            if let Some(state) = app_handle.try_state::<SidecarState>() {
                if let Ok(mut sc) = state.0.lock() {
                    sc.kill();
                }
            }
        }
    });
}
