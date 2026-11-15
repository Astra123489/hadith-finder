import { useCallback, useEffect, useState, type ReactNode } from "react";
import "./styles.css";
import { api } from "./api";
import type { Book, Citation } from "./types";
import AskView from "./views/AskView";
import LibraryView from "./views/LibraryView";
import SettingsView from "./views/SettingsView";
import PdfViewer from "./views/PdfViewer";

type View = "ask" | "library" | "settings";

interface ViewerState {
  filePath: string;
  title: string;
  page: number;
  spans: Citation["spans"];
  tag?: string;
}

const NAV: { key: View; label: string; icon: ReactNode; kbd: string }[] = [
  {
    key: "ask",
    label: "Ask",
    kbd: "1",
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
        <path d="M12 3a9 9 0 1 0 3.5 17.3L21 21l-1-4.6A8.96 8.96 0 0 0 12 3Z" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round" />
        <path d="M9.5 10.2a2.5 2.5 0 0 1 4.9.6c0 1.6-2.4 2-2.4 3.4" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
        <circle cx="12" cy="16.6" r="0.9" fill="currentColor" />
      </svg>
    ),
  },
  {
    key: "library",
    label: "Library",
    kbd: "2",
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
        <path d="M4 19V5a2 2 0 0 1 2-2h13a1 1 0 0 1 1 1v15a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2Z" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round" />
        <path d="M8 7h8M8 11h5" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    key: "settings",
    label: "Settings",
    kbd: "3",
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
        <path d="M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z" stroke="currentColor" strokeWidth="1.7" />
        <path d="M19.4 13.5a7.8 7.8 0 0 0 0-3l1.9-1.5-2-3.4-2.3.9a8 8 0 0 0-2.6-1.5L14 2.5h-4L9.6 5.4a8 8 0 0 0-2.6 1.5l-2.3-.9-2 3.4 1.9 1.5a7.8 7.8 0 0 0 0 3l-1.9 1.5 2 3.4 2.3-.9a8 8 0 0 0 2.6 1.5l.4 3h4l.4-3a8 8 0 0 0 2.6-1.5l2.3.9 2-3.4-1.9-1.5Z" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" />
      </svg>
    ),
  },
];

export default function App() {
  const [view, setView] = useState<View>("ask");
  const [books, setBooks] = useState<Book[]>([]);
  const [scope, setScope] = useState<number[] | null>(null);
  const [viewer, setViewer] = useState<ViewerState | null>(null);

  const refreshBooks = useCallback(async () => {
    try {
      const list = await api.listBooks();
      setBooks(list);
    } catch (e) {
      console.error("list_books failed", e);
    }
  }, []);

  useEffect(() => {
    refreshBooks();
  }, [refreshBooks]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLInputElement) return;
      if (e.key === "1") setView("ask");
      if (e.key === "2") setView("library");
      if (e.key === "3") setView("settings");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const openCitation = useCallback(
    (citation: Citation) => {
      const book = books.find((b) => b.id === citation.book_id);
      if (!book) return;
      setViewer({
        filePath: book.file_path,
        title: `${citation.book_title} — p. ${citation.page_start + 1}`,
        page: citation.page_start,
        spans: citation.spans,
        tag: `S${citation.source_index}`,
      });
    },
    [books],
  );

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <svg width="40" height="40" viewBox="0 0 48 48" fill="none" aria-hidden="true">
            <rect width="48" height="48" rx="11" fill="#1c3a2a" />
            <path d="M13 32V15.5A2.5 2.5 0 0 1 15.5 13H23v20a2.3 2.3 0 0 0-2.3-2.3H13Z" fill="#e7dcc3" />
            <path d="M35 32V15.5A2.5 2.5 0 0 0 32.5 13H23v20a2.3 2.3 0 0 1 2.3-2.3H35Z" fill="#cbb98e" />
            <circle cx="37.5" cy="16" r="5.4" stroke="#d9a441" strokeWidth="2.4" fill="#1c3a2a" />
            <path d="M41.2 19.6 44 22.4" stroke="#d9a441" strokeWidth="2.4" strokeLinecap="round" />
          </svg>
          <div>
            <div className="brand-name">Hadith Finder</div>
            <div className="brand-sub">Local · Cited · Free</div>
          </div>
        </div>

        {NAV.map((item) => (
          <button
            key={item.key}
            className={`nav-btn ${view === item.key ? "active" : ""}`}
            onClick={() => setView(item.key)}
          >
            {item.icon}
            {item.label}
            <span className="nav-kbd">{item.kbd}</span>
          </button>
        ))}

        <div className="sidebar-foot">
          <div><span className="dot-live" /><strong>Fully local</strong> — nothing leaves this device.</div>
          <div style={{ marginTop: 8 }}>Always verify answers against the highlighted source passage.</div>
        </div>
      </aside>

      <main className="main">
        {view === "ask" && (
          <AskView
            books={books}
            initialScope={scope}
            onScopeChange={setScope}
            onOpenCitation={openCitation}
          />
        )}
        {view === "library" && <LibraryView books={books} onChanged={refreshBooks} />}
        {view === "settings" && <SettingsView />}
      </main>

      {viewer && (
        <PdfViewer
          filePath={viewer.filePath}
          title={viewer.title}
          initialPage={viewer.page}
          spans={viewer.spans}
          sourceTag={viewer.tag}
          onClose={() => setViewer(null)}
        />
      )}
    </div>
  );
}
