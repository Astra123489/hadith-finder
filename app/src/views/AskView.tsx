import {
  FormEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
  type RefObject,
} from "react";
import { createPortal } from "react-dom";
import { api } from "../api";
import type { LlmOption } from "../api";
import type {
  Book,
  ChatFolder,
  ChatSummary,
  Citation,
  NarrationPlace,
  Parallel,
  ThreadState,
  ThreadTurn,
  ThreadUsage,
} from "../types";
import CitationCard, { citationLabel } from "../components/CitationCard";

const EXAMPLES = [
  "Explain the virtue of intentions in one's deeds",
  "What does the Prophet say about mercy to others?",
  "Find hadiths about feeding the poor",
  "Summarize the chapter on intentions",
];

const NOT_FOUND_PREFIX = "I could not find this in the loaded books.";
const RING = 2 * Math.PI * 12;

interface ChatRowProps {
  chat: ChatSummary;
  inFolder: boolean;
  busy: boolean;
  renaming: boolean;
  value: string;
  inputRef: RefObject<HTMLInputElement | null>;
  onOpen: () => void;
  onMenu: (event: ReactMouseEvent) => void;
  onValue: (value: string) => void;
  onCommit: () => void;
  onCancel: () => void;
  onBlur: () => void;
}

function ChatRow({
  chat,
  inFolder,
  busy,
  renaming,
  value,
  inputRef,
  onOpen,
  onMenu,
  onValue,
  onCommit,
  onCancel,
  onBlur,
}: ChatRowProps) {
  return (
    <div
      className={`chat-item ${chat.active ? "active" : ""} ${renaming ? "renaming" : ""}${
        inFolder ? " in-folder" : ""
      }`}
      onContextMenu={onMenu}
    >
      {renaming ? (
        <>
          <ChatIcon />
          <input
            ref={inputRef}
            className="chat-rename"
            value={value}
            onChange={(e) => onValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                onCommit();
              } else if (e.key === "Escape") {
                e.preventDefault();
                onCancel();
              }
            }}
            onBlur={onBlur}
            aria-label="Conversation name"
          />
        </>
      ) : (
        <button
          className="chat-open"
          onClick={onOpen}
          disabled={busy || chat.active}
          title={chat.title || "New conversation"}
        >
          <ChatIcon />
          <span className="chat-name">{chat.title || "New conversation"}</span>
        </button>
      )}
    </div>
  );
}

function FolderIcon({ size = 13 }: { size?: number }) {
  return (
    <svg
      className="folder-icon"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M3.6 6.9a1.8 1.8 0 0 1 1.8-1.8h3.4l1.9 2.3h7.9a1.8 1.8 0 0 1 1.8 1.8v7.9a1.8 1.8 0 0 1-1.8 1.8H5.4a1.8 1.8 0 0 1-1.8-1.8V6.9Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function ChatIcon() {
  return (
    <svg className="chat-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M4.8 4.5h14.4a1.8 1.8 0 0 1 1.8 1.8v9a1.8 1.8 0 0 1-1.8 1.8H10l-4.4 3.2v-3.2H4.8A1.8 1.8 0 0 1 3 15.3v-9a1.8 1.8 0 0 1 1.8-1.8Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
    </svg>
  );
}

const ICON = { width: 14, height: 14, viewBox: "0 0 24 24", fill: "none", "aria-hidden": true } as const;

function IconPencil() {
  return (
    <svg {...ICON}>
      <path d="M4 20h4L18.5 9.5a2.1 2.1 0 0 0-3-3L5 17v3Z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
      <path d="M13.5 6.5l4 4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

function IconRetry() {
  return (
    <svg {...ICON}>
      <path d="M20 12a8 8 0 1 1-2.6-5.9" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <path d="M20 4.5V9h-4.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function IconUndo() {
  return (
    <svg {...ICON}>
      <path d="M4 11h9a5 5 0 0 1 0 10h-2" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <path d="M7.5 7.5 4 11l3.5 3.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function IconCopy() {
  return (
    <svg {...ICON}>
      <rect x="9" y="9" width="11" height="11" rx="2" stroke="currentColor" strokeWidth="1.6" />
      <path
        d="M15 6.5V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h.5"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    </svg>
  );
}

function IconCheck() {
  return (
    <svg {...ICON}>
      <path d="M5 13l4.5 4.5L19 7" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

async function writeClipboard(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  // Older webviews have no async clipboard; this still works on a user gesture.
  const scratch = document.createElement("textarea");
  scratch.value = text;
  scratch.style.position = "fixed";
  scratch.style.opacity = "0";
  document.body.appendChild(scratch);
  scratch.select();
  document.execCommand("copy");
  scratch.remove();
}

function formatTokens(n: number): string {
  if (n >= 10000) return `${Math.round(n / 1000)}k`;
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

interface Props {
  books: Book[];
  initialScope: number[] | null;
  onScopeChange: (ids: number[] | null) => void;
  onOpenCitation: (citation: Citation) => void;
  onOpenPassage: (place: NarrationPlace) => void;
  onBookmark?: (citation: Citation, description: string) => Promise<void>;
}

export default function AskView({
  books,
  initialScope,
  onScopeChange,
  onOpenCitation,
  onOpenPassage,
  onBookmark,
}: Props) {
  const [query, setQuery] = useState("");
  const [scope, setScope] = useState<number[] | null>(initialScope);
  const [llms, setLlms] = useState<LlmOption[]>([]);
  const [modelPath, setModelPath] = useState("");
  const [modelBusy, setModelBusy] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [streamed, setStreamed] = useState("");
  const [thinking, setThinking] = useState("");
  // The conversation, owned by the sidecar: every answer returns it whole.
  const [turns, setTurns] = useState<ThreadTurn[]>([]);
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [summary, setSummary] = useState("");
  const [usage, setUsage] = useState<ThreadUsage | null>(null);
  const [pending, setPending] = useState<string | null>(null);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [suggestBusy, setSuggestBusy] = useState(false);
  const [compacting, setCompacting] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [showCtx, setShowCtx] = useState(false);
  // The sidebar renders after this view mounts, so the portal target is looked
  // up once and re-rendered into when the tab is revisited.
  const [sidebarSlot, setSidebarSlot] = useState<HTMLElement | null>(null);
  const [menu, setMenu] = useState<{
    id: string;
    x: number;
    y: number;
    kind: "chat" | "folder";
  } | null>(null);
  const [folders, setFolders] = useState<ChatFolder[]>([]);
  const [closedFolders, setClosedFolders] = useState<string[]>([]);
  const [folderRenaming, setFolderRenaming] = useState<string | null>(null);
  const [folderName, setFolderName] = useState("");
  const folderInputRef = useRef<HTMLInputElement>(null);
  const [scopeOpen, setScopeOpen] = useState(false);
  const [focusTick, setFocusTick] = useState(0);
  const [copiedTurn, setCopiedTurn] = useState<number | null>(null);
  const [modelOpen, setModelOpen] = useState(false);
  const scopeRef = useRef<HTMLDivElement>(null);
  const modelRef = useRef<HTMLDivElement>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const renameInputRef = useRef<HTMLInputElement>(null);
  const renameCancelled = useRef(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const ctxRef = useRef<HTMLDivElement>(null);
  const askIdRef = useRef<string | null>(null);

  useEffect(() => setScope(initialScope), [initialScope]);

  useEffect(() => {
    setSidebarSlot(document.getElementById("chat-slot"));
  }, []);

  const loadModels = useCallback(async () => {
    try {
      const list = await api.listLlms();
      setLlms(list.llms);
      setModelPath(list.llms.find((m) => m.selected)?.path ?? list.active);
    } catch {
      /* the model picker is optional: answering still works without it */
    }
  }, []);

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  const pickModel = async (option: LlmOption) => {
    setModelBusy(true);
    setError(null);
    try {
      await api.setLlm(option.default ? null : option.path);
      await loadModels();
    } catch (e) {
      setError(String(e));
      await loadModels();
    } finally {
      setModelBusy(false);
    }
  };

  // Live tokens from the sidecar: the answer appears as it is written instead
  // of after the whole (often multi-minute) CPU generation finishes.
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    api.onProgress((p) => {
      if (p.op === "token") {
        if (!askIdRef.current || p.ask_id !== askIdRef.current) return;
        if (p.phase === "thinking") setThinking((cur) => cur + p.text);
        else setStreamed((cur) => cur + p.text);
        return;
      }
      if (p.op === "thread") {
        if (p.stage === "compacting") {
          setCompacting(true);
          return;
        }
        setCompacting(false);
        if (p.stage === "compacted") {
          const n = p.dropped ?? 0;
          setNotice(`Summarised ${n} earlier ${n === 1 ? "turn" : "turns"} to free context.`);
        }
      }
    }).then((fn) => (unlisten = fn));
    return () => unlisten?.();
  }, []);

  // The context menu closes on any click, scroll or Escape.
  useEffect(() => {
    if (!menu) return;
    const close = () => setMenu(null);
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    window.addEventListener("mousedown", close);
    window.addEventListener("scroll", close, true);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", close);
      window.removeEventListener("scroll", close, true);
      window.removeEventListener("keydown", onKey);
    };
  }, [menu]);

  // The model picker closes on an outside click or Escape.
  useEffect(() => {
    if (!modelOpen) return;
    const onDown = (e: MouseEvent) => {
      if (!modelRef.current?.contains(e.target as Node)) setModelOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setModelOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [modelOpen]);

  // The book picker closes on an outside click or Escape.
  useEffect(() => {
    if (!scopeOpen) return;
    const onDown = (e: MouseEvent) => {
      if (!scopeRef.current?.contains(e.target as Node)) setScopeOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setScopeOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [scopeOpen]);

  // The context panel closes when you click anywhere else.
  useEffect(() => {
    if (!showCtx) return;
    const onDown = (e: MouseEvent) => {
      if (!ctxRef.current?.contains(e.target as Node)) setShowCtx(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [showCtx]);

  const noneSelected = Array.isArray(scope) && scope.length === 0 && books.length > 0;

  const scopeLabel = useMemo(() => {
    if (!scope || scope.length === books.length) return "All books";
    if (scope.length === 0) return "No books";
    return scope.length === 1
      ? books.find((b) => b.id === scope[0])?.title ?? "1 book"
      : `${scope.length} of ${books.length}`;
  }, [scope, books]);

  const toggleScope = (id: number) => {
    setScope((prev) => {
      const base = prev ?? books.map((b) => b.id);
      const next = base.includes(id) ? base.filter((x) => x !== id) : [...base, id];
      const normalized = next.length === books.length ? null : next;
      onScopeChange(normalized);
      return normalized;
    });
  };

  const selectAll = () => {
    setScope(null);
    onScopeChange(null);
  };

  const applyThread = useCallback((thread: ThreadState | undefined) => {
    if (!thread) return;
    setTurns(thread.turns);
    setSummary(thread.summary);
    setUsage(thread.usage);
  }, []);

  // The conversation outlives this view: coming back from another tab shows
  // the same thread instead of an empty page.
  useEffect(() => {
    let alive = true;
    api
      .threadState()
      .then((res) => {
        if (!alive) return;
        setChats(res.chats);
        setFolders(res.folders ?? []);
        if (res.thread.turns.length > 0) applyThread(res.thread);
      })
      .catch(() => {
        /* a fresh sidecar simply has nothing to restore */
      });
    return () => {
      alive = false;
    };
  }, [applyThread]);

  const scrollToEnd = () => requestAnimationFrame(() => endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }));

  const loadSuggestions = useCallback(async () => {
    setSuggestBusy(true);
    try {
      setSuggestions(await api.suggestFollowups(3));
    } catch {
      setSuggestions([]);
    } finally {
      setSuggestBusy(false);
    }
  }, []);

  const submit = async (e?: FormEvent, preset?: string) => {
    e?.preventDefault();
    const q = (preset ?? query).trim();
    if (q.length < 2 || busy || noneSelected) return;
    const askId =
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID()
        : `ask-${Date.now()}`;
    askIdRef.current = askId;
    setBusy(true);
    setError(null);
    setPending(q);
    setStreamed("");
    setThinking("");
    setSuggestions([]);
    setNotice(null);
    setQuery("");
    scrollToEnd();
    try {
      // new_thread stays false: this question joins the conversation.
      const res = await api.ask(q, scope, askId, false);
      applyThread(res.thread);
      if (res.chats) setChats(res.chats);
      setPending(null);
      scrollToEnd();
      void loadSuggestions();
    } catch (err) {
      setError(String(err));
      setPending(null);
      setQuery(q);
    } finally {
      askIdRef.current = null;
      setBusy(false);
      setStreamed("");
      setThinking("");
    }
  };

  /** Start a new conversation; an untouched one is reused by the sidecar. */
  const startNewChat = async () => {
    if (busy) return;
    setError(null);
    setNotice(null);
    setSuggestions([]);
    setPending(null);
    setStreamed("");
    setThinking("");
    setQuery("");
    setShowCtx(false);
    try {
      const res = await api.newChat();
      applyThread(res.thread);
      setChats(res.chats);
    } catch (e) {
      setError(String(e));
    }
    inputRef.current?.focus();
  };

  const openChatItem = async (id: string) => {
    if (busy || id === activeChatId) return;
    setError(null);
    setNotice(null);
    setSuggestions([]);
    setPending(null);
    setStreamed("");
    setThinking("");
    setQuery("");
    setShowCtx(false);
    try {
      const res = await api.openChat(id);
      applyThread(res.thread);
      setChats(res.chats);
    } catch (e) {
      setError(String(e));
    }
  };

  /** Cut the conversation back to just before this turn. */
  const rewindTo = async (turnId: number): Promise<string | null> => {
    if (busy) return null;
    setError(null);
    setNotice(null);
    setSuggestions([]);
    try {
      const res = await api.rewindThread(turnId);
      applyThread(res.thread);
      if (res.chats) setChats(res.chats);
      return res.question;
    } catch (e) {
      setError(String(e));
      return null;
    }
  };

  /** Take the question back into the box and forget the answer it got. */
  const editTurn = async (turn: ThreadTurn) => {
    const question = await rewindTo(turn.id);
    if (question === null) return;
    setQuery(turn.question);
    setFocusTick((n) => n + 1);
  };

  // Put the caret at the end of a question that was taken back for editing.
  useEffect(() => {
    if (!focusTick) return;
    const el = inputRef.current;
    if (!el) return;
    el.focus();
    el.setSelectionRange(el.value.length, el.value.length);
  }, [focusTick]);

  /** Ask the same question again from the same point in the conversation. */
  const retryTurn = async (turn: ThreadTurn) => {
    const question = await rewindTo(turn.id);
    if (question === null) return;
    await submit(undefined, turn.question);
  };

  const dropTurn = async (turn: ThreadTurn) => {
    await rewindTo(turn.id);
  };

  const copyQuestion = async (turn: ThreadTurn) => {
    try {
      await writeClipboard(turn.question);
      setCopiedTurn(turn.id);
      window.setTimeout(() => setCopiedTurn((cur) => (cur === turn.id ? null : cur)), 1400);
    } catch (e) {
      setError(String(e));
    }
  };

  /** Follow a "same narration" line into the reader's own copy of the book. */
  const openNarration = async (p: Parallel) => {
    try {
      const place = await api.locateNarration(p.collection, p.number);
      if (place.found && place.book_id != null) {
        onOpenPassage(place);
      } else {
        setError(
          `${place.collection_name} is not in your library, so ${p.label} cannot open in a book yet.`,
        );
      }
    } catch (e) {
      setError(String(e));
    }
  };

  const startRename = (id: string) => {
    const chat = chats.find((c) => c.id === id);
    renameCancelled.current = false;
    setMenu(null);
    setRenamingId(id);
    setRenameValue(chat?.title ?? "");
  };

  // Once the field is on screen, put the caret in it with the name selected,
  // so typing replaces the old title.
  useEffect(() => {
    if (renamingId) renameInputRef.current?.select();
  }, [renamingId]);

  const commitRename = async () => {
    const id = renamingId;
    if (!id) return;
    const title = renameValue.trim();
    setRenamingId(null);
    if (!title) return; // an empty name keeps the old one
    try {
      const res = await api.renameChat(id, title);
      setChats(res.chats);
    } catch (e) {
      setError(String(e));
    }
  };

  const createFolder = async () => {
    try {
      const res = await api.addFolder("New folder");
      setFolders(res.folders);
      setFolderName("New folder");
      setFolderRenaming(res.folder);
    } catch (e) {
      setError(String(e));
    }
  };

  const commitFolderRename = async () => {
    const id = folderRenaming;
    if (!id) return;
    const name = folderName.trim();
    setFolderRenaming(null);
    if (!name) return;
    try {
      const res = await api.renameFolder(id, name);
      setFolders(res.folders);
    } catch (e) {
      setError(String(e));
    }
  };

  // Select the whole default name, so typing replaces "New folder".
  useEffect(() => {
    if (folderRenaming) folderInputRef.current?.select();
  }, [folderRenaming]);

  const removeFolder = async (id: string) => {
    setMenu(null);
    if (busy) return;
    try {
      const res = await api.deleteFolder(id);
      setFolders(res.folders);
      setChats(res.chats);
    } catch (e) {
      setError(String(e));
    }
  };

  const fileChat = async (id: string, folderId: string | null) => {
    setMenu(null);
    try {
      const res = await api.moveChat(id, folderId);
      setChats(res.chats);
      setFolders(res.folders);
    } catch (e) {
      setError(String(e));
    }
  };

  const toggleFolder = (id: string) =>
    setClosedFolders((cur) =>
      cur.includes(id) ? cur.filter((f) => f !== id) : [...cur, id],
    );

  const removeChat = async (id: string) => {
    if (busy) return;
    try {
      const res = await api.deleteChat(id);
      setChats(res.chats);
      applyThread(res.thread);
      setSuggestions([]);
      setNotice(null);
    } catch (e) {
      setError(String(e));
    }
  };

  const compactNow = async () => {
    setCompacting(true);
    setError(null);
    try {
      const res = await api.compactThread();
      applyThread(res.thread);
      setNotice(
        res.compacted
          ? `Summarised ${res.dropped} earlier ${res.dropped === 1 ? "turn" : "turns"} to free context.`
          : "Nothing older to summarise yet — compaction keeps your two most recent turns.",
      );
    } catch (e) {
      setError(String(e));
    } finally {
      setCompacting(false);
    }
  };

  const renderedAnswer = (answer: string, citations: Citation[], prefix: string) => {
    const parts = answer.split(/(\[S\d+\])/g);
    return parts.map((part, i) => {
      const m = part.match(/^\[S(\d+)\]$/);
      if (m) {
        const num = parseInt(m[1], 10);
        const cite = citations.find((c) => c.source_index === num);
        if (cite) {
          return (
            <button
              key={i}
              className="cite-badge"
              title={citationLabel(cite)}
              onClick={() => {
                document.getElementById(`${prefix}-${num}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
              }}
            >
              S{num}
            </button>
          );
        }
        return <span key={i} className="cite-badge">{part}</span>;
      }
      return <span key={i}>{part}</span>;
    });
  };

  const pct = usage?.pct ?? 0;
  const level = pct >= 0.9 ? "hot" : pct >= 0.7 ? "warm" : "";
  const activeChatId = chats.find((c) => c.active)?.id;
  const unfiledChats = chats.filter((c) => !c.folder_id);

  const chatRow = (chat: ChatSummary, inFolder: boolean) => (
    <ChatRow
      key={chat.id}
      chat={chat}
      inFolder={inFolder}
      busy={busy}
      renaming={renamingId === chat.id}
      value={renameValue}
      inputRef={renameInputRef}
      onOpen={() => openChatItem(chat.id)}
      onMenu={(e) => {
        e.preventDefault();
        setMenu({ id: chat.id, x: e.clientX, y: e.clientY, kind: "chat" });
      }}
      onValue={setRenameValue}
      onCommit={() => void commitRename()}
      onCancel={() => {
        renameCancelled.current = true;
        setRenamingId(null);
      }}
      onBlur={() => {
        const cancelled = renameCancelled.current;
        renameCancelled.current = false;
        if (!cancelled) void commitRename();
      }}
    />
  );

  return (
    <div className="page ask-page">
      <div className="ask-head">
        <div>
          <h1 className="page-title">Ask the books</h1>
          <p className="page-sub">
            Question the library in plain language, then follow up — the conversation keeps its place
            and every claim stays tied to a highlighted passage.
          </p>
        </div>
      </div>

      <div className="thread">
        {summary && (
          <div className="thread-summary">
            <div className="thread-summary-label">Earlier turns, summarised</div>
            <p>{summary}</p>
          </div>
        )}

        {turns.map((turn, index) => {
          const notFound = turn.answer.startsWith(NOT_FOUND_PREFIX);
          const prefix = `citation-t${turn.id}`;
          const latest = index === turns.length - 1;
          return (
            <div className="turn" key={turn.id}>
              <div className="turn-q-wrap">
                <div className="turn-q" dir="auto">{turn.question}</div>
                <div className="turn-actions">
                  <button
                    onClick={() => copyQuestion(turn)}
                    className={copiedTurn === turn.id ? "copied" : ""}
                    title={copiedTurn === turn.id ? "Copied" : "Copy this question"}
                    aria-label="Copy this question"
                  >
                    {copiedTurn === turn.id ? <IconCheck /> : <IconCopy />}
                  </button>
                  <button
                    onClick={() => editTurn(turn)}
                    disabled={busy}
                    title="Edit - put this question back in the box"
                    aria-label="Edit this question"
                  >
                    <IconPencil />
                  </button>
                  <button
                    onClick={() => retryTurn(turn)}
                    disabled={busy}
                    title="Retry - ask this question again"
                    aria-label="Retry this question"
                  >
                    <IconRetry />
                  </button>
                  <button
                    onClick={() => dropTurn(turn)}
                    disabled={busy}
                    title="Roll back - remove this question and everything after it"
                    aria-label="Roll back to before this question"
                  >
                    <IconUndo />
                  </button>
                </div>
              </div>
              <div className={`card answer ${notFound ? "notfound" : ""}`} dir="auto">
                <div className="answer-label">Answer</div>
                <p>{renderedAnswer(turn.answer, turn.citations, prefix)}</p>
              </div>
              {turn.unverified && !notFound && (
                <div className="error-banner">
                  The model answered without citing a passage. Treat this answer as unsourced until you
                  find it in the text yourself.
                </div>
              )}
              {turn.citations.length > 0 && (
                <div className="sources-block">
                  <div className="sources-head">
                    Sources <span className="sources-count">{turn.citations.length}</span>
                  </div>
                  <div className="citations">
                    {turn.citations.map((c) => (
                      <CitationCard
                        key={c.chunk_id}
                        idPrefix={prefix}
                        citation={c}
                        onHighlight={onOpenCitation}
                        onBookmark={onBookmark}
                        onOpenNarration={openNarration}
                      />
                    ))}
                  </div>
                </div>
              )}

              {latest && (suggestBusy || suggestions.length > 0) && (
                <div className="suggest-row">
                  <span className="suggest-label">
                    {suggestBusy ? "Looking for gaps to fill…" : "Keep digging"}
                  </span>
                  {!suggestBusy &&
                    suggestions.map((s) => (
                      <button
                        key={s}
                        className="suggest-chip"
                        onClick={() => {
                          setQuery(s);
                          inputRef.current?.focus();
                        }}
                        title="Put this question in the box"
                      >
                        {s}
                      </button>
                    ))}
                </div>
              )}
            </div>
          );
        })}

        {busy && pending && (
          <div className="turn">
            <div className="turn-q" dir="auto">{pending}</div>
            <div className="card answer streaming" dir="auto">
              <div className="answer-label">Answer</div>
              {thinking && (
                <details className="thinking-box" open={!streamed}>
                  <summary>{streamed ? "Reasoning" : "Thinking…"}</summary>
                  <p className="thinking-text">
                    {thinking}
                    {!streamed && <span className="cursor">▍</span>}
                  </p>
                </details>
              )}
              {streamed ? (
                <p>
                  {streamed}
                  <span className="cursor">▍</span>
                </p>
              ) : (
                !thinking && (
                  <p className="stream-wait">
                    <span className="spinner" /> Searching the books and writing…
                  </p>
                )
              )}
              <div className="stream-note">Still writing — sources appear when the answer is finished.</div>
            </div>
          </div>
        )}

        {(compacting || notice) && (
          <div className={`thread-note ${compacting ? "busy" : ""}`}>
            {compacting ? (
              <>
                <span className="spinner" /> Compacting older turns…
              </>
            ) : (
              <>
                {notice}
                <button className="thread-note-close" onClick={() => setNotice(null)} aria-label="Dismiss">
                  ×
                </button>
              </>
            )}
          </div>
        )}

        {turns.length === 0 && !busy && books.length === 0 && (
          <div className="empty-state">
            <div style={{ fontSize: 44, color: "var(--gold)" }}>☾</div>
            <h2>Your library is empty</h2>
            <p>Import a text-based hadith PDF from the Library tab — everything stays on this device.</p>
          </div>
        )}

        {turns.length === 0 && !busy && books.length > 0 && (
          <div className="examples">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                className="example-chip"
                onClick={() => {
                  setQuery(ex);
                  inputRef.current?.focus();
                }}
              >
                {ex}
              </button>
            ))}
          </div>
        )}

        {noneSelected && turns.length === 0 && (
          <div className="error-banner">Select at least one book to search.</div>
        )}
        {error && <div className="error-banner">{error}</div>}
        <div ref={endRef} />
      </div>

      <div className="thread-composer">
        <form className="card ask-form" onSubmit={submit}>
          <textarea
            ref={inputRef}
            className="ask-input"
            dir="auto"
            placeholder={
              turns.length > 0
                ? "Ask a follow-up — e.g. which collection grades this narration?"
                : "e.g. Explain what the Prophet ﷺ taught about mercy…"
            }
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              // Enter asks, Shift+Enter starts a new line. Never swallow the Enter
              // that confirms an IME composition (Arabic input).
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                submit();
              }
            }}
          />
          <div className="ask-row">
            {llms.length > 0 && (
              <div className="model-pick" ref={modelRef}>
                <span>Model</span>
                <button
                  type="button"
                  className="model-pick-btn"
                  onClick={() => setModelOpen((v) => !v)}
                  disabled={busy || modelBusy}
                  aria-expanded={modelOpen}
                  aria-haspopup="listbox"
                  title="The local model that writes the answer"
                >
                  <span className="model-pick-name">
                    {llms.find((m) => m.path === modelPath)?.name ?? "Choose a model"}
                  </span>
                  <svg width="10" height="6" viewBox="0 0 10 6" fill="none" aria-hidden="true">
                    <path d="M1 1l4 4 4-4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </button>
                {modelOpen && (
                  <div className="model-menu" role="listbox" aria-label="Answer model">
                    {llms.map((m) => (
                      <button
                        key={m.path}
                        type="button"
                        role="option"
                        aria-selected={m.path === modelPath}
                        className={`model-opt ${m.path === modelPath ? "on" : ""}`}
                        disabled={!m.installed}
                        onClick={() => {
                          pickModel(m);
                          setModelOpen(false);
                        }}
                      >
                        <span className="model-opt-name">{m.name}</span>
                        {!m.installed && <span className="model-opt-note">not downloaded</span>}
                        {m.path === modelPath && <span className="scope-tick">✓</span>}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
            {books.length > 0 && (
              <div className="scope-wrap" ref={scopeRef}>
                <button
                  type="button"
                  className="scope-chip scope-pick"
                  onClick={() => setScopeOpen((v) => !v)}
                  disabled={busy}
                  aria-expanded={scopeOpen}
                  aria-haspopup="true"
                  title="Which books to search"
                >
                  <strong>{scopeLabel}</strong>
                </button>
                {scopeOpen && (
                  <div className="scope-menu" role="menu" aria-label="Books to search">
                    <button
                      type="button"
                      role="menuitemradio"
                      aria-checked={scope === null}
                      className={`scope-opt ${scope === null ? "on" : ""}`}
                      onClick={() => {
                        selectAll();
                        setScopeOpen(false);
                      }}
                    >
                      <span className="scope-tick">{scope === null ? "✓" : ""}</span> All loaded books
                    </button>
                    {books.map((b) => {
                      const on = scope === null || scope.includes(b.id);
                      return (
                        <button
                          key={b.id}
                          type="button"
                          role="menuitemcheckbox"
                          aria-checked={on}
                          className={`scope-opt ${on ? "on" : ""}`}
                          onClick={() => toggleScope(b.id)}
                          title={b.title}
                        >
                          <span className="scope-tick">{on ? "✓" : ""}</span>
                          <span className="scope-opt-name">{b.title}</span>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
            <div className="spacer" />

            <div className="ctx-wrap" ref={ctxRef}>
              <button
                type="button"
                className={`ctx-btn ${level}`}
                onClick={() => setShowCtx((v) => !v)}
                title="Context window"
                aria-expanded={showCtx}
              >
                <svg width="22" height="22" viewBox="0 0 30 30" aria-hidden="true">
                  <circle cx="15" cy="15" r="12" className="ctx-ring-bg" />
                  <circle
                    cx="15"
                    cy="15"
                    r="12"
                    className="ctx-ring-fill"
                    strokeDasharray={`${RING * Math.min(pct, 1)} ${RING}`}
                  />
                </svg>
                <span className="ctx-pct">{Math.round(pct * 100)}%</span>
              </button>

              {showCtx && (
                <div className="ctx-pop" role="dialog" aria-label="Context window">
                  {usage ? (
                    <>
                      <div className="ctx-line ctx-title">
                        <span>Context</span>
                        <strong>{formatTokens(usage.used)}</strong>
                      </div>
                      <div className="ctx-line">
                        <span>Compacts at <strong>{formatTokens(usage.compact_at)}</strong></span>
                        <button
                          className="ctx-compact"
                          onClick={compactNow}
                          disabled={compacting || busy || usage.turns <= 2}
                          title={
                            usage.turns <= 2
                              ? "Compact keeps your two most recent turns; there is nothing older yet."
                              : "Summarise older turns now"
                          }
                        >
                          {compacting ? "Compacting…" : "Compact"}
                        </button>
                      </div>
                      <div className="ctx-divider" />
                      <div className="ctx-line ctx-title">
                        <span>Thread usage</span>
                        <strong>{usage.turns} {usage.turns === 1 ? "turn" : "turns"}</strong>
                      </div>
                      <div className="ctx-line"><span>Prompt</span><span>{formatTokens(usage.used)}</span></div>
                      <div className="ctx-line"><span>Excerpts</span><span>{formatTokens(usage.excerpt_tokens)}</span></div>
                      <div className="ctx-line"><span>Conversation</span><span>{formatTokens(usage.history_tokens)}</span></div>
                      <div className="ctx-line"><span>Window</span><span>{formatTokens(usage.ctx)}</span></div>
                      <div className="ctx-line"><span>Summaries</span><span>{usage.compactions}</span></div>
                      <p className="ctx-note">
                        Older turns are summarised automatically at {Math.round(usage.compact_at / usage.ctx * 100)}% of
                        the window. The window size is set in Settings → Answer model.
                      </p>
                    </>
                  ) : (
                    <p className="ctx-note">Ask a question to start filling the context window.</p>
                  )}
                </div>
              )}
            </div>

            <button
              className="btn btn-primary"
              type="submit"
              disabled={busy || query.trim().length < 2 || books.length === 0 || noneSelected}
            >
              {busy ? (<><span className="spinner" /> Thinking…</>) : turns.length > 0 ? "Follow up →" : "Ask →"}
            </button>
          </div>
        </form>
      </div>

      {sidebarSlot &&
        createPortal(
          <>
            <div className="chat-slot-head">
              <span className="chat-slot-title">Conversations</span>
              <button
                className="chat-new chat-folder"
                onClick={() => void createFolder()}
                title="New folder"
                aria-label="New folder"
              >
                <FolderIcon size={14} />
              </button>
              <button
                className="chat-new"
                onClick={startNewChat}
                disabled={busy}
                title="New conversation"
                aria-label="New conversation"
              >
                +
              </button>
            </div>
            <div className="chat-list">
              {chats.length === 0 && (
                <div className="chat-empty">No conversations yet. Ask a question to start one.</div>
              )}
              {folders.map((folder) => {
                const items = chats.filter((c) => (c.folder_id ?? null) === folder.id);
                const closed = closedFolders.includes(folder.id);
                return (
                  <div className="folder-group" key={folder.id}>
                    <div
                      className="folder-row"
                      onContextMenu={(e) => {
                        e.preventDefault();
                        setMenu({ id: folder.id, x: e.clientX, y: e.clientY, kind: "folder" });
                      }}
                    >
                      {folderRenaming === folder.id ? (
                        <input
                          ref={folderInputRef}
                          className="folder-rename"
                          value={folderName}
                          onChange={(e) => setFolderName(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") {
                              e.preventDefault();
                              void commitFolderRename();
                            } else if (e.key === "Escape") {
                              e.preventDefault();
                              setFolderRenaming(null);
                            }
                          }}
                          onBlur={() => void commitFolderRename()}
                          aria-label="Folder name"
                        />
                      ) : (
                        <button
                          className="folder-toggle"
                          onClick={() => toggleFolder(folder.id)}
                          aria-expanded={!closed}
                          title={closed ? "Show conversations" : "Hide conversations"}
                        >
                          <svg
                            className={`folder-chevron ${closed ? "closed" : ""}`}
                            width="8"
                            height="8"
                            viewBox="0 0 10 10"
                            fill="none"
                            aria-hidden="true"
                          >
                            <path
                              d="M2.5 1.5 6 5l-3.5 3.5"
                              stroke="currentColor"
                              strokeWidth="1.6"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                            />
                          </svg>
                          <FolderIcon />
                          <span className="folder-name">{folder.name}</span>
                          <span className="folder-count">{items.length}</span>
                        </button>
                      )}
                    </div>
                    {!closed && items.map((chat) => chatRow(chat, true))}
                  </div>
                );
              })}
              {folders.length > 0 && unfiledChats.length > 0 && (
                <div className="chat-group-label">Unfiled</div>
              )}
              {unfiledChats.map((chat) => chatRow(chat, false))}
            </div>
          </>,
          sidebarSlot,
        )}

      {menu &&
        createPortal(
          <div
            className="chat-menu"
            role="menu"
            style={{
              left: Math.min(menu.x, window.innerWidth - 172),
              top: Math.min(menu.y, window.innerHeight - 96),
            }}
            onMouseDown={(e) => e.stopPropagation()}
            onContextMenu={(e) => e.preventDefault()}
          >
            {menu.kind === "folder" ? (
              <>
                <button
                  role="menuitem"
                  onClick={() => {
                    const folder = folders.find((f) => f.id === menu.id);
                    setMenu(null);
                    setFolderName(folder?.name ?? "");
                    setFolderRenaming(menu.id);
                  }}
                >
                  Rename folder
                </button>
                <button
                  role="menuitem"
                  className="danger"
                  onClick={() => void removeFolder(menu.id)}
                >
                  Delete folder
                </button>
              </>
            ) : (
              <>
                <button role="menuitem" onClick={() => startRename(menu.id)}>
                  Rename
                </button>
                <button
                  role="menuitem"
                  className="danger"
                  onClick={() => {
                    const id = menu.id;
                    setMenu(null);
                    void removeChat(id);
                  }}
                >
                  Delete
                </button>
                <div className="menu-sep" />
                <div className="menu-label">Move to</div>
                <button role="menuitem" onClick={() => void fileChat(menu.id, null)}>
                  No folder
                </button>
                {folders.map((folder) => (
                  <button
                    key={folder.id}
                    role="menuitem"
                    onClick={() => void fileChat(menu.id, folder.id)}
                  >
                    {folder.name}
                  </button>
                ))}
              </>
            )}
          </div>,
          document.body,
        )}
    </div>
  );
}
