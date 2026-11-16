import { useSyncExternalStore } from "react";
import { isTauri } from "./api";

export interface AvailableUpdate {
  version: string;
  notes: string;
}

let available: AvailableUpdate | null = null;
let started = false;
const listeners = new Set<() => void>();

function publish(next: AvailableUpdate | null) {
  available = next;
  listeners.forEach((listener) => listener());
}

async function lookForUpdate() {
  try {
    // Loaded on demand so the updater never sits in the startup path.
    const { check } = await import("@tauri-apps/plugin-updater");
    const found = await check();
    if (found) publish({ version: found.version, notes: found.body ?? "" });
  } catch {
    // Offline, no release yet, or the endpoint is unreachable: stay quiet.
  }
}

/**
 * Ask GitHub Releases once, a few seconds after launch, so an update can
 * surface on its own instead of waiting for someone to open Settings.
 * Skipped entirely in the browser preview; if the machine is offline the check
 * waits for the connection rather than failing at startup.
 */
export function startUpdateCheck(delayMs = 6000) {
  if (started || !isTauri()) return;
  started = true;

  const run = () => {
    if (navigator.onLine === false) {
      window.addEventListener("online", run, { once: true });
      return;
    }
    void lookForUpdate();
  };

  window.setTimeout(run, delayMs);
}

export function useAvailableUpdate() {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => available,
    () => null,
  );
}
