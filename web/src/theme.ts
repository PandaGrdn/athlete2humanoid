import { useEffect, useState } from "react";

/** Two designs over the same pages: "aether" (the default: black and white, after Nous Research's Hermes
 *  Agent) and "classic" (the original). Either can be picked from the footer, or with ?ui=classic / ?ui=aether;
 *  the choice is remembered per browser. Classic stays the fallback: anything aether doesn't restyle
 *  looks as it always did. VITE_DEFAULT_UI=classic builds with classic as the default again. */
export type Ui = "classic" | "aether";
const KEY = "a2h-ui";
const DEFAULT: Ui = import.meta.env.VITE_DEFAULT_UI === "classic" ? "classic" : "aether";
const isUi = (v: unknown): v is Ui => v === "classic" || v === "aether";

function initial(): Ui {
  const q = new URLSearchParams(window.location.search).get("ui");
  if (isUi(q)) {
    store(q);
    return q;
  }
  try {
    const s = window.localStorage.getItem(KEY);
    if (isUi(s)) return s;
  } catch { /* storage blocked: the default */ }
  return DEFAULT;
}

function store(ui: Ui) {
  try { window.localStorage.setItem(KEY, ui); } catch { /* not remembered, still applied */ }
}

/** Set before the first render (main.tsx), so the page never flashes the other design. */
export function applyInitialUi() {
  document.documentElement.dataset.ui = initial();
}

export function useUi(): [Ui, (ui: Ui) => void] {
  const [ui, setUi] = useState<Ui>(() => (isUi(document.documentElement.dataset.ui) ? document.documentElement.dataset.ui : DEFAULT));
  useEffect(() => {
    document.documentElement.dataset.ui = ui;
    document.querySelector('meta[name="theme-color"]')?.setAttribute("content", ui === "aether" ? "#000000" : "#0a0c0f");
  }, [ui]);
  return [ui, (next) => { store(next); setUi(next); }];
}
