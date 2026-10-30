import { useEffect, useState } from "react";

/** Three designs over the same pages: "aurora" (the default: midnight, electric blue and cantaloupe, editorial
 *  serif headings and mono labels), "aether" (black and white, after Nous Research's Hermes Agent) and "classic"
 *  (the original). Any can be picked from the footer, or with ?ui=aurora / ?ui=aether / ?ui=classic; the choice
 *  is remembered per browser. Classic stays the fallback: anything a design doesn't restyle looks as it always
 *  did. VITE_DEFAULT_UI=aether or classic builds with that one as the default instead. */
export type Ui = "classic" | "aether" | "aurora";
const KEY = "a2h-ui";
const isUi = (v: unknown): v is Ui => v === "classic" || v === "aether" || v === "aurora";
const DEFAULT: Ui = isUi(import.meta.env.VITE_DEFAULT_UI) ? import.meta.env.VITE_DEFAULT_UI : "aurora";
const THEME_COLOR: Record<Ui, string> = { classic: "#0a0c0f", aether: "#000000", aurora: "#02070A" };

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
    document.querySelector('meta[name="theme-color"]')?.setAttribute("content", THEME_COLOR[ui]);
  }, [ui]);
  return [ui, (next) => { store(next); setUi(next); }];
}
