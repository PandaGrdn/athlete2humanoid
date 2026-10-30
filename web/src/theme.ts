import { createContext, useContext, useEffect, useState } from "react";

/** Four designs over the same pages: "pixel" (the default: white paper, black ink, dot-matrix headings, a glyph
 *  field in the hero, after Sequoia's bot tournament page), "aurora" (midnight, electric blue and cantaloupe,
 *  editorial serif headings), "aether" (black and white, after Nous Research's Hermes Agent) and "classic" (the
 *  original). Any can be picked from the footer, or with ?ui=pixel / ?ui=aurora / ?ui=aether / ?ui=classic; the
 *  choice is remembered per browser. Classic stays the fallback: anything a design doesn't restyle looks as it
 *  always did. VITE_DEFAULT_UI=aurora (or another) builds with that one as the default instead. */
export type Ui = "classic" | "aether" | "aurora" | "pixel";
// v3: a choice saved before pixel existed (aurora was the default then) no longer overrides it
const KEY = "a2h-ui-v3";
const isUi = (v: unknown): v is Ui => v === "classic" || v === "aether" || v === "aurora" || v === "pixel";
const DEFAULT: Ui = isUi(import.meta.env.VITE_DEFAULT_UI) ? import.meta.env.VITE_DEFAULT_UI : "pixel";
const THEME_COLOR: Record<Ui, string> = { classic: "#0a0c0f", aether: "#000000", aurora: "#02070A", pixel: "#FFFFFF" };

/** The design in use, for components that render differently per design (App provides it). */
export const UiCtx = createContext<Ui>("classic");
export const useUiMode = () => useContext(UiCtx);

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
