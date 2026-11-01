import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import type { AttemptMeta, AttemptsView, Training } from "../api";
import { url } from "../api";
import { LineChart } from "../components/charts";
import { Icon } from "../components/ui";
import type { Frames } from "../showcase/data";
import { loadAttempt, type Attempt } from "./npz";
import type { BuildScene, LearnActor } from "./scene";

// Stage 06, "watch it learn": the policy's attempts while it trained (an .npz every N iterations, a2h_env/attempts.py),
// played one after another on the G1 in the policy slot with the retarget as a ghost. A live job's attempts arrive
// while the GPU trains (the page polls); a stored run's are all here. The slider picks one; "Play all" steps
// through them; "Latest" follows a training as it goes.

export type LearnMode = "follow" | "play" | "hold";

export type LearnPlayer = {
  rows: AttemptMeta[];
  sel: number;
  setSel: (i: number) => void;
  mode: LearnMode;
  setMode: (m: LearnMode) => void;
  actor: LearnActor | null;
  current: AttemptMeta | null;
  loading: boolean;
  err: string | null;
};

const pad = (n: number) => String(n).padStart(4, "0");

/** Drives the scene's learning actor from the attempts: which one plays, loading it, stepping on. */
export function useLearnPlayer(scene: BuildScene | null, view: AttemptsView | null, live: boolean, ref: Frames | null,
  refFps: number, active: boolean): LearnPlayer {
  const rows = useMemo(() => view?.attempts ?? [], [view]);
  const [sel, setSelRaw] = useState(0);
  const [mode, setMode] = useState<LearnMode>(live ? "follow" : "play");
  const [actor, setActor] = useState<LearnActor | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const cache = useRef(new Map<string, Promise<Attempt>>());
  const modeRef = useRef(mode);
  modeRef.current = mode;
  const n = rows.length;
  const setSel = (i: number) => { setSelRaw(Math.max(0, Math.min(n - 1, i))); setMode("hold"); };

  // a training that is still going: follow its newest attempt
  useEffect(() => { if (mode === "follow" && n) setSelRaw(n - 1); }, [mode, n]);
  useEffect(() => { if (!live && mode === "follow") setMode("play"); }, [live, mode]);
  useEffect(() => { if (live) setMode("follow"); }, [live]); // a training that starts on this page: follow it

  const row = rows[Math.min(sel, n - 1)] ?? null;
  const get = (r: AttemptMeta) => {
    let p = cache.current.get(r.url);
    if (!p) { p = loadAttempt(url(r.url)); cache.current.set(r.url, p); p.catch(() => cache.current.delete(r.url)); }
    return p;
  };

  useEffect(() => {
    if (!scene || !active || !row) { scene?.setAttempt(null, 50, null, 30, ""); setActor(null); return; }
    let on = true;
    setLoading(true);
    setErr(null);
    get(row).then((a) => {
      if (!on) return;
      const total = view?.iterations ?? row.iteration;
      const label = `attempt · iteration ${row.iteration} of ${total}`;
      const act = scene.setAttempt(a.frames, a.fps, ref, refFps, label);
      if (act) {
        act.loop = modeRef.current !== "play";
        act.onEnd = () => {
          if (modeRef.current !== "play") return;
          setSelRaw((i) => (i + 1 < n ? i + 1 : live ? i : 0));
          if (sel + 1 >= n && !live && n === 1) act.seek(0); // one attempt: play it again
        };
        if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) act.setPlaying(false);
      }
      setActor(act);
      setLoading(false);
      const next = rows[Math.min(sel, n - 1) + 1];
      if (next) get(next).catch(() => undefined);
    }).catch((e) => { if (on) { setErr(String(e.message ?? e)); setLoading(false); } });
    return () => { on = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scene, active, row?.url, ref, refFps]);
  useEffect(() => { if (actor) actor.loop = mode !== "play"; }, [actor, mode]);
  useEffect(() => () => { scene?.setAttempt(null, 50, null, 30, ""); }, [scene]);

  return { rows, sel: Math.min(sel, Math.max(0, n - 1)), setSel, mode, setMode, actor, current: row, loading, err };
}

export function outcome(a: AttemptMeta) {
  const parts: string[] = [];
  if (a.reward != null) parts.push(`reward ${a.reward.toFixed(1)}`);
  if (a.stages_total) parts.push(`${a.stages_completed ?? 0}/${a.stages_total} key moments`);
  parts.push(a.fell ? `fell${a.fell_at_s != null ? ` at ${a.fell_at_s.toFixed(1)} s` : ""}` : a.success ? "completed the clip" : "stayed up");
  return parts.join(" · ");
}

/** The slider and the play buttons, over the stage. */
export function LearnControls({ p, live, training, total }: { p: LearnPlayer; live: boolean; training: Training | null; total: number | null }) {
  const st = useSyncExternalStore(p.actor?.subscribe ?? (() => () => undefined), p.actor?.getState ?? (() => null));
  const a = p.current;
  const n = p.rows.length;
  return (
    <div className="bv-learn" role="group" aria-label="The policy's attempts while it trained">
      <div className="bv-learn-row">
        <button className="btn sm" onClick={() => p.actor?.setPlaying(!st?.playing)} disabled={!p.actor} aria-label={st?.playing ? "Pause the attempt" : "Play the attempt"}>
          <Icon name={st?.playing ? "pause" : "play"} size={13} />
        </button>
        <input type="range" min={0} max={Math.max(0, n - 1)} value={p.sel} onChange={(e) => p.setSel(Number(e.target.value))}
          aria-label="Which attempt" aria-valuetext={a ? `iteration ${a.iteration}` : ""} disabled={n < 2} />
        <span className="mono bv-learn-it">{a ? `iter ${pad(a.iteration)}${total ? ` / ${total}` : ""}` : "no attempt yet"}</span>
      </div>
      <div className="bv-learn-row">
        <span className="bv-seg" role="group" aria-label="How the attempts play">
          <button className={p.mode === "play" ? "on" : ""} aria-pressed={p.mode === "play"} onClick={() => p.setMode("play")} disabled={n < 2}>Play all</button>
          {live && <button className={p.mode === "follow" ? "on" : ""} aria-pressed={p.mode === "follow"} onClick={() => p.setMode("follow")}>Latest</button>}
        </span>
        <span className="bv-learn-out mono" role="status">
          {p.err ? `couldn't load the attempt: ${p.err}` : p.loading && !p.actor ? "loading…" : a ? outcome(a) : live ? "the first attempt arrives after the first iterations" : ""}
          {live && training && training.state === "running" ? ` · training: iteration ${training.iteration} of ${training.iterations}` : ""}
        </span>
      </div>
    </div>
  );
}

/** The reward while it trained, with the attempt being played marked; under the stage. */
export function LearnCurve({ view, current, training }: { view: AttemptsView; current: AttemptMeta | null; training: Training | null }) {
  const pts = view.curve.length ? view.curve : [];
  const xMax = view.iterations ?? training?.iterations ?? undefined;
  const n = view.attempts.length;
  return (
    <div className="card card-pad bv-learn-curve">
      <div className="eyebrow">Reward while training · {n} attempt{n === 1 ? "" : "s"}{view.attempt_every ? `, one every ${view.attempt_every} iterations` : ""}</div>
      {pts.length > 1 ? (
        <LineChart series={[{ key: "policy", label: "mean episode reward", color: "var(--accent)", points: pts }]} height={160}
          xLabel="iteration" yLabel="reward" xMax={xMax} marker={current?.iteration ?? null} />
      ) : (
        <p className="faint" style={{ fontSize: 13 }}>{training && training.state === "running" ? "The curve appears as the GPU logs its first iterations." : "No reward curve recorded."}</p>
      )}
      <p className="muted" style={{ fontSize: 13, marginTop: 8 }}>
        mjlab's mean episode return over the training environments (every variant end to end). Each attempt is one try on
        the clip itself, so the first ones fall at once and the later ones follow the reference, the ghost, more closely.
        {view.task ? <> Task: <span className="mono">{view.task}</span>.</> : null}
      </p>
    </div>
  );
}
