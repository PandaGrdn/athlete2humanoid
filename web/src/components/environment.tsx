import { useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import type { TaskSpec, TaskStage } from "../api";
import { Icon } from "./ui";

const pct = (t: number, dur: number) => Math.min(100, Math.max(0, (t / dur) * 100));
const secs = (t: number) => `${t.toFixed(t < 10 ? 2 : 1)} s`;

function feetNote(s: TaskStage) {
  if (s.kind === "flight") return s.clearance_m != null ? `both feet off, ${Math.round(s.clearance_m * 100)} cm up` : "both feet off";
  if (s.kind === "settle") return "feet down, steady";
  const f = s.feet_down;
  if (!f) return null;
  if (!f[0] && !f[1]) return "both feet off";
  if (!f[0]) return "left foot lifted";
  if (!f[1]) return "right foot lifted";
  return null;
}

const startOf = (s: TaskStage) => (s.kind === "flight" ? s.start_s : s.at_s);

/** The RL environment compiled from the retarget: key moments on the clip's timeline, then the list.
 *  With `t` and `onSeek` it follows a video: a playhead, moments light up as it passes them, and a click seeks. */
export function EnvironmentPanel({ spec, live = false, t, onSeek }: { spec: TaskSpec; live?: boolean; t?: number; onSeek?: (t: number) => void }) {
  const [on, setOn] = useState<number | null>(null);
  const inner = useRef<HTMLDivElement>(null);
  const dur = spec.duration_s > 0 ? spec.duration_s : Math.max(1, ...spec.stages.map((s) => s.end_s));
  const c = spec.counts;
  const sc = spec.self_check;

  // Key moments are numbered in order; a marker too close to the one before drops to the second lane.
  const number = new Map<number, number>();
  const lane = new Map<number, number>();
  let last = -Infinity;
  let lastLane = 1;
  spec.stages.forEach((s, i) => {
    if (s.kind !== "pose") return;
    number.set(i, number.size + 1);
    const x = pct(s.at_s, dur);
    lastLane = x - last < 3.2 ? 1 - lastLane : 0;
    lane.set(i, lastLane);
    last = x;
  });
  const hover = (i: number | null) => () => setOn(i);
  // The stage playback is in: the last one started (a flight only while it lasts).
  let cur = -1;
  if (t != null) spec.stages.forEach((s, i) => { if (startOf(s) <= t + 1e-3 && (s.kind !== "flight" || t <= s.end_s)) cur = i; });
  const state = (s: TaskStage) => (t == null ? "" : t >= s.at_s && t - s.at_s < 0.35 ? "hit past" : t >= s.at_s ? "past" : "");
  const seekTrack = (e: PointerEvent<HTMLDivElement>) => {
    const r = inner.current?.getBoundingClientRect();
    if (!onSeek || !r || (e.target as HTMLElement).closest("button")) return;
    onSeek(Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)) * dur);
  };
  const rowKey = (s: TaskStage) => (e: KeyboardEvent) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSeek?.(startOf(s)); }
  };

  return (
    <div className="card card-pad env">
      <div className="env-head">
        <div>
          <div className="eyebrow">RL environment{live && <span className="pill accent" style={{ marginLeft: 8 }}>ready</span>}</div>
          <div className="env-counts">
            {c.moments} key moment{c.moments === 1 ? "" : "s"}
            {c.optional > 0 && <span className="faint"> ({c.required} required, {c.optional} optional)</span>}
            {c.flights > 0 && <> · {c.flights} flight{c.flights === 1 ? "" : "s"}</>}
            <span className="faint"> · {secs(dur)} clip</span>
          </div>
        </div>
        <div className="env-legend" aria-hidden="true">
          <span><i className="env-dot req" />required</span>
          <span><i className="env-dot opt" />optional</span>
          {c.flights > 0 && <span><i className="env-swatch" />flight</span>}
        </div>
      </div>
      <p className="env-lede">
        The robot is rewarded for reaching these moments in order, at its own timing, without falling.
        Required moments (solid) must all be reached; optional ones (hollow) can be skipped.
        {c.flights > 0 && " Flights (shaded) need both feet off the ground."}
      </p>

      <div className="env-timeline" role="img"
        aria-label={`Timeline of ${c.moments} key moments over ${secs(dur)}: ${spec.stages.filter((s) => s.kind === "pose").map((s) => s.name).join(", ")}`}>
        <div className={`env-track ${onSeek ? "seek" : ""}`} onPointerDown={seekTrack}><div className="env-inner" ref={inner}>
          <div className="env-base" />
          {spec.stages.map((s, i) => s.kind === "flight" && (
            <div key={i} className={`env-flight ${on === i ? "on" : ""}`} onMouseEnter={hover(i)} onMouseLeave={hover(null)}
              style={{ left: `${pct(s.start_s, dur)}%`, width: `${Math.max(0.6, pct(s.end_s, dur) - pct(s.start_s, dur))}%` }}
              title={`${s.name}: ${secs(s.start_s)} – ${secs(s.end_s)}`} />
          ))}
          {spec.stages.map((s, i) => s.kind === "settle" && (
            <i key={i} className="env-tick" style={{ left: `${pct(s.at_s, dur)}%` }} title={`${s.name} · ${secs(s.at_s)}`} />
          ))}
          {spec.stages.map((s, i) => {
            if (s.kind !== "pose") return null;
            const props = {
              className: `env-mark ${s.optional ? "opt" : "req"} lane-${lane.get(i)} ${on === i ? "on" : ""} ${state(s)}`,
              style: { left: `${pct(s.at_s, dur)}%` }, onMouseEnter: hover(i), onMouseLeave: hover(null),
              title: `${number.get(i)}. ${s.name} · ${secs(s.at_s)}${s.optional ? " · optional" : ""}`,
            };
            return onSeek
              ? <button key={i} type="button" {...props} onClick={() => onSeek(s.at_s)} aria-label={`Seek to ${props.title}`}>{number.get(i)}</button>
              : <span key={i} {...props}>{number.get(i)}</span>;
          })}
          {t != null && <i className="env-playhead" style={{ left: `${pct(t, dur)}%` }} />}
        </div></div>
        <div className="env-axis mono"><span>0 s</span><span>{secs(dur / 2)}</span><span>{secs(dur)}</span></div>
      </div>

      <ol className="env-list">
        {spec.stages.map((s, i) => {
          const note = feetNote(s);
          return (
            <li key={i} className={`env-row k-${s.kind} ${s.optional ? "opt" : ""} ${on === i || cur === i ? "on" : ""} ${onSeek ? "seek" : ""}`}
              onMouseEnter={hover(i)} onMouseLeave={hover(null)}
              {...(onSeek && { onClick: () => onSeek(startOf(s)), onKeyDown: rowKey(s), tabIndex: 0, title: `Seek to ${secs(startOf(s))}` })}>
              <span className="env-bullet">
                {s.kind === "pose" ? <i className={`env-dot ${s.optional ? "opt" : "req"}`}>{number.get(i)}</i>
                  : s.kind === "flight" ? <i className="env-swatch" /> : <i className="env-tick-sm" />}
              </span>
              <span className="env-what">
                <span className="env-name">{s.name}</span>
                {s.optional && <span className="env-tag">optional</span>}
                {(s.body_parts.length > 0 || note) && (
                  <span className="env-parts">
                    {s.body_parts.map((p) => (
                      <span key={p.part} className="env-part" title={`goal relevance ${p.weight.toFixed(2)}`}
                        style={{ opacity: 0.55 + 0.45 * p.weight }}>{p.part}</span>
                    ))}
                    {note && <span className="faint">{note}</span>}
                  </span>
                )}
              </span>
              <span className="env-time mono">{s.kind === "flight" ? `${secs(s.start_s)}–${secs(s.end_s)}` : secs(s.at_s)}</span>
            </li>
          );
        })}
      </ol>

      {sc && (
        <div className={`env-check ${sc.success ? "ok" : "bad"}`}>
          <Icon name={sc.success ? "check" : "x"} size={15} />
          <span>
            The retarget completes {sc.stages_completed}/{sc.stages_total} stages of its own spec
            {!sc.success && sc.fell_at_s != null && <>, falling at {secs(sc.fell_at_s)}</>}
            <span className="faint">{sc.success ? " (the G1 replaying the clip reaches every moment, so the task is achievable)." : " (the reference motion itself misses some, so those stages may be unreachable)."}</span>
          </span>
        </div>
      )}
      {(spec.format === "phases" || spec.warnings.length > 0) && (
        <div className="env-notes">
          {spec.format === "phases" && <span className="faint">Older stand / squat / jump / land spec; newer runs name key moments for any action.</span>}
          {spec.warnings.map((w) => <span key={w} className="pill warn">{w}</span>)}
        </div>
      )}
    </div>
  );
}
