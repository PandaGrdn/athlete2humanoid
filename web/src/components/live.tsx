import { useEffect, useMemo, useRef, type PointerEvent } from "react";
import type { LiveData, TaskSpec, TaskStage } from "../api";

// Canvases can't read CSS variables cheaply; these mirror styles.css (left / right as in the charts).
const C = { left: "#3987e5", right: "#d95926", mid: "#e8edf2", grid: "#222a33", gridStrong: "#2e3843", shadow: "rgba(0, 0, 0, 0.55)", text: "#6d7884" };
const side = (name: string) => (/(^|[_\s])left/.test(name) ? C.left : /(^|[_\s])right/.test(name) ? C.right : C.mid);

const frameAt = (t: number, fps: number, n: number) => Math.max(0, Math.min(n - 1, Math.floor(t * fps + 1e-3)));
const secs = (t: number) => `${t.toFixed(t < 10 ? 2 : 1)} s`;

/** Redraws a canvas every animation frame at the device's pixel ratio; `draw` gets CSS pixels. */
function useCanvas(draw: (c: CanvasRenderingContext2D, w: number, h: number) => void) {
  const ref = useRef<HTMLCanvasElement>(null);
  const fn = useRef(draw);
  fn.current = draw;
  useEffect(() => {
    let raf = 0;
    const loop = () => {
      const el = ref.current;
      const ctx = el?.getContext("2d");
      if (el && ctx && el.clientWidth && el.clientHeight) {
        const dpr = Math.min(2, window.devicePixelRatio || 1);
        const w = el.clientWidth;
        const h = el.clientHeight;
        if (el.width !== Math.round(w * dpr) || el.height !== Math.round(h * dpr)) {
          el.width = Math.round(w * dpr);
          el.height = Math.round(h * dpr);
        }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, w, h);
        fn.current(ctx, w, h);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);
  return ref;
}

/** The pose model's 2D landmarks over the processed video, at the video's current frame. The video is
 *  object-fit: contain, so the landmarks go in the letterboxed picture, not the element's box. */
export function PoseOverlay({ video, data }: { video: HTMLVideoElement | null; data: LiveData }) {
  const lm = data.landmarks;
  const ref = useCanvas((c, w, h) => {
    if (!video || !lm || !lm.frames.length) return;
    const pts = lm.frames[frameAt(video.currentTime, data.fps, lm.frames.length)];
    if (!pts) return;
    const vw = video.videoWidth || data.video?.width || 1;
    const vh = video.videoHeight || data.video?.height || 1;
    const s = Math.min(w / vw, h / vh);
    const ox = (w - vw * s) / 2;
    const oy = (h - vh * s) / 2;
    const xy = (p: [number, number]) => [ox + p[0] * vw * s, oy + p[1] * vh * s] as const;
    c.lineCap = "round";
    c.lineWidth = Math.max(2, Math.min(4, (vh * s) / 180));
    for (const [a, b] of lm.edges) {
      const pa = pts[a];
      const pb = pts[b];
      if (!pa || !pb) continue;
      const ca = side(lm.names[a] ?? "");
      c.strokeStyle = ca === side(lm.names[b] ?? "") ? ca : C.mid;
      c.globalAlpha = 0.9;
      c.beginPath();
      c.moveTo(...xy(pa));
      c.lineTo(...xy(pb));
      c.stroke();
    }
    c.globalAlpha = 1;
    pts.forEach((p, i) => {
      if (!p) return;
      const [x, y] = xy(p);
      c.fillStyle = side(lm.names[i] ?? "");
      c.beginPath();
      c.arc(x, y, c.lineWidth * 0.8, 0, Math.PI * 2);
      c.fill();
    });
  });
  return <canvas ref={ref} className="live-overlay" aria-hidden="true" />;
}

type V3 = [number, number, number];

/** The retargeted G1 as a stick figure: a fixed orthographic camera (drag to turn it), z up, the ground
 *  as a grid with the robot's shadow on it, and the bodies at the video's current frame. */
export function RobotView({ video, data }: { video: HTMLVideoElement | null; data: LiveData }) {
  const robot = data.robot;
  const view = useRef({ az: (35 * Math.PI) / 180, el: (14 * Math.PI) / 180, drag: null as null | { x: number; az: number } });
  // Fixed framing for the whole clip (any azimuth): the horizontal radius around the mean, the height range.
  const box = useMemo(() => {
    if (!robot?.frames.length) return null;
    let cx = 0, cy = 0, n = 0, zMin = Infinity, zMax = -Infinity;
    for (const f of robot.frames) for (const p of f) { cx += p[0]; cy += p[1]; n++; zMin = Math.min(zMin, p[2]); zMax = Math.max(zMax, p[2]); }
    cx /= n;
    cy /= n;
    let r = 0;
    for (const f of robot.frames) for (const p of f) r = Math.max(r, Math.hypot(p[0] - cx, p[1] - cy));
    return { cx, cy, r: r + 0.15, ground: Math.min(0, zMin), zMax };
  }, [robot]);

  const ref = useCanvas((c, w, h) => {
    if (!robot || !box || !robot.frames.length) return;
    const { az, el } = view.current;
    const t = video?.currentTime ?? 0;
    const f = robot.frames[frameAt(t, data.fps, robot.frames.length)];
    const [ca, sa, ce, se] = [Math.cos(az), Math.sin(az), Math.cos(el), Math.sin(el)];
    // Camera on the (cos az, sin az) side looking back at the robot: right = (-sin, cos, 0).
    const proj = (p: V3) => {
      const x = p[0] - box.cx;
      const y = p[1] - box.cy;
      const z = p[2] - box.ground;
      return { u: -x * sa + y * ca, v: z * ce - (x * ca + y * sa) * se, d: x * ca * ce + y * sa * ce + z * se };
    };
    const spanU = 2 * box.r;
    const spanV = (box.zMax - box.ground) * ce + 2 * box.r * se;
    const s = Math.min((w - 32) / spanU, (h - 40) / spanV);
    const ox = w / 2;
    const oy = h - 20 - box.r * se * s;
    const sx = (q: { u: number; v: number }) => [ox + q.u * s, oy - q.v * s] as const;

    // Ground grid, 0.25 m, around the clip's footprint.
    const g = Math.ceil((box.r + 0.25) / 0.25) * 0.25;
    c.lineWidth = 1;
    for (let k = -g; k <= g + 1e-6; k += 0.25) {
      const major = Math.abs(k) < 1e-6;
      c.strokeStyle = major ? C.gridStrong : C.grid;
      for (const [a, b] of [[[k, -g], [k, g]], [[-g, k], [g, k]]] as const) {
        c.beginPath();
        c.moveTo(...sx(proj([box.cx + a[0], box.cy + a[1], box.ground])));
        c.lineTo(...sx(proj([box.cx + b[0], box.cy + b[1], box.ground])));
        c.stroke();
      }
    }

    // Shadow: the same skeleton flattened onto the ground.
    c.lineCap = "round";
    c.strokeStyle = C.shadow;
    c.lineWidth = 5;
    for (const [a, b] of robot.edges) {
      if (!f[a] || !f[b]) continue;
      c.beginPath();
      c.moveTo(...sx(proj([f[a][0], f[a][1], box.ground])));
      c.lineTo(...sx(proj([f[b][0], f[b][1], box.ground])));
      c.stroke();
    }

    // Bones far to near, the far ones dimmer.
    const q = f.map(proj);
    const bones = robot.edges.filter(([a, b]) => q[a] && q[b]).map(([a, b]) => ({ a, b, d: (q[a].d + q[b].d) / 2 })).sort((x, y) => x.d - y.d);
    const dMin = Math.min(...q.map((p) => p.d));
    const dMax = Math.max(...q.map((p) => p.d));
    c.lineWidth = Math.max(2.5, s * 0.035);
    for (const { a, b, d } of bones) {
      const ca2 = side(robot.names[a] ?? "");
      c.strokeStyle = ca2 === side(robot.names[b] ?? "") ? ca2 : side(robot.names[b] ?? "");
      c.globalAlpha = 0.5 + 0.5 * (dMax > dMin ? (d - dMin) / (dMax - dMin) : 1);
      c.beginPath();
      c.moveTo(...sx(q[a]));
      c.lineTo(...sx(q[b]));
      c.stroke();
    }
    c.globalAlpha = 1;
    q.forEach((p, i) => {
      c.fillStyle = side(robot.names[i] ?? "");
      c.beginPath();
      c.arc(...sx(p), Math.max(2, s * 0.018), 0, Math.PI * 2);
      c.fill();
    });
    c.fillStyle = C.text;
    c.font = "11px ui-monospace, Menlo, monospace";
    c.fillText(`${secs(t)} · frame ${frameAt(t, data.fps, robot.frames.length) + 1}/${robot.frames.length}`, 10, 18);
  });

  const down = (e: PointerEvent<HTMLCanvasElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    view.current.drag = { x: e.clientX, az: view.current.az };
  };
  const move = (e: PointerEvent<HTMLCanvasElement>) => {
    const d = view.current.drag;
    if (d) view.current.az = d.az - ((e.clientX - d.x) / 160);
  };
  const up = () => { view.current.drag = null; };
  return (
    <canvas ref={ref} className="live-robot" onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up}
      role="img" aria-label="The Unitree G1 replaying the retargeted motion, in sync with the video. Drag to turn the view." />
  );
}

function feet(s: TaskStage) {
  if (s.kind === "flight") return "both feet off";
  if (!s.feet_down) return s.kind === "settle" ? "feet down" : "any";
  const [l, r] = s.feet_down;
  return l && r ? "both down" : l ? "left down, right up" : r ? "right down, left up" : "both off";
}

/** What each stage of the task asks of the robot, in order; a click seeks the video there. */
export function TaskBreakdown({ spec, t, onSeek }: { spec: TaskSpec; t: number; onSeek: (t: number) => void }) {
  let moment = 0;
  return (
    <div className="card card-pad live-tasks">
      <div className="eyebrow">Task breakdown · {spec.stages.length} stages</div>
      {spec.action && <p style={{ fontSize: 14 }}>{spec.action}</p>}
      {spec.summary && <p className="muted" style={{ fontSize: 13.5 }}>{spec.summary}</p>}
      {spec.named_by && <p className="faint" style={{ fontSize: 12 }}>Moments named by {spec.named_by} from the video; the reward never reads the names.</p>}
      <ol className="live-task-list">
        {spec.stages.map((s, i) => {
          const n = s.kind === "pose" ? ++moment : null;
          const start = s.kind === "flight" ? s.start_s : s.at_s;
          const now = s.kind === "flight" ? t >= s.start_s && t <= s.end_s : Math.abs(t - s.at_s) < 0.35;
          return (
            <li key={i} className={`live-task k-${s.kind} ${now ? "on" : ""} ${t > (s.kind === "flight" ? s.end_s : s.at_s) ? "past" : ""}`}>
              <button type="button" className="live-task-head" onClick={() => onSeek(start)} title={`Seek to ${secs(start)}`}>
                <span className="live-task-n mono">{n ?? (s.kind === "flight" ? "↑" : "·")}</span>
                <span className="live-task-name">{s.name}</span>
                <span className="mono faint live-task-time">{s.kind === "flight" ? `${secs(s.start_s)}–${secs(s.end_s)}` : secs(s.at_s)}</span>
              </button>
              {s.description && <p className="live-task-desc muted">{s.description}</p>}
              <div className="live-task-tags">
                <span className={`env-tag k-${s.kind}`}>{s.kind}</span>
                {s.kind !== "settle" && <span className={`env-tag ${s.optional ? "" : "req"}`}>{s.optional ? "optional" : "required"}</span>}
                <span className="env-tag">feet: {feet(s)}</span>
                {s.clearance_m != null && <span className="env-tag">clearance {Math.round(s.clearance_m * 100)} cm</span>}
                {s.pelvis_rise_m ? <span className="env-tag">pelvis +{Math.round(s.pelvis_rise_m * 100)} cm</span> : null}
                {s.hold_s ? <span className="env-tag">hold {s.hold_s.toFixed(2)} s</span> : null}
                {s.measured && <span className="env-tag" title="what keypose.py measured changing most">measured: {s.measured}</span>}
              </div>
              {s.body_parts.length > 0 && (
                <div className="live-parts">
                  {s.body_parts.map((p) => (
                    <div key={p.part} className="live-part" title={`relevance ${p.weight.toFixed(2)}`}>
                      <span>{p.part}</span>
                      <i style={{ width: `${Math.max(4, Math.min(1, p.weight) * 100)}%` }} />
                      <span className="mono faint">{p.weight.toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}

/** How the reward machine pays: its weights, then the server's notes (or a generic account without them). */
export function RewardBox({ spec }: { spec: TaskSpec }) {
  const w = spec.reward?.weights;
  const notes = spec.reward?.notes ?? [];
  const terms = w ? ([["progress", w.progress], ["moment", w.stage], ["jump", w.jump], ["finish", w.done], ["hold / s", w.hold]] as const) : [];
  return (
    <div className="card card-pad live-reward">
      <div className="eyebrow">How the reward works</div>
      {terms.length > 0 && (
        <div className="live-weights">
          {terms.filter(([, x]) => x).map(([k, x]) => <span key={k} className="env-part">{k} <span className="mono">×{x}</span></span>)}
        </div>
      )}
      {notes.length ? (
        <ul>{notes.map((n) => <li key={n}>{n}</li>)}</ul>
      ) : (
        <ul>
          <li><b>Progress:</b> paid as it closes in on the next moment's pose.</li>
          <li><b>Moment bonus:</b> for each key moment reached, in order{spec.counts.flights > 0 && "; a jump pays extra for its height"}.</li>
          <li><b>Finish bonus:</b> once the last moment is reached; then holding the end pose pays per second.</li>
          <li><b>A fall</b> gives back what that attempt earned.</li>
          <li><b>Timing is free:</b> only the order of the moments counts.</li>
        </ul>
      )}
    </div>
  );
}
