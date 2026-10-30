import { useEffect, useRef, type RefObject } from "react";
import { demo, type LiveData } from "../api";

/** The pixel hero's backdrop: the page as a grid of glyphs, ordered-dithered (4 x 4 Bayer). A recorded clip plays
 *  inside `stage`, the person from the video as round dots and the retargeted G1 as square pixels, side by side and
 *  frame-locked; the mouse leaves a trail that fades. One canvas, paused off screen, a still frame (and no trail)
 *  under reduced motion. `counter` gets the frame number as text. */

const CLIP = "squat-jump";
const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5].map((v) => (v + 0.5) / 16);
const INK = "#000000";
const DUST = "#c4c4c4";
const SIGNAL = "#0297de";
const LEVELS = 4;
const TRAIL_S = 1.1; // seconds for the trail to fade to ~3%
const AZ = 0.62; // the G1's camera: a three-quarter view, slightly above
const EL = 0.1;

type XY = [number, number];
type Box = { x0: number; y0: number; x1: number; y1: number };
type Figure = { frames: (XY[] | null)[]; box: Box; head: number[]; torso: number[]; edges: [number, number][]; depth: (number[] | null)[] };

const bound = (pts: Iterable<XY>): Box => {
  const b = { x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity };
  for (const [x, y] of pts) { b.x0 = Math.min(b.x0, x); b.y0 = Math.min(b.y0, y); b.x1 = Math.max(b.x1, x); b.y1 = Math.max(b.y1, y); }
  return b;
};

/** The person: the video's 2D keypoints in pixels (y down), a head point added just above the nose. */
function personFigure(d: LiveData): Figure | null {
  const lm = d.landmarks;
  if (!lm || !d.video) return null;
  const { width: W, height: H } = d.video;
  const frames = lm.frames.map((f) => {
    if (!f || f.some((p, i) => !p && [0, 1, 2, 7, 8].includes(i))) return null;
    const pts = f.map((p) => (p ? [p[0] * W, p[1] * H] as XY : [NaN, NaN] as XY));
    const sw = Math.hypot(pts[1][0] - pts[2][0], pts[1][1] - pts[2][1]);
    pts.push([pts[0][0], pts[0][1] - sw * 0.12]); // head centre, a little above the nose
    return pts;
  });
  const all = frames.flatMap((f) => (f ? f.filter((p) => Number.isFinite(p[0])) : []));
  if (!all.length) return null;
  return { frames, box: bound(all), head: [lm.names.length], torso: [1, 2, 8, 7], edges: lm.edges, depth: frames.map(() => null) };
}

/** The G1: bodies in metres, z up, projected orthographically from a fixed three-quarter view (y down on screen). */
function robotFigure(d: LiveData): Figure | null {
  const r = d.robot;
  if (!r?.frames.length) return null;
  const [ca, sa, ce, se] = [Math.cos(AZ), Math.sin(AZ), Math.cos(EL), Math.sin(EL)];
  const depth: number[][] = [];
  const frames = r.frames.map((f) => {
    const neck = [0, 1, 2].map((k) => (f[8][k] + f[11][k]) / 2);
    const bodies = [...f, [neck[0], neck[1], neck[2] + 0.17] as [number, number, number]];
    depth.push(bodies.map(([x, y, z]) => x * ca * ce + y * sa * ce + z * se));
    return bodies.map(([x, y, z]) => [-x * sa + y * ca, -(z * ce - (x * ca + y * sa) * se)] as XY);
  });
  return { frames, box: bound(frames.flat()), head: [r.names.length], torso: [8, 11, 4, 1], edges: r.edges, depth };
}

export function GlyphField({ stage, counter }: { stage: RefObject<HTMLElement | null>; counter?: RefObject<HTMLElement | null> }) {
  const host = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const root = host.current;
    const cv = canvas.current;
    const ctx = cv?.getContext("2d");
    if (!root || !cv || !ctx) return;
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const fig = document.createElement("canvas");
    const fctx = fig.getContext("2d", { willReadFrequently: true });
    if (!fctx) return;

    let data: LiveData | null = null;
    let person: Figure | null = null;
    let robot: Figure | null = null;
    let stillFrame = 0;
    let cell = 10, cols = 0, rows = 0, w = 0, h = 0, dpr = 1;
    let trail = new Float32Array(0);
    let spare = new Float32Array(0);
    let slot: Box | null = null; // the stage, in cells
    let visible = true;
    let raf = 0;
    let t0 = performance.now();
    let last = t0;
    let trailAlive = false;
    let p: XY | null = null; // pointer, in cells
    let q: XY | null = null;

    const measure = () => {
      const r = root.getBoundingClientRect();
      w = r.width;
      h = r.height;
      dpr = Math.min(2, window.devicePixelRatio || 1);
      cell = w < 640 ? 8 : 10;
      cols = Math.ceil(w / cell);
      rows = Math.ceil(h / cell);
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
      fig.width = cols;
      fig.height = rows;
      trail = new Float32Array(cols * rows);
      spare = new Float32Array(cols * rows);
      const s = stage.current?.getBoundingClientRect();
      slot = s && s.width > 0 && s.height > 0
        ? { x0: (s.left - r.left) / cell, y0: (s.top - r.top) / cell, x1: (s.right - r.left) / cell, y1: (s.bottom - r.top) / cell }
        : null;
    };

    /** One figure into the low-res canvas: torso filled, limbs as thick round strokes, nearer bones brighter. */
    const drawFigure = (f: Figure, i: number, into: Box, channel: "255,0,0" | "0,255,0") => {
      const pts = f.frames[i];
      if (!pts) return;
      const bw = f.box.x1 - f.box.x0;
      const bh = f.box.y1 - f.box.y0;
      const k = Math.min(((into.x1 - into.x0) * 0.86) / bw, ((into.y1 - into.y0) * 0.94) / bh);
      const ox = (into.x0 + into.x1) / 2 - ((f.box.x0 + f.box.x1) / 2) * k;
      const oy = into.y1 - f.box.y1 * k;
      const at = (j: number): XY => [ox + pts[j][0] * k, oy + pts[j][1] * k];
      const size = bh * k; // figure height, in cells
      const dep = f.depth[i];
      const [dMin, dMax] = dep ? [Math.min(...dep), Math.max(...dep)] : [0, 1];
      const shade = (j: number) => (dep && dMax > dMin ? 0.45 + 0.55 * ((dep[j] - dMin) / (dMax - dMin)) : 1);

      fctx.fillStyle = `rgba(${channel},0.55)`;
      fctx.beginPath();
      f.torso.forEach((j, n) => (n ? fctx.lineTo(...at(j)) : fctx.moveTo(...at(j))));
      fctx.fill();
      fctx.lineCap = "round";
      fctx.lineJoin = "round";
      fctx.lineWidth = Math.max(1.2, size * 0.05);
      for (const [a, b] of f.edges) {
        if (!Number.isFinite(pts[a][0]) || !Number.isFinite(pts[b][0])) continue;
        fctx.strokeStyle = `rgba(${channel},${Math.min(shade(a), shade(b))})`;
        fctx.beginPath();
        fctx.moveTo(...at(a));
        fctx.lineTo(...at(b));
        fctx.stroke();
      }
      const [hx, hy] = at(f.head[0]);
      fctx.fillStyle = `rgba(${channel},1)`;
      fctx.beginPath();
      fctx.arc(hx, hy, Math.max(1, size * 0.065), 0, Math.PI * 2);
      fctx.fill();
    };

    const stamp = (x: number, y: number) => {
      const r = 4.2;
      for (let yy = Math.max(0, Math.floor(y - r)); yy <= Math.min(rows - 1, Math.ceil(y + r)); yy++) {
        for (let xx = Math.max(0, Math.floor(x - r)); xx <= Math.min(cols - 1, Math.ceil(x + r)); xx++) {
          const d = ((xx - x) ** 2 + (yy - y) ** 2) / (r * r);
          if (d >= 1) continue;
          const i = yy * cols + xx;
          trail[i] = Math.max(trail[i], 0.9 * (1 - d) ** 2);
        }
      }
    };

    const render = (time: number, frame: number) => {
      // the figures, at this frame
      let px: Uint8ClampedArray | null = null;
      if (slot && data) {
        fctx.clearRect(0, 0, cols, rows);
        const mid = (slot.x0 + slot.x1) / 2;
        if (person) drawFigure(person, Math.min(frame, person.frames.length - 1), { ...slot, x1: mid }, "255,0,0");
        if (robot) drawFigure(robot, Math.min(frame, robot.frames.length - 1), { ...slot, x0: mid }, "0,255,0");
        px = fctx.getImageData(0, 0, cols, rows).data;
      }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const dust = new Path2D();
      const dots = new Path2D();
      const squares = new Path2D();
      const marks = new Path2D();
      const u = cell / 10;
      const arm = (n: number) => Math.round(n * u / 2) * 2;
      for (let y = 0; y < rows; y++) {
        const row = (y & 3) * 4;
        const row2 = ((y + 2) & 3) * 4;
        for (let x = 0; x < cols; x++) {
          const i = y * cols + x;
          const th = BAYER[row + (x & 3)];
          const X = x * cell;
          const Y = y * cell;
          const tl = Math.min(LEVELS, Math.floor(trail[i] * LEVELS + BAYER[row2 + ((x + 2) & 3)] - 0.2));
          if (tl > 0) {
            const len = arm(2 + tl * 2);
            const bar = Math.max(2, arm(2));
            marks.rect(X + (cell - len) / 2, Y + (cell - bar) / 2, len, bar);
            marks.rect(X + (cell - bar) / 2, Y + (cell - len) / 2, bar, len);
            continue;
          }
          if (px) {
            const a = px[i * 4 + 3] / 255;
            const person = (a * px[i * 4]) / 255;
            const robotV = (a * px[i * 4 + 1]) / 255;
            if (person > 0.02 || robotV > 0.02) {
              const isRobot = robotV >= person;
              const lv = Math.min(LEVELS, Math.floor((isRobot ? robotV : person) * LEVELS + th));
              if (lv > 0) {
                if (isRobot) {
                  const s = arm(lv * 2);
                  squares.rect(X + (cell - s) / 2, Y + (cell - s) / 2, s, s);
                } else {
                  const r = (0.1 + lv * 0.08) * cell;
                  dots.moveTo(X + cell / 2 + r, Y + cell / 2);
                  dots.arc(X + cell / 2, Y + cell / 2, r, 0, Math.PI * 2);
                }
                continue;
              }
            }
          }
          // the dust: a slow interference pattern, sparse on the left (behind the copy), denser to the right
          const ramp = 0.35 + 0.65 * Math.min(1, x / Math.max(1, cols * 0.8));
          const v = (0.07 + 0.09 * (0.5 + 0.5 * Math.sin(x * 0.09 + time * 0.25) * Math.cos(y * 0.12 - time * 0.18))) * ramp;
          if (v * 2.4 + th > 1) dust.rect(X + cell / 2 - u, Y + cell / 2 - u, 2 * u, 2 * u);
        }
      }
      ctx.fillStyle = DUST;
      ctx.fill(dust);
      ctx.fillStyle = INK;
      ctx.fill(dots);
      ctx.fill(squares);
      ctx.fillStyle = SIGNAL;
      ctx.fill(marks);
      if (counter?.current && data) counter.current.textContent = `${String(frame + 1).padStart(3, "0")}/${data.frames}`;
    };

    const tick = (now: number) => {
      raf = 0;
      if (!visible) return;
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      const time = (now - t0) / 1000;
      if (q) {
        if (p) {
          const steps = Math.max(1, Math.ceil(Math.hypot(q[0] - p[0], q[1] - p[1]) / 1.2));
          for (let s = 1; s <= steps; s++) stamp(p[0] + ((q[0] - p[0]) * s) / steps, p[1] + ((q[1] - p[1]) * s) / steps);
        } else stamp(q[0], q[1]);
        if (!p || p[0] !== q[0] || p[1] !== q[1]) trailAlive = true;
        p = q;
      }
      if (trailAlive) {
        // fade, and bleed a little into the four neighbours
        const decay = Math.pow(0.03, dt / TRAIL_S);
        let max = 0;
        for (let y = 0; y < rows; y++) {
          for (let x = 0; x < cols; x++) {
            const i = y * cols + x;
            const c = trail[i];
            const n = (trail[x > 0 ? i - 1 : i] + trail[x < cols - 1 ? i + 1 : i] + trail[y > 0 ? i - cols : i] + trail[y < rows - 1 ? i + cols : i]) / 4;
            const v = (c + (n - c) * 0.18) * decay;
            spare[i] = v;
            if (v > max) max = v;
          }
        }
        [trail, spare] = [spare, trail];
        if (max < 0.01) { trail.fill(0); trailAlive = false; }
      }
      const frame = data ? Math.floor(time * data.fps) % data.frames : 0;
      render(time, frame);
      raf = requestAnimationFrame(tick);
    };

    const start = () => { if (!raf && visible && !still) { last = performance.now(); raf = requestAnimationFrame(tick); } };
    const paintStill = () => { if (still) render(0, stillFrame); };

    measure();
    paintStill();
    const ro = new ResizeObserver(() => { measure(); paintStill(); });
    ro.observe(root);
    if (stage.current) ro.observe(stage.current);
    const io = new IntersectionObserver(([e]) => {
      visible = e.isIntersecting && document.visibilityState === "visible";
      if (visible) start();
    });
    io.observe(root);
    const onVis = () => { visible = document.visibilityState === "visible"; if (visible) start(); };
    document.addEventListener("visibilitychange", onVis);
    const onMove = (e: PointerEvent) => {
      if (e.pointerType !== "mouse" || still) return;
      const r = root.getBoundingClientRect();
      const inside = e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom;
      q = inside ? [(e.clientX - r.left) / cell, (e.clientY - r.top) / cell] : null;
      if (!q) p = null;
    };
    const onLeave = () => { p = q = null; };
    window.addEventListener("pointermove", onMove, { passive: true });
    window.addEventListener("blur", onLeave);

    let cancelled = false;
    demo.live(CLIP).then((d) => {
      if (cancelled) return;
      data = d;
      person = personFigure(d);
      robot = robotFigure(d);
      // the still: the top of the jump
      let best = -Infinity;
      d.robot?.frames.forEach((f, i) => { if (f[0][2] > best) { best = f[0][2]; stillFrame = i; } });
      paintStill();
      start();
    }).catch(() => start()); // no recording: the dust and the trail still work
    start();

    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
      document.removeEventListener("visibilitychange", onVis);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("blur", onLeave);
    };
  }, [stage, counter]);

  return (
    <div ref={host} className="px-field" aria-hidden="true">
      <canvas ref={canvas} />
    </div>
  );
}
