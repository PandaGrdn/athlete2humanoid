// The 3D showcase's static data, built by scripts/build_showcase.py into /showcase/<id>/: the G1 (meshes +
// kinematic tree), the trained policy's recorded run, the retarget it tracks, and its recorded pushes.

export type Body = { name: string; parent: number; free: boolean; pos: V3; quat: Q; joint: number; jpos: V3; axis: V3 };
export type Part = { body: number; tone: "light" | "dark"; v: [number, number]; i: [number, number] };
export type RobotJson = { bodies: Body[]; parts: Part[]; bin: string; positions: [number, number]; indices: [number, number] };
export type Robot = RobotJson & { pos: Float32Array; idx: Uint16Array };

/** One recording: clip frames start .. start + frames - 1 (its env's own run, a few frames before the push on). */
export type PushRow = { angle: number; mps: number; start: number; frames: number; fall: number | null; rejoin: number | null; offset: number };
export type Moment = { step: number; file: string; pushes: PushRow[] };
export type Showcase = {
  id: string; example: string; policy: string; fps: number; frames: number;
  nominal: string; reference: string; push_stride: number;
  video: { src: string; poster: string; size: [number, number]; crop: [number, number]; offset_s: number; label?: string };
  stages?: { kind: string; label: string; start_s: number; end_s: number }[] | null;
  clean: { falls: number; joint_err_rad: number; torso_err_m: number };
  pushes: {
    kind: string; strengths_mps: number[]; directions: number; fall_rate: Record<string, number>; rejoined: number;
    rejoin: { m: number; rad: number }; after_fall_s: number; checkpoint: string; moments: Moment[];
    pre_frames: number; drift: { median: number; max: number };
  };
};

export type V3 = [number, number, number];
export type Q = [number, number, number, number];

/** qpos frames: 36 numbers each (root position, root quaternion wxyz, 29 joint angles), decoded from int16. */
export type Frames = { n: number; data: Float32Array };
export const NQ = 36;
const SCALE = [...Array(3).fill(1e-3), ...Array(4).fill(1 / 32767), ...Array(29).fill(1e-4)];

function decode(buf: ArrayBuffer, byteOffset = 0, frames?: number): Frames {
  const n = frames ?? (buf.byteLength - byteOffset) / (NQ * 2);
  const raw = new Int16Array(buf, byteOffset, n * NQ);
  const data = new Float32Array(n * NQ);
  for (let i = 0; i < raw.length; i++) data[i] = raw[i] * SCALE[i % NQ];
  return { n, data };
}

export const BASE = import.meta.env.BASE_URL;
const dir = (id: string) => `${BASE}showcase/${encodeURIComponent(id)}/`;

async function get(url: string, kind: "json" | "bin") {
  const r = await fetch(url);
  // A missing file comes back as the SPA's index.html on most static hosts.
  const type = r.headers.get("content-type") ?? "";
  if (!r.ok || type.includes("text/html")) throw new Error(`${url} is missing: run scripts/build_showcase.py`);
  return kind === "json" ? r.json() : r.arrayBuffer();
}

/** The showcased clips (public/showcase/index.json): the switcher's tabs and each one's words. */
export type ClipEntry = {
  id: string; tab: string; title: [string, string]; kicker: string; lede: string; video: string; policy: string;
  facts: { v: string; l: string }[]; research?: string | null; fallback?: string; caveat?: string;
};
export const loadIndex = () => get(`${BASE}showcase/index.json`, "json") as Promise<{ clips: ClipEntry[] }>;

/** The G1's meshes and kinematic tree, as a showcase bundle ships them (the same robot in every bundle). */
export async function loadRobot(id: string): Promise<Robot> {
  const d = dir(id);
  const robotJson = await (get(`${d}g1.json`, "json") as Promise<RobotJson>);
  const bin = await (get(d + robotJson.bin, "bin") as Promise<ArrayBuffer>);
  return {
    ...robotJson,
    pos: new Float32Array(bin, robotJson.positions[0] * 4, robotJson.positions[1]),
    idx: new Uint16Array(bin, robotJson.indices[0], robotJson.indices[1]),
  };
}

export async function loadShowcase(id: string) {
  const d = dir(id);
  const [show, robot] = await Promise.all([get(`${d}showcase.json`, "json") as Promise<Showcase>, loadRobot(id)]);
  const [nominal, reference] = await Promise.all([
    get(d + show.nominal, "bin") as Promise<ArrayBuffer>,
    get(d + show.reference, "bin") as Promise<ArrayBuffer>,
  ]);
  return { show, robot, nominal: decode(nominal), reference: decode(reference), pushes: new PushStore(d, show) };
}
export type Loaded = Awaited<ReturnType<typeof loadShowcase>>;

/** The recorded pushes, one file per push moment, fetched when first needed. */
export class PushStore {
  private cache = new Map<number, Promise<ArrayBuffer>>();
  constructor(private dir: string, private show: Showcase) {}

  /** The first recorded moment at or after `frame` (frames ahead counted modulo the clip), or null. */
  next(frame: number): { moment: Moment; wait: number } | null {
    const ms = this.show.pushes.moments;
    if (!ms.length) return null;
    const m = ms.find((x) => x.step >= frame);
    return m ? { moment: m, wait: m.step - frame } : { moment: ms[0], wait: ms[0].step + this.show.frames - frame };
  }

  fetch(m: Moment) {
    if (!this.cache.has(m.step)) {
      const p = get(this.dir + m.file, "bin") as Promise<ArrayBuffer>;
      p.catch(() => this.cache.delete(m.step));
      this.cache.set(m.step, p);
    }
    return this.cache.get(m.step)!;
  }

  /** The recording at moment m nearest to `angle` (rad, world frame) at strength `mps`. */
  pick(m: Moment, angle: number, mps: number) {
    let best = m.pushes[0];
    let bestD = Infinity;
    for (const r of m.pushes) {
      if (Math.abs(r.mps - mps) > 1e-3) continue;
      const d = Math.abs(Math.atan2(Math.sin(r.angle - angle), Math.cos(r.angle - angle)));
      if (d < bestD) { bestD = d; best = r; }
    }
    return best;
  }

  /** A recording at the clip's frame rate (stored every push_stride-th frame; interpolated back here). */
  async frames(m: Moment, r: PushRow): Promise<Frames> {
    const buf = await this.fetch(m);
    const k = this.show.push_stride || 1;
    const raw = decode(buf, r.offset * NQ * 2, r.frames);
    if (k === 1) return raw;
    const n = (raw.n - 1) * k + 1;
    const out = { n, data: new Float32Array(n * NQ) };
    const q = newQpos();
    for (let i = 0; i < n; i++) out.data.set(sample(raw, i / k, q), i * NQ);
    return out;
  }

  /** Warm the cache for the next moments from `frame` (pointer over the stage: a click may follow). */
  prefetch(frame: number, count = 2) {
    const ms = this.show.pushes.moments;
    const i = ms.findIndex((x) => x.step >= frame);
    for (let j = 0; j < count && ms.length; j++) this.fetch(ms[((i < 0 ? 0 : i) + j) % ms.length]).catch(() => undefined);
  }
}

/** qpos at fractional frame f of `fr` (clamped), linearly interpolated (quaternion: normalised lerp). */
export function sample(fr: Frames, f: number, out: Float32Array) {
  const n = fr.n;
  const x = Math.max(0, Math.min(n - 1, f));
  const i0 = Math.floor(x);
  const i1 = Math.min(n - 1, i0 + 1);
  const w = x - i0;
  const a = i0 * NQ;
  const b = i1 * NQ;
  const d = fr.data;
  let dot = 0;
  for (let k = 3; k < 7; k++) dot += d[a + k] * d[b + k];
  const s = dot < 0 ? -1 : 1;
  for (let k = 0; k < NQ; k++) out[k] = d[a + k] * (1 - w) + (k >= 3 && k < 7 ? s : 1) * d[b + k] * w;
  const qn = Math.hypot(out[3], out[4], out[5], out[6]) || 1;
  for (let k = 3; k < 7; k++) out[k] /= qn;
  return out;
}

export const newQpos = () => new Float32Array(NQ);

/** Blend b into a by w (0: a, 1: b), in place in a. */
export function blend(a: Float32Array, b: Float32Array, w: number) {
  let dot = 0;
  for (let k = 3; k < 7; k++) dot += a[k] * b[k];
  const s = dot < 0 ? -1 : 1;
  for (let k = 0; k < NQ; k++) a[k] = a[k] * (1 - w) + (k >= 3 && k < 7 ? s : 1) * b[k] * w;
  const qn = Math.hypot(a[3], a[4], a[5], a[6]) || 1;
  for (let k = 3; k < 7; k++) a[k] /= qn;
  return a;
}
