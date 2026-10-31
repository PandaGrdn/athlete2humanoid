import { api, url, type LiveData, type RunDetail, type TaskSpec, type VariantsView } from "../api";
import { NQ, type ClipEntry, type Frames } from "../showcase/data";

// What the build view draws, from whichever source has it: a live job (GET /api/jobs/{id}/live), or a stored
// run's files (GET /api/runs/{id}/files/...). Every piece is optional: the view shows each stage as its piece arrives.

export type StageKey = "video" | "skeleton" | "retarget" | "moments" | "variants" | "policy";
export type StageStatus = "waiting" | "working" | "ready" | "offer" | "missing";
export const STAGES: { key: StageKey; label: string; short: string }[] = [
  { key: "video", label: "Video and mask", short: "Video" },
  { key: "skeleton", label: "3D skeleton", short: "Skeleton" },
  { key: "retarget", label: "Retarget onto the G1", short: "Retarget" },
  { key: "moments", label: "Key moments", short: "Moments" },
  { key: "variants", label: "Variants", short: "Variants" },
  { key: "policy", label: "Policy", short: "Policy" },
];

/** A 3D skeleton: n frames of j joints, metres, MuJoCo's frame (z up, the athlete facing +x); NaN where missing. */
export type Skeleton = { names: string[]; edges: [number, number][]; n: number; j: number; data: Float32Array };
export type Ghost = { name: string; label: string; fps: number; frames: Frames };
export type Variants = Omit<VariantsView, "ghosts"> & { ghosts: Ghost[] };

export type BuildInput = {
  fps: number; // of the skeleton, the retarget and the video's frames
  video: { src: string; sideBySide: boolean; label: string } | null; // sideBySide: the left part of render.side_by_side
  mask: { src: string } | null; // SAM 3.1's silhouette video (a crop that follows the athlete)
  landmarks: LiveData["landmarks"]; // 2D, over the video
  segmenter: string | null; // "SAM 3.1", or null: the whole frame (a CPU build)
  poseModel: string | null; // "SAM 3D Body" or "MediaPipe"
  skeleton: Skeleton | null;
  qpos: Frames | null;
  retargetCm: number | null; // mean end-effector error
  spec: TaskSpec | null;
  variants: Variants | null;
  clip: string | null; // the clip's name (front_kick), to find its policy
};
export const emptyInput = (): BuildInput => ({
  fps: 30, video: null, mask: null, landmarks: null, segmenter: null, poseModel: null, skeleton: null, qpos: null,
  retargetCm: null, spec: null, variants: null, clip: null,
});

// MediaPipe's 33 landmarks -> the body's 17 (server/live_worker.py LANDMARKS), and its bones
const KEEP = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
const NAMES = ["nose", "left shoulder", "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist", "left hip",
  "right hip", "left knee", "right knee", "left ankle", "right ankle", "left heel", "right heel", "left toe", "right toe"];
const EDGES: [number, number][] = ([[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24], [23, 25],
  [25, 27], [27, 29], [29, 31], [27, 31], [24, 26], [26, 28], [28, 30], [30, 32], [28, 32]] as [number, number][])
  .map(([a, b]) => [KEEP.indexOf(a), KEEP.indexOf(b)]);

const qposFrames = (rows: number[][]): Frames => {
  const data = new Float32Array(rows.length * NQ);
  rows.forEach((r, i) => data.set(r.slice(0, NQ), i * NQ));
  return { n: rows.length, data };
};

function variantsFrom(v: VariantsView | null | undefined): Variants | null {
  if (!v || !Array.isArray(v.rows)) return null;
  return { ...v, ghosts: (v.ghosts ?? []).filter((g) => g.qpos?.length).map((g) => ({ name: g.name, label: g.label, fps: g.fps, frames: qposFrames(g.qpos) })) };
}

const meanCm = (e: Record<string, [number, number] | number> | null | undefined) => {
  const v = Object.values(e ?? {}).map((x) => (Array.isArray(x) ? x[0] : x));
  return v.length ? v.reduce((a, x) => a + x, 0) / v.length : null;
};

/** A live job's data, as far as it has got. */
export function fromLive(d: LiveData, video: string | null): Partial<BuildInput> {
  let skeleton: Skeleton | null = null;
  const s = d.skeleton;
  if (s?.frames?.length) {
    const j = s.names.length;
    const data = new Float32Array(s.frames.length * j * 3).fill(NaN);
    s.frames.forEach((f, i) => f?.forEach((p, k) => { if (p) data.set(p, (i * j + k) * 3); }));
    skeleton = { names: s.names, edges: s.edges, n: s.frames.length, j, data };
  }
  return {
    fps: d.fps,
    video: video ? { src: video, sideBySide: false, label: "Processed clip" } : null,
    landmarks: d.landmarks,
    segmenter: null,
    poseModel: d.pose_model ?? (d.landmarks ? "MediaPipe" : null),
    skeleton,
    qpos: d.qpos?.length ? qposFrames(d.qpos) : null,
    retargetCm: meanCm(d.retarget_error_cm),
    variants: variantsFrom(d.variants),
  };
}

/** A .npy file (little-endian float32/float64, C order) -> its shape and values. */
export function parseNpy(buf: ArrayBuffer): { shape: number[]; data: Float32Array } {
  const u8 = new Uint8Array(buf);
  if (u8[0] !== 0x93 || String.fromCharCode(...u8.subarray(1, 6)) !== "NUMPY") throw new Error("not a .npy file");
  const v2 = u8[6] >= 2;
  const dv = new DataView(buf);
  const hlen = v2 ? dv.getUint32(8, true) : dv.getUint16(8, true);
  const start = (v2 ? 12 : 10) + hlen;
  const header = new TextDecoder().decode(u8.subarray(v2 ? 12 : 10, start));
  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  const shape = (/'shape':\s*\(([^)]*)\)/.exec(header)?.[1] ?? "").split(",").map((x) => x.trim()).filter(Boolean).map(Number);
  if (/'fortran_order':\s*True/.test(header)) throw new Error("Fortran-order .npy");
  const body = buf.slice(start);
  if (descr === "<f8") return { shape, data: Float32Array.from(new Float64Array(body)) };
  if (descr === "<f4") return { shape, data: new Float32Array(body) };
  throw new Error(`unsupported .npy dtype ${descr}`);
}

/** human.npy (T, 33, 3) -> the 17 body joints the live view draws. */
export function skeletonFromNpy(buf: ArrayBuffer): Skeleton {
  const { shape, data } = parseNpy(buf);
  const [n, all] = shape;
  if (shape.length !== 3 || shape[2] !== 3 || all < 33) throw new Error(`unexpected skeleton shape ${shape}`);
  const j = KEEP.length;
  const out = new Float32Array(n * j * 3);
  for (let i = 0; i < n; i++) KEEP.forEach((src, k) => out.set(data.subarray((i * all + src) * 3, (i * all + src) * 3 + 3), (i * j + k) * 3));
  return { names: NAMES, edges: EDGES, n, j, data: out };
}

/** motion.csv (mjlab: root xyz, quaternion xyzw, 29 joints) -> qpos frames (quaternion wxyz). */
export function framesFromCsv(text: string): Frames {
  const rows = text.trim().split(/\r?\n/).map((l) => l.split(",").map(Number)).filter((r) => r.length >= NQ && r.every(Number.isFinite));
  const data = new Float32Array(rows.length * NQ);
  rows.forEach((r, i) => {
    const o = i * NQ;
    data.set(r.slice(0, 3), o);
    data[o + 3] = r[6];
    data.set(r.slice(3, 6), o + 4);
    data.set(r.slice(7, NQ), o + 7);
  });
  return { n: rows.length, data };
}

async function fetchOk(path: string) {
  const r = await fetch(url(path));
  const type = r.headers.get("content-type") ?? "";
  if (!r.ok || type.includes("text/html")) throw new Error(`${path}: ${r.status}`);
  return r;
}

/** A stored run's clip: its target, else the one its task spec is named after (an older API has no target for
 *  runs started without --target), else the first. */
export function runClip(run: RunDetail, specName?: string | null) {
  const target = run.target ?? specName;
  return run.clips.find((c) => c.name === target) ?? run.clips.find((c) => c.is_target) ?? run.clips[0] ?? null;
}

/** A stored run's build, piece by piece: `set` gets each piece as it loads (failures leave it out). */
export async function loadRun(run: RunDetail, set: (patch: Partial<BuildInput>) => void) {
  const spec = run.has_task_spec ? await api.runTaskSpec(run.id).catch(() => null) : null;
  const c = runClip(run, spec?.name);
  set({
    fps: c?.fps ?? 30, clip: c?.name ?? run.target, spec,
    video: c?.side_by_side ? { src: url(c.side_by_side.url), sideBySide: true, label: "Phone video" } : null,
    mask: c?.silhouette ? { src: url(c.silhouette.url) } : null,
    segmenter: c?.sam ? "SAM 3.1" : null,
    poseModel: c?.frontend === "sam3d" ? "SAM 3D Body" : c?.frontend ? "MediaPipe" : null,
    retargetCm: c?.retarget_error_mean_cm ?? null,
  });
  const jobs: Promise<unknown>[] = [];
  if (c?.skeleton) jobs.push(fetchOk(c.skeleton.url).then((r) => r.arrayBuffer()).then((b) => set({ skeleton: skeletonFromNpy(b) })));
  if (c?.motion_csv) jobs.push(fetchOk(c.motion_csv.url).then((r) => r.text()).then((t) => set({ qpos: framesFromCsv(t) })));
  if (run.variants) jobs.push(fetchOk(run.variants.url).then((r) => r.json()).then((v) => set({ variants: variantsFrom(v) })));
  return Promise.allSettled(jobs);
}

// --- the policy for stage 6 --------------------------------------------------------------------------
// The showcase bundles (public/showcase/<id>/) are the trained policies with recorded pushes. A clip that has one
// plays its own; any other gets the nearest by what its key moments ask for, labelled as pre-trained.
const OWN: Record<string, string> = { front_kick: "front-kick", squat_jump: "squat-jump", squat: "squat" };

export type PolicyPick = { id: string; own: boolean; tab: string; why: string };

export function pickPolicy(clips: ClipEntry[], clip: string | null, spec: TaskSpec | null): PolicyPick | null {
  if (!clips.length) return null;
  const has = (id: string) => clips.find((c) => c.id === id);
  const own = clip ? has(OWN[clip] ?? "") : undefined;
  if (own) return { id: own.id, own: true, tab: own.tab, why: "trained on this clip" };
  const poses = spec?.stages.filter((s) => s.kind === "pose") ?? [];
  const oneFoot = poses.some((s) => s.feet_down && s.feet_down[0] !== s.feet_down[1]);
  const [want, why] = spec?.counts.flights ? ["squat-jump", "it jumps"] : oneFoot ? ["front-kick", "it lifts one foot"]
    : ["squat", "both feet stay down"];
  const c = has(want) ?? clips[0];
  return { id: c.id, own: false, tab: c.tab, why: c.id === want ? `the nearest stored policy: ${why}` : "a stored policy" };
}
