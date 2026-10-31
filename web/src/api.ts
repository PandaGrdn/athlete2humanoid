declare const __API_BASE__: string;

export const API_BASE: string = __API_BASE__;

export type FileRef = { label: string; path: string; url: string; size: number; kind: string | null };

export type Metrics = {
  aligned: number | null;
  dtw: number | null;
  chamfer: number | null;
  dtw_rank_vs_bank: number | null;
  reward_net_aligned: number | null;
  real_aligned_in_reward_space: number | null;
  steps: number | null;
  fps: number | null;
  falls: number | null;
  mean_joint_error_rad: number | null;
  mean_torso_position_error_m: number | null;
  max_torso_position_error_m: number | null;
};

export type Policy = {
  key: string;
  base: string;
  label: string;
  description: string;
  previous: boolean;
  metrics: Metrics;
  video: FileRef | null;
  checkpoint: FileRef | null;
  onnx: FileRef | null;
  curve: [number, number][];
  iterations: number | null;
};

export type Clip = {
  name: string;
  is_target: boolean;
  fps: number | null;
  frames: number | null;
  view_azimuth: number | null;
  sam: boolean | null;
  retarget_error_cm: Record<string, number>;
  retarget_error_mean_cm: number | null;
  side_by_side: FileRef | null;
  silhouette: FileRef | null;
  motion_csv: FileRef | null;
  frontend?: string | null; // the pose model that ran: "sam3d" (SAM 3D Body) or MediaPipe otherwise
  skeleton?: FileRef | null; // human.npy: the 3D pose the retarget read, (T, 33, 3)
};

export type RobustCell = {
  fall_rate: number | null;
  survival: number | null;
  torso_err_m: number | null;
  joint_err_rad: number | null;
  vjepa_aligned: number | null;
  vjepa_dtw: number | null;
};

export type Robust = {
  conditions: { suite: string; level: string; params: Record<string, unknown>; policies: Record<string, RobustCell> }[];
  summary: Record<string, Record<string, Partial<RobustCell> & { vjepa_retained?: number }>>;
  reps: number | null;
  step_hz: number | null;
};

export type RunSummary = {
  id: string;
  title: string;
  description: string;
  target: string | null;
  tag: string | null;
  iterations: number | null;
  started: string | null;
  finished: string | null;
  state: string;
  hero: FileRef | null;
  video: FileRef | null;
  poster: FileRef | null;
  policies_trained: string[];
  headline: { retarget_aligned: number | null; best_policy: string | null; best_aligned: number | null; falls: number | null };
  has_robust: boolean;
  has_task_spec?: boolean;
  clip_count: number;
};

/** The RL environment compiled from the retarget (GET /api/runs/{id}/task_spec, /api/jobs/{id}/task_spec). */
export type TaskStage = {
  name: string;
  kind: "pose" | "flight" | "settle";
  start_s: number;
  end_s: number;
  at_s: number;
  optional: boolean;
  level: number | null;
  frame: number | null;
  feet_down: [boolean, boolean] | null; // [left, right]
  body_parts: { part: string; weight: number }[];
  clearance_m: number | null;
  // Muse's one sentence on the moment, and keypose.py's measured name it replaced (named specs only)
  description?: string | null;
  measured?: string | null;
  hold_s?: number | null; // how long a key pose must be held
  pelvis_rise_m?: number | null; // how far a jump must lift the pelvis above standing
};

export type TaskSpec = {
  format: "keypose" | "phases";
  name: string | null;
  summary: string | null;
  action?: string | null; // Muse's one-sentence description of the whole action
  named_by?: string | null; // the model that named the moments
  duration_s: number;
  stages: TaskStage[];
  counts: { moments: number; required: number; optional: number; flights: number };
  body_parts: string[];
  self_check: {
    stages_completed: number;
    stages_total: number;
    success: boolean;
    stages_skipped: number;
    fell_at_s: number | null;
    source: "spec" | "eval";
  } | null;
  warnings: string[];
  // The reward machine's weights (live jobs; newer specs) and plain-text notes on how it pays.
  reward?: { weights: { progress: number; stage: number; jump: number; done: number; hold: number }; notes: string[] };
};

export type RunDetail = RunSummary & {
  reference: Record<string, number | null>;
  policies: Policy[];
  bank_top5: [string, number][];
  distill: Record<string, unknown> | null;
  robust: Robust | null;
  clips: Clip[];
  downloads: FileRef[];
  variants?: FileRef | null; // variants.json (scripts/build_variants_view.py)
};

export type StageState = {
  key: string;
  label: string;
  description: string;
  status: "pending" | "running" | "done" | "failed" | "skipped";
  progress: number;
  started_at: number | null;
  finished_at: number | null;
};

export type Job = {
  id: string;
  runner: string;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  stage: string | null;
  progress: number;
  stages: StageState[];
  logs: [number, string, string][];
  curves: Record<string, [number, number][]>;
  input: { filename: string; size: number; url: string } | null;
  source_run: string | null;
  result_run: string | null;
  error: string | null;
  created_at: number;
  updated_at: number;
  demo: boolean;
};

export type Health = {
  ok: boolean;
  runners: string[];
  default_runner: string;
  demo_run: string | null;
  demo_seconds: number;
  run_count: number;
  limits: { max_upload_mb: number; extensions: string[] };
  live?: { max_seconds: number; concurrency: number } | null; // set when runners include "live"
  stages: { key: string; label: string; description: string }[];
};

/** A live job's data (GET /api/jobs/{id}/live), growing as its stages finish. Times are seconds of the
 *  processed video (GET /api/jobs/{id}/live/video), frames at `fps`. */
export type LiveData = {
  fps: number;
  frames: number;
  duration_s: number;
  video: { width: number; height: number } | null; // null only before the video stage is done
  // 2D, 0..1 of the processed frame (x right, y down); null where not detected.
  landmarks: { names: string[]; edges: [number, number][]; frames: (([number, number] | null)[] | null)[] } | null;
  pose_found: number | null;
  // which 3D pose model ran (server/live_worker.py): "MediaPipe" on the CPU, "SAM 3D Body" in recorded builds
  pose_model?: string;
  // The 3D skeleton the retarget read (the landmarks' joints): metres, z up, the athlete facing +x.
  skeleton?: { names: string[]; edges: [number, number][]; frames: ([number, number, number] | null)[][] } | null;
  // G1 bodies in metres, world frame, z up.
  robot: { names: string[]; edges: [number, number][]; frames: [number, number, number][][] } | null;
  qpos?: number[][] | null; // the retarget: G1 qpos (root xyz, quaternion wxyz, 29 joints) per frame
  variants?: VariantsView | null; // skill/variations.py's preview, after the environment
  retarget_error_cm: Record<string, [number, number]> | null;
  timings_s: Record<string, number>;
};

/** A clip's variants as the build view shows them (skill/variations.py preview): every variant's verdict, and
 *  a few kept ones' qpos (at their own fps) to draw as ghosts. */
export type VariantsView = {
  total: number;
  kept: number;
  rows: { name: string; params: { mirror: boolean; tempo: number; amplitude: number; warp: number }; frames: number; ok: boolean; rules: string[]; why: string }[];
  ghosts: { name: string; label: string; fps: number; qpos: number[][] }[];
  source?: string; // stored runs: the variants run it comes from, or "generated"
  trained_on?: boolean;
  policies?: string[];
  clip?: string;
};

/** A path the API serves (/api/...) on the API's origin; the site's own static files (/examples, /live-demo, ...) stay here. */
export const url = (path: string) => (path.startsWith("/api/") ? API_BASE + path : path);

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url(path), init);
  // A static host without the API answers /api/* with the SPA's index.html: treat that as "no API".
  if (r.ok && !(r.headers.get("content-type") ?? "").includes("json")) throw new Error("API not reachable");
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const body = await r.json();
      if (body?.detail) msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not JSON */
    }
    throw new Error(msg);
  }
  return r.json() as Promise<T>;
}

export const api = {
  // A sleeping Space can take a while to answer; give up after 10 s and retry later.
  health: () => request<Health>("/api/health", typeof AbortSignal.timeout === "function" ? { signal: AbortSignal.timeout(10_000) } : {}),
  runs: () => request<RunSummary[]>("/api/runs"),
  run: (id: string) => request<RunDetail>(`/api/runs/${encodeURIComponent(id)}`),
  runTaskSpec: (id: string) => request<TaskSpec>(`/api/runs/${encodeURIComponent(id)}/task_spec`),
  job: (id: string) => request<Job>(`/api/jobs/${encodeURIComponent(id)}`),
  live: (id: string) => request<LiveData>(`/api/jobs/${encodeURIComponent(id)}/live`),
  liveVideo: (id: string) => url(`/api/jobs/${encodeURIComponent(id)}/live/video`),
  jobTaskSpec: (id: string) => request<TaskSpec>(`/api/jobs/${encodeURIComponent(id)}/task_spec`),
  cancel: (id: string) => request<Job>(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" }),
  createJob: (opts: { file?: File; runId?: string; runner?: string }) => {
    const fd = new FormData();
    if (opts.file) fd.append("file", opts.file);
    if (opts.runId) fd.append("run_id", opts.runId);
    fd.append("runner", opts.runner ?? "demo");
    return request<Job>("/api/jobs", { method: "POST", body: fd });
  },
};

/** Client-side upload check, for a quick message only: the server stays the authority. */
export function checkVideo(f: File, health: Health | null): string | null {
  const exts = health?.limits.extensions ?? [".mp4", ".mov", ".m4v", ".webm"];
  const maxMb = health?.limits.max_upload_mb ?? 100;
  const ext = f.name.includes(".") ? f.name.slice(f.name.lastIndexOf(".")).toLowerCase() : "";
  // Some browsers leave .mov/.m4v untyped, so an empty type passes on the extension alone.
  if (f.type && !f.type.startsWith("video/")) return `That isn't a video (${f.type}). Use ${exts.join(", ")}.`;
  if (!exts.includes(ext)) return `Unsupported file type ${ext || "(no extension)"}. Use ${exts.join(", ")}.`;
  if (f.size > maxMb * 1024 * 1024) return `That file is ${fmt.bytes(f.size)}; the limit is ${maxMb} MB.`;
  return null;
}

/** Live job updates over SSE, falling back to polling if the stream fails. Returns a stop function. */
export function watchJob(id: string, onJob: (j: Job) => void): () => void {
  let stopped = false;
  let timer: number | undefined;
  const poll = async () => {
    if (stopped) return;
    try {
      const j = await api.job(id);
      onJob(j);
      if (["succeeded", "failed", "cancelled"].includes(j.status)) return;
    } catch {
      /* transient; keep polling */
    }
    timer = window.setTimeout(poll, 1000);
  };
  let es: EventSource | null = null;
  if (typeof EventSource !== "undefined") {
    es = new EventSource(url(`/api/jobs/${encodeURIComponent(id)}/events`));
    es.onmessage = (e) => {
      const j = JSON.parse(e.data) as Job;
      onJob(j);
      if (["succeeded", "failed", "cancelled"].includes(j.status)) es?.close();
    };
    es.onerror = () => {
      if (es && es.readyState === EventSource.CLOSED) return;
      es?.close();
      es = null;
      poll();
    };
  } else {
    poll();
  }
  return () => {
    stopped = true;
    es?.close();
    window.clearTimeout(timer);
  };
}

export const fmt = {
  num: (v: number | null | undefined, d = 3) => (v == null ? "—" : v.toFixed(d)),
  pct: (v: number | null | undefined, d = 0) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`),
  cm: (m: number | null | undefined) => (m == null ? "—" : `${(m * 100).toFixed(1)} cm`),
  bytes: (b: number) => (b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1e3))} kB`),
  date: (s: string | null) => {
    if (!s) return "—";
    const d = new Date(s);
    return isNaN(d.getTime()) ? s : d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  },
  duration: (a: string | null, b: string | null) => {
    if (!a || !b) return null;
    const s = (new Date(b).getTime() - new Date(a).getTime()) / 1000;
    if (!isFinite(s) || s <= 0) return null;
    const h = Math.floor(s / 3600);
    const m = Math.round((s % 3600) / 60);
    return h ? `${h} h ${m} min` : `${m} min`;
  },
};

// Categorical identity per policy: validated on the dark surface (blue / orange / aqua).
export const POLICY_COLOR: Record<string, string> = { tracking: "#3987e5", hybrid: "#d95926", latent: "#199e70", task: "#8e5cd9", taskhybrid: "#b08a1e", taskphase: "#c2569b" };
export const policyColor = (base: string) => POLICY_COLOR[base] ?? "#8a96a3";

/** Recorded live builds (scripts/build_live_demo.py), served with the site: /live replays them when no
 *  pipeline API is reachable. Each is a real run of the live worker, its stage messages at the times
 *  they arrived, plus the policies trained on that clip and their training curves. */
export type ReplayEvent = { t: number; stage: string; progress?: number; done?: boolean; log?: string };
export type ReplayPolicy = {
  key: string;
  label: string;
  video: string;
  poster: string;
  run: string;
  curve: { iterations: number; points: [number, number, number][] } | null; // [iteration, mean reward, mean episode length]
  checker: { stages_completed: number; stages_total: number; success: boolean; fell_at_s: number | null };
};
export type Replay = {
  id: string;
  title: string;
  clip: string;
  recorded: string;
  stages: { key: string; label: string }[];
  events: ReplayEvent[];
  policies: ReplayPolicy[];
  named_by: string | null;
};
export type ReplayIndex = { clips: { id: string; title: string; summary: string | null; duration_s: number; poster: string; policies: string[] }[] };

const DEMO = "/live-demo";
async function staticJson<T>(path: string): Promise<T> {
  const r = await fetch(`${DEMO}/${path}`);
  if (!r.ok || !(r.headers.get("content-type") ?? "").includes("json")) throw new Error(`${path}: not found`);
  return r.json() as Promise<T>;
}
export const demo = {
  index: () => staticJson<ReplayIndex>("index.json"),
  replay: (id: string) => staticJson<Replay>(`${encodeURIComponent(id)}/replay.json`),
  live: (id: string) => staticJson<LiveData>(`${encodeURIComponent(id)}/live.json`),
  spec: (id: string) => staticJson<TaskSpec>(`${encodeURIComponent(id)}/task_spec.json`),
  file: (id: string, name: string) => `${DEMO}/${encodeURIComponent(id)}/${name}`,
};
