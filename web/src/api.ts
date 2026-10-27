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
  clip_count: number;
};

export type RunDetail = RunSummary & {
  reference: Record<string, number | null>;
  policies: Policy[];
  bank_top5: [string, number][];
  distill: Record<string, unknown> | null;
  robust: Robust | null;
  clips: Clip[];
  downloads: FileRef[];
};

export type StageState = {
  key: string;
  label: string;
  description: string;
  status: "pending" | "running" | "done" | "failed";
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
  stages: { key: string; label: string; description: string }[];
};

export const url = (path: string) => (path.startsWith("http") ? path : API_BASE + path);

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
  job: (id: string) => request<Job>(`/api/jobs/${encodeURIComponent(id)}`),
  cancel: (id: string) => request<Job>(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" }),
  createJob: (opts: { file?: File; runId?: string; runner?: string }) => {
    const fd = new FormData();
    if (opts.file) fd.append("file", opts.file);
    if (opts.runId) fd.append("run_id", opts.runId);
    fd.append("runner", opts.runner ?? "demo");
    return request<Job>("/api/jobs", { method: "POST", body: fd });
  },
};

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
export const POLICY_COLOR: Record<string, string> = { tracking: "#3987e5", hybrid: "#d95926", latent: "#199e70" };
export const policyColor = (base: string) => POLICY_COLOR[base] ?? "#8a96a3";
