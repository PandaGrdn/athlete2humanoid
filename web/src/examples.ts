// Curated demo examples: static JSON + videos under /examples, built by scripts/build_examples.py.
// They're served by the frontend itself, never the API, so the deployed site is a full demo offline.

export type Panel = { label: string; from: number; to: number; tone?: "good" | "bad" | "warn" };

export type ExVideo = {
  src: string;
  poster: string;
  label: string;
  width: number;
  height: number;
  duration_s: number;
  panels?: Panel[];
  color?: string;
  policy?: string;
};

export type Stage = { kind: string; label: string; start_s: number; end_s: number };
export type Metric = { label: string; display: string; unit?: string; note?: string; tone?: "good" | "bad" | null };
export type Cell = { value: number | string | number[]; display: string; good?: boolean };
export type PolicyRow = { key: string; label: string; color: string; reward: string; values: Record<string, Cell> };
export type Column = { key: string; label: string; note: string };
export type CompareRow = {
  label: string; unit: string; before: number; after: number; ratio: number | null;
  clip_before: number; clip_after: number;
};
export type ImportanceChart = {
  fps: number;
  weight: number[];
  arms: number[];
  legs: number[];
  arm_motion: number[];
  floor: number;
  hype: [number, number] | null;
  events: { t: number; label: string; peak: number }[];
};
export type RobustTable = {
  policies: { key: string; label: string; color: string }[];
  conditions: { suite: string; suite_label: string; level: string; cells: Record<string, { fall_rate: number | null; vjepa: number | null }> }[];
  reps: number;
};

export type Example = {
  id: string;
  kicker: string;
  title: string;
  tagline: string;
  status: "ready" | "pending";
  story: { shown: string; learned: string; how: string };
  duration_s: number | null;
  stages: Stage[] | null;
  primary: ExVideo | null;
  grid: ExVideo[];
  grid_caption?: string;
  extra: ExVideo[];
  metrics: Metric[];
  policies?: PolicyRow[];
  columns?: Column[];
  compare?: CompareRow[];
  compare_labels?: { before: string; after: string };
  chart?: ImportanceChart;
  robust?: RobustTable;
  sources: string[];
  notes: string[];
};

export type ExampleSummary = Pick<Example, "id" | "kicker" | "title" | "tagline" | "status"> & {
  poster: string | null;
  video: string | null;
};

const BASE = `${import.meta.env.BASE_URL}examples/`;
export const exUrl = (path: string) => BASE + path;

async function getJson<T>(path: string): Promise<T> {
  const r = await fetch(exUrl(path));
  // A missing file comes back as the SPA's index.html (200, text/html) on most static hosts.
  if (!r.ok || !(r.headers.get("content-type") ?? "").includes("json")) {
    throw new Error(`examples/${path} is missing: run scripts/build_examples.py`);
  }
  return r.json() as Promise<T>;
}

let indexCache: Promise<ExampleSummary[]> | null = null;
const exampleCache = new Map<string, Promise<Example>>();

export const examples = {
  index: () => {
    indexCache ??= getJson<{ examples: ExampleSummary[] }>("index.json").then((d) => d.examples);
    indexCache.catch(() => (indexCache = null));
    return indexCache;
  },
  get: (id: string) => {
    if (!exampleCache.has(id)) {
      const p = getJson<Example>(`${encodeURIComponent(id)}/example.json`);
      p.catch(() => exampleCache.delete(id));
      exampleCache.set(id, p);
    }
    return exampleCache.get(id)!;
  },
};
