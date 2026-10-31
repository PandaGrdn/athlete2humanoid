import { useEffect, useState } from "react";
import { api, type Job, type RunDetail } from "../api";
import { emptyInput, loadRun, STAGES, type BuildInput, type StageKey, type StageStatus } from "./data";

// Where the build view's input and each stage's status come from: a stored run (its files, revealed on a timer
// or by a demo job's replay), or a live job (its events and live.json). The pages hold these; BuildView only draws.

export type Statuses = Record<StageKey, StageStatus>;
const all = (s: StageStatus) => Object.fromEntries(STAGES.map((x) => [x.key, s])) as Statuses;

/** A stored run and its build input, loaded piece by piece; `done` once every piece has settled. */
export function useRunInput(rid: string | null) {
  const [run, setRun] = useState<RunDetail | null>(null);
  const [input, setInput] = useState<BuildInput>(emptyInput);
  const [done, setDone] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    setRun(null);
    setInput(emptyInput());
    setDone(false);
    setErr(null);
    if (!rid) return;
    let live = true;
    api.run(rid).then((r) => {
      if (!live) return;
      setRun(r);
      return loadRun(r, (patch) => live && setInput((x) => ({ ...x, ...patch })));
    }).then(() => live && setDone(true)).catch((e) => live && setErr(e.message));
    return () => { live = false; };
  }, [rid]);
  return { run, input, done, err };
}

/** Whether a stage's piece of the input is here. */
export function has(input: BuildInput, k: StageKey) {
  return k === "video" ? !!input.video : k === "skeleton" ? !!input.skeleton : k === "retarget" ? !!input.qpos
    : k === "moments" ? !!input.spec : k === "variants" ? !!input.variants : true;
}

/** A stage reached by the replay (`reached` of them so far): ready if its piece is here, missing if it never came. */
function reachedStatus(input: BuildInput, loaded: boolean, k: StageKey, reached: number, i: number): StageStatus {
  if (i > reached) return "waiting";
  if (i === reached) return "working";
  return has(input, k) ? "ready" : loaded ? "missing" : "working";
}

/** A stored run replayed on a timer: one stage every `step` seconds from t0. */
export function timedStatus(input: BuildInput, loaded: boolean, elapsed: number, step: number): Statuses {
  const reached = Math.floor(elapsed / step);
  const out = all("waiting");
  STAGES.forEach((s, i) => { out[s.key] = reachedStatus(input, loaded, s.key, reached, i); });
  return out;
}

const jobStage = (job: Job | null, key: string) => job?.stages.find((s) => s.key === key)?.status ?? "pending";

/** A demo-runner job (server/stages.py STAGES) replaying a stored run: each build stage with its pipeline stage. */
export function demoJobStatus(job: Job | null, input: BuildInput, loaded: boolean): Statuses {
  const map: Record<StageKey, string[]> = {
    video: ["silhouette"], skeleton: ["pose"], retarget: ["retarget"], moments: ["environment"], variants: ["latent", "train"], policy: ["train", "eval"],
  };
  const out = all("waiting");
  const failed = job?.status === "failed" || job?.status === "cancelled";
  for (const s of STAGES) {
    const [first, last = first] = map[s.key];
    const a = jobStage(job, first);
    const b = jobStage(job, last);
    // variants arrive when training starts (they're its data); the policy once training is done
    const isDone = s.key === "variants" ? b !== "pending" : s.key === "policy" ? jobStage(job, "train") === "done" : b === "done";
    const started = a !== "pending";
    out[s.key] = isDone ? (has(input, s.key) ? "ready" : loaded ? "missing" : "working") : started ? (failed ? "missing" : "working") : failed ? "missing" : "waiting";
  }
  return out;
}

/** A live job (server/stages.py LIVE_STAGES): its stages, and live.json's pieces as they arrive. */
export function liveJobStatus(job: Job | null, input: BuildInput, liveVersionLoaded: (k: StageKey) => boolean): Statuses {
  const st = (k: string) => jobStage(job, k);
  const ended = job?.status === "succeeded" || job?.status === "failed" || job?.status === "cancelled";
  const piece = (k: StageKey, stage: string): StageStatus => {
    const s = st(stage);
    if (s === "done") return has(input, k) ? "ready" : liveVersionLoaded(k) ? "missing" : "working";
    if (s === "running") return "working";
    return s === "failed" || (ended && s === "pending") ? "missing" : "waiting";
  };
  const env = st("environment");
  const policy = st("policy");
  return {
    video: input.video && st("video") !== "done" ? "working" : piece("video", "video"),
    skeleton: piece("skeleton", "pose"),
    retarget: piece("retarget", "retarget"),
    moments: piece("moments", "environment"),
    // variants come after the environment, before the job ends (best effort: an older API has none)
    variants: input.variants ? "ready" : env === "done" ? (ended ? "missing" : "working") : env === "failed" || ended ? "missing" : "waiting",
    policy: policy === "done" ? "ready" : policy === "skipped" ? "offer" : policy === "running" ? "working" : ended ? "missing" : "waiting",
  };
}
