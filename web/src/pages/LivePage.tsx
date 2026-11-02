import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from "react";
import { Link, useNavigate, useParams } from "react-router";
import { api, checkVideo, demo, fmt, url, watchJob, type AttemptsView, type Health, type Job, type LiveData, type Replay, type ReplayIndex, type RunSummary, type TaskSpec } from "../api";
import { useHealth } from "../App";
import { emptyInput, fromLive, loadRun, type BuildInput, type StageKey } from "../buildview/data";
import { liveJobStatus } from "../buildview/hooks";
import { RewardBox, TaskBreakdown } from "../components/live";
import { EnvDownload, Icon, StatusPill } from "../components/ui";
import { Typed } from "../components/pixel";

// three.js loads with the build view, after the page
const BuildView = lazy(() => import("../buildview/BuildView"));
const BuildFallback = <div className="skeleton" style={{ height: "min(74svh, 720px)", minHeight: 500, margin: "8px 0 28px" }} />;

const STEPS = [
  { key: "video", label: "Video check", model: "ffmpeg" },
  { key: "pose", label: "Pose", model: "MediaPipe" },
  { key: "retarget", label: "Retarget", model: "mink IK" },
  { key: "environment", label: "Task environment", model: "key moments" },
  { key: "policy", label: "Policy", model: "PPO · GPU" },
];

// The visitor's own file, kept across /live -> /live/:id so the local preview and the GPU re-upload
// survive the navigation (not a reload: then the job's processed video stands in).
const localFiles = new Map<string, File>();

/** Upload a clip as a live build and open its page, where the clip plays as each stage arrives.
 *  Used by /live and by the home page's drop zone, so a dropped video always lands on its live view. */
export async function startLiveBuild(f: File, nav: (to: string) => void) {
  const j = await api.createJob({ file: f, runner: "live" });
  localFiles.set(j.id, f);
  nav(`/live/${j.id}`);
  return j;
}

const done = (job: Job | null, key: string) => job?.stages.find((s) => s.key === key)?.status === "done";
const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

/** Fetches `load` once `ready`, again whenever `version` changes, retrying every 3 s while the job runs.
 *  Returns the data and whether it is for the current version. */
function useWhenReady<T>(ready: boolean, version: string, running: boolean, load: () => Promise<T>): [T | null, boolean] {
  const [data, setData] = useState<T | null>(null);
  const [got, setGot] = useState<string | null>(null);
  const fn = useRef(load);
  fn.current = load;
  useEffect(() => {
    if (!ready || got === version) return;
    let stopped = false;
    let timer: number | undefined;
    const go = () => fn.current().then((d) => { if (!stopped) { setData(d); setGot(version); } }).catch(() => {
      if (!stopped && running) timer = window.setTimeout(go, 3000);
    });
    go();
    return () => { stopped = true; window.clearTimeout(timer); };
  }, [ready, version, running, got]);
  return [data, got === version];
}

/** The video's clock for React, ~10 times a second (canvases read currentTime themselves every frame). */
function useVideoTime(video: HTMLVideoElement | null) {
  const [t, setT] = useState(0);
  useEffect(() => {
    if (!video) return;
    let raf = 0;
    let last = -1;
    const loop = () => {
      if (Math.abs(video.currentTime - last) > 0.08) setT((last = video.currentTime));
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [video]);
  return t;
}

export default function LivePage() {
  const { id } = useParams();
  const nav = useNavigate();
  const [file, setFile] = useState<File | null>(id ? localFiles.get(id) ?? null : null);
  const [job, setJob] = useState<Job | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [watch, setWatch] = useState(0); // bumped when a finished job starts again (training): subscribe anew

  // Back on /live (the nav link): start over.
  useEffect(() => {
    if (id) return;
    setJob(null);
    setFile(null);
    setErr(null);
  }, [id]);
  useEffect(() => {
    if (!id) return;
    setFile((f) => f ?? localFiles.get(id) ?? null);
    api.job(id).then(setJob).catch((e) => setErr(e.message));
    return watchJob(id, setJob);
  }, [id, watch]);

  const start = async (f: File) => {
    setErr(null);
    setFile(f);
    setBusy(true);
    try {
      setJob(await startLiveBuild(f, nav));
    } catch (e) {
      setErr((e as Error).message);
      setFile(null);
    } finally {
      setBusy(false);
    }
  };

  if (!id && !file) return <LiveStart onFile={start} err={err} />;
  if (id && err && !job) return <div className="container section"><div className="error-box">Job {id}: {err}</div></div>;
  return <LiveView job={job} file={file} uploading={busy} err={err} onTrained={(j) => { setJob(j); setWatch((w) => w + 1); }} />;
}

function LiveStart({ onFile, err }: { onFile: (f: File) => void; err: string | null }) {
  const { health, error: apiError } = useHealth();
  const input = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [bad, setBad] = useState<string | null>(null);
  const exts = health?.limits.extensions ?? [".mp4", ".mov", ".m4v", ".webm"];
  const maxMb = health?.limits.max_upload_mb ?? 100;
  const available = !!health?.runners.includes("live");
  const pick = (f: File | undefined) => {
    setBad(null);
    if (!f) return;
    const why = checkVideo(f, health);
    if (why) return setBad(why);
    onFile(f);
  };

  return (
    <div className="container">
      <div className="ex-intro">
        <div className="eyebrow">Live demo</div>
        <h1><Typed text="Watch a video become an RL environment" when="load" /></h1>
        <p className="lede">Upload a clip. The pose, the G1 and its task appear stage by stage, in about a minute.</p>
        <div className="live-chips">
          {STEPS.map((s, i) => <span key={s.key} className="pill"><span className="mono faint">{i + 1}</span>{s.label}</span>)}
        </div>
      </div>

      {!health && !apiError && <div className="skeleton" style={{ height: 220 }} />}
      {!health && apiError && <OfflineDrop />}
      {health && !available && (
        <div className="notice">
          <Icon name="cube" />
          <div><strong>This deployment can't build live.</strong> The <Link to="/research" style={{ color: "var(--accent)" }}>research examples</Link> show recorded runs.</div>
        </div>
      )}
      {health && available && (
        <div className="card card-pad" style={{ maxWidth: 780 }}>
          <div className={`dropzone ${drag ? "drag" : ""}`} role="button" tabIndex={0}
            onClick={() => input.current?.click()}
            onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0]); }}>
            <div className="icon"><Icon name="upload" size={22} /></div>
            <h3>Drop a video, or click to choose</h3>
            <p>{exts.join(" ")} · up to {maxMb} MB · one person, full body in frame{health.live?.max_seconds ? ` · the first ${Math.round(health.live.max_seconds)} s are used` : ", 3–15 s"}</p>
            <input ref={input} type="file" accept={"video/*," + exts.join(",")} hidden onChange={(e) => pick(e.target.files?.[0])} />
          </div>
          {(bad || err) && <div className="error-box" style={{ marginTop: 14 }}>{bad ?? err}</div>}
          <p className="faint" style={{ fontSize: 12.5, marginTop: 12 }}>Uploads go only to this site's API and are kept briefly.</p>
        </div>
      )}
      <RecordedBuilds live={!!health && available} />
    </div>
  );
}

/** No API: a dropped video stays in the browser; the page says what would happen and offers the recordings. */
function OfflineDrop() {
  const [name, setName] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  return (
    <div className="card card-pad" style={{ maxWidth: 780 }}>
      <div className={`dropzone ${drag ? "drag" : ""}`} role="button" tabIndex={0}
        onClick={() => input.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); setName(e.dataTransfer.files[0]?.name ?? null); }}>
        <div className="icon"><Icon name="upload" size={22} /></div>
        <h3>Drop a video, or click to choose</h3>
        <p>The pipeline API is offline on this static site, so nothing is uploaded.</p>
        <input ref={input} type="file" accept="video/*" hidden onChange={(e) => setName(e.target.files?.[0]?.name ?? null)} />
      </div>
      <div className="notice" style={{ marginTop: 14 }}>
        <Icon name="cube" />
        <div>
          {name ? <><strong>{name} stays on your computer.</strong> </> : <><strong>The pipeline API isn't running right now.</strong> </>}
          Pick a recording below to watch a build end to end.
        </div>
      </div>
    </div>
  );
}

/** The recorded builds (scripts/build_live_demo.py), each a real run of this page's pipeline. */
function RecordedBuilds({ live }: { live: boolean }) {
  const [index, setIndex] = useState<ReplayIndex | null>(null);
  useEffect(() => { demo.index().then(setIndex).catch(() => setIndex({ clips: [] })); }, []);
  if (index && !index.clips.length) return null;
  return (
    <div className="section" style={{ paddingTop: 28 }}>
      <div className="eyebrow" style={{ marginBottom: 14 }}>{live ? "Or replay a recorded build" : "Recorded builds"}</div>
      <div className="replay-grid">
        {(index?.clips ?? Array.from({ length: 3 }, () => null)).map((c, i) => c ? (
          <Link key={c.id} to={`/live/demo/${c.id}`} className="card replay-card">
            <div className="replay-thumb"><img src={demo.file(c.id, c.poster)} alt="" loading="lazy" /><span className="pill">{c.duration_s.toFixed(1)} s</span></div>
            <div className="card-pad">
              <h3>{c.title}</h3>
              {c.summary && <p className="faint" style={{ fontSize: 12.5 }}>{c.summary.split(":")[0]}</p>}
              <p className="muted" style={{ fontSize: 13 }}>Trained: {c.policies.join(", ") || "none yet"}</p>
            </div>
          </Link>
        ) : <div key={i} className="skeleton" style={{ height: 220 }} />)}
      </div>
    </div>
  );
}

/** Where LiveView reads a build's data: the API (a live job), or a recording (Replay). A recording's live.json has
 *  no 3D skeleton, qpos or variants: `extra` fetches them from the stored run of the same clip, when the API is up. */
type Source = {
  live: () => Promise<LiveData>; spec: () => Promise<TaskSpec>; video: string; policy: ReactNode; title: string; clip: string;
  extra?: (set: (p: Partial<BuildInput>) => void) => Promise<string | null>; // resolves to the run it used
};

function LiveView({ job, file, uploading, err, source, onTrained }: { job: Job | null; file: File | null; uploading: boolean; err: string | null; source?: Source; onTrained?: (j: Job) => void }) {
  const [video, setVideo] = useState<HTMLVideoElement | null>(null);
  const [blob, setBlob] = useState<string | null>(null);
  const [extra, setExtra] = useState<Partial<BuildInput>>({});
  const [extraRun, setExtraRun] = useState<string | null | undefined>(undefined); // undefined: still loading
  const t = useVideoTime(video);
  const { health } = useHealth();  // which pose model this host runs, for the stage badge before the pose is in
  const running = job?.status === "running" || job?.status === "queued";
  const id = job?.id ?? "";

  useEffect(() => {
    if (!file) return setBlob(null);
    const u = URL.createObjectURL(file);
    setBlob(u);
    return () => URL.revokeObjectURL(u);
  }, [file]);
  const loadExtra = source?.extra; // stable per recording (the page re-creates `source` on every tick)
  useEffect(() => {
    setExtra({});
    setExtraRun(loadExtra ? undefined : null);
    if (!loadExtra) return;
    let live = true;
    loadExtra((p) => live && setExtra((x) => ({ ...x, ...p }))).then((r) => live && setExtraRun(r)).catch(() => live && setExtraRun(null));
    return () => { live = false; };
  }, [loadExtra]);

  const st = (k: string) => job?.stages.find((s) => s.key === k)?.status ?? "pending";
  // live.json grows: refetched as the retarget and the environment finish, and once more at the end (the variants)
  const version = `${st("retarget")}/${st("environment")}/${job?.status}`;
  const [live, fresh] = useWhenReady<LiveData>(done(job, "pose"), version, running, () => source ? source.live() : api.live(id));
  const [spec] = useWhenReady<TaskSpec>(done(job, "environment"), "1", running, () => source ? source.spec() : api.jobTaskSpec(id));
  const attempts = useAttempts(source ? null : id, job?.training ?? null);

  // The processed copy once it exists (trimmed and rescaled, so the landmarks line up); until then the local file.
  const processed = done(job, "video") ? (source ? source.video : api.liveVideo(id)) : null;
  const src = processed ?? blob ?? (source ? source.video : job?.input ? url(job.input.url) : null);
  const aligned = !!processed && src === processed;

  // live.json is refetched as it grows: keep each piece's previous object while its content is the same, so the
  // scene doesn't replay its arrival (a skeleton growing in, the G1 snapping on) on every refetch
  const prev = useRef<Partial<BuildInput>>({});
  const input = useMemo<BuildInput>(() => {
    const base = { ...emptyInput(), ...(live ? fromLive(live, null, source ? null : api.liveEnv(id), source ? null : api.liveMask(id)) : {}) };
    for (const k of ["skeleton", "qpos", "variants"] as const) {
      const was = prev.current[k];
      if (was && base[k] && signature(was) === signature(base[k])) (base as Record<string, unknown>)[k] = was;
      else (prev.current as Record<string, unknown>)[k] = base[k];
    }
    const own = (k: keyof BuildInput) => (base[k] == null ? extra[k] : base[k]);
    return {
      ...base,
      video: src ? {
        src, sideBySide: false, label: source?.title ? "Recorded clip" : aligned ? "Processed clip" : file ? `Your upload · ${file.name}` : "Your upload",
        // the processed clip won't play: the visitor's own file, or the job's copy of the upload
        fallback: aligned && !source ? blob ?? (job?.input ? url(job.input.url) : null) : null,
      } : null,
      landmarks: aligned ? base.landmarks : null,
      skeleton: own("skeleton") as BuildInput["skeleton"],
      qpos: own("qpos") as BuildInput["qpos"],
      variants: own("variants") as BuildInput["variants"],
      spec,
      clip: source?.clip ?? null,
      attempts,
      training: job?.training ?? null,
    };
  }, [live, extra, src, aligned, source?.title, source?.clip, file, spec, blob, job?.input, id, !!source, attempts, job?.training]);
  const status = liveJobStatus(job, input, (k: StageKey) => fresh && (!source?.extra || extraRun !== undefined || k === "video"));
  const fromRun = extraRun ? <>3D from the stored run <Link to={`/build/${extraRun}`}>{extraRun}</Link> of the same clip (same SAM 3D Body front end); this recording kept the 2D landmarks only.</> : null;

  const seek = useCallback((s: number) => {
    if (video) video.currentTime = Math.max(0, Math.min(s, (video.duration || s + 1) - 0.02));
  }, [video]);

  const elapsed = job ? (running ? Date.now() / 1000 : job.updated_at) - job.created_at : 0;
  const stageLabel = job?.stages.find((s) => s.key === job.stage)?.label;
  const tr = job?.training ?? null;
  const title = !job ? "Uploading…" : job.status === "failed" ? "Build failed" : job.status === "cancelled" ? "Build cancelled"
    : tr?.state === "done" ? "Policy trained" : tr?.state === "failed" ? "Training failed" : tr?.state === "cancelled" ? "Training cancelled"
    : tr ? "Watch the robot learn" : spec ? "Environment ready" : job.status === "queued" ? "Waiting for the builder…" : stageLabel ? `${stageLabel}…` : "Starting…";
  // a training that failed or was cancelled can be started again: the offer comes back
  const canTrainAgain = !!tr && (tr.state === "failed" || tr.state === "cancelled");
  const offer = !source && job && (st("policy") === "skipped" || (canTrainAgain && st("policy") !== "running")) && health?.train && onTrained
    ? <TrainOffer job={job} train={health.train} onStarted={onTrained} again={canTrainAgain} /> : null;

  return (
    <div className="container">
      <div className="job-head">
        <div style={{ flex: 1, minWidth: 240 }}>
          <div className="eyebrow">{source ? source.title : <>Live build{job && <> · job {job.id}</>}</>}</div>
          <h1 style={{ marginTop: 4 }}><Typed text={title} when="load" /></h1>
        </div>
        {job && <StatusPill status={job.status} />}
        {job && <span className="mono muted" style={{ fontSize: 14 }}>{clock(Math.max(0, elapsed))}</span>}
        {job && running && !source && <button className="btn sm" onClick={() => api.cancel(job.id).catch(() => undefined)}>Cancel</button>}
        <Link className="btn sm ghost" to="/live"><Icon name="upload" size={14} />New video</Link>
      </div>
      <div className="progress" style={{ marginBottom: 20 }}><div style={{ width: `${(job?.progress ?? 0) * 100}%` }} /></div>
      {job?.status === "queued" && job.logs[0] && (
        <div className="notice" style={{ marginBottom: 20 }}><span className="dot live" style={{ marginTop: 6 }} /><div>{job.logs[0][2]}</div></div>
      )}
      {(job?.error || (err && !uploading)) && <div className="error-box" style={{ marginBottom: 20 }}>{job?.error ?? err}</div>}

      <Suspense fallback={BuildFallback}>
        <BuildView input={input} status={status} follow onVideo={setVideo} offer={offer}
          kicker={source ? <>Recorded build · {source.clip.replace(/_/g, " ")}</> : <>Live build{file ? <> · {file.name}</> : null}</>}
          notes={{
            video: !source && !aligned && src ? "Your local copy; the landmarks appear on the processed clip." : null,
            skeleton: fromRun, retarget: fromRun, variants: fromRun ?? (input.variants && !source ? "Generated on this host from your retarget, the same way the GPU runs make their training set." : null),
            policy: source && !["front_kick", "squat_jump", "squat"].includes(source.clip) ? "This clip's own trained policies are below, as videos; the site has 3D recordings for three clips." : null,
          }} />
      </Suspense>

      <div className="live-grid">
        <div className="live-left">
          <div className="card stepper live-steps">
            {(job?.stages.length ? job.stages : STEPS.map((s) => ({ ...s, status: "pending" as const, progress: 0, description: "", started_at: null, finished_at: null }))).map((s, i) => {
              const took = live?.timings_s[s.key] ?? (s.started_at && s.finished_at ? s.finished_at - s.started_at : null);
              const model = (s.key === "pose" && (live?.pose_model || (health?.live?.pose === "modal" ? "SAM 3D Body · GPU" : null)))
                || STEPS.find((x) => x.key === s.key)?.model;
              return (
                <div className={`stage ${s.status}`} key={s.key}>
                  <div className="marker">{s.status === "done" ? <Icon name="check" size={14} /> : s.status === "skipped" ? "–" : i + 1}</div>
                  <div>
                    <div className="head">
                      <h4>{s.label}</h4>
                      <span className="mono faint" style={{ fontSize: 12 }}>
                        {s.status === "running" ? `${Math.round(s.progress * 100)}%` : s.status === "skipped" ? "skipped" : s.status === "done" && took != null ? `${took.toFixed(1)} s` : model}
                      </span>
                    </div>
                    {s.description && <div className="desc">{s.description}</div>}
                    {s.status === "running" && <div className="progress"><div style={{ width: `${s.progress * 100}%` }} /></div>}
                  </div>
                </div>
              );
            })}
          </div>
          {live && (
            <dl className="kv card card-pad live-stats">
              <dt>Clip</dt><dd>{live.duration_s.toFixed(1)} s · {live.frames} frames @ {Math.round(live.fps)} fps</dd>
              {live.video && <><dt>Processed size</dt><dd>{live.video.width}×{live.video.height}</dd></>}
              <dt>Pose found</dt><dd>{fmt.pct(live.pose_found)}{live.pose_model ? ` · ${live.pose_model}` : ""}</dd>
              {live.retarget_error_cm && <><dt>Retarget error (mean / max)</dt><dd>{retargetErr(live.retarget_error_cm)}</dd></>}
              {input.variants && <><dt>Variants kept</dt><dd>{input.variants.kept} of {input.variants.total}</dd></>}
            </dl>
          )}
        </div>

        <div className="live-right">
          {spec ? (
            <div className="fade-in live-env">
              <TaskBreakdown spec={spec} t={t} onSeek={seek} />
              <RewardBox spec={spec} />
              {input.env && (
                <div className="card card-pad">
                  <EnvDownload env={input.env} />
                  <p className="faint" style={{ fontSize: 12.5, marginTop: 10 }}>
                    Your upload as a training environment: the G1 reference, its key moments, the variants made here and a loader.
                    Train it anywhere mjlab runs; the file is kept with the upload for a short while.
                  </p>
                </div>
              )}
            </div>
          ) : (
            <div className="card card-pad faint" style={{ fontSize: 13.5 }}>
              {st("environment") === "running" ? "Finding the key moments and compiling the reward…" : "The task environment (key moments, contacts, reward) appears after the retarget."}
            </div>
          )}

          {job && (source ? source.policy : <PolicyCard job={job} file={file} attempts={attempts} />)}

          {job && job.logs.length > 0 && (
            <details className="live-log">
              <summary>Log · {job.logs.length} line{job.logs.length === 1 ? "" : "s"}</summary>
              <div className="console">
                {job.logs.map(([ts, stage, line], i) => (
                  <div className="line" key={i}>
                    <span className="t">{clock(Math.max(0, ts - job.created_at))}</span>
                    <span className="s">{stage}</span>
                    <span className="m">{line}</span>
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
      </div>
    </div>
  );
}

/** A live job's attempts (GET /api/jobs/{id}/attempts): fetched when its training starts, every 8 s while the GPU
 *  trains, and once more when it ends. */
function useAttempts(id: string | null, training: Job["training"] | null): AttemptsView | null {
  const [view, setView] = useState<AttemptsView | null>(null);
  const state = training?.state ?? null;
  useEffect(() => {
    if (!id || !state) return;
    let stopped = false;
    let timer: number | undefined;
    const go = async () => {
      try {
        const v = await api.attempts(id);
        if (!stopped) setView(v);
      } catch { /* not yet, or a blip: next time */ }
      if (!stopped && (state === "starting" || state === "running")) timer = window.setTimeout(go, 8000);
    };
    go();
    return () => { stopped = true; window.clearTimeout(timer); };
  }, [id, state]);
  useEffect(() => { setView(null); }, [id]);
  return view;
}

/** "Train on a GPU": the modal runner's gate, over the build view once the environment is built. */
function TrainOffer({ job, train, onStarted, again = false }: { job: Job; train: NonNullable<Health["train"]>; onStarted: (j: Job) => void; again?: boolean }) {
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const needCode = train.code_required || (train.per_day > 0 && train.left_today <= 0 && train.code_accepted);
  const go = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      onStarted(await api.train(job.id, code.trim()));
    } catch (x) {
      setErr((x as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <form onSubmit={go} className="bv-train">
      <p><strong>{again ? "The training stopped." : "The environment is built."}</strong> Train the robot on it{again ? " again" : ""}: about {train.minutes} min on a GPU, and you watch it learn.{train.per_day > 0 && !train.code_required ? ` ${train.left_today} of ${train.per_day} free today.` : ""}</p>
      <div className="bv-train-row">
        {(needCode || train.code_accepted) && (
          <input type="text" value={code} onChange={(e) => setCode(e.target.value)} placeholder={needCode ? "Invite code" : "Invite code (optional)"}
            aria-label="Invite code" autoComplete="off" spellCheck={false} required={needCode} maxLength={200} />
        )}
        <button className="btn sm primary" type="submit" disabled={busy || train.busy || (needCode && !code.trim())}>
          <Icon name="play" size={14} />{busy ? "Starting…" : train.busy ? "GPU busy, try again soon" : "Train on a GPU"}
        </button>
      </div>
      {err && <div className="error-box" style={{ marginTop: 8 }}>{err}</div>}
    </form>
  );
}

/** A cheap content signature of a build piece: frames by length and end values, variants by counts. */
function signature(x: BuildInput["skeleton"] | BuildInput["qpos"] | BuildInput["variants"] | undefined): string {
  if (!x) return "";
  if ("rows" in x) return `v${x.total}/${x.kept}/${x.ghosts.length}`;
  const a = x.data;
  return `${a.length}:${a[0]}:${a[a.length - 1]}`;
}

function retargetErr(e: Record<string, [number, number]>) {
  const v = Object.values(e);
  if (!v.length) return "—";
  const mean = v.reduce((a, x) => a + x[0], 0) / v.length;
  const max = Math.max(...v.map((x) => x[1]));
  return `${mean.toFixed(1)} / ${max.toFixed(1)} cm`;
}

/** Training needs a GPU: the modal runner's training when there is one, else offer the colab runner when this host
 *  has it, else point at trained gallery runs. */
function PolicyCard({ job, file, attempts }: { job: Job; file: File | null; attempts: AttemptsView | null }) {
  const { health } = useHealth();
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [gpu, setGpu] = useState<Job | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const policy = job.stages.find((s) => s.key === "policy");
  const colab = !!health?.runners.includes("colab");
  const modal = !!health?.train;
  useEffect(() => {
    if (!colab && !runs) api.runs().then(setRuns).catch(() => setRuns([]));
  }, [colab, runs]);
  const train = async () => {
    if (!file) return;
    setBusy(true);
    setErr(null);
    try {
      setGpu(await api.createJob({ file, runner: "colab" }));
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const trained = runs?.filter((r) => r.has_task_spec) ?? [];
  const tr = job.training;

  if (tr) {
    const last = attempts?.attempts[attempts.attempts.length - 1] ?? null;
    const going = tr.state === "starting" || tr.state === "running";
    return (
      <div className="card card-pad live-policy">
        <div className="eyebrow">Policy · {tr.state === "done" ? "trained on a GPU" : going ? "training on a GPU" : tr.state}</div>
        <p className="muted" style={{ fontSize: 14 }}>
          {tr.state === "starting" ? "Uploading the environment to the GPU…"
            : going ? `Iteration ${tr.iteration.toLocaleString()} of ${tr.iterations.toLocaleString()}${tr.reward != null ? `, mean reward ${tr.reward.toFixed(2)}` : ""}. A new attempt lands above every ${tr.attempt_every} iterations.`
            : tr.state === "done" ? `${tr.iterations.toLocaleString()} iterations, ${tr.attempts} attempts.${last?.stages_total ? ` The last one reaches ${last.stages_completed ?? 0} of ${last.stages_total} key moments${last.fell ? " and falls" : last.success ? " and completes the clip" : ""}.` : ""}`
            : tr.error ?? job.error ?? "The training stopped."}
        </p>
        {tr.checkpoint && <a className="btn sm" href={url(api.policy(job.id))} download><Icon name="download" size={14} />Download the policy (rsl_rl .pt)</a>}
        {tr.state === "done" && <p className="faint" style={{ fontSize: 12.5, marginTop: 8 }}>Loads in mjlab with this build's training environment.</p>}
      </div>
    );
  }
  if (policy?.status === "running" || policy?.status === "done") {
    return (
      <div className="card card-pad live-policy">
        <div className="eyebrow">Policy</div>
        <p className="muted" style={{ fontSize: 14 }}>
          {policy.status === "done" ? "The policy is trained." : `PPO is training on this environment (${Math.round(policy.progress * 100)}%).`}
        </p>
        {job.result_run && <Link className="btn sm primary" to={`/runs/${job.result_run}`}>Open results <Icon name="arrow" size={14} /></Link>}
      </div>
    );
  }
  return (
    <div className="card card-pad live-policy">
      <div className="eyebrow">Policy</div>
      <p className="muted" style={{ fontSize: 14 }}>
        {modal && policy?.status === "skipped"
          ? <>Nothing trained yet. "Train on a GPU" above trains a policy in about {health?.train?.minutes} min{health?.train?.code_required ? " (invite code needed)" : ""}.</>
          : <>Training takes 1–2 h on a GPU{policy?.status !== "skipped" ? "." : colab ? "; the GPU runner can take it from here." : ", which this host lacks."}</>}
      </p>
      {colab ? (
        gpu ? (
          <Link className="btn sm primary" to={`/jobs/${gpu.id}`}>Follow the GPU job <Icon name="arrow" size={14} /></Link>
        ) : file ? (
          <div><button className="btn sm primary" onClick={train} disabled={busy}><Icon name="play" size={14} />{busy ? "Uploading…" : "Train a policy on the GPU"}</button></div>
        ) : (
          <p className="faint" style={{ fontSize: 13 }}>To train on the GPU, <Link to="/live" style={{ color: "var(--accent)" }}>upload the video again</Link>.</p>
        )
      ) : modal ? null : (
        <>
          <p className="faint" style={{ fontSize: 13 }}>Runs trained on a task like this one:</p>
          {runs == null ? <div className="skeleton" style={{ height: 40 }} /> : trained.length ? (
            <ul className="live-runs">
              {trained.map((r) => (
                <li key={r.id}><Link to={`/runs/${r.id}`}><span>{r.title}</span><Icon name="arrow" size={14} /></Link></li>
              ))}
            </ul>
          ) : (
            <p className="faint" style={{ fontSize: 13 }}>None yet; see the <Link to="/showcase" style={{ color: "var(--accent)" }}>showcase</Link>.</p>
          )}
        </>
      )}
      {err && <div className="error-box">{err}</div>}
    </div>
  );
}

// the stored run of each recorded build's clip (same clip, same SAM 3D Body front end): its 3D skeleton, qpos and variants
const DEMO_RUN: Record<string, string> = { "front-kick": "latent_s3_kick", "squat-jump": "latent_s3_sj", "jumping-jacks": "latent_s3_jacks", lunge: "latent_s3_lunge", punches: "latent_s3_punch" };

const TRAIN_REPLAY_S = 24; // the recorded 1,500 PPO iterations (~1 h on an A100), replayed in this many seconds

/** A recorded build as a Job that grows with the clock, as the API's would: stages start and finish at
 *  the times the worker reported them; then the policy stage replays training. */
function replayJob(r: Replay, e: number, t0: number): Job {
  const seen = r.events.filter((x) => x.t <= e);
  const doneAt = (k: string) => r.events.find((x) => x.stage === k && x.done)?.t ?? Infinity;
  const envDone = doneAt("environment");
  const trains = r.policies.length > 0;
  const end = trains ? envDone + TRAIN_REPLAY_S : envDone;
  let prevDone = 0;
  const stages = r.stages.map((s) => {
    let status: Job["stages"][number]["status"] = "pending";
    let progress = 0;
    let started: number | null = null;
    let finished: number | null = null;
    if (s.key === "policy") {
      if (e >= envDone) {
        started = envDone;
        if (!trains) status = "skipped";
        else {
          progress = Math.min(1, (e - envDone) / TRAIN_REPLAY_S);
          status = progress >= 1 ? "done" : "running";
          finished = progress >= 1 ? end : null;
        }
      }
    } else {
      const mine = seen.filter((x) => x.stage === s.key);
      const d = doneAt(s.key);
      if (e >= d) { status = "done"; progress = 1; finished = d; }
      else if (mine.length || e >= prevDone) { status = "running"; progress = mine.filter((x) => x.progress != null).at(-1)?.progress ?? 0; }
      if (status !== "pending") started = prevDone;
      prevDone = d;
    }
    return { key: s.key, label: s.label, description: "", status, progress,
             started_at: started == null ? null : t0 + started, finished_at: finished == null ? null : t0 + finished };
  });
  const logs: Job["logs"] = seen.filter((x) => x.log).map((x) => [t0 + x.t, x.stage, x.log as string]);
  if (trains && e >= envDone) {
    const p = r.policies[0];
    logs.push([t0 + envDone, "policy", `PPO on 4,096 simulated G1s in mjlab, ${p.curve?.iterations ?? 1500} iterations (recorded run ${p.run}; ~1 h on an A100, replayed in ${TRAIN_REPLAY_S} s)`]);
    if (e >= end) logs.push([t0 + end, "policy", `trained: ${r.policies.map((x) => x.label).join(", ")}`]);
  }
  const running = stages.find((s) => s.status === "running");
  return {
    id: r.id, runner: "replay", status: e >= end ? "succeeded" : "running", stage: running?.key ?? null,
    progress: stages.reduce((a, s) => a + (s.status === "done" || s.status === "skipped" ? 1 : s.status === "running" ? s.progress : 0), 0) / stages.length,
    stages, logs, curves: {}, input: null, source_run: null, result_run: null, error: null,
    created_at: t0, updated_at: t0 + Math.min(e, end), demo: true,
  };
}

/** /live/demo/:demoId: a recorded build (scripts/build_live_demo.py) replayed through the live view. */
export function LiveReplayPage() {
  const { demoId = "" } = useParams();
  const [replay, setReplay] = useState<Replay | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [t0, setT0] = useState(() => Date.now() / 1000);
  const [now, setNow] = useState(t0);
  useEffect(() => {
    setReplay(null);
    setErr(null);
    demo.replay(demoId).then((r) => { setReplay(r); const t = Date.now() / 1000; setT0(t); setNow(t); }).catch((e) => setErr(e.message));
  }, [demoId]);
  const job = useMemo(() => (replay ? replayJob(replay, now - t0, t0) : null), [replay, now, t0]);
  const finished = job?.status === "succeeded";
  useEffect(() => {
    if (!replay || finished) return;
    const i = window.setInterval(() => setNow(Date.now() / 1000), 200);
    return () => window.clearInterval(i);
  }, [replay, finished]);
  const retargeted = useRef(false);
  retargeted.current = done(job, "retarget");
  const source = useMemo<Source | null>(() => replay && {
    title: `Recorded build · ${replay.title} · ${replay.recorded}`,
    clip: replay.clip.replace(/\.[a-z0-9]+$/i, ""),
    extra: async (set) => {
      const rid = DEMO_RUN[demoId];
      if (!rid) return null;
      const run = await api.run(rid);
      await loadRun(run, (p) => {
        const keep = Object.fromEntries(Object.entries(p).filter(([k, v]) => (k === "skeleton" || k === "qpos" || k === "variants") && v));
        if (Object.keys(keep).length) set(keep);
      });
      return rid;
    },
    // the recording has everything; hand it out stage by stage, as the API would
    live: () => demo.live(demoId).then((d) => (retargeted.current ? d : { ...d, robot: null, retarget_error_cm: null })),
    spec: () => demo.spec(demoId),
    video: demo.file(demoId, "processed.mp4"),
    policy: null,
  }, [replay, demoId]);

  if (err) return <div className="container section"><div className="error-box">Recorded build {demoId}: {err}</div></div>;
  if (!replay || !job || !source) return <div className="container section"><div className="skeleton" style={{ height: 320 }} /></div>;
  return (
    <>
      <div className="container"><div className="notice" style={{ marginTop: 18 }}>
        <Icon name="cube" />
        <div><strong>Recorded build.</strong> This replays a real run of the pipeline on {replay.clip}, at the speed it ran;
          training is replayed fast. <button className="linklike" onClick={() => { const t = Date.now() / 1000; setT0(t); setNow(t); }}>Replay from the start</button></div>
      </div></div>
      <LiveView job={job} file={null} uploading={false} err={null}
        source={{ ...source, policy: <ReplayPolicies replay={replay} job={job} demoId={demoId} /> }} />
    </>
  );
}

function ReplayPolicies({ replay, job, demoId }: { replay: Replay; job: Job; demoId: string }) {
  const stage = job.stages.find((s) => s.key === "policy");
  if (!replay.policies.length) {
    return <div className="card card-pad live-policy"><div className="eyebrow">Policy</div>
      <p className="muted" style={{ fontSize: 14 }}>No policy has been trained on this clip yet.</p></div>;
  }
  if (!stage || stage.status === "pending") {
    return <div className="card card-pad live-policy"><div className="eyebrow">Policy</div>
      <p className="muted" style={{ fontSize: 14 }}>Training starts once the task environment is ready.</p></div>;
  }
  const lead = replay.policies[0];
  if (stage.status === "running") {
    const iters = lead.curve?.iterations ?? 1500;
    return (
      <div className="card card-pad live-policy">
        <div className="eyebrow">Policy · training (replay)</div>
        <p className="muted" style={{ fontSize: 14 }}>
          PPO on 4,096 simulated G1s: iteration {Math.round(stage.progress * iters).toLocaleString()} of {iters.toLocaleString()}.
        </p>
        {lead.curve && <Curve points={lead.curve.points} upTo={stage.progress * iters} label={`${lead.label}: mean episode reward`} />}
      </div>
    );
  }
  return (
    <div className="card card-pad live-policy">
      <div className="eyebrow">Policies trained on this environment</div>
      <div className="replay-policies">
        {replay.policies.map((p) => (
          <figure key={p.key} className="replay-policy">
            <div className="video-frame"><video src={demo.file(demoId, p.video)} poster={demo.file(demoId, p.poster)} muted loop playsInline autoPlay controls /></div>
            <figcaption>
              <strong>{p.label}</strong>
              <span className={`env-tag ${p.checker.success ? "req" : ""}`}>
                {p.checker.success ? `passes today's checks: ${p.checker.stages_completed}/${p.checker.stages_total}` : `today's checks: ${p.checker.stages_completed}/${p.checker.stages_total}`}
              </span>
              {!p.checker.success && <span className="faint" style={{ fontSize: 12 }}>Trained under an earlier, looser checker.</span>}
              {p.curve && <Curve points={p.curve.points} upTo={Infinity} label="mean episode reward" small />}
            </figcaption>
          </figure>
        ))}
      </div>
      <p className="faint" style={{ fontSize: 12.5 }}>Solid: the trained policy. Green ghost: the reference.</p>
    </div>
  );
}

/** Mean episode reward over PPO iterations, drawn up to iteration `upTo`. */
function Curve({ points, upTo, label, small }: { points: [number, number, number][]; upTo: number; label: string; small?: boolean }) {
  const W = 320, H = small ? 56 : 96;
  const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs) || 1, y0 = Math.min(...ys), y1 = Math.max(...ys);
  const sx = (x: number) => ((x - x0) / (x1 - x0 || 1)) * (W - 8) + 4;
  const sy = (y: number) => H - 4 - ((y - y0) / (y1 - y0 || 1)) * (H - 8);
  const shown = points.filter((p) => p[0] <= upTo);
  const last = shown.at(-1);
  return (
    <div className="curve">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${label}, ${last ? `${last[1].toFixed(1)} at iteration ${last[0]}` : "starting"}`}>
        <polyline points={points.map((p) => `${sx(p[0])},${sy(p[1])}`).join(" ")} className="ghost" />
        {shown.length > 1 && <polyline points={shown.map((p) => `${sx(p[0])},${sy(p[1])}`).join(" ")} />}
        {last && <circle cx={sx(last[0])} cy={sy(last[1])} r={3} />}
      </svg>
      <div className="faint mono" style={{ fontSize: 11.5 }}>{label}{last ? ` · ${last[1].toFixed(1)}` : ""}</div>
    </div>
  );
}
