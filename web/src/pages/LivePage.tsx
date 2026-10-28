import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router";
import { api, checkVideo, fmt, url, watchJob, type Job, type LiveData, type RunSummary, type TaskSpec } from "../api";
import { useHealth } from "../App";
import { EnvironmentPanel } from "../components/environment";
import { PoseOverlay, RewardBox, RobotView, TaskBreakdown } from "../components/live";
import { Icon, StatusPill } from "../components/ui";

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

const done = (job: Job | null, key: string) => job?.stages.find((s) => s.key === key)?.status === "done";
const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

/** Fetches `load` once `ready`, again whenever `version` changes, retrying every 3 s while the job runs. */
function useWhenReady<T>(ready: boolean, version: string, running: boolean, load: () => Promise<T>) {
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
  return data;
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
  }, [id]);

  const start = async (f: File) => {
    setErr(null);
    setFile(f);
    setBusy(true);
    try {
      const j = await api.createJob({ file: f, runner: "live" });
      localFiles.set(j.id, f);
      setJob(j);
      nav(`/live/${j.id}`);
    } catch (e) {
      setErr((e as Error).message);
      setFile(null);
    } finally {
      setBusy(false);
    }
  };

  if (!id && !file) return <LiveStart onFile={start} err={err} />;
  if (id && err && !job) return <div className="container section"><div className="error-box">Job {id}: {err}</div></div>;
  return <LiveView job={job} file={file} uploading={busy} err={err} />;
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
        <h1>Watch a video become an RL environment</h1>
        <p className="lede">
          Upload any athletic movement. Your clip plays on the left with its pose landmarks; on the right, the
          Unitree G1 retarget, the key moments and the reward the robot will train against appear as each stage
          finishes, usually within a minute.
        </p>
        <div className="live-chips">
          {STEPS.map((s, i) => <span key={s.key} className="pill"><span className="mono faint">{i + 1}</span>{s.label}</span>)}
        </div>
      </div>

      {!health && !apiError && <div className="skeleton" style={{ height: 220 }} />}
      {!health && apiError && (
        <div className="notice">
          <Icon name="cube" />
          <div><strong>The pipeline API isn't reachable right now.</strong> The hosted API sleeps when idle; this page keeps
            checking. Meanwhile the <Link to="/examples" style={{ color: "var(--accent)" }}>examples</Link> show finished results.</div>
        </div>
      )}
      {health && !available && (
        <div className="notice">
          <Icon name="cube" />
          <div><strong>This deployment can't build environments live.</strong> Its API has no live runner (it may be an
            older version, or the host lacks the pose and IK models). The <Link to="/examples" style={{ color: "var(--accent)" }}>examples</Link> walk
            through recorded runs, environments included.</div>
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
          <p className="faint" style={{ fontSize: 12.5, marginTop: 12 }}>
            Your video is sent only to this site's pipeline API, which keeps uploads for a short while after the build.
          </p>
        </div>
      )}
    </div>
  );
}

function LiveView({ job, file, uploading, err }: { job: Job | null; file: File | null; uploading: boolean; err: string | null }) {
  const [video, setVideo] = useState<HTMLVideoElement | null>(null);
  const [blob, setBlob] = useState<string | null>(null);
  const [procFailed, setProcFailed] = useState(false);
  const [ratio, setRatio] = useState<number | null>(null);
  const t = useVideoTime(video);
  const running = job?.status === "running" || job?.status === "queued";
  const id = job?.id ?? "";

  useEffect(() => {
    if (!file) return setBlob(null);
    const u = URL.createObjectURL(file);
    setBlob(u);
    return () => URL.revokeObjectURL(u);
  }, [file]);
  useEffect(() => setProcFailed(false), [id]);

  const st = (k: string) => job?.stages.find((s) => s.key === k)?.status ?? "pending";
  const live = useWhenReady<LiveData>(done(job, "pose"), `${st("retarget")}/${st("environment")}`, running, () => api.live(id));
  const spec = useWhenReady<TaskSpec>(done(job, "environment"), "1", running, () => api.jobTaskSpec(id));

  // The processed copy once it exists (trimmed and rescaled, so the landmarks line up); until then the local file.
  const processed = done(job, "video") && !procFailed ? api.liveVideo(id) : null;
  const src = processed ?? blob ?? (job?.input ? url(job.input.url) : null);
  const aligned = !!processed && src === processed;
  const aspect = live?.video && aligned ? live.video.width / live.video.height : ratio;

  const seek = useCallback((s: number) => {
    if (video) video.currentTime = Math.max(0, Math.min(s, (video.duration || s + 1) - 0.02));
  }, [video]);

  const elapsed = job ? (running ? Date.now() / 1000 : job.updated_at) - job.created_at : 0;
  const stageLabel = job?.stages.find((s) => s.key === job.stage)?.label;
  const title = !job ? "Uploading…" : job.status === "failed" ? "Build failed" : job.status === "cancelled" ? "Build cancelled"
    : spec ? "Environment ready" : job.status === "queued" ? "Waiting for the builder…" : stageLabel ? `${stageLabel}…` : "Starting…";

  return (
    <div className="container">
      <div className="job-head">
        <div style={{ flex: 1, minWidth: 240 }}>
          <div className="eyebrow">Live build{job && <> · job {job.id}</>}</div>
          <h1 style={{ marginTop: 4 }}>{title}</h1>
        </div>
        {job && <StatusPill status={job.status} />}
        {job && <span className="mono muted" style={{ fontSize: 14 }}>{clock(Math.max(0, elapsed))}</span>}
        {job && running && <button className="btn sm" onClick={() => api.cancel(job.id).catch(() => undefined)}>Cancel</button>}
        <Link className="btn sm ghost" to="/live"><Icon name="upload" size={14} />New video</Link>
      </div>
      <div className="progress" style={{ marginBottom: 20 }}><div style={{ width: `${(job?.progress ?? 0) * 100}%` }} /></div>
      {job?.status === "queued" && job.logs[0] && (
        <div className="notice" style={{ marginBottom: 20 }}><span className="dot live" style={{ marginTop: 6 }} /><div>{job.logs[0][2]}</div></div>
      )}
      {(job?.error || (err && !uploading)) && <div className="error-box" style={{ marginBottom: 20 }}>{job?.error ?? err}</div>}

      <div className="live-grid">
        <div className="live-left">
          <div className="video-frame live-video" style={{ aspectRatio: aspect ? String(aspect) : "16 / 10", maxWidth: aspect ? `calc(var(--live-h) * ${aspect.toFixed(4)})` : undefined }}>
            {src && (
              <video key={src} ref={setVideo} src={src} muted loop playsInline autoPlay controls
                onLoadedMetadata={(e) => e.currentTarget.videoWidth && setRatio(e.currentTarget.videoWidth / e.currentTarget.videoHeight)}
                onError={() => aligned && setProcFailed(true)} />
            )}
            {live && aligned && <PoseOverlay video={video} data={live} />}
            <div className="tag">
              <span className="pill">{aligned ? "Processed clip" : file ? `Your upload · ${file.name}` : "Your upload"}</span>
              {live?.landmarks && aligned && <span className="pill accent">pose landmarks</span>}
            </div>
          </div>
          <div className="live-legend">
            <span><i className="dot" style={{ background: "#3987e5" }} />left</span>
            <span><i className="dot" style={{ background: "#d95926" }} />right</span>
            <span className="faint">{aligned ? "The video is the only clock: the robot and the timeline follow it." : "Local preview; landmarks appear on the processed clip."}</span>
          </div>
          {live && (
            <dl className="kv card card-pad live-stats">
              <dt>Clip</dt><dd>{live.duration_s.toFixed(1)} s · {live.frames} frames @ {Math.round(live.fps)} fps</dd>
              {live.video && <><dt>Processed size</dt><dd>{live.video.width}×{live.video.height}</dd></>}
              <dt>Pose found</dt><dd>{fmt.pct(live.pose_found)}</dd>
              {live.retarget_error_cm && <><dt>Retarget error (mean / max)</dt><dd>{retargetErr(live.retarget_error_cm)}</dd></>}
            </dl>
          )}
        </div>

        <div className="live-right">
          <div className="card stepper live-steps">
            {(job?.stages.length ? job.stages : STEPS.map((s) => ({ ...s, status: "pending" as const, progress: 0, description: "", started_at: null, finished_at: null }))).map((s, i) => {
              const took = live?.timings_s[s.key] ?? (s.started_at && s.finished_at ? s.finished_at - s.started_at : null);
              const model = STEPS.find((x) => x.key === s.key)?.model;
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

          <div className="card live-robot-card">
            <div className="live-card-head">
              <div className="eyebrow">Unitree G1 · retarget</div>
              {live?.robot && <span className="faint" style={{ fontSize: 12 }}>drag to turn</span>}
            </div>
            {live?.robot ? <RobotView video={video} data={live} /> : (
              <div className="ex-pending-media live-pending">{done(job, "pose") || st("retarget") === "running" ? "Retargeting onto the G1…" : "The robot appears when the retarget finishes"}</div>
            )}
          </div>

          {spec ? (
            <div className="fade-in live-env">
              <EnvironmentPanel spec={spec} live={running} t={t} onSeek={seek} />
              <TaskBreakdown spec={spec} t={t} onSeek={seek} />
              <RewardBox spec={spec} />
            </div>
          ) : (
            <div className="card card-pad faint" style={{ fontSize: 13.5 }}>
              {st("environment") === "running" ? "Finding the key moments and compiling the reward…" : "The task environment (key moments, contacts, reward) appears after the retarget."}
            </div>
          )}

          {job && <PolicyCard job={job} file={file} />}

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

function retargetErr(e: Record<string, [number, number]>) {
  const v = Object.values(e);
  if (!v.length) return "—";
  const mean = v.reduce((a, x) => a + x[0], 0) / v.length;
  const max = Math.max(...v.map((x) => x[1]));
  return `${mean.toFixed(1)} / ${max.toFixed(1)} cm`;
}

/** Training needs a GPU: offer the colab runner when this host has it, else point at trained gallery runs. */
function PolicyCard({ job, file }: { job: Job; file: File | null }) {
  const { health } = useHealth();
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [gpu, setGpu] = useState<Job | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const policy = job.stages.find((s) => s.key === "policy");
  const colab = !!health?.runners.includes("colab");
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
        Training a policy in this environment takes 1–2 h on a GPU (PPO across thousands of simulated G1s){policy?.status !== "skipped" ? "." : colab ? "; this build stops at the environment, and the GPU runner can take it from there." : "; this host has none, so the build stops here."}
      </p>
      {colab ? (
        gpu ? (
          <Link className="btn sm primary" to={`/jobs/${gpu.id}`}>Follow the GPU job <Icon name="arrow" size={14} /></Link>
        ) : file ? (
          <div><button className="btn sm primary" onClick={train} disabled={busy}><Icon name="play" size={14} />{busy ? "Uploading…" : "Train a policy on the GPU"}</button></div>
        ) : (
          <p className="faint" style={{ fontSize: 13 }}>To train on the GPU, <Link to="/live" style={{ color: "var(--accent)" }}>upload the video again</Link> (it isn't kept in this tab after a reload).</p>
        )
      ) : (
        <>
          <p className="faint" style={{ fontSize: 13 }}>Runs already trained against a keypose task like this one:</p>
          {runs == null ? <div className="skeleton" style={{ height: 40 }} /> : trained.length ? (
            <ul className="live-runs">
              {trained.map((r) => (
                <li key={r.id}><Link to={`/runs/${r.id}`}><span>{r.title}</span><Icon name="arrow" size={14} /></Link></li>
              ))}
            </ul>
          ) : (
            <p className="faint" style={{ fontSize: 13 }}>None in this gallery yet; the <Link to="/examples" style={{ color: "var(--accent)" }}>examples</Link> show trained policies.</p>
          )}
        </>
      )}
      {err && <div className="error-box">{err}</div>}
    </div>
  );
}
