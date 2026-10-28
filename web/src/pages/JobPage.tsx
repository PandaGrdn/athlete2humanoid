import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, fmt, policyColor, url, watchJob, type Job, type RunSummary, type TaskSpec } from "../api";
import { LineChart } from "../components/charts";
import { EnvironmentPanel } from "../components/environment";
import { Icon, LoopVideo, StatusPill } from "../components/ui";

const MODEL: Record<string, string> = {
  silhouette: "SAM 3.1", pose: "MediaPipe", retarget: "mink", environment: "key poses", latent: "V-JEPA 2", train: "mjlab PPO", eval: "V-JEPA 2",
};

/** The job's task environment, fetched once its stage has started; retried while the job runs (a GPU job's
 *  spec reaches the server some time after its log line). */
function useTaskSpec(job: Job | null) {
  const [spec, setSpec] = useState<TaskSpec | null>(null);
  const env = job?.stages.find((s) => s.key === "environment")?.status;
  const live = job?.status === "running" || job?.status === "queued";
  const id = job?.id;
  useEffect(() => setSpec(null), [id]);
  useEffect(() => {
    if (!id || spec || !env || env === "pending") return;
    let stopped = false;
    let timer: number | undefined;
    const load = () => api.jobTaskSpec(id).then((s) => !stopped && setSpec(s)).catch(() => {
      if (!stopped && live) timer = window.setTimeout(load, 10_000);
    });
    load();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [id, env, live, spec]);
  return spec;
}

function useNow(active: boolean) {
  const [now, setNow] = useState(Date.now() / 1000);
  useEffect(() => {
    if (!active) return;
    const t = window.setInterval(() => setNow(Date.now() / 1000), 250);
    return () => window.clearInterval(t);
  }, [active]);
  return now;
}

const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

export default function JobPage() {
  const { id = "" } = useParams();
  const [job, setJob] = useState<Job | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [source, setSource] = useState<RunSummary | null>(null);
  const consoleRef = useRef<HTMLDivElement>(null);
  const spec = useTaskSpec(job);
  const live = job?.status === "running" || job?.status === "queued";
  const now = useNow(!!live);

  useEffect(() => {
    api.job(id).then(setJob).catch((e) => setErr(e.message));
    return watchJob(id, setJob);
  }, [id]);
  useEffect(() => {
    if (job?.source_run && !source) api.runs().then((rs) => setSource(rs.find((r) => r.id === job.source_run) ?? null)).catch(() => undefined);
  }, [job?.source_run, source]);
  useEffect(() => {
    const c = consoleRef.current;
    if (c) c.scrollTop = c.scrollHeight;
  }, [job?.logs.length]);

  if (err) return <div className="container section"><div className="error-box">Job {id}: {err}</div></div>;
  if (!job) return <div className="container section"><div className="skeleton" style={{ height: 420 }} /></div>;

  const elapsed = (live ? now : job.updated_at) - job.created_at;
  const curves = Object.entries(job.curves).filter(([, c]) => c.length > 0);
  const xMax = source?.iterations ? source.iterations - 1 : Math.max(1, ...curves.flatMap(([, c]) => c.map((p) => p[0])));
  const resultId = job.result_run;
  const stageLabel = job.stages.find((s) => s.key === job.stage)?.label;

  return (
    <div className="container">
      <div className="job-head">
        <div style={{ flex: 1, minWidth: 260 }}>
          <div className="eyebrow">Job {job.id}</div>
          <h1 style={{ marginTop: 4 }}>
            {job.status === "succeeded" ? "Policy trained" : job.status === "failed" ? "Job failed" : job.status === "cancelled" ? "Job cancelled" : stageLabel ? `${stageLabel}…` : "Queued…"}
          </h1>
        </div>
        <StatusPill status={job.status} />
        <span className="mono muted" style={{ fontSize: 14 }}>{clock(Math.max(0, elapsed))}</span>
        {live && <button className="btn sm" onClick={() => api.cancel(job.id).then(setJob)}>Cancel</button>}
        {resultId && <Link className="btn primary" to={`/runs/${resultId}`}>View results <Icon name="arrow" size={16} /></Link>}
      </div>

      <div className="progress" style={{ marginBottom: 20 }}><div style={{ width: `${job.progress * 100}%` }} /></div>

      {job.demo && (
        <div className="notice" style={{ marginBottom: 20 }}>
          <Icon name="cube" />
          <div>
            <strong>Demo runner.</strong> This server has no GPU, so the job replays the recorded run{" "}
            <Link to={`/runs/${job.source_run}`} style={{ color: "var(--accent)" }}>{source?.title ?? job.source_run}</Link>{" "}
            (its real log lines and training curves) on a compressed clock.
            {job.input && " Results come from that run, not from your upload."}
          </div>
        </div>
      )}
      {job.error && <div className="error-box" style={{ marginBottom: 20 }}>{job.error}</div>}

      <div className="job-grid">
        <div className="card stepper">
          {job.stages.map((s, i) => {
            const dur = s.started_at ? (s.finished_at ?? (s.status === "running" ? now : s.started_at)) - s.started_at : null;
            return (
              <div className={`stage ${s.status}`} key={s.key}>
                <div className="marker">{s.status === "done" ? <Icon name="check" size={14} /> : i + 1}</div>
                <div>
                  <div className="head">
                    <h4>{s.label}</h4>
                    <span className="mono faint" style={{ fontSize: 12 }}>
                      {s.status === "running" ? `${Math.round(s.progress * 100)}%` : dur != null && s.status === "done" ? `${dur.toFixed(1)} s` : MODEL[s.key]}
                    </span>
                  </div>
                  <div className="desc">{s.description}</div>
                  {s.status === "running" && <div className="progress"><div style={{ width: `${s.progress * 100}%` }} /></div>}
                </div>
              </div>
            );
          })}
        </div>

        <div style={{ display: "grid", gap: 20, minWidth: 0 }}>
          <div className="two-col">
            <div className="video-frame" style={{ aspectRatio: "16 / 10" }}>
              {job.input ? (
                <video src={url(job.input.url)} muted loop playsInline autoPlay style={{ height: "100%", objectFit: "contain" }} />
              ) : source?.hero ? (
                <LoopVideo src={source.hero.url} label="Source clip" />
              ) : null}
              <div className="tag"><span className="pill">{job.input ? `Your upload · ${job.input.filename}` : "Source clip"}</span></div>
            </div>
            <div className="card card-pad" style={{ display: "flex", flexDirection: "column", minWidth: 0 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 8 }}>
                <div className="eyebrow">Training · mean episode reward</div>
                {curves.length > 0 && <span className="mono faint" style={{ fontSize: 12 }}>iter {Math.max(...curves.map(([, c]) => c[c.length - 1][0])) + 1}</span>}
              </div>
              {curves.length > 0 ? (
                // Small multiples: each policy optimises a different reward, so they don't share a y-scale.
                <div style={{ display: "grid", gap: 6 }}>
                  {curves.map(([k, c]) => (
                    <div key={k}>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12.5 }}>
                        <span style={{ display: "inline-flex", alignItems: "center", gap: 6, color: "var(--text-2)" }}><i className="dot" style={{ background: policyColor(k) }} />{k}</span>
                        <span className="mono">{c[c.length - 1][1].toFixed(2)}</span>
                      </div>
                      <LineChart compact height={44} xLabel="iteration" yLabel="mean reward" xMax={xMax}
                        series={[{ key: k, label: k, color: policyColor(k), points: c }]} />
                    </div>
                  ))}
                </div>
              ) : (
                <div className="faint" style={{ margin: "auto", fontSize: 13, textAlign: "center" }}>
                  {job.status === "succeeded" ? "No training curve recorded" : "Curves appear when PPO training starts"}
                </div>
              )}
            </div>
          </div>
          {spec && (
            <div className="fade-in">
              <EnvironmentPanel spec={spec} live={!!live} />
              {live && <p className="faint" style={{ fontSize: 12.5, marginTop: 8 }}>This is the environment the policy is training in right now; the trained policy follows when training finishes.</p>}
            </div>
          )}
          <div className="console" ref={consoleRef} aria-live="polite">
            {job.logs.map(([t, stage, line], i) => (
              <div className="line" key={i}>
                <span className="t">{clock(Math.max(0, t - job.created_at))}</span>
                <span className="s">{stage}</span>
                <span className="m">{line}</span>
              </div>
            ))}
            {live && <div className="line"><span className="t" /><span className="s" /><span className="m faint">▍</span></div>}
          </div>
          {job.status === "succeeded" && resultId && (
            <div className="card card-pad fade-in" style={{ display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
              <div style={{ flex: 1 }}>
                <h3 style={{ fontSize: 17 }}>Results are ready</h3>
                <p className="muted" style={{ fontSize: 14, marginTop: 4 }}>Side-by-side video, V-JEPA 2 scores, robustness and downloads.</p>
              </div>
              <Link className="btn primary" to={`/runs/${resultId}`}>Open results <Icon name="arrow" size={16} /></Link>
            </div>
          )}
          <p className="faint" style={{ fontSize: 12.5 }}>Started {fmt.date(new Date(job.created_at * 1000).toISOString())} · runner: {job.runner}</p>
        </div>
      </div>
    </div>
  );
}
