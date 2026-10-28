import { useEffect, useState, type ReactNode } from "react";
import { Link, useNavigate, useParams } from "react-router";
import { api, fmt, policyColor, url, type FileRef, type Policy, type Robust, type RunDetail, type TaskSpec } from "../api";
import { BarChart, LineChart } from "../components/charts";
import { EnvironmentPanel } from "../components/environment";
import { Icon, LoopVideo } from "../components/ui";

const NEUTRAL = "#4a5562";

export default function RunPage() {
  const { id = "" } = useParams();
  const nav = useNavigate();
  const [run, setRun] = useState<RunDetail | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [spec, setSpec] = useState<TaskSpec | null>(null);

  useEffect(() => {
    setRun(null);
    setSpec(null);
    api.run(id).then((r) => {
      setRun(r);
      if (r.has_task_spec) api.runTaskSpec(id).then(setSpec).catch(() => undefined);
    }).catch((e) => setErr(e.message));
  }, [id]);

  if (err) return <div className="container section"><div className="error-box">Run "{id}": {err}</div></div>;
  if (!run) return <div className="container section"><div className="skeleton" style={{ height: 480 }} /></div>;

  const current = run.policies.filter((p) => !p.previous);
  const best = [...current].sort((a, b) => (b.metrics.aligned ?? -1) - (a.metrics.aligned ?? -1))[0];
  const tightest = [...current].filter((p) => p.metrics.mean_torso_position_error_m != null)
    .sort((a, b) => a.metrics.mean_torso_position_error_m! - b.metrics.mean_torso_position_error_m!)[0];
  const ranks = current.map((p) => p.metrics.dtw_rank_vs_bank).filter((r): r is number => r != null);
  const falls = current.reduce((s, p) => s + (p.metrics.falls ?? 0), 0);
  const nearest = run.bank_top5[0];
  const duration = fmt.duration(run.started, run.finished);
  const replay = async () => {
    setBusy(true);
    try {
      nav(`/jobs/${(await api.createJob({ runId: run.id })).id}`);
    } catch (e) {
      alert((e as Error).message);
      setBusy(false);
    }
  };

  return (
    <div className="container fade-in">
      <div className="run-head">
        <div>
          <div className="eyebrow"><Link to="/#gallery">Gallery</Link> / {run.id}</div>
          <h1 style={{ marginTop: 6 }}>{run.title}</h1>
          <div className="meta">
            {run.target && <span className="pill">target clip: {run.target}</span>}
            {run.iterations && <span className="pill mono">{run.iterations.toLocaleString()} PPO iterations</span>}
            {duration && <span className="pill">{duration} on one A100</span>}
            <span className="pill">{fmt.date(run.finished)}</span>
          </div>
          {run.description && <p className="muted" style={{ marginTop: 12, maxWidth: 720 }}>{run.description}</p>}
        </div>
        <button className="btn" onClick={replay} disabled={busy}><Icon name="play" size={16} />Replay pipeline</button>
      </div>

      <div className="result-hero">
        <div className="video-frame">
          {run.hero && <video src={url(run.hero.url)} poster={run.poster ? url(run.poster.url) : undefined} controls muted loop autoPlay playsInline />}
          <div className="tag"><span className="pill">Source clip</span><span className="pill accent">G1 retarget</span></div>
        </div>
        <div className="tiles">
          <Tile k="V-JEPA 2 alignment" v={fmt.num(best?.metrics.aligned)}
            sub={<>best policy: <i className="dot" style={{ background: policyColor(best?.base ?? ""), width: 7, height: 7 }} /> {best?.label ?? "—"} · retarget ceiling {fmt.num(run.reference.aligned)}</>} />
          <Tile k="DTW rank vs motion bank" v={ranks.length ? `#${Math.max(...ranks)}` : "—"}
            sub={nearest ? <>worst policy rank; nearest other motion: {nearest[0]} ({nearest[1].toFixed(3)})</> : "1 = closer to the source clip than any bank motion"} />
          <Tile k="Falls" v={<>{falls}<small>/ {current.length} rollouts</small></>}
            sub={current[0]?.metrics.steps ? `${current[0].metrics.steps} steps at ${current[0].metrics.fps} Hz per rollout` : "evaluation rollouts"} />
          <Tile k="Torso position error" v={fmt.cm(tightest?.metrics.mean_torso_position_error_m)}
            sub={tightest ? `mean, ${tightest.label.toLowerCase()} policy; ` + current.filter((p) => p !== tightest).map((p) => `${p.base} ${fmt.cm(p.metrics.mean_torso_position_error_m)}`).join(", ") : ""} />
        </div>
      </div>

      {spec && (
        <section className="section">
          <div className="section-head">
            <div>
              <div className="eyebrow">Task environment</div>
              <h2 style={{ marginTop: 6 }}>What the robot was asked to do</h2>
              <p>Compiled from the retarget before any training: the action's key moments, found statistically from the motion, plus flights and landings from foot contacts. This is the reward machine the task policies train against.</p>
            </div>
          </div>
          <EnvironmentPanel spec={spec} />
        </section>
      )}

      {run.video && (
        <section className="section">
          <div className="section-head">
            <div>
              <div className="eyebrow">What the reward sees</div>
              <h2 style={{ marginTop: 6 }}>V-JEPA 2 compares silhouettes, not joint angles</h2>
              <p>The source video and each robot rollout are rendered as silhouettes from the clip's camera angle and embedded by V-JEPA 2. Their similarity over time is the latent reward and the evaluation score.</p>
            </div>
          </div>
          <div className="video-frame"><LoopVideo src={run.video.url} label="Silhouettes compared by V-JEPA 2" /></div>
        </section>
      )}

      <section className="section">
        <div className="section-head">
          <div>
            <div className="eyebrow">Policies</div>
            <h2 style={{ marginTop: 6 }}>Three rewards, same robot</h2>
            <p>Each policy is trained with PPO on 4,096 parallel G1s in mjlab, then rolled out once and scored. The ghost in each video is the retargeted reference.</p>
          </div>
        </div>
        <div className="policies">{current.map((p) => <PolicyCard key={p.key} p={p} />)}</div>
      </section>

      <section className="section">
        <div className="two-col">
          <div className="card card-pad">
            <div className="eyebrow">V-JEPA 2 alignment (time-aligned cosine)</div>
            <p className="muted" style={{ fontSize: 13, margin: "6px 0 14px" }}>How much each rollout looks like the source video to V-JEPA 2. The retarget is the kinematic ceiling.</p>
            <BarChart max={1} bars={[
              ...(run.reference.aligned != null ? [{ label: "retarget", value: run.reference.aligned, color: NEUTRAL, note: "kinematic reference" }] : []),
              ...run.policies.filter((p) => p.metrics.aligned != null).map((p) => ({
                label: p.previous ? `${p.base} (prev)` : p.base, value: p.metrics.aligned!, color: p.previous ? NEUTRAL : policyColor(p.base),
              })),
            ]} />
          </div>
          <div className="card card-pad">
            <div className="eyebrow">DTW similarity vs the motion bank</div>
            <p className="muted" style={{ fontSize: 13, margin: "6px 0 14px" }}>Policies against the closest other motions in the bank. Rank #1 means the rollout is closer to its source clip than to any other motion.</p>
            <BarChart max={1} bars={[
              ...current.filter((p) => p.metrics.dtw != null).map((p) => ({ label: p.base, value: p.metrics.dtw!, color: policyColor(p.base), note: `rank #${p.metrics.dtw_rank_vs_bank ?? "?"}` })),
              ...run.bank_top5.map(([n, v]) => ({ label: n, value: v, color: NEUTRAL, note: "bank motion" })),
            ]} />
          </div>
        </div>
      </section>

      {run.robust && <RobustSection robust={run.robust} policies={current} />}

      {run.clips.length > 0 && (
        <section className="section">
          <div className="section-head">
            <div>
              <div className="eyebrow">Clips</div>
              <h2 style={{ marginTop: 6 }}>Retargeted clips ({run.clips.length})</h2>
              <p>Every clip in the run goes through SAM 3.1, pose and IK. They join the motion bank used to distill the reward net and to rank rollouts.</p>
            </div>
          </div>
          <div className="clips">
            {run.clips.map((c) => (
              <div className="card clip" key={c.name}>
                <div className="media">{c.side_by_side && <LoopVideo src={c.side_by_side.url} label={`${c.name} retarget`} />}</div>
                <div className="body">
                  <span style={{ display: "flex", gap: 6, alignItems: "center" }}>{c.name.replace(/_/g, " ")}{c.is_target && <span className="pill accent">target</span>}</span>
                  <span className="mono faint" title="Mean retarget error across tracked keypoints">{c.retarget_error_mean_cm != null ? `${c.retarget_error_mean_cm.toFixed(1)} cm` : ""}</span>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="section">
        <div className="section-head">
          <div>
            <div className="eyebrow">Downloads</div>
            <h2 style={{ marginTop: 6 }}>Take the skill with you</h2>
            <p>ONNX policies run anywhere (observation in, 29 joint actions out). Checkpoints are rsl_rl format for mjlab. The motion CSV is the G1 root pose and joint angles per frame.</p>
          </div>
        </div>
        <div className="downloads">{run.downloads.map((d) => <Download key={d.path} d={d} />)}</div>
        {run.distill && (
          <p className="faint" style={{ fontSize: 13, marginTop: 16 }}>
            Reward net: distilled from {String(run.distill.motions ?? "?")} motions ({String(run.distill.samples ?? "?")} windows), {String(run.distill.pca_k ?? "?")} PCA dims,
            held-out cosine {typeof run.distill.val_cosine_mean === "number" ? run.distill.val_cosine_mean.toFixed(3) : "—"}.
          </p>
        )}
      </section>
    </div>
  );
}

function Tile({ k, v, sub }: { k: string; v: ReactNode; sub: ReactNode }) {
  return (
    <div className="card tile">
      <div className="k">{k}</div>
      <div className="v">{v}</div>
      <div className="sub">{sub}</div>
    </div>
  );
}

function PolicyCard({ p }: { p: Policy }) {
  const m = p.metrics;
  return (
    <div className="card policy-card">
      <div className="media">{p.video && <LoopVideo src={p.video.url} label={`${p.label} policy rollout`} />}</div>
      <div className="body">
        <div className="name"><i className="dot" style={{ background: policyColor(p.base), width: 10, height: 10 }} />{p.label}</div>
        <div className="blurb">{p.description}</div>
        <dl className="kv">
          <dt>V-JEPA aligned</dt><dd>{fmt.num(m.aligned)}</dd>
          <dt>DTW · rank vs bank</dt><dd>{fmt.num(m.dtw)} · #{m.dtw_rank_vs_bank ?? "—"}</dd>
          <dt>Torso error (mean / max)</dt><dd>{fmt.cm(m.mean_torso_position_error_m)} / {fmt.cm(m.max_torso_position_error_m)}</dd>
          <dt>Joint error</dt><dd>{fmt.num(m.mean_joint_error_rad, 2)} rad</dd>
          <dt>Falls</dt><dd>{m.falls ?? "—"}</dd>
        </dl>
        {p.curve.length > 1 && (
          <div style={{ marginTop: 14 }}>
            <div className="faint" style={{ fontSize: 12, marginBottom: 4 }}>Mean reward over {p.iterations?.toLocaleString() ?? ""} iterations</div>
            <LineChart height={110} xLabel="iteration" yLabel="mean reward" series={[{ key: p.key, label: p.base, color: policyColor(p.base), points: p.curve }]} />
          </div>
        )}
      </div>
    </div>
  );
}

function heat(fr: number | null) {
  // One hue (red), more opaque as more rollouts fall; the number is always printed.
  return fr == null ? "transparent" : `rgba(230, 103, 103, ${0.06 + 0.5 * fr})`;
}

function RobustSection({ robust, policies }: { robust: Robust; policies: Policy[] }) {
  const keys = policies.map((p) => p.base).filter((k) => robust.conditions.some((c) => c.policies[k]));
  const suites: string[] = [];
  robust.conditions.forEach((c) => !suites.includes(c.suite) && suites.push(c.suite));
  const SUITE: Record<string, string> = { clean: "Nominal", push: "Pushes", init: "Noisy start state", dynamics: "Friction & mass", speed: "Playback speed" };
  return (
    <section className="section">
      <div className="section-head">
        <div>
          <div className="eyebrow">Robustness</div>
          <h2 style={{ marginTop: 6 }}>Under perturbation</h2>
          <p>Fall rate across {robust.reps ?? "repeated"} rollouts per cell, with V-JEPA 2 alignment underneath. Darker red means more falls.</p>
        </div>
      </div>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Condition</th>
              {keys.map((k) => <th key={k} className="num"><span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}><i className="dot" style={{ background: policyColor(k) }} />{k}</span></th>)}
            </tr>
          </thead>
          <tbody>
            {suites.map((s) => [
              <tr className="group" key={`g-${s}`}><td colSpan={keys.length + 1}>{SUITE[s] ?? s}</td></tr>,
              ...robust.conditions.filter((c) => c.suite === s).map((c) => (
                <tr key={`${s}-${c.level}`}>
                  <td>{c.level}</td>
                  {keys.map((k) => {
                    const cell = c.policies[k];
                    return (
                      <td key={k} className="num">
                        <span className="heat" style={{ background: heat(cell?.fall_rate ?? null) }}
                          title={cell ? `fall rate ${fmt.pct(cell.fall_rate, 1)}, torso error ${fmt.cm(cell.torso_err_m)}, V-JEPA ${fmt.num(cell.vjepa_aligned)}` : ""}>
                          <span className="fr mono">{fmt.pct(cell?.fall_rate, 0)}</span>
                          <span className="vj mono">vj {fmt.num(cell?.vjepa_aligned, 2)}</span>
                        </span>
                      </td>
                    );
                  })}
                </tr>
              )),
            ])}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Download({ d }: { d: FileRef }) {
  const kind = d.kind === "onnx" ? "ONNX policy" : d.kind === "checkpoint" ? "Checkpoint" : d.kind === "motion" ? "G1 motion (CSV)" : d.kind === "reward" ? "Reward net" : "Metrics";
  const policy = d.path.match(/^policy_(\w+)\//)?.[1];
  return (
    <a className="card dl" href={url(d.url) + "?download=1"} download>
      <div className="ico"><Icon name={d.kind === "onnx" || d.kind === "checkpoint" ? "cube" : d.kind === "motion" ? "chart" : "file"} /></div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div className="name" style={{ display: "flex", alignItems: "center", gap: 6 }}>
          {policy && <i className="dot" style={{ background: policyColor(policy) }} />}
          {policy ? `${policy} · ${kind}` : kind}
        </div>
        <div className="size">{d.label} · {fmt.bytes(d.size)}</div>
      </div>
      <Icon name="download" />
    </a>
  );
}
