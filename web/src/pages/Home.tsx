import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router";
import { api, checkVideo, fmt, policyColor, type RunSummary } from "../api";
import { useHealth } from "../App";
import { Icon, LoopVideo } from "../components/ui";
import { exUrl } from "../examples";
import { ExampleCards, useExampleIndex } from "./Examples";

const PIPELINE = [
  { title: "Silhouette", text: "Segment and track the person through every frame.", model: "SAM 3.1" },
  { title: "Pose", text: "3D body landmarks per frame, lifted to world space.", model: "MediaPipe" },
  { title: "Retarget", text: "Inverse kinematics onto a 29-DoF Unitree G1.", model: "mink IK" },
  { title: "Task environment", text: "Key moments, contacts and the reward machine, from the retarget.", model: "key poses" },
  { title: "Latent reward", text: "Embed video and robot renders; distill a reward net.", model: "V-JEPA 2" },
  { title: "Policy", text: "Reinforcement learning across 4,096 simulated G1s.", model: "PPO · mjlab" },
  { title: "Evaluate", text: "Re-render rollouts; score them against the source clip.", model: "V-JEPA 2 · DTW" },
];

export default function Home() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const { health, error: apiError } = useHealth();
  const exIdx = useExampleIndex();

  useEffect(() => {
    if (health) api.runs().then(setRuns).catch((e) => setErr(e.message));
  }, [health]);
  const featured = runs?.find((r) => r.id === health?.demo_run) ?? runs?.[0];
  // The hero is the lead curated example (static, so it shows at once and offline); else the newest run.
  const lead = exIdx.items?.find((e) => e.status === "ready" && e.video);
  const offline = !!apiError && !health;

  return (
    <div className="container">
      <section className="hero">
        <div className="fade-in">
          <div className="models">
            <span className="pill accent">Meta V-JEPA 2</span>
            <span className="pill">SAM 3.1</span>
            <span className="pill">mjlab · MuJoCo Warp</span>
            <span className="pill">Unitree G1</span>
          </div>
          <h1>Any video in.<br /><em>A humanoid skill</em> out.</h1>
          <p className="lede">
            Film a person moving. The pipeline retargets the motion to a Unitree G1 and trains a control
            policy whose reward comes from Meta's V-JEPA 2 video model: the robot is rewarded for looking
            like the clip, not only for matching its joint angles.
          </p>
          <div className="cta">
            <Link className="btn primary" to="/examples"><Icon name="play" size={16} />See the examples</Link>
            <a className="btn" href="#run"><Icon name="upload" size={16} />Run a video</a>
            {featured && <Link className="btn ghost" to={`/runs/${featured.id}`}>Latest run <Icon name="arrow" size={16} /></Link>}
          </div>
        </div>
        <div className="fade-in">
          {lead?.video ? (
            <Link to={`/examples/${lead.id}`} className="video-frame" style={{ display: "block" }}>
              <LoopVideo src={exUrl(lead.video)} poster={lead.poster ? exUrl(lead.poster) : undefined} label={lead.title} />
              <div className="tag"><span className="pill">Phone video</span><span className="pill accent">Unitree G1</span></div>
            </Link>
          ) : featured?.hero ? (
            <div className="video-frame">
              <LoopVideo src={featured.hero.url} poster={featured.poster?.url} label="Source clip next to the G1 retarget" />
              <div className="tag"><span className="pill">Source clip</span><span className="pill accent">Unitree G1</span></div>
            </div>
          ) : (
            <div className="skeleton" style={{ aspectRatio: "751 / 480" }} />
          )}
        </div>
      </section>

      <section className="section" id="examples">
        <div className="section-head">
          <div>
            <div className="eyebrow">Examples</div>
            <h2 style={{ marginTop: 6 }}>What the robot learned, and how we know</h2>
            <p>Recorded results, each as a short story with synchronized videos and the numbers behind it.</p>
          </div>
          <Link className="btn sm" to="/examples">All examples <Icon name="arrow" size={14} /></Link>
        </div>
        <ExampleCards {...exIdx} />
      </section>

      <section className="section" id="how">
        <div className="section-head">
          <div>
            <div className="eyebrow">Pipeline</div>
            <h2 style={{ marginTop: 6 }}>Seven stages from pixels to a policy</h2>
          </div>
        </div>
        <div className="pipeline">
          {PIPELINE.map((s, i) => (
            <div className="step" key={s.title}>
              <div className="n">0{i + 1}</div>
              <h4>{s.title}</h4>
              <p>{s.text}</p>
              <div className="model">{s.model}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="section" id="run">
        <div className="section-head">
          <div>
            <div className="eyebrow">Run</div>
            <h2 style={{ marginTop: 6 }}>Start a job</h2>
            <p>Upload a clip of one person, full body in frame, or replay one of the finished runs below.</p>
          </div>
        </div>
        {offline && (
          <div className="notice" style={{ marginBottom: 16 }}>
            <Icon name="cube" />
            <div><strong>The live pipeline is offline right now.</strong> The hosted API sleeps when idle; this page
              keeps checking and uploads come back when it answers. Everything under{" "}
              <Link to="/examples" style={{ color: "var(--accent)" }}>Examples</Link> works without it.</div>
          </div>
        )}
        <Link to="/live" className="card live-cta">
          <span className="live-cta-ico"><Icon name="cube" size={20} /></span>
          <span style={{ flex: 1, minWidth: 0 }}>
            <strong>Watch it build live.</strong>{" "}
            <span className="muted">Upload any athletic movement and see the pose, the G1 retarget and its RL task
              environment appear next to your video, stage by stage.</span>
          </span>
          <span className="btn sm primary">Open live demo <Icon name="arrow" size={14} /></span>
        </Link>
        <div className="upload-grid">
          <UploadPanel offline={offline} />
          <div className="card card-pad" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <div className="eyebrow">What a job produces</div>
            <ul style={{ margin: 0, paddingLeft: 18, color: "var(--text-2)", fontSize: 14, display: "grid", gap: 8 }}>
              <li>The clip retargeted onto the G1, side by side with the source</li>
              <li>Three policies: joint tracking, hybrid, and V-JEPA latent reward</li>
              <li>V-JEPA 2 alignment, DTW rank against a bank of other motions, falls, torso error</li>
              <li>Robustness under pushes, noisy starts, friction, mass and speed changes</li>
              <li>Downloads: policy checkpoints (and ONNX), the G1 motion as CSV</li>
            </ul>
            {health && !health.runners.includes("colab") && (
              <div className="notice" style={{ marginTop: "auto" }}>
                <Icon name="cube" />
                <div><strong>Hosted demo.</strong> This deployment has no GPU, so jobs replay a recorded A100 run
                  with its real logs and training curves on a ~{Math.round(health.demo_seconds)}-second clock. A full run takes about 3 hours.</div>
              </div>
            )}
          </div>
        </div>
      </section>

      <section className="section" id="gallery">
        <div className="section-head">
          <div>
            <div className="eyebrow">Gallery</div>
            <h2 style={{ marginTop: 6 }}>Finished runs</h2>
            <p>Each was trained on a Colab A100 from ordinary video clips. Open one for the full results.</p>
          </div>
        </div>
        {err && <div className="error-box">Couldn't load runs: {err}</div>}
        {offline && !runs && (
          <div className="notice"><Icon name="film" /><div>Finished runs are listed by the pipeline API, which isn't
            reachable. The <Link to="/examples" style={{ color: "var(--accent)" }}>examples</Link> show the same runs' results.</div></div>
        )}
        {!runs && !err && !offline && <div className="gallery">{[0, 1].map((i) => <div key={i} className="skeleton" style={{ height: 380 }} />)}</div>}
        {runs && runs.length === 0 && <div className="notice">No finished runs found in the runs folder yet.</div>}
        {runs && runs.length > 0 && <div className="gallery">{runs.map((r) => <RunCard key={r.id} run={r} />)}</div>}
      </section>
    </div>
  );
}

function RunCard({ run }: { run: RunSummary }) {
  const nav = useNavigate();
  const [busy, setBusy] = useState(false);
  const replay = async () => {
    setBusy(true);
    try {
      const job = await api.createJob({ runId: run.id });
      nav(`/jobs/${job.id}`);
    } catch (e) {
      alert((e as Error).message);
      setBusy(false);
    }
  };
  const h = run.headline;
  return (
    <div className="card run-card">
      <Link to={`/runs/${run.id}`} className="media">
        {run.hero && <LoopVideo src={run.hero.url} poster={run.poster?.url} label={`${run.title} clip`} />}
      </Link>
      <div className="body">
        <div className="title-row">
          <h3><Link to={`/runs/${run.id}`}>{run.title}</Link></h3>
          <span className="faint" style={{ fontSize: 12.5 }}>{fmt.date(run.finished)}</span>
        </div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {run.policies_trained.map((p) => (
            <span className="pill" key={p}><i className="dot" style={{ background: policyColor(p) }} />{p}</span>
          ))}
          {run.has_robust && <span className="pill accent">robustness</span>}
        </div>
        <div className="stats">
          <div className="stat"><div className="k">V-JEPA aligned</div><div className="v">{fmt.num(h.best_aligned)}</div></div>
          <div className="stat"><div className="k">Retarget ceiling</div><div className="v">{fmt.num(h.retarget_aligned)}</div></div>
          <div className="stat"><div className="k">Falls</div><div className="v">{h.falls ?? "—"}</div></div>
        </div>
        <div className="actions">
          <Link className="btn sm" to={`/runs/${run.id}`}>Results <Icon name="arrow" size={14} /></Link>
          <button className="btn sm ghost" onClick={replay} disabled={busy}><Icon name="play" size={14} />Replay pipeline</button>
        </div>
      </div>
    </div>
  );
}

function UploadPanel({ offline }: { offline: boolean }) {
  const { health } = useHealth();
  const nav = useNavigate();
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [runner, setRunner] = useState("demo");
  const exts = health?.limits.extensions ?? [".mp4", ".mov", ".m4v", ".webm"];
  const maxMb = health?.limits.max_upload_mb ?? 100;
  const runners = (health?.runners ?? []).filter((r) => r !== "live"); // live builds have their own page

  const pick = (f: File | undefined) => {
    setErr(null);
    if (!f) return;
    const bad = checkVideo(f, health);
    if (bad) return setErr(bad);
    if (preview) URL.revokeObjectURL(preview);
    setFile(f);
    setPreview(URL.createObjectURL(f));
  };
  const submit = async () => {
    if (!file) return;
    setBusy(true);
    setErr(null);
    try {
      const job = await api.createJob({ file, runner });
      nav(`/jobs/${job.id}`);
    } catch (e) {
      setErr((e as Error).message);
      setBusy(false);
    }
  };

  return (
    <div className="card card-pad">
      <div
        className={`dropzone ${drag ? "drag" : ""}`}
        onClick={() => input.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0]); }}
        role="button" tabIndex={0}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
      >
        <div className="icon"><Icon name="upload" size={22} /></div>
        <h3>Drop a video, or click to choose</h3>
        <p>{exts.join(" ")} · up to {maxMb} MB · one person, full body, 5–20 s works best</p>
        <input ref={input} type="file" accept={exts.join(",") + ",video/*"} hidden onChange={(e) => pick(e.target.files?.[0])} />
      </div>
      {file && preview && (
        <div className="file-chip fade-in">
          <video src={preview} muted playsInline autoPlay loop />
          <div style={{ minWidth: 0, flex: 1 }}>
            <div style={{ fontWeight: 550, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{file.name}</div>
            <div className="faint mono" style={{ fontSize: 12 }}>{fmt.bytes(file.size)}</div>
          </div>
          <button className="btn ghost sm" onClick={() => { setFile(null); setPreview(null); }} aria-label="Remove file"><Icon name="x" size={14} /></button>
        </div>
      )}
      {err && <div className="error-box" style={{ marginTop: 14 }}>{err}</div>}
      <div className="upload-actions">
        <button className="btn primary" disabled={!file || busy || offline} onClick={submit}
          title={offline ? "The pipeline API is offline" : undefined}>
          <Icon name="play" size={16} />{busy ? "Uploading…" : "Run pipeline"}
        </button>
        {runners.length > 1 && (
          <select className="select" value={runner} onChange={(e) => setRunner(e.target.value)} aria-label="Runner">
            {runners.map((r) => <option key={r} value={r}>{r === "colab" ? "Colab A100 (real, ~3 h)" : "Demo replay"}</option>)}
          </select>
        )}
      </div>
    </div>
  );
}
