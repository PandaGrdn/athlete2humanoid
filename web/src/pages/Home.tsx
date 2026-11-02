import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router";
import { api, checkVideo, fmt, policyColor, type RunSummary } from "../api";
import { useHealth, useUiMode } from "../App";
import { AetherHero } from "../components/aether";
import { AuroraHero, Figures, MethodTimeline, StackMarquee } from "../components/aurora";
import { MethodTable, PixelHero, Readout, Typed } from "../components/pixel";
import { Icon, LoopVideo } from "../components/ui";
import { exUrl } from "../examples";
import { useExampleIndex } from "./Examples";
import { startLiveBuild } from "./LivePage";

const PIPELINE = [
  { title: "Silhouette", text: "Segment and track the person.", model: "SAM 3.1" },
  { title: "Pose", text: "3D keypoints per frame.", model: "SAM 3D Body" },
  { title: "Retarget", text: "IK onto the 29-joint G1.", model: "mink IK" },
  { title: "Task environment", text: "Key moments and the reward.", model: "key poses" },
  { title: "Latent reward", text: "Does it look like the clip?", model: "V-JEPA 2" },
  { title: "Policy", text: "RL across 4,096 simulated G1s.", model: "PPO · mjlab" },
  { title: "Evaluate", text: "Score rollouts against the clip.", model: "V-JEPA 2 · DTW" },
];

const AETHER_HERO = "squat-jump-variants/policy_tracking_var.mp4";

const FIGURES = [
  { v: "7", k: "Stages", sub: "Video in, evaluated policy out, all automatic." },
  { v: "29", k: "Degrees of freedom", sub: "Every joint of the G1, driven by the network." },
  { v: "4,096", k: "Simulated robots", sub: "Learning in parallel on one GPU." },
  { v: "82%", k: "Held-out variants", sub: "Completed without ever training on them." },
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
  const ui = useUiMode();
  const aurora = ui === "aurora";
  const pixel = ui === "pixel";

  return (
    <>
    {pixel && <PixelHero latest={featured?.id} />}
    {aurora && <>
      <div className="container"><AuroraHero latest={featured?.id} /></div>
      <StackMarquee />
    </>}
    <div className="container">
      {/* aether: the robot alone at the centre of the halo, the best policy's own rollout (static, so it shows offline) */}
      {ui === "aether" && <AetherHero video={exUrl(AETHER_HERO)} poster={exUrl(AETHER_HERO.replace(/mp4$/, "jpg"))}
        title="The G1 squat-jumping: the policy trained on variants of one clip" to="/showcase" />}
      {ui === "classic" && <section className="hero">
        <div className="fade-in">
          <div className="models">
            <span className="pill accent">Meta V-JEPA 2</span>
            <span className="pill">SAM 3.1</span>
            <span className="pill">mjlab · MuJoCo Warp</span>
            <span className="pill">Unitree G1</span>
          </div>
          <h1>Any video in.<br /><em>A humanoid skill</em> out.</h1>
          <div className="cta" style={{ marginTop: 24 }}>
            <Link className="btn primary" to="/showcase"><Icon name="cube" size={16} />Push the robot, in 3D</Link>
            <a className="btn" href="#run"><Icon name="upload" size={16} />Run a video</a>
            {featured && <Link className="btn ghost" to={`/runs/${featured.id}`}>Latest run <Icon name="arrow" size={16} /></Link>}
          </div>
        </div>
        <div className="fade-in">
          {lead?.video ? (
            <Link to="/showcase" className="video-frame" style={{ display: "block" }}>
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
      </section>}

      <section className="section" id="examples">
        <ShowcaseTeaser more={exIdx.items?.filter((e) => e.status === "ready").length ?? null} />
      </section>

      </div>
      <div className="container">
      {aurora && <Figures title="One clip, a whole training run" items={FIGURES} />}
      {pixel && <Readout title="One clip, a whole training run" items={FIGURES} />}
      {pixel ? <MethodTable steps={PIPELINE} /> : aurora ? <MethodTimeline steps={PIPELINE} /> : <section className="section" id="how">
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
      </section>}

      <section className="section" id="run">
        <div className="section-head">
          <div>
            <div className="eyebrow">Run</div>
            <h2 style={{ marginTop: 6 }}><Typed text="Start a job" /></h2>
          </div>
        </div>
        {offline && (
          <div className="notice" style={{ marginBottom: 16 }}>
            <Icon name="cube" />
            <div><strong>The pipeline API is asleep.</strong> Uploads work once it answers; the{" "}
              <Link to="/showcase" style={{ color: "var(--accent)" }}>showcase</Link> works now.</div>
          </div>
        )}
        <Link to="/live" className="card live-cta">
          <span className="live-cta-ico"><Icon name="cube" size={20} /></span>
          <span style={{ flex: 1, minWidth: 0 }}>
            <strong>Watch it build live.</strong>{" "}
            <span className="muted">Your video becomes a robot and its task, stage by stage.</span>
          </span>
          <span className="btn sm primary">Open live demo <Icon name="arrow" size={14} /></span>
        </Link>
        <div className="upload-grid">
          <UploadPanel offline={offline} />
          <div className="card card-pad" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <div className="eyebrow">What a job produces</div>
            <ul style={{ margin: 0, paddingLeft: 18, color: "var(--text-2)", fontSize: 14, display: "grid", gap: 8 }}>
              <li>The clip retargeted onto the G1</li>
              <li>Three trained policies</li>
              <li>V-JEPA 2 and DTW scores, falls</li>
              <li>Robustness under pushes and noise</li>
              <li>Checkpoints, ONNX, motion CSV</li>
            </ul>
            {health && !health.runners.includes("colab") && (
              <div className="notice" style={{ marginTop: "auto" }}>
                <Icon name="cube" />
                <div><strong>Hosted demo.</strong> Jobs replay a recorded A100 run in about {Math.round(health.demo_seconds)} s.</div>
              </div>
            )}
          </div>
        </div>
      </section>

      <section className="section" id="gallery">
        <div className="section-head">
          <div>
            <div className="eyebrow">Gallery</div>
            <h2 style={{ marginTop: 6 }}><Typed text="Finished runs" /></h2>
          </div>
        </div>
        {err && <div className="error-box">Couldn't load runs: {err}</div>}
        {offline && !runs && (
          <div className="notice">The gallery needs the pipeline API, which is asleep. Results are
            on the <Link to="/research" style={{ color: "var(--accent)" }}>research page</Link>.</div>
        )}
        {!runs && !err && !offline && <div className="gallery">{[0, 1].map((i) => <div key={i} className="skeleton" style={{ height: 380 }} />)}</div>}
        {runs && runs.length === 0 && <div className="notice">No finished runs found in the runs folder yet.</div>}
        {runs && runs.length > 0 && <div className="gallery">{runs.map((r) => <RunCard key={r.id} run={r} />)}</div>}
      </section>
    </div>
    </>
  );
}

/** One example, big: the 3D showcase. The rest are on /research. */
function ShowcaseTeaser({ more }: { more: number | null }) {
  return (
    <div className="home-sc card">
      <Link to="/showcase" className="home-sc-media" aria-label="Open the 3D showcase">
        <LoopVideo src={exUrl(AETHER_HERO)} poster={exUrl(AETHER_HERO.replace(/mp4$/, "jpg"))} label="The trained squat-jump policy" />
        <span className="pill">Trained policy · mjlab</span>
      </Link>
      <div className="home-sc-body">
        <div className="eyebrow">Showcase</div>
        <h2><Typed text="Push the robot" /></h2>
        <p>The trained policy in 3D. Orbit it, then click it to shove it.</p>
        <div className="home-sc-cta">
          <Link className="btn primary" to="/showcase"><Icon name="cube" size={15} />Open the showcase</Link>
          <Link className="btn ghost" to="/research">All results{more ? ` (${more})` : ""} <Icon name="arrow" size={14} /></Link>
        </div>
      </div>
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
  const live = !!health?.runners.includes("live");

  const pick = (f: File | undefined) => {
    setErr(null);
    if (!f) return;
    const bad = checkVideo(f, health);
    if (bad) return setErr(bad);
    if (live) {  // straight to the live view: the clip, its pose, the G1 and the task, stage by stage
      setBusy(true);
      startLiveBuild(f, nav).catch((e) => { setErr((e as Error).message); setBusy(false); });
      return;
    }
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
        <h3>{busy && live ? "Uploading…" : "Drop a video, or click to choose"}</h3>
        <p>{exts.join(" ")} · up to {maxMb} MB · one person, full body, 5–20 s</p>
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
