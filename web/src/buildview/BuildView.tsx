import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactNode } from "react";
import { EnvironmentPanel } from "../components/environment";
import { PoseOverlay } from "../components/live";
import { Icon } from "../components/ui";
import { loadIndex, loadRobot, loadShowcase, type ClipEntry, type Robot } from "../showcase/data";
import type { PolicyActor, PushState } from "../showcase/scene";
import { pickPolicy, STAGES, type BuildInput, type PolicyPick, type StageKey, type StageStatus } from "./data";
import { BuildScene } from "./scene";
import "./build.css";

// The 3D build view: a clip going through the pipeline, one stage at a time, on one floor (buildview/scene.ts).
// The page decides each stage's status (a live job's events, a demo job's replay, or a stored run's timed
// reveal); this draws whatever has arrived. The video is the clock for the skeleton, the retarget and the key
// moments; the policy runs on its own and takes pushes, as on the showcase.

export type BuildViewProps = {
  input: BuildInput;
  status: Record<StageKey, StageStatus>;
  kicker: ReactNode;
  /** Follow the newest stage with the camera until the visitor picks one. */
  follow?: boolean;
  /** Extra words under a stage's caption (e.g. where its data came from). */
  notes?: Partial<Record<StageKey, ReactNode>>;
  /** The clip's video element (the clock), for the page's own synced panels. */
  onVideo?: (v: HTMLVideoElement | null) => void;
};

const STRENGTHS = [{ mps: 0.5, label: "Nudge" }, { mps: 1, label: "Shove" }];
const COMPASS = ["front", "front-left", "left", "back-left", "back", "back-right", "right", "front-right"];
const STATUS_WORD: Record<StageStatus, string> = { waiting: "waiting", working: "working", ready: "ready", offer: "ready", missing: "n/a" };

/** The video's clock for React, ~12 times a second. */
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

function useRobot() {
  const [clips, setClips] = useState<ClipEntry[] | null>(null);
  const [robot, setRobot] = useState<Robot | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    loadIndex().then(async (i) => {
      if (!live) return;
      setClips(i.clips);
      if (i.clips[0]) { const r = await loadRobot(i.clips[0].id); if (live) setRobot(r); }
    }).catch((e) => live && setErr(String(e.message ?? e)));
    return () => { live = false; };
  }, []);
  return { clips, robot, err };
}

export default function BuildView({ input, status, kicker, follow = false, notes = {}, onVideo }: BuildViewProps) {
  const { clips, robot, err } = useRobot();
  const canvas = useRef<HTMLCanvasElement>(null);
  const [video, setVideo] = useState<HTMLVideoElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  videoRef.current = video;
  const [scene, setScene] = useState<BuildScene | null>(null);
  const [glErr, setGlErr] = useState(false);
  const [focus, setFocus] = useState<StageKey | null>(null);
  const [picked, setPicked] = useState(false); // the visitor chose a stage: the camera stops following
  const [actor, setActor] = useState<PolicyActor | null>(null);
  const [accepted, setAccepted] = useState(false); // took the offered pre-trained policy
  const [mps, setMps] = useState(STRENGTHS[0].mps);
  const [policyErr, setPolicyErr] = useState<string | null>(null);
  const [hint, setHint] = useState<string | null>(null);
  const [crop, setCrop] = useState<number | null>(null); // side-by-side videos: the share of the width that is the person
  const [aspect, setAspect] = useState<number | null>(null);
  const [paused, setPaused] = useState(false);
  const t = useVideoTime(video);
  const ready = (k: StageKey) => status[k] === "ready";
  useEffect(() => { onVideo?.(video); }, [video, onVideo]);

  useEffect(() => {
    if (!robot || !canvas.current) return;
    let s: BuildScene;
    try {
      s = new BuildScene(canvas.current, robot, () => videoRef.current?.currentTime ?? 0);
    } catch {
      setGlErr(true);
      return;
    }
    setScene(s);
    return () => { s.dispose(); setScene(null); setActor(null); };
  }, [robot]);

  // each stage's actor goes in once its stage is ready and its data is here
  useEffect(() => { scene?.setFps(input.fps); }, [scene, input.fps]);
  const sk = ready("skeleton") ? input.skeleton : null;
  useEffect(() => { scene?.setSkeleton(sk, input.poseModel); }, [scene, sk, input.poseModel]);
  const q = ready("retarget") ? input.qpos : null;
  useEffect(() => { scene?.setRetarget(q); }, [scene, q]);
  const variants = ready("variants") ? input.variants : null;
  useEffect(() => { scene?.setVariants(variants); }, [scene, variants]);
  const spec = ready("moments") ? input.spec : null;
  useEffect(() => {
    scene?.setMoments(spec ? spec.stages.filter((s) => s.kind === "pose").map((s) => s.at_s) : []);
  }, [scene, spec]);

  const pick: PolicyPick | null = useMemo(() => (clips ? pickPolicy(clips, input.clip, input.spec) : null), [clips, input.clip, input.spec]);
  const wantPolicy = status.policy === "ready" || (status.policy === "offer" && accepted);
  useEffect(() => {
    if (!scene || !pick || !wantPolicy) return;
    let live = true;
    setPolicyErr(null);
    loadShowcase(pick.id).then((d) => {
      if (!live) return;
      const a = scene.setPolicy(d, mps, pick.own ? `policy · trained on this clip` : `pre-trained policy · ${pick.tab}`);
      if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) a?.setPlaying(false);
      setActor(a);
    }).catch((e) => live && setPolicyErr(String(e.message ?? e)));
    return () => { live = false; scene.setPolicy(null, mps, ""); setActor(null); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scene, pick?.id, wantPolicy]);
  useEffect(() => { actor?.setStrength(mps); }, [actor, mps]);
  useEffect(() => {
    scene?.onClickPolicy((hit) => { if (!hit) setHint("Click the robot on the right to push it."); });
    return () => scene?.onClickPolicy(null);
  }, [scene]);
  useEffect(() => {
    if (!hint) return;
    const id = window.setTimeout(() => setHint(null), 2800);
    return () => window.clearTimeout(id);
  }, [hint]);

  // the camera follows the newest stage until the visitor picks one
  const newest = [...STAGES].reverse().find((s) => status[s.key] === "ready" || status[s.key] === "offer")?.key ?? null;
  const shown: StageKey | null = picked ? focus : newest;
  useEffect(() => {
    if (!scene || !follow || picked || !newest) return;
    scene.focus(newest === "video" ? "all" : newest === "policy" && !actor ? "all" : newest);
  }, [scene, follow, picked, newest, actor]);
  const choose = (k: StageKey) => {
    setPicked(true);
    setFocus(k);
    scene?.focus(k === "video" ? "all" : k);
  };

  const seek = useCallback((s: number) => {
    if (video) video.currentTime = Math.max(0, Math.min(s, (video.duration || s + 1) - 0.02));
  }, [video]);

  const vsrc = input.video?.src ?? null;
  const c1 = input.video?.sideBySide ? crop : 1;
  const videoReady = status.video === "ready" || status.video === "working";

  return (
    <section className="bv" aria-label="The pipeline, stage by stage, in 3D">
      <ol className="bv-rail" aria-label="Pipeline stages">
        {STAGES.map((s, i) => (
          <li key={s.key}>
            <button type="button" className={`bv-chip ${status[s.key]} ${shown === s.key ? "on" : ""}`}
              onClick={() => choose(s.key)} disabled={status[s.key] === "waiting"} aria-pressed={shown === s.key}>
              <span className="bv-n mono">{String(i + 1).padStart(2, "0")}</span>
              <span className="bv-l">{s.short}</span>
              <span className="bv-s mono">{s.key === "policy" && status.policy === "offer" ? "pre-trained" : STATUS_WORD[status[s.key]]}</span>
            </button>
          </li>
        ))}
      </ol>

      <div className="bv-stage">
        <div className="bv-view">
        {!glErr && <canvas ref={canvas} className="bv-canvas"
          aria-label="The skeleton, the retargeted G1, its variants and the policy on one floor. Drag to orbit, pinch or ctrl-scroll to zoom, click the policy robot to push it." />}
        {glErr && <div className="bv-empty"><div className="notice">This view needs WebGL, which this browser has turned off.</div></div>}
        {!robot && !err && !glErr && <div className="bv-empty mono"><span className="dot live" />Loading the G1</div>}
        {err && <div className="bv-empty"><div className="error-box">Couldn't load the G1: {err}</div></div>}

        <div className="bv-kicker mono">{kicker}</div>

        {vsrc && videoReady && (
          <div className="bv-media">
            <figure className="bv-pip">
              <div className="bv-crop" style={{ aspectRatio: aspect ? String(aspect * (c1 ?? 1)) : "9 / 16" }}>
                <video key={vsrc} ref={setVideo} src={vsrc} muted loop playsInline autoPlay preload="auto"
                  onPlay={() => setPaused(false)} onPause={() => setPaused(true)}
                  style={c1 && c1 < 1 ? { width: `${100 / c1}%`, maxWidth: "none" } : undefined}
                  onLoadedMetadata={(e) => {
                    const v = e.currentTarget;
                    if (!v.videoWidth) return;
                    setAspect(v.videoWidth / v.videoHeight);
                    if (input.video?.sideBySide) setCrop(Math.max(0.2, Math.min(1, (v.videoWidth - v.videoHeight) / v.videoWidth)));
                  }} aria-label={input.video?.label ?? "The clip"} />
                {input.landmarks && !input.video?.sideBySide && video && <PoseOverlay video={video} data={{ fps: input.fps, frames: 0, duration_s: 0, video: null, landmarks: input.landmarks, pose_found: null, robot: null, retarget_error_cm: null, timings_s: {} }} />}
              </div>
              <figcaption className="mono">{input.video?.label ?? "Clip"}{input.landmarks && !input.video?.sideBySide ? " · 2D landmarks" : ""}</figcaption>
            </figure>
            {input.mask && status.video === "ready" && <MaskVideo src={input.mask.src} master={video} />}
          </div>
        )}

        {spec && <MomentTag spec={spec} t={t} />}

        <div className="bv-play">
          {video && (
            <button className="btn sm" onClick={() => (video.paused ? video.play().catch(() => undefined) : video.pause())} aria-label="Play or pause the clip">
              <Icon name={paused ? "play" : "pause"} size={13} /><span className="mono">{t.toFixed(1)} s</span>
            </button>
          )}
          <span className="bv-hint mono">drag to orbit · ⌘/ctrl + scroll to zoom</span>
        </div>
        </div>

        {/* over the view on a wide screen, under it on a phone */}
        <div className="bv-hud">
        <Caption k={shown} input={input} status={status} pick={pick} accepted={accepted} note={shown ? notes[shown] : null} />

        {status.policy === "offer" && !accepted && pick && (
          <div className="bv-offer">
            <p><strong>No GPU on this host:</strong> the build stops at the environment.</p>
            <button className="btn sm primary" onClick={() => { setAccepted(true); choose("policy"); }}>
              <Icon name="play" size={14} />Play a pre-trained policy ({pick.tab.toLowerCase()})
            </button>
          </div>
        )}
        {policyErr && <div className="bv-offer"><div className="error-box">Couldn't load the policy: {policyErr}</div></div>}
        {actor && <PolicyControls actor={actor} mps={mps} setMps={setMps} onPush={() => scene?.pushFromView()} pick={pick} hint={hint} />}
        </div>
      </div>

      {(spec || variants) && (
        <div className="bv-below">
          {spec && <EnvironmentPanel spec={spec} t={t} onSeek={seek} />}
          {variants && <VariantList v={variants} />}
        </div>
      )}
    </section>
  );
}

/** SAM 3.1's mask video, kept on the main video's clock. */
function MaskVideo({ src, master }: { src: string; master: HTMLVideoElement | null }) {
  const ref = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    let raf = 0;
    const loop = () => {
      const el = ref.current;
      if (el && master) {
        if (Math.abs(el.currentTime - master.currentTime) > 0.12) el.currentTime = master.currentTime;
        if (master.paused !== el.paused) (master.paused ? el.pause() : el.play().catch(() => undefined));
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [master]);
  return (
    <figure className="bv-pip bv-mask">
      <div className="bv-crop" style={{ aspectRatio: "1" }}>
        <video ref={ref} src={src} muted playsInline preload="auto" aria-label="SAM 3.1's mask of the athlete" />
      </div>
      <figcaption className="mono">SAM 3.1 mask</figcaption>
    </figure>
  );
}

/** The key moment the clip just passed (Muse's name when the spec has one), for a moment after it passes. */
function MomentTag({ spec, t }: { spec: NonNullable<BuildInput["spec"]>; t: number }) {
  const poses = spec.stages.filter((s) => s.kind === "pose");
  const i = poses.findIndex((s) => t >= s.at_s - 0.05 && t < s.at_s + 0.9);
  if (i < 0) return null;
  const s = poses[i];
  return (
    <div className="bv-moment" key={i} aria-hidden="true">
      <span className="mono">Key moment {i + 1}/{poses.length}{s.optional ? " · optional" : ""}</span>
      <strong>{s.name}</strong>
    </div>
  );
}

function Caption({ k, input, status, pick, accepted, note }: { k: StageKey | null; input: BuildInput; status: BuildViewProps["status"]; pick: PolicyPick | null; accepted: boolean; note: ReactNode }) {
  if (!k) return <div className="bv-caption"><span className="bv-cap-k mono">Build</span><p>Waiting for the clip.</p></div>;
  const n = STAGES.findIndex((s) => s.key === k) + 1;
  const st = status[k];
  const v = input.variants;
  const spec = input.spec;
  const text: Record<StageKey, ReactNode> = {
    video: input.segmenter
      ? <>{input.segmenter} segments and tracks the athlete; the pose model sees only the masked person.</>
      : <>This CPU build skips {"SAM 3.1"}: the pose model reads the whole frame, so film one person.</>,
    skeleton: <>{input.poseModel ?? "The pose model"} lifts the athlete to 3D: {input.skeleton?.j ?? 17} joints in metres, feet on the floor, in step with the video.</>,
    retarget: <>mink IK puts the motion on the G1's 29 joints and bone lengths. The lines join each wrist, elbow, knee and ankle to the body it drives{input.retargetCm != null ? <>; mean end-effector error {input.retargetCm.toFixed(1)} cm</> : null}.</>,
    moments: spec ? <>{spec.counts.moments} key moments{spec.counts.flights ? ` and ${spec.counts.flights} flight${spec.counts.flights > 1 ? "s" : ""}` : ""}, found in the motion itself{spec.named_by ? <>, named by {spec.named_by}</> : null}. The reward pays for reaching them in order; a ring pulses as each passes.</> : null,
    variants: v ? <>{v.kept} of {v.total} variants kept: mirrored, re-timed and rescaled copies, each passing its own key-moment check and the physics filter{v.trained_on ? "; the stored policies trained on these" : v.trained_on === false ? " (generated from this retarget; its stored policy trained on the clip alone)" : ""}. Ghosts: {v.ghosts.map((g) => g.label).join(", ") || "none"}.</> : null,
    policy: pick ? (status.policy === "offer" && !accepted
      ? <>Training a policy on this clip takes PPO in mjlab 1–2 h on a GPU, and this host has none. You can play the nearest stored policy instead ({pick.tab.toLowerCase()}: {pick.why.replace("the nearest stored policy: ", "")}), labelled pre-trained.</>
      : pick.own
      ? <>The policy trained on this clip, rolled out in MuJoCo and recorded. Click it, or press Push, to shove it: you see its recorded recovery.</>
      : <>Pre-trained, not trained on this clip: {pick.tab.toLowerCase()}, {pick.why}. Click it or press Push to shove it.</>) : null,
  };
  const waiting: Record<StageKey, string> = {
    video: "Checking the video…", skeleton: "Finding the pose…", retarget: "Retargeting onto the G1…", moments: "Compiling the key moments…",
    variants: "Generating variants…", policy: "Training the policy…",
  };
  return (
    <div className="bv-caption" aria-live="polite">
      <span className="bv-cap-k mono">{String(n).padStart(2, "0")} · {STAGES[n - 1].label}</span>
      <p>{st === "ready" || st === "offer" ? text[k] : st === "missing" ? "Not in this build." : waiting[k]}</p>
      {note && <p className="bv-note">{note}</p>}
    </div>
  );
}

function pushText(p: PushState) {
  const dir = COMPASS[((Math.round(p.angle / (Math.PI / 4)) % 8) + 8) % 8];
  const what = `${p.mps} m/s toward the ${dir}`;
  if (p.status === "queued") return `Push: ${what}…`;
  if (p.status === "live") return p.fall != null ? `Recorded response: ${what}, this one falls` : `Recorded response: ${what}`;
  if (p.status === "fell") return "Fell: restarting the clip";
  const back = p.rejoin != null ? ((p.rejoin - p.step) / 50).toFixed(1) : null;
  return back ? `Recovered: back on its run ${back} s after the push` : "Recovered";
}

function PolicyControls({ actor, mps, setMps, onPush, pick, hint }: {
  actor: PolicyActor; mps: number; setMps: (m: number) => void; onPush: () => void; pick: PolicyPick | null; hint: string | null;
}) {
  const s = useSyncExternalStore(actor.subscribe, actor.getState);
  const msg = hint ?? s.note ?? (s.push ? pushText(s.push) : null);
  return (
    <div className="bv-policy">
      {pick && <span className={`bv-badge mono ${pick.own ? "" : "pre"}`}>{pick.own ? "trained on this clip" : "pre-trained"}</span>}
      <span className="bv-seg" role="group" aria-label="Push strength">
        {STRENGTHS.map((x) => (
          <button key={x.mps} className={mps === x.mps ? "on" : ""} aria-pressed={mps === x.mps} onClick={() => setMps(x.mps)}>
            {x.label} <span className="mono">{x.mps}</span>
          </button>
        ))}
      </span>
      <button className="btn sm primary" onClick={onPush}>Push</button>
      {msg && <span className="bv-push mono" role="status">{msg}</span>}
    </div>
  );
}

function VariantList({ v }: { v: NonNullable<BuildInput["variants"]> }) {
  const dropped = v.rows.filter((r) => !r.ok);
  const why = new Map<string, number>();
  dropped.forEach((r) => (r.rules.length ? r.rules : ["other"]).forEach((x) => why.set(x, (why.get(x) ?? 0) + 1)));
  return (
    <div className="card card-pad bv-variants">
      <div className="eyebrow">Variants · {v.kept} of {v.total} kept</div>
      <p className="muted" style={{ fontSize: 13.5 }}>
        Every variant is defined on the G1's joints and the clip's own contacts, never on the action: mirrored, played
        at 0.75–1.33× tempo, scaled to 0.85–1.1× amplitude, some with a time warp. Flights keep the airtime gravity
        sets. A variant is dropped if its own reference fails its key-moment check or breaks the physics filter.
        Cells: tempo, amplitude; <span className="mono">mir</span> mirrored, <span className="mono">~</span> time-warped, dashed dropped.
      </p>
      {why.size > 0 && (
        <p className="bv-why">{[...why].map(([k, n]) => <span key={k} className="env-tag">{n} dropped: {k}</span>)}</p>
      )}
      <div className="bv-grid" role="list">
        {v.rows.map((r) => (
          <span key={r.name} role="listitem" className={`bv-cell ${r.ok ? "ok" : "no"} ${r.params.mirror ? "m" : ""}`}
            title={`${r.name}: ${r.params.mirror ? "mirrored, " : ""}${r.params.tempo}x tempo, ${r.params.amplitude}x amplitude${r.params.warp ? `, warp ${r.params.warp}` : ""} — ${r.ok ? "kept" : `dropped: ${r.why}`}`}>
            <span className="mono">{r.params.mirror ? "mir " : ""}{r.params.tempo}×</span>
            <span className="mono faint">amp {r.params.amplitude}{r.params.warp ? " ~" : ""}</span>
          </span>
        ))}
      </div>
      {v.source && <p className="faint" style={{ fontSize: 12 }}>{v.source === "generated" ? "Generated by skill/variations.py from this run's retarget." : `From ${v.source}${v.policies?.length ? `, which trained ${v.policies.join(", ")}` : ""}.`}</p>}
    </div>
  );
}
