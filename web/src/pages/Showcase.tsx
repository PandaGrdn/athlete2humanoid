import { useEffect, useMemo, useRef, useState, useSyncExternalStore, type PointerEvent } from "react";
import { Link, useNavigate, useParams } from "react-router";
import { Icon } from "../components/ui";
import { Typed } from "../components/pixel";
import { examples, type Example, type Stage } from "../examples";
import { BASE, loadIndex, loadShowcase, type ClipEntry, type Loaded } from "../showcase/data";
import { ShowcasePlayer, type PlayerState, type PushState } from "../showcase/scene";
import "../showcase/showcase.css";

// Trained policies in 3D, one clip at a time (public/showcase/index.json; /showcase/<id>), each next to the
// video it came from. A click on the robot plays the policy's recorded response to a push there
// (scripts/build_showcase.py).

const STRENGTHS = [{ mps: 0.5, label: "Nudge" }, { mps: 1, label: "Shove" }];
const COMPASS = ["front", "front-left", "left", "back-left", "back", "back-right", "right", "front-right"];
const pct = (x: number) => `${Math.round(x * 100)}%`;

function useShowcase(id: string | null) {
  const [data, setData] = useState<Loaded | null>(null);
  const [ex, setEx] = useState<Example | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    if (!id) return;
    let live = true;
    setData(null);
    setEx(null);
    setErr(null);
    loadShowcase(id).then((d) => {
      if (!live) return;
      setData(d);
      if (!d.show.stages) examples.get(d.show.example).then((e) => live && setEx(e)).catch(() => undefined);
    }).catch((e) => live && setErr(String(e.message ?? e)));
    return () => { live = false; };
  }, [id]);
  return { data, ex, err };
}

function useClips() {
  const [clips, setClips] = useState<ClipEntry[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { loadIndex().then((i) => setClips(i.clips)).catch((e) => setErr(String(e.message ?? e))); }, []);
  return { clips, err };
}

export default function Showcase() {
  const { clip } = useParams();
  const nav = useNavigate();
  const { clips, err: idxErr } = useClips();
  const entry = clips?.find((c) => c.id === clip) ?? clips?.[0] ?? null;
  const { data, ex, err: loadErr } = useShowcase(entry?.id ?? null);
  const err = idxErr ?? loadErr;
  const switcher = useRef<HTMLElement>(null);
  useEffect(() => {  // on a phone the tab row scrolls: keep the open clip's tab in view
    const tab = switcher.current?.querySelector<HTMLElement>(".on");
    const row = switcher.current;
    if (tab && row) row.scrollTo({ left: tab.offsetLeft - (row.clientWidth - tab.clientWidth) / 2 });
  }, [entry?.id]);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [player, setPlayer] = useState<ShowcasePlayer | null>(null);
  const [glErr, setGlErr] = useState(false);
  const [mps, setMps] = useState(STRENGTHS[0].mps);

  useEffect(() => {
    if (!data || !canvas.current) return;
    let p: ShowcasePlayer;
    try {
      p = new ShowcasePlayer(canvas.current, data, mps);
    } catch {
      setGlErr(true);
      return;
    }
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) p.setPlaying(false);
    setPlayer(p);
    return () => { p.dispose(); setPlayer(null); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);
  useEffect(() => { player?.setStrength(mps); }, [player, mps]);

  const show = data?.show;
  const fellRate = show ? show.pushes.fall_rate : null;
  const nPushes = show ? show.pushes.moments.reduce((n, m) => n + m.pushes.length, 0) : 0;
  const stages = show?.stages ?? ex?.stages ?? [];

  return (
    <div className="sc">
      <section className="sc-stage" aria-label="The trained G1 policy in 3D">
        <div className="sc-glow" aria-hidden="true" />
        {!glErr && <canvas ref={canvas} className="sc-canvas" aria-label={`The Unitree G1 performing the ${entry?.tab.toLowerCase() ?? "clip"}. Drag to orbit, pinch or ctrl-scroll to zoom, click the robot to push it.`} />}
        {glErr && entry?.fallback && (
          <video className="sc-fallback" src={`${BASE}${entry.fallback}`} autoPlay muted loop playsInline />
        )}
        {glErr && !entry?.fallback && <div className="sc-loading"><div className="notice">This view needs WebGL, which this browser has turned off.</div></div>}
        {!data && !err && <div className="sc-loading mono"><span className="dot live" />Loading the G1 and its recordings</div>}
        {err && <div className="sc-loading"><div className="error-box">Couldn't load the showcase: {err}</div></div>}

        {entry && (
          <div className="sc-head" key={entry.id}>
            <div className="sc-kicker mono">Showcase · {entry.kicker}</div>
            <h1><Typed text={entry.title.join("\n")} when="load" /></h1>
            {entry.caveat && <div className="sc-caveat mono">{entry.caveat}</div>}
            <p className="sc-lede">{entry.lede}</p>
          </div>
        )}
        {clips && clips.length > 1 && (
          <nav ref={switcher} className="sc-switch" aria-label="Showcased clips">
            {clips.map((c) => (
              <button key={c.id} className={c.id === entry?.id ? "on" : ""} aria-current={c.id === entry?.id ? "page" : undefined}
                onClick={() => nav(`/showcase/${c.id}`)}>{c.tab}</button>
            ))}
          </nav>
        )}

        {player && <PushChip player={player} />}
        {player && <Hint />}
        {data && player && <SourceVideo data={data} player={player} />}
      </section>

      {player && data && (
        <Controls player={player} stages={stages} duration={data.show.frames / data.show.fps} mps={mps} setMps={setMps} />
      )}

      <div className="container">
        <section className="sc-steps">
          <div className="sc-step">
            <span className="sc-n mono">01</span>
            <h3>Video</h3>
            <p>{entry?.video}</p>
            {entry?.credit && (
              <p className="muted" style={{ fontSize: 13 }}>
                Source: <a href={entry.credit.url} target="_blank" rel="noopener noreferrer">{entry.credit.text}</a> ({entry.credit.licence})
              </p>
            )}
          </div>
          <div className="sc-step">
            <span className="sc-n mono">02</span>
            <h3>Retarget</h3>
            <p>Inverse kinematics puts the motion on the G1's 29 joints and bone lengths. Turn on <em>Retarget</em> to see it
              as a ghost: the reference the policy tracks.</p>
          </div>
          <div className="sc-step">
            <span className="sc-n mono">03</span>
            <h3>Policy</h3>
            <p>{entry?.policy}</p>
          </div>
        </section>

        <section className="sc-facts">
          {entry?.facts.map((f) => <div className="sc-fact" key={f.l}><span className="v">{f.v}</span><span className="l">{f.l}</span></div>)}
          <div className="sc-fact"><span className="v">{show ? show.clean.falls : 0}</span><span className="l">falls on the clip</span></div>
          {fellRate && Object.entries(fellRate).map(([v, r]) => (
            <div className="sc-fact" key={v}><span className="v">{pct(1 - r)}</span><span className="l">stay up after a {v} m/s push</span></div>
          ))}
        </section>

        <section className="sc-honest card card-pad">
          <div className="eyebrow">What you're watching</div>
          <p>
            Recorded responses of the trained policy, replayed in your browser: nothing is simulated live. The robot's
            pose comes from the policy's own rollout in mjlab (checkpoint <span className="mono">{show?.pushes.checkpoint ?? "model_1499.pt"}</span>,
            50 Hz), drawn with the G1's meshes from MuJoCo Menagerie.
          </p>
          {show && (() => {
            const p = show.pushes;
            const gap = (p.moments[1]?.step - p.moments[0]?.step) / show.fps || 0.25;
            const lead = (gap + 6 / show.fps).toFixed(2);
            return (
              <>
                <p>
                  Pushes: {nPushes} rollouts of the same policy, each shoved once ({p.kind}) at one of {p.moments.length} moments
                  {" "}{gap} s apart, from one of {p.directions} directions, at {p.strengths_mps.join(" or ")} m/s. A click plays the
                  recording for the next moment (up to {lead} s later), from the recorded direction nearest the one you pushed.
                  The arrow shows that direction.
                </p>
                <p>
                  Joins: parallel runs of one policy drift apart a little through the jumps (float noise; median
                  {" "}{p.drift.median.toFixed(2)}, at most {p.drift.max.toFixed(2)} in qpos before their push), so each recording starts
                  {" "}{(p.pre_frames / show.fps).toFixed(1)} s before its push and the view crossfades into it. A recording ends at the
                  clip's end, {p.after_fall_s} s after a fall, or once the robot is back within {Math.round(p.rejoin.m * 100)} cm and
                  {" "}{Math.round((p.rejoin.rad * 180) / Math.PI)}° of every joint of the unpushed run, which {pct(p.rejoined)} of the upright
                  ones do; it then eases back onto that run. Most keep the ground they lost: the policy recovers its balance and the
                  motion, not its spot.
                </p>
              </>
            );
          })()}
          <div className="sc-links">
            {entry?.research && <Link className="btn sm" to={entry.research}>The numbers behind it <Icon name="arrow" size={14} /></Link>}
            <Link className="btn sm ghost" to="/research">More results <Icon name="arrow" size={14} /></Link>
          </div>
        </section>
      </div>
    </div>
  );
}

function usePlayer(player: ShowcasePlayer): PlayerState {
  return useSyncExternalStore(player.subscribe, player.getState);
}

/** The phone video, cropped to the person, kept on the robot's clock. */
function SourceVideo({ data, player }: { data: Loaded; player: ShowcasePlayer }) {
  const { t, playing, speed } = usePlayer(player);
  const ref = useRef<HTMLVideoElement>(null);
  const v = data.show.video;
  const [c0, c1] = v.crop;
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const want = t + v.offset_s;
    if (Math.abs(el.currentTime - want) > 0.2 || !playing) el.currentTime = Math.min(want, (el.duration || 99) - 0.01);
    el.playbackRate = speed;
    if (playing && el.paused) el.play().catch(() => undefined);
    if (!playing && !el.paused) el.pause();
  }, [t, playing, speed, v.offset_s]);
  return (
    <figure className="sc-source">
      <div className="sc-crop" style={{ aspectRatio: `${(c1 - c0) * v.size[0]} / ${v.size[1]}` }}>
        <video ref={ref} src={`${BASE}${v.src}`} poster={`${BASE}${v.poster}`} muted playsInline preload="auto"
          style={{ width: `${100 / (c1 - c0)}%`, marginLeft: `${(-c0 / (c1 - c0)) * 100}%` }} aria-label={`The source: ${v.label ?? "phone video"}`} />
      </div>
      <figcaption><span className="mono">Source</span> {v.label ?? "Phone video"}, in sync</figcaption>
    </figure>
  );
}

function Hint() {
  const [gone, setGone] = useState(false);
  useEffect(() => { const id = window.setTimeout(() => setGone(true), 9000); return () => window.clearTimeout(id); }, []);
  return (
    <div className={`sc-hint mono ${gone ? "gone" : ""}`} aria-hidden="true">
      <span>Drag to orbit</span><span>Pinch or ⌘/ctrl + scroll to zoom</span><span>Click the robot to push</span>
    </div>
  );
}

function pushText(p: PushState) {
  const dir = COMPASS[((Math.round(p.angle / (Math.PI / 4)) % 8) + 8) % 8];
  const what = `${p.mps} m/s push toward the ${dir}`;
  if (p.status === "queued") return { k: "Push", v: `${what}…` };
  if (p.status === "live") return { k: "Recorded response", v: p.fall != null ? `${what}: this one falls` : what };
  if (p.status === "fell") return { k: "Fell", v: "The policy lost this one. Restarting the clip." };
  const back = p.rejoin != null ? ((p.rejoin - p.step) / 50).toFixed(1) : null;
  return { k: "Recovered", v: back ? `Back on its run ${back} s after the push` : "Recovered" };
}

function PushChip({ player }: { player: ShowcasePlayer }) {
  const { push, note } = usePlayer(player);
  useEffect(() => {
    if (!note) return;
    const id = window.setTimeout(() => player.clearNote(), 3200);
    return () => window.clearTimeout(id);
  }, [note, player]);
  if (!push && !note) return null;
  const t = push ? pushText(push) : { k: "Note", v: note! };
  return (
    <div className={`sc-chip ${push?.status ?? "note"}`} role="status" aria-live="polite">
      <span className="k mono">{t.k}</span><span className="v">{t.v}</span>
    </div>
  );
}

function Controls({ player, stages, duration, mps, setMps }: {
  player: ShowcasePlayer; stages: Stage[]; duration: number; mps: number; setMps: (m: number) => void;
}) {
  const s = usePlayer(player);
  const track = useRef<HTMLDivElement>(null);
  const scrub = (e: PointerEvent<HTMLDivElement>) => {
    const r = track.current!.getBoundingClientRect();
    player.seek(Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * duration);
  };
  const stage = useMemo(() => stages.find((x) => s.t >= x.start_s && s.t < x.end_s) ?? null, [stages, s.t]);
  return (
    <div className="sc-controls">
      <div className="container sc-controls-inner">
        <button className="btn sm sc-play" onClick={() => player.setPlaying(!s.playing)} aria-label={s.playing ? "Pause" : "Play"}>
          <Icon name={s.playing ? "pause" : "play"} size={14} />
        </button>
        <div className="sc-time">
          <div className="sc-stage-name"><span className="mono faint">{s.t.toFixed(2)} s</span> {stage?.label ?? ""}</div>
          <div ref={track} className="sc-track" role="slider" aria-label="Clip time" aria-valuemin={0} aria-valuemax={duration}
            aria-valuenow={Math.round(s.t * 10) / 10} tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "ArrowRight") player.seek(s.t + 0.25);
              if (e.key === "ArrowLeft") player.seek(s.t - 0.25);
            }}
            onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); scrub(e); }}
            onPointerMove={(e) => { if (e.buttons) scrub(e); }}>
            {stages.map((x, i) => (
              <i key={i} className={`seg k-${x.kind}`} style={{ left: `${(x.start_s / duration) * 100}%`, width: `${((x.end_s - x.start_s) / duration) * 100}%` }} title={x.label} />
            ))}
            {s.push && <b className="mark" style={{ left: `${(s.push.step / 50 / duration) * 100}%` }} />}
            <span className="head" style={{ left: `${(s.t / duration) * 100}%` }} />
          </div>
        </div>
        <div className="sc-opts">
          <span className="sc-seg" role="group" aria-label="Push strength">
            {STRENGTHS.map((x) => (
              <button key={x.mps} className={mps === x.mps ? "on" : ""} aria-pressed={mps === x.mps} onClick={() => setMps(x.mps)}>
                {x.label} <span className="mono">{x.mps}</span>
              </button>
            ))}
          </span>
          <button className="btn sm" onClick={() => player.pushFromView()}>Push</button>
          <span className="sc-seg">
            <button className={s.ghost ? "on" : ""} aria-pressed={s.ghost} onClick={() => player.setGhost(!s.ghost)}>Retarget</button>
            <button className={s.speed !== 1 ? "on" : ""} aria-pressed={s.speed !== 1} onClick={() => player.setSpeed(s.speed === 1 ? 0.5 : 1)}>0.5×</button>
          </span>
        </div>
      </div>
    </div>
  );
}
