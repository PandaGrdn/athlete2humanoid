import { Link } from "react-router";
import { Icon, LoopVideo } from "./ui";

/** The aether design's own pieces (theme.ts). Everything else is the shared pages, restyled by styles-aether.css. */

// Deterministic rays (no Math.random: the same halo on every render and in every browser).
function rays(n: number) {
  let s = 7;
  const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  return Array.from({ length: n }, (_, i) => {
    const a = (i / n) * Math.PI * 2 + rnd() * 0.02;
    const r0 = 168 + rnd() * 10;
    const r1 = r0 + 30 + rnd() ** 2.2 * 150;
    return { x0: Math.cos(a) * r0, y0: Math.sin(a) * r0, x1: Math.cos(a) * r1, y1: Math.sin(a) * r1, o: 0.12 + rnd() * 0.5 };
  });
}
const RAYS = rays(240);

/** Radiating hairlines around a circle, after Hermes Agent's engraved figure: the robot at the centre. */
export function Halo({ children }: { children?: React.ReactNode }) {
  return (
    <div className="ae-halo">
      <svg viewBox="-340 -340 680 680" aria-hidden="true" className="ae-rays">
        <circle r="164" className="ae-ring" />
        <circle r="300" className="ae-ring faint" />
        <circle r="326" className="ae-ring faint dashed" />
        {RAYS.map((r, i) => <line key={i} x1={r.x0} y1={r.y0} x2={r.x1} y2={r.y1} strokeOpacity={r.o} />)}
      </svg>
      <div className="ae-core">{children}</div>
    </div>
  );
}

export function AetherHero({ video, poster, title, to }: { video?: string; poster?: string; title?: string; to?: string }) {
  return (
    <section className="ae-hero">
      <div className="ae-hero-copy fade-in">
        <div className="ae-kicker">Video <span>→</span> policy · Unitree G1</div>
        <h1>Watch once.<br />Move forever.</h1>
        <p className="ae-lede">
          One clip of a person moving. The pipeline sees it, understands it in V-JEPA 2's latent space, and
          trains a humanoid to perform it: in simulation, under pushes, at any tempo.
        </p>
        <div className="ae-cta">
          <Link className="btn primary" to="/examples"><Icon name="play" size={14} />See the examples</Link>
          <a className="btn" href="#run"><Icon name="upload" size={14} />Run a video</a>
        </div>
        <div className="ae-stack mono">
          <span>SAM 3.1</span><span>MediaPipe</span><span>mink IK</span><span>V-JEPA 2</span><span>PPO · mjlab</span>
        </div>
      </div>
      <div className="fade-in">
        <Halo>
          {video ? (
            to ? <Link to={to} aria-label={title}><LoopVideo src={video} poster={poster} label={title} /></Link>
              : <LoopVideo src={video} poster={poster} label={title} />
          ) : <div className="skeleton" style={{ width: "100%", height: "100%" }} />}
        </Halo>
      </div>
    </section>
  );
}

export function UiSwitch({ ui, setUi }: { ui: string; setUi: (ui: "classic" | "aether") => void }) {
  return (
    <span className="ui-switch" role="group" aria-label="Interface">
      <span className="faint">Interface</span>
      <button className={ui === "aether" ? "on" : ""} onClick={() => setUi("aether")} aria-pressed={ui === "aether"}>Aether</button>
      <button className={ui === "classic" ? "on" : ""} onClick={() => setUi("classic")} aria-pressed={ui === "classic"}>Classic</button>
    </span>
  );
}
