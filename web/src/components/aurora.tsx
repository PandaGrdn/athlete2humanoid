import { useEffect, useRef, useState, type ReactNode } from "react";
import { Link, NavLink, useLocation } from "react-router";
import { useHealth } from "../App";
import { exUrl } from "../examples";
import { Icon, LoopVideo } from "./ui";

/** The aurora design's own pieces (theme.ts): a floating nav, the home page's hero, statement, figures and method
 *  timeline, and the footer. Everything else is the shared pages, restyled by styles-aurora.css. */

export const REPO_URL = "https://github.com/PandaGrdn/athlete2humanoid";

const LINKS = [
  { to: "/live", label: "Live" },
  { to: "/showcase", label: "Showcase" },
  { to: "/research", label: "Research" },
];

export function useApiStatus() {
  const { health, error } = useHealth();
  const label = health?.runners.includes("colab") ? "A100 online" : health ? "API online" : error ? "Offline demo" : "Connecting";
  const title = error && !health ? `The pipeline API isn't reachable (${error}). The showcase and research pages work without it.`
    : health && !health.runners.includes("colab") ? "This deployment has no GPU: jobs replay a recorded run" : undefined;
  return { label, title, live: !!health };
}

/** A pill that floats over the page: wordmark, links, the API's state and one call to action. A menu on phones. */
export function AuroraNav() {
  const [open, setOpen] = useState(false);
  const { pathname } = useLocation();
  const status = useApiStatus();
  useEffect(() => setOpen(false), [pathname]);
  useEffect(() => {
    if (!open) return;
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [open]);

  return (
    <header className={`au-nav ${open ? "open" : ""}`}>
      <div className="au-nav-bar">
        <Link to="/" className="au-brand" aria-label="athlete2humanoid, home">
          <img src="/favicon.svg" alt="" />
          <span>Athlete<b>2</b>Humanoid</span>
        </Link>
        <nav className="au-nav-links" aria-label="Main">
          {LINKS.map((l) => <NavLink key={l.to} to={l.to}>{l.label}</NavLink>)}
          <Link to="/#how">Method</Link>
        </nav>
        <span className="au-status" title={status.title}><i className={`dot ${status.live ? "live" : ""}`} />{status.label}</span>
        <a className="au-icon-link" href={REPO_URL} target="_blank" rel="noreferrer" aria-label="Code on GitHub"><Icon name="github" size={16} /></a>
        <Link className="btn primary sm au-nav-cta" to="/live">Run a video</Link>
        <button className="au-burger" aria-label={open ? "Close menu" : "Open menu"} aria-expanded={open} aria-controls="au-menu"
          onClick={() => setOpen((o) => !o)}>
          <span /><span />
        </button>
      </div>
      <div className="au-menu" id="au-menu" hidden={!open}>
        {LINKS.map((l) => <NavLink key={l.to} to={l.to}>{l.label}<Icon name="arrow" size={16} /></NavLink>)}
        <Link to="/#how">Method<Icon name="arrow" size={16} /></Link>
        <div className="au-menu-foot">
          <span className="au-status" title={status.title}><i className={`dot ${status.live ? "live" : ""}`} />{status.label}</span>
          <a className="btn sm" href={REPO_URL} target="_blank" rel="noreferrer"><Icon name="github" size={14} />Code</a>
          <Link className="btn primary sm" to="/live">Run a video</Link>
        </div>
      </div>
    </header>
  );
}

const SOURCE = "squat-jump-variants/source.mp4";
const POLICY = "squat-jump-variants/policy_tracking_var.mp4";
const poster = (v: string) => exUrl(v.replace(/mp4$/, "jpg"));

/** The opening: the claim on the left; on the right the phone clip and the policy it became, side by side. */
export function AuroraHero({ latest }: { latest?: string }) {
  return (
    <section className="au-hero">
      <div className="au-glow" aria-hidden="true"><i /><i /><i /></div>
      <div className="au-hero-copy">
        <div className="au-eyebrow">Video <span>→</span> policy · Unitree G1</div>
        <h1>Any video in.<br /><em>A humanoid skill</em> out.</h1>
        <p className="au-lede">
          Film a person moving. The pipeline retargets the motion to a Unitree G1 and trains a control policy whose
          reward comes from Meta's V-JEPA 2 video model, in simulation, under pushes.
        </p>
        <div className="au-cta">
          <Link className="btn primary" to="/showcase"><Icon name="cube" size={15} />Push the robot in 3D</Link>
          <Link className="btn" to="/live"><Icon name="upload" size={15} />Run your video</Link>
          {latest && <Link className="btn ghost" to={`/runs/${latest}`}>Latest run<Icon name="arrow" size={15} /></Link>}
        </div>
      </div>
      <div className="au-cluster">
        <figure className="au-shot source">
          <LoopVideo src={exUrl(SOURCE)} poster={poster(SOURCE)} label="The phone video: a person squat-jumping" />
          <figcaption><span>01</span>Phone video</figcaption>
        </figure>
        <span className="au-cluster-arrow" aria-hidden="true"><Icon name="arrow" size={18} /></span>
        <Link to="/showcase" className="au-shot policy" aria-label="Open the 3D showcase of this policy">
          <LoopVideo src={exUrl(POLICY)} poster={poster(POLICY)} label="The trained G1 policy performing the squat jump" />
          <figcaption><span>02</span>Trained policy · G1</figcaption>
        </Link>
      </div>
    </section>
  );
}

const STACK = ["SAM 3.1", "SAM 3D Body", "mink IK", "V-JEPA 2", "PPO", "mjlab", "MuJoCo Warp", "Unitree G1", "ONNX"];

/** The models the pipeline runs, scrolling past. */
export function StackMarquee() {
  const row = (hidden: boolean) => (
    <ul aria-hidden={hidden || undefined}>{STACK.map((s) => <li key={s}>{s}</li>)}</ul>
  );
  return (
    <div className="au-marquee" aria-label={`Built with ${STACK.join(", ")}`}>
      <div className="au-marquee-track">{row(false)}{row(true)}</div>
    </div>
  );
}

/** One idea in large type, its words lighting up as it scrolls into view. */
export function Statement({ eyebrow, text }: { eyebrow: string; text: string }) {
  return (
    <section className="au-band">
      <div className="container au-statement">
        <div className="au-eyebrow">{eyebrow}</div>
        <p>{text.split(" ").map((w, i) => <span key={i}>{w} </span>)}</p>
      </div>
    </section>
  );
}

/** Counts up to `value` (e.g. "4,096" or "78%") the first time it is on screen. */
function CountUp({ value }: { value: string }) {
  const m = value.match(/^([^\d]*)([\d,]+)(.*)$/);
  const target = m ? Number(m[2].replace(/,/g, "")) : NaN;
  const [shown, setShown] = useState(value);
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el || !m || !Number.isFinite(target) || typeof IntersectionObserver === "undefined") return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const fmt = (n: number) => `${m[1]}${m[2].includes(",") ? n.toLocaleString("en-US") : n}${m[3]}`;
    let raf = 0;
    const io = new IntersectionObserver(([e]) => {
      if (!e.isIntersecting) return;
      io.disconnect();
      const t0 = performance.now();
      const step = (t: number) => {
        const k = Math.min(1, (t - t0) / 1400);
        setShown(fmt(Math.round(target * (1 - (1 - k) ** 3))));
        if (k < 1) raf = requestAnimationFrame(step);
      };
      setShown(fmt(0));
      raf = requestAnimationFrame(step);
    }, { threshold: 0.4 });
    io.observe(el);
    return () => { io.disconnect(); cancelAnimationFrame(raf); };
  }, [value]);
  return <span ref={ref} aria-label={value}>{shown}</span>;
}

export function Figures({ title, lede, items }: { title: string; lede: ReactNode; items: { v: string; k: string; sub: string }[] }) {
  return (
    <section className="section au-figures">
      <div className="au-split-head">
        <h2>{title}</h2>
        <p>{lede}</p>
      </div>
      <div className="au-figure-grid">
        {items.map((f) => (
          <div className="card au-figure au-reveal" key={f.k}>
            <div className="v"><CountUp value={f.v} /></div>
            <div>
              <h3>{f.k}</h3>
              <p>{f.sub}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

/** The pipeline as a timeline: the heading stays put on the left while the stages scroll past on the right. */
export function MethodTimeline({ steps }: { steps: { title: string; text: string; model: string }[] }) {
  return (
    <section className="section au-method" id="how">
      <div className="au-method-head">
        <div className="au-eyebrow">Method</div>
        <h2>Seven stages from pixels to a policy</h2>
        <p>Every clip takes the same path. The first four stages run in about a minute; the reward, training and evaluation take a GPU a few hours.</p>
        <Link className="btn" to="/live">Watch one build live<Icon name="arrow" size={15} /></Link>
      </div>
      <ol className="au-steps">
        {steps.map((s, i) => (
          <li key={s.title} className="au-reveal">
            <span className="au-node" aria-hidden="true" />
            <div className="au-step-k">0{i + 1} · {s.model}</div>
            <h3>{s.title}</h3>
            <p>{s.text}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}

export function AuroraFooter({ switcher }: { switcher: ReactNode }) {
  return (
    <footer className="au-footer">
      <div className="container">
        <div className="au-footer-main">
          <div className="au-footer-brand">
            <Link to="/" className="au-brand"><img src="/favicon.svg" alt="" /><span>Athlete<b>2</b>Humanoid</span></Link>
            <p>One video of a person moving in, a Unitree G1 control policy out.</p>
          </div>
          <nav className="au-footer-col" aria-label="Explore">
            <span className="au-eyebrow">Explore</span>
            {LINKS.map((l) => <Link key={l.to} to={l.to}>{l.label}</Link>)}
            <Link to="/#how">Method</Link>
          </nav>
          <nav className="au-footer-col" aria-label="Project">
            <span className="au-eyebrow">Project</span>
            <a href={REPO_URL} target="_blank" rel="noreferrer">Code on GitHub</a>
            <Link to="/#run">Start a job</Link>
            <Link to="/#gallery">Finished runs</Link>
          </nav>
        </div>
        <div className="au-footer-base">
          <span>Video → SAM 3.1 → SAM 3D Body → mink IK → V-JEPA 2 reward → PPO (mjlab) → Unitree G1</span>
          <span>Meta Global AI Developer Hackathon · Push the Frontier</span>
          {switcher}
        </div>
      </div>
    </footer>
  );
}
