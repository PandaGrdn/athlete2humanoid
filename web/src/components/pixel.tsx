import { useEffect, useRef, useState, type ReactNode } from "react";
import { Link, NavLink, useLocation } from "react-router";
import { useUiMode } from "../theme";
import { REPO_URL, useApiStatus } from "./aurora";
import { GlyphField } from "./glyphfield";
import { Icon, PixelIcon } from "./ui";

/** The pixel design's own pieces (theme.ts): the nav, the home page's glyph-field hero, the readout, the method
 *  table and the footer, plus <Typed>, which types a heading out once. Everything else is the shared pages,
 *  restyled by styles-pixel.css. */

/** Doto's full stop is a small cross; in typed headings it is drawn as one square pixel instead (styles-pixel.css). */
const stops = (s: string) => s.split(/(\.)/).map((part, i) => (part === "." ? <i key={i} className="px-stop">.</i> : part));

const reduced = () => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Types `text` out behind a block caret, once: on load, or when it first scrolls into view. The full text stays
 *  in the layout (hidden) so nothing reflows, and screen readers get it whole. Plain text in the other designs. */
export function Typed({ text, when = "view", delay = 0, caret = false }: { text: string; when?: "load" | "view"; delay?: number; caret?: boolean }) {
  const ui = useUiMode();
  const ref = useRef<HTMLSpanElement>(null);
  const [n, setN] = useState(() => (reduced() ? text.length : 0));
  useEffect(() => {
    const el = ref.current;
    if (ui !== "pixel" || !el || reduced()) return;
    setN(0);
    let timer = 0;
    let io: IntersectionObserver | undefined;
    const run = () => {
      let i = 0;
      const step = () => {
        i++;
        setN(i);
        if (i < text.length) timer = window.setTimeout(step, /[.,:\n]/.test(text[i - 1]) ? 200 : text[i - 1] === " " ? 46 : 30);
      };
      timer = window.setTimeout(step, delay);
    };
    if (when === "load" || typeof IntersectionObserver === "undefined") run();
    else {
      io = new IntersectionObserver(([e]) => { if (e.isIntersecting) { io?.disconnect(); run(); } }, { threshold: 0.6 });
      io.observe(el);
    }
    return () => { window.clearTimeout(timer); io?.disconnect(); };
  }, [text, ui, when, delay]);

  if (ui !== "pixel") return <>{text}</>;
  const done = n >= text.length;
  return (
    <span ref={ref} className={`px-typed ${done ? "done" : ""} ${caret ? "keep" : ""}`}>
      <span className="px-sr">{text}</span>
      <span className="px-typed-ghost" aria-hidden="true">{stops(text)}</span>
      <span className="px-typed-live" aria-hidden="true">{stops(text.slice(0, n))}<i className="px-caret" /></span>
    </span>
  );
}

const LINKS = [
  { to: "/live", label: "Live" },
  { to: "/showcase", label: "Showcase" },
  { to: "/research", label: "Research" },
];

function Brand() {
  return (
    <Link to="/" className="px-brand" aria-label="athlete2humanoid, home">
      <span className="px-brand-mark"><PixelIcon name="mark" size={22} /></span>
      <span>athlete2humanoid</span>
    </Link>
  );
}

export function PixelNav() {
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
  const led = <span className="px-status" title={status.title}><i className={`px-led ${status.live ? "on" : ""}`} />{status.label}</span>;

  return (
    <header className={`px-nav ${open ? "open" : ""}`}>
      <div className="container px-nav-bar">
        <Brand />
        <nav className="px-links" aria-label="Main">
          {LINKS.map((l) => <NavLink key={l.to} to={l.to}>{l.label}</NavLink>)}
          <Link to="/#how">Method</Link>
        </nav>
        {led}
        <a className="px-icon-link" href={REPO_URL} target="_blank" rel="noreferrer" aria-label="Code on GitHub"><Icon name="github" size={18} /></a>
        <Link className="btn primary sm px-nav-cta" to="/live">Run a video</Link>
        <button className="px-burger" aria-label={open ? "Close menu" : "Open menu"} aria-expanded={open} aria-controls="px-menu"
          onClick={() => setOpen((o) => !o)}>
          <PixelIcon name={open ? "x" : "menu"} size={18} />
        </button>
      </div>
      <div className="px-menu" id="px-menu" hidden={!open}>
        <div className="container">
          {LINKS.map((l) => <NavLink key={l.to} to={l.to}>{l.label}<Icon name="arrow" size={16} /></NavLink>)}
          <Link to="/#how">Method<Icon name="arrow" size={16} /></Link>
          <div className="px-menu-foot">
            {led}
            <a className="btn sm" href={REPO_URL} target="_blank" rel="noreferrer"><Icon name="github" size={14} />Code</a>
          </div>
        </div>
      </div>
    </header>
  );
}

const SPEC = [
  { k: "Input", v: "One video of one person, full body, 5 to 20 seconds" },
  { k: "Reward", v: "Meta V-JEPA 2, comparing the clip with renders of the robot" },
  { k: "Training", v: "PPO across 4,096 simulated G1s in mjlab, on one GPU" },
  { k: "Output", v: "A control policy for the 29-joint Unitree G1, with ONNX" },
];

/** The opening: the claim, typed; the spec as a key-value sheet; and, drawn in the glyph field, the recorded clip
 *  as the person's keypoints next to the G1 they became. */
export function PixelHero({ latest }: { latest?: string }) {
  const stage = useRef<HTMLDivElement>(null);
  const counter = useRef<HTMLSpanElement>(null);
  return (
    <section className="px-hero">
      <GlyphField stage={stage} counter={counter} />
      <div className="container px-hero-grid">
        <div className="px-hero-copy">
          <h1><Typed text={"Any video in.\nA humanoid skill out."} when="load" delay={300} caret /></h1>
          <p className="px-lede">
            Film a person moving. The pipeline retargets the motion to a Unitree G1 and trains a control policy in
            simulation, rewarded for looking like your clip.
          </p>
          <div className="px-cta">
            <Link className="btn primary" to="/showcase"><Icon name="cube" size={16} />Push the robot in 3D</Link>
            <Link className="btn" to="/live"><Icon name="upload" size={16} />Run your video</Link>
            {latest && <Link className="btn ghost" to={`/runs/${latest}`}>Latest run<Icon name="arrow" size={16} /></Link>}
          </div>
          <dl className="px-spec">
            {SPEC.map((s) => <div key={s.k}><dt>{s.k}</dt><dd>{s.v}</dd></div>)}
          </dl>
        </div>
        <div className="px-stage-wrap">
          <div ref={stage} className="px-stage" role="img"
            aria-label="A recorded squat jump: the person's keypoints from the video, and the Unitree G1 retargeted from them, moving in step" />
          <div className="px-stage-cap">
            <span><i className="px-key dot" />Person, 17 keypoints from the video</span>
            <span><i className="px-key sq" />Unitree G1, retargeted</span>
            <span className="px-frame mono">Squat jump, frame <span ref={counter}>001/270</span></span>
          </div>
        </div>
      </div>
    </section>
  );
}

/** One idea, set large on a black band. */
export function PixelStatement({ text }: { text: string }) {
  return (
    <section className="px-band">
      <div className="container"><p className="px-statement">{text.split(/(V-JEPA \d)/).map((s, i) => (i % 2 ? <span key={i} style={{ whiteSpace: "nowrap" }}>{s}</span> : s))}</p></div>
    </section>
  );
}

/** The numbers behind a run, as an instrument readout: label, dotted leader, value. */
export function Readout({ title, lede, items }: { title: string; lede: ReactNode; items: { v: string; k: string; sub: string }[] }) {
  return (
    <section className="section px-readout">
      <div className="px-head">
        <h2><Typed text={title} /></h2>
        <p>{lede}</p>
      </div>
      <dl className="px-readout-list">
        {items.map((f) => (
          <div className="px-readout-row" key={f.k}>
            <dt>{f.k}</dt>
            <span className="px-leader" aria-hidden="true" />
            <dd>{f.v}</dd>
            <p>{f.sub}</p>
          </div>
        ))}
      </dl>
    </section>
  );
}

/** The pipeline, in order, as a numbered table. */
export function MethodTable({ steps }: { steps: { title: string; text: string; model: string }[] }) {
  return (
    <section className="section px-method" id="how">
      <div className="px-head">
        <h2><Typed text="Seven stages from pixels to a policy" /></h2>
        <p>Every clip takes the same path. The first four stages run in about a minute; the reward, training and evaluation take a GPU a few hours.</p>
      </div>
      <ol className="px-steps">
        {steps.map((s, i) => (
          <li key={s.title}>
            <span className="px-n">{String(i + 1).padStart(2, "0")}</span>
            <h3>{s.title}</h3>
            <p>{s.text}</p>
            <span className="px-model">{s.model}</span>
          </li>
        ))}
      </ol>
      <Link className="btn" to="/live">Watch one build live<Icon name="arrow" size={16} /></Link>
    </section>
  );
}

export function PixelFooter({ switcher }: { switcher: ReactNode }) {
  return (
    <footer className="px-footer">
      <div className="container">
        <div className="px-footer-main">
          <div className="px-footer-brand">
            <Brand />
            <p>One video of a person moving in, a Unitree G1 control policy out.</p>
          </div>
          <nav className="px-footer-col" aria-label="Explore">
            <span>Explore</span>
            {LINKS.map((l) => <Link key={l.to} to={l.to}>{l.label}</Link>)}
            <Link to="/#how">Method</Link>
          </nav>
          <nav className="px-footer-col" aria-label="Project">
            <span>Project</span>
            <a href={REPO_URL} target="_blank" rel="noreferrer">Code on GitHub</a>
            <Link to="/#run">Start a job</Link>
            <Link to="/#gallery">Finished runs</Link>
          </nav>
        </div>
        <div className="px-footer-base">
          <span>Video, SAM 3.1, SAM 3D Body, mink IK, V-JEPA 2 reward, PPO in mjlab, Unitree G1</span>
          <span>Meta Global AI Developer Hackathon: Push the Frontier</span>
          {switcher}
        </div>
      </div>
    </footer>
  );
}
