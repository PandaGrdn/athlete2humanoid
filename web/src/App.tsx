import { createContext, lazy, Suspense, useContext, useEffect, useState } from "react";
import { Link, Navigate, NavLink, Route, Routes, useLocation, useParams } from "react-router";
import { api, type Health } from "./api";
import { UiSwitch } from "./components/aether";
import { AuroraFooter, AuroraNav } from "./components/aurora";
import { Icon } from "./components/ui";
import { ExampleDetail, ExamplesIndex } from "./pages/Examples";
import Home from "./pages/Home";
import JobPage from "./pages/JobPage";
import LivePage, { LiveReplayPage } from "./pages/LivePage";
import RunPage from "./pages/RunPage";
import { useUi, type Ui } from "./theme";

// three.js only loads with the showcase
const Showcase = lazy(() => import("./pages/Showcase"));
const ShowcaseFallback = <div className="sc-stage-skeleton skeleton" style={{ height: "min(calc(100svh - 68px), 900px)" }} />;

/** The old /examples/:id links: the examples now live under /research. */
function OldExample() {
  const { id = "" } = useParams();
  return <Navigate to={`/research/${id}`} replace />;
}

const HealthCtx = createContext<{ health: Health | null; error: string | null }>({ health: null, error: null });
export const useHealth = () => useContext(HealthCtx);
const UiCtx = createContext<Ui>("classic");
export const useUiMode = () => useContext(UiCtx);

const REPO_URL = "https://github.com/PandaGrdn/athlete2humanoid";

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { pathname, hash } = useLocation();
  const [ui, setUi] = useUi();

  // Without the API the site is an offline demo (the examples are static); keep checking so uploads
  // come back as soon as a sleeping API wakes up.
  useEffect(() => {
    let timer: number | undefined;
    let stopped = false;
    const check = () => api.health()
      .then((h) => { if (!stopped) { setHealth(h); setError(null); } })
      .catch((e) => {
        if (stopped) return;
        setError(String(e.message ?? e));
        timer = window.setTimeout(check, 30_000);
      });
    check();
    return () => { stopped = true; window.clearTimeout(timer); };
  }, []);
  useEffect(() => {
    if (hash) document.getElementById(hash.slice(1))?.scrollIntoView({ behavior: "smooth" });
    else window.scrollTo(0, 0);
  }, [pathname, hash]);

  return (
    <HealthCtx.Provider value={{ health, error }}>
      <UiCtx.Provider value={ui}>
      {ui === "aurora" ? <AuroraNav /> : ui === "aether" ? <AetherNav /> : <header className="nav">
        <div className="container nav-inner">
          <Link to="/" className="brand">
            <img src="/favicon.svg" alt="" />
            <span className="brand-text">athlete2humanoid</span>
          </Link>
          <nav className="nav-links">
            <NavLink to="/live" className="nav-ex">Live</NavLink>
            <NavLink to="/showcase" className="nav-ex">Showcase</NavLink>
            <NavLink to="/research" className="nav-ex">Research</NavLink>
            <Link to="/#run">Run</Link>
            <Link to="/#gallery">Gallery</Link>
            <Link to="/#how">How it works</Link>
          </nav>
          <div className="nav-right">
            {health && !health.runners.includes("colab") && (
              <span className="pill" title="This deployment has no GPU: jobs replay a recorded run">Demo runner</span>
            )}
            {health?.runners.includes("colab") && <span className="pill accent">Colab A100 enabled</span>}
            {error && !health && (
              <Link to="/showcase" className="pill warn" title={`The pipeline API isn't reachable (${error}). The showcase and research pages work without it.`}>
                Offline demo
              </Link>
            )}
            <a className="btn ghost sm" href={REPO_URL} target="_blank" rel="noreferrer"><Icon name="github" size={16} />Code</a>
          </div>
        </div>
      </header>}
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/live/demo/:demoId" element={<LiveReplayPage />} />
          <Route path="/live/:id?" element={<LivePage />} />
          <Route path="/jobs/:id" element={<JobPage />} />
          <Route path="/runs/:id" element={<RunPage />} />
          <Route path="/showcase/:clip?" element={<Suspense fallback={ShowcaseFallback}><Showcase /></Suspense>} />
          <Route path="/research" element={<ExamplesIndex />} />
          <Route path="/research/:id" element={<ExampleDetail />} />
          <Route path="/examples" element={<Navigate to="/research" replace />} />
          <Route path="/examples/:id" element={<OldExample />} />
          <Route path="*" element={<div className="container section"><h2>Not found</h2><p className="muted" style={{ marginTop: 8 }}><Link to="/">Back to the start</Link></p></div>} />
        </Routes>
      </main>
      {ui === "aurora" ? <AuroraFooter switcher={<UiSwitch ui={ui} setUi={setUi} />} /> : <footer className="site">
        <div className="container">
          <span>Video → SAM 3.1 → SAM 3D Body → mink IK → V-JEPA 2 reward → PPO (mjlab) → Unitree G1</span>
          <span>Meta Global AI Developer Hackathon · Push the Frontier</span>
          <UiSwitch ui={ui} setUi={setUi} />
        </div>
      </footer>}
      </UiCtx.Provider>
    </HealthCtx.Provider>
  );
}

/** The aether design's header: a stacked wordmark at the centre, the links either side (after Hermes Agent). */
function AetherNav() {
  const { health, error } = useHealth();
  const status = health?.runners.includes("colab") ? "A100 online" : health ? "Demo runner" : error ? "Offline demo" : "Connecting";
  return (
    <header className="nav ae-nav">
      <div className="container ae-nav-inner">
        <nav className="ae-links">
          <NavLink to="/live">Live</NavLink>
          <NavLink to="/showcase">Showcase</NavLink>
          <NavLink to="/research">Research</NavLink>
        </nav>
        <Link to="/" className="ae-brand" aria-label="athlete2humanoid, home">
          <span>Athlete</span><span>2 Humanoid</span>
        </Link>
        <nav className="ae-links right">
          <Link to="/#how">Method</Link>
          <span className="ae-status mono" title={error && !health ? `The pipeline API isn't reachable (${error}). The showcase and research pages work without it.` : undefined}>
            <i className={`dot ${health ? "live" : ""}`} />{status}
          </span>
          <a href={REPO_URL} target="_blank" rel="noreferrer" aria-label="Code on GitHub"><Icon name="github" size={15} /></a>
        </nav>
      </div>
    </header>
  );
}
