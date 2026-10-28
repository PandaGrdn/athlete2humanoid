import { createContext, useContext, useEffect, useState } from "react";
import { Link, NavLink, Route, Routes, useLocation } from "react-router";
import { api, type Health } from "./api";
import { Icon } from "./components/ui";
import { ExampleDetail, ExamplesIndex } from "./pages/Examples";
import Home from "./pages/Home";
import JobPage from "./pages/JobPage";
import LivePage, { LiveReplayPage } from "./pages/LivePage";
import RunPage from "./pages/RunPage";

const HealthCtx = createContext<{ health: Health | null; error: string | null }>({ health: null, error: null });
export const useHealth = () => useContext(HealthCtx);

const REPO_URL = "https://github.com/PandaGrdn/athlete2humanoid";

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { pathname, hash } = useLocation();

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
      <header className="nav">
        <div className="container nav-inner">
          <Link to="/" className="brand">
            <img src="/favicon.svg" alt="" />
            <span className="brand-text">athlete2humanoid</span>
          </Link>
          <nav className="nav-links">
            <NavLink to="/live" className="nav-ex">Live</NavLink>
            <NavLink to="/examples" className="nav-ex">Examples</NavLink>
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
              <Link to="/examples" className="pill warn" title={`The pipeline API isn't reachable (${error}). The examples work without it.`}>
                Offline demo
              </Link>
            )}
            <a className="btn ghost sm" href={REPO_URL} target="_blank" rel="noreferrer"><Icon name="github" size={16} />Code</a>
          </div>
        </div>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/live/demo/:demoId" element={<LiveReplayPage />} />
          <Route path="/live/:id?" element={<LivePage />} />
          <Route path="/jobs/:id" element={<JobPage />} />
          <Route path="/runs/:id" element={<RunPage />} />
          <Route path="/examples" element={<ExamplesIndex />} />
          <Route path="/examples/:id" element={<ExampleDetail />} />
          <Route path="*" element={<div className="container section"><h2>Not found</h2><p className="muted" style={{ marginTop: 8 }}><Link to="/">Back to the start</Link></p></div>} />
        </Routes>
      </main>
      <footer className="site">
        <div className="container">
          <span>Video → SAM 3.1 → MediaPipe → mink IK → V-JEPA 2 reward → PPO (mjlab) → Unitree G1</span>
          <span>Meta Global AI Developer Hackathon · Push the Frontier</span>
        </div>
      </footer>
    </HealthCtx.Provider>
  );
}
