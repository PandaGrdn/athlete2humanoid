import { createContext, useContext, useEffect, useState } from "react";
import { Link, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { api, type Health } from "./api";
import { Icon } from "./components/ui";
import Home from "./pages/Home";
import JobPage from "./pages/JobPage";
import RunPage from "./pages/RunPage";

const HealthCtx = createContext<{ health: Health | null; error: string | null }>({ health: null, error: null });
export const useHealth = () => useContext(HealthCtx);

const REPO_URL = "https://github.com/PandaGrdn/athlete2humanoid";

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { pathname, hash } = useLocation();

  useEffect(() => {
    api.health().then(setHealth).catch((e) => setError(String(e.message ?? e)));
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
            athlete2humanoid
          </Link>
          <nav className="nav-links">
            <NavLink to="/#run" end>Run</NavLink>
            <NavLink to="/#gallery">Gallery</NavLink>
            <NavLink to="/#how">How it works</NavLink>
          </nav>
          <div className="nav-right">
            {health && !health.runners.includes("colab") && (
              <span className="pill" title="This deployment has no GPU: jobs replay a recorded run">Demo runner</span>
            )}
            {health?.runners.includes("colab") && <span className="pill accent">Colab A100 enabled</span>}
            {error && <span className="pill bad" title={error}>API offline</span>}
            <a className="btn ghost sm" href={REPO_URL} target="_blank" rel="noreferrer"><Icon name="github" size={16} />Code</a>
          </div>
        </div>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/jobs/:id" element={<JobPage />} />
          <Route path="/runs/:id" element={<RunPage />} />
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
