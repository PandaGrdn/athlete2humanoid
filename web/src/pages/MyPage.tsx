import { useEffect, useState, type FormEvent } from "react";
import { Link } from "react-router";
import { api, fmt, OWNER_KEY_RE, ownerKey, setOwnerKey, url, type MyJob } from "../api";
import { useHealth } from "../App";
import { Icon, StatusPill } from "../components/ui";
import { Typed } from "../components/pixel";

// /my: this visitor's environments and trainings, kept by the API's store (server/store.py) under the key the browser
// made on its first visit (localStorage, sent as X-A2H-Owner). A link with #key= restores the key on another device:
// a fragment, so the key never reaches the host's logs or a proxy, and it is taken out of the address at once.

/** The key a /my#key=... link carries, taken out of the address bar (history and logs never keep it). */
function takeKeyFromHash(): string | null {
  const m = /(?:^#|[#&])key=([a-f0-9]{32})\b/i.exec(window.location.hash);
  if (!m) return null;
  window.history.replaceState(null, "", window.location.pathname + window.location.search);
  return m[1].toLowerCase();
}

export default function MyPage() {
  const [restored, setRestored] = useState(0);
  useEffect(() => {
    const k = takeKeyFromHash();
    if (k && OWNER_KEY_RE.test(k)) { setOwnerKey(k); setRestored((n) => n + 1); }
  }, []);
  return <MyList key={restored} />;
}

function MyList() {
  const { health, error: apiError } = useHealth();
  const key = ownerKey();
  const [jobs, setJobs] = useState<MyJob[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [paste, setPaste] = useState("");
  const [pasteErr, setPasteErr] = useState<string | null>(null);
  const link = `${window.location.origin}/my#key=${key}`;

  useEffect(() => {
    if (!health) return;
    let live = true;
    api.mine().then((r) => live && setJobs(r.jobs)).catch((e) => live && setErr(e.message));
    return () => { live = false; };
  }, [health, key]);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(link);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch { /* the field below is selectable */ }
  };
  const useKey = (e: FormEvent) => {
    e.preventDefault();
    const k = paste.trim().toLowerCase().replace(/^.*[?&#]key=/, "");
    if (!OWNER_KEY_RE.test(k)) return setPasteErr("A key is 32 letters and digits, or the link that carries one.");
    setOwnerKey(k);
    window.location.assign("/my");
  };

  return (
    <div className="container">
      <div className="ex-intro">
        <div className="eyebrow">My environments</div>
        <h1><Typed text="Your builds and trainings" when="load" /></h1>
        <p className="lede">Everything this browser built and trained, kept for you. No account: a key identifies you.</p>
      </div>

      <div className="card card-pad" style={{ maxWidth: 820, marginBottom: 20 }}>
        <div className="eyebrow">Your key</div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center", marginTop: 8 }}>
          <input className="my-key mono" type="text" readOnly value={link} aria-label="A link that restores your key" onFocus={(e) => e.currentTarget.select()} />
          <button className="btn sm" onClick={copy}><Icon name="file" size={14} />{copied ? "Copied" : "Copy link"}</button>
        </div>
        <p className="faint" style={{ fontSize: 12.5, marginTop: 8 }}>Open this link on another device to see the same list there. Keep it to yourself.</p>
        <form onSubmit={useKey} style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
          <input className="my-key mono" type="text" value={paste} onChange={(e) => setPaste(e.target.value)} placeholder="Paste a key or a link" aria-label="A key or link from another device" />
          <button className="btn sm" type="submit" disabled={!paste.trim()}>Use this key</button>
        </form>
        {pasteErr && <div className="error-box" style={{ marginTop: 8 }}>{pasteErr}</div>}
      </div>

      {!health && apiError && <div className="notice"><Icon name="cube" /><div>The pipeline API isn't reachable, so the list can't load right now.</div></div>}
      {health && !health.store?.durable && <div className="notice"><Icon name="cube" /><div>This deployment keeps builds only until its API restarts.</div></div>}
      {err && <div className="error-box">{err}</div>}
      {jobs && jobs.length === 0 && (
        <div className="card card-pad" style={{ maxWidth: 820 }}>
          <p className="muted" style={{ fontSize: 14 }}>Nothing yet. <Link to="/live" style={{ color: "var(--accent)" }}>Upload a video</Link> and it lands here.</p>
        </div>
      )}
      {jobs && jobs.length > 0 && (
        <ul className="my-list" aria-label="Your environments">
          {jobs.map((j) => <li key={j.id}><MyCard j={j} /></li>)}
        </ul>
      )}
      {!jobs && !err && health && <div className="skeleton" style={{ height: 160, maxWidth: 820 }} />}
    </div>
  );
}

function MyCard({ j }: { j: MyJob }) {
  const t = j.training;
  const when = new Date(j.created_at * 1000);
  const trained = t?.state === "done";
  return (
    <div className="card card-pad my-card">
      <div className="my-head">
        <div>
          <h3>{j.filename ?? "Upload"}</h3>
          <p className="faint" style={{ fontSize: 12.5 }}>{when.toLocaleDateString()} {when.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}{j.duration_s ? ` · ${j.duration_s.toFixed(1)} s clip` : ""}</p>
        </div>
        <StatusPill status={j.status} />
      </div>
      {j.summary && <p className="muted" style={{ fontSize: 13.5 }}>{j.summary}</p>}
      <p className="muted" style={{ fontSize: 13 }}>
        {j.status === "succeeded" && !t ? "Environment built." : null}
        {t ? (trained ? `Policy trained: ${t.iterations.toLocaleString()} iterations, ${t.attempts} attempts recorded.`
          : t.state === "running" || t.state === "starting" ? `Training on a GPU: iteration ${t.iteration} of ${t.iterations}.` : `Training ${t.state}.`) : null}
        {j.status === "failed" ? ` ${j.error ?? "The build failed."}` : null}
        {!j.kept && j.status === "succeeded" ? " Its files have expired on this host." : null}
      </p>
      <div className="my-actions">
        <Link className="btn sm primary" to={`/live/${j.id}`}><Icon name="cube" size={14} />{trained ? "Watch it learn" : "Open the build"}</Link>
        {j.env && <a className="btn sm" href={url(j.env.url)} download={j.env.filename}><Icon name="download" size={14} />Environment{j.env.size ? ` (${fmt.bytes(j.env.size)})` : ""}</a>}
        {trained && t?.checkpoint && <a className="btn sm" href={url(api.policy(j.id))} download><Icon name="download" size={14} />Policy</a>}
      </div>
    </div>
  );
}
