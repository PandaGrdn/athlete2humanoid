import { useEffect, useState } from "react";
import { Link, useParams } from "react-router";
import BuildView from "../buildview/BuildView";
import { timedStatus, useRunInput } from "../buildview/hooks";
import { Typed } from "../components/pixel";
import { Icon } from "../components/ui";

// /build/:rid: a stored run's build replayed in 3D from its files, one stage every few seconds, the way an
// upload's live build arrives. Nothing is recomputed: every stage is that run's own output.

const STEP_S = 2.6;

export default function BuildPage() {
  const { rid = "" } = useParams();
  const { run, input, done, err } = useRunInput(rid);
  const [t0, setT0] = useState(() => performance.now());
  const [now, setNow] = useState(t0);
  const elapsed = (now - t0) / 1000;
  const finished = elapsed > STEP_S * 7;
  useEffect(() => { if (run) { const t = performance.now(); setT0(t); setNow(t); } }, [run]);
  useEffect(() => {
    if (finished) return;
    const id = window.setInterval(() => setNow(performance.now()), 250);
    return () => window.clearInterval(id);
  }, [finished, t0]);
  const status = timedStatus(input, done, run ? elapsed : -1, STEP_S);
  const clip = (input.clip ?? run?.target ?? rid).replace(/_/g, " ");

  if (err) return <div className="container section"><div className="error-box">Run {rid}: {err}</div></div>;
  return (
    <div className="container">
      <div className="job-head">
        <div style={{ flex: 1, minWidth: 240 }}>
          <div className="eyebrow">Recorded build · {run?.title ?? rid}</div>
          <h1 style={{ marginTop: 4 }}><Typed text={`Watch the ${clip} become a policy`} when="load" /></h1>
        </div>
        <button className="btn sm" onClick={() => { const t = performance.now(); setT0(t); setNow(t); }}><Icon name="play" size={14} />Replay the build</button>
        {run && <Link className="btn sm ghost" to={`/runs/${run.id}`}>Results <Icon name="arrow" size={14} /></Link>}
      </div>
      <p className="muted" style={{ fontSize: 14, maxWidth: 820, margin: "0 0 14px" }}>
        Each stage of the pipeline, from this run's own files, shown in the order it ran: the video and its SAM 3.1 mask,
        the 3D skeleton, the G1 retargeted onto it, the key moments the reward is built from, the variants it trains on,
        and a trained policy you can push. Click a stage to look at it; drag to orbit.
      </p>
      <BuildView input={input} status={status} follow kicker={<>Recorded build · {run?.id ?? rid}</>}
        notes={{ policy: run ? <>This run trained: {run.policies_trained.join(", ") || "none"} (<Link to={`/runs/${run.id}`}>scores and videos</Link>). The 3D robot is the showcase recording with pushes.</> : null }} />
    </div>
  );
}
