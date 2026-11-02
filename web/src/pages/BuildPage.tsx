import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Link, useParams } from "react-router";
import type { RunDetail } from "../api";
import BuildView from "../buildview/BuildView";
import type { BuildInput } from "../buildview/data";
import { timedStatus, useRunInput } from "../buildview/hooks";
import { planReplay, replayAt, STAGE_COUNT, type ReplayPosition } from "../buildview/replay";
import { Typed } from "../components/pixel";
import { CreditLine, Icon } from "../components/ui";

// /build/:rid: a stored run's build replayed in 3D from its files, the way an upload's live build arrives, fast: the
// stages one after another, then the policy's recorded attempts (every ~100th iteration, a couple of seconds each),
// then the trained policy; 25-30 s in all (buildview/replay.ts; the run's web.json "replay" sets the pacing). Nothing
// is recomputed: every stage is that run's own output. The showcase's replay tabs use the same component.

/** The replay's clock and position for a loaded run: `pos` while it runs, null once it is over or stopped. */
export function useReplay(run: RunDetail | null, input: BuildInput, done: boolean) {
  const plan = useMemo(() => planReplay(input.attempts?.attempts ?? [], run?.replay ?? undefined, input.attempts?.fps ?? 50),
    [input.attempts, run?.replay]);
  const [t0, setT0] = useState<number | null>(null);
  const [now, setNow] = useState(0);
  const [stopped, setStopped] = useState(false);
  const started = useRef<string | null>(null);
  useEffect(() => {  // starts once the run (and its attempts list) is here; restart() runs it again
    if (!run || !done || started.current === run.id) return;
    started.current = run.id;
    const t = performance.now();
    setT0(t);
    setNow(t);
    setStopped(false);
  }, [run, done]);
  const elapsed = t0 == null ? -1 : (now - t0) / 1000;
  const over = t0 == null || elapsed > plan.seconds + 0.5;
  useEffect(() => {
    if (over || stopped) return;
    const id = window.setInterval(() => setNow(performance.now()), 100);
    return () => window.clearInterval(id);
  }, [over, stopped, t0]);
  const restart = useCallback(() => { const t = performance.now(); setT0(t); setNow(t); setStopped(false); }, []);
  const stop = useCallback(() => setStopped(true), []);
  const pos: ReplayPosition | null = t0 != null && !stopped && !over ? replayAt(plan, elapsed) : null;
  // the stages: the replay reveals them one by one; after it (or when stopped) everything that is here shows
  const reached = pos ? (pos.stagesReached >= STAGE_COUNT ? STAGE_COUNT + (pos.closing ? 1 : 0) : pos.stagesReached) : STAGE_COUNT + 1;
  const status = timedStatus(input, done, run ? reached * plan.stageStep + 1e-3 : -1, plan.stageStep);
  if (pos && pos.stagesReached >= STAGE_COUNT && !pos.closing && status.policy === "waiting") status.policy = "working";
  return { plan, pos, status, restart, stop, running: pos != null, elapsed };
}

/** A recorded build's replay: the build view driven by the run's replay plan, with a Replay button. */
export function BuildReplay({ rid, kicker, notes, head }: { rid: string; kicker: ReactNode; notes?: (run: RunDetail) => Record<string, ReactNode>; head?: (run: RunDetail | null, restart: () => void) => ReactNode }) {
  const { run, input, done, err } = useRunInput(rid);
  const { plan, pos, status, restart, stop, running } = useReplay(run, input, done);
  if (err) return <div className="error-box">Run {rid}: {err}</div>;
  return (
    <>
      {head?.(run, restart)}
      <BuildView input={input} status={status} follow kicker={<>{kicker}{running ? <> · replay {Math.round(plan.seconds)} s</> : null}</>}
        replay={pos} onReplayInteract={stop} notes={run && notes ? notes(run) : {}} />
    </>
  );
}

export default function BuildPage() {
  const { rid = "" } = useParams();
  return (
    <div className="container">
      <BuildReplay rid={rid} kicker={<>Recorded build · {rid}</>}
        head={(run, restart) => {
          const clip = (run?.target ?? rid).replace(/_/g, " ");
          return (
            <>
              <div className="job-head">
                <div style={{ flex: 1, minWidth: 240 }}>
                  <div className="eyebrow">Recorded build · {run?.title ?? rid}</div>
                  <h1 style={{ marginTop: 4 }}><Typed text={`Watch the ${clip} become a policy`} when="load" /></h1>
                </div>
                <button className="btn sm" onClick={restart}><Icon name="play" size={14} />Replay the build</button>
                {run && <Link className="btn sm ghost" to={`/runs/${run.id}`}>Results <Icon name="arrow" size={14} /></Link>}
              </div>
              <p className="muted" style={{ fontSize: 14, maxWidth: 820, margin: "0 0 14px" }}>
                Every stage of the build from this run's own files, then the robot learning attempt by attempt. Click a stage; drag to orbit.
              </p>
              <CreditLine credit={run?.credit} />
            </>
          );
        }}
        notes={(run) => ({ policy: run.attempts?.attempts.length
          ? <>{run.attempts.iterations?.toLocaleString() ?? ""} iterations on a GPU. <Link to={`/runs/${run.id}`}>Scores and videos</Link>.</>
          : <>Trained: {run.policies_trained.join(", ") || "none"}. <Link to={`/runs/${run.id}`}>Scores and videos</Link>.</> })} />
    </div>
  );
}
