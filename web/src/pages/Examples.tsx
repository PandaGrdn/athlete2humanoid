import { useEffect, useMemo, useState, type PointerEvent } from "react";
import { Link, useParams } from "react-router";
import {
  examples, exUrl,
  type CompareRow, type ExVideo, type Example, type ExampleSummary, type ImportanceChart, type PolicyRow, type RobustTable,
} from "../examples";
import { useWidth } from "../components/charts";
import { SyncControls, stageAt, useSyncGroup, type SyncGroup } from "../components/synced";
import { Icon, LoopVideo } from "../components/ui";

// --- index ---------------------------------------------------------------------------------------

export function useExampleIndex() {
  const [items, setItems] = useState<ExampleSummary[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    examples.index().then(setItems).catch((e) => setErr(e.message));
  }, []);
  return { items, err };
}

export function ExampleCards({ items, err }: { items: ExampleSummary[] | null; err: string | null }) {
  if (err) return <div className="error-box">Couldn't load the examples: {err}</div>;
  if (!items) return <div className="ex-cards">{[0, 1, 2].map((i) => <div key={i} className="skeleton" style={{ height: 360 }} />)}</div>;
  return (
    <div className="ex-cards">
      {items.map((e, i) => <ExampleCard key={e.id} ex={e} n={i + 1} />)}
    </div>
  );
}

function ExampleCard({ ex, n }: { ex: ExampleSummary; n: number }) {
  const pending = ex.status !== "ready";
  const body = (
    <>
      <div className="media">
        {ex.video ? <LoopVideo src={exUrl(ex.video)} poster={ex.poster ? exUrl(ex.poster) : undefined} label={ex.title} />
          : ex.poster ? <img src={exUrl(ex.poster)} alt="" />
          : <div className="ex-pending-media"><span className="dot live" /> Training</div>}
        <span className="ex-num mono">{String(n).padStart(2, "0")}</span>
      </div>
      <div className="body">
        <div className="eyebrow">{ex.kicker}</div>
        <h3>{ex.title}</h3>
        <p>{ex.tagline}</p>
        {pending ? <span className="pill warn" style={{ alignSelf: "flex-start" }}><span className="dot live" style={{ background: "var(--warn)" }} />Results pending</span>
          : <span className="ex-open">Open the example <Icon name="arrow" size={14} /></span>}
      </div>
    </>
  );
  return pending
    ? <div className="card ex-card pending">{body}</div>
    : <Link to={`/research/${ex.id}`} className="card ex-card">{body}</Link>;
}

export function ExamplesIndex() {
  const idx = useExampleIndex();
  return (
    <div className="container">
      <section className="ex-intro">
        <div className="eyebrow">Research</div>
        <h1>What the robot was shown, what it learned, and how we know</h1>
        <p className="lede">
          Recorded results from the full pipeline: phone videos in, simulated Unitree G1 policies out. Every number
          is read from the runs' own evaluation files. These pages need no server, so they work even when the live
          pipeline is asleep.
        </p>
      </section>
      <div className="card ex-showcase">
        <p><strong>In 3D:</strong> the squat-jump policy next to its source video, with recorded responses to pushes you
          trigger by clicking it.</p>
        <Link className="btn sm" to="/showcase">Open the showcase <Icon name="arrow" size={14} /></Link>
      </div>
      <ExampleCards {...idx} />
    </div>
  );
}

// --- one example ---------------------------------------------------------------------------------

export function ExampleDetail() {
  const { id = "" } = useParams();
  const [ex, setEx] = useState<Example | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const { items } = useExampleIndex();
  useEffect(() => {
    setEx(null);
    setErr(null);
    examples.get(id).then(setEx).catch((e) => setErr(e.message));
  }, [id]);

  if (err) return <div className="container section"><div className="error-box">Example "{id}": {err}</div></div>;
  if (!ex) return <div className="container section"><div className="skeleton" style={{ height: 520 }} /></div>;

  const ready = items?.filter((e) => e.status === "ready") ?? [];
  const i = ready.findIndex((e) => e.id === id);
  const prev = i > 0 ? ready[i - 1] : null;
  const next = i >= 0 && i < ready.length - 1 ? ready[i + 1] : null;
  return (
    <div className="container">
      <div className="ex-crumbs"><Link to="/research">Research</Link><span className="faint">/</span><span>{ex.kicker}</span></div>
      <Story key={ex.id} ex={ex} />
      <nav className="ex-pager">
        {prev ? <Link className="card" to={`/research/${prev.id}`}><span className="faint">← Previous</span><strong>{prev.title}</strong></Link> : <span />}
        {next ? <Link className="card next" to={`/research/${next.id}`}><span className="faint">Next →</span><strong>{next.title}</strong></Link> : <span />}
      </nav>
    </div>
  );
}

function Story({ ex }: { ex: Example }) {
  const vids = [ex.primary, ...ex.grid, ...ex.extra].filter((v): v is ExVideo => !!v);
  const duration = Math.max(ex.duration_s ?? 0, ...vids.map((v) => v.duration_s));
  const group = useSyncGroup(duration);
  const hasPlayer = vids.length > 0;

  if (ex.status !== "ready") {
    return (
      <article className="story">
        <StoryHead ex={ex} />
        <div className="notice"><span className="dot live" style={{ background: "var(--warn)", marginTop: 6 }} />
          <div><strong>Results pending.</strong> {ex.story.learned} {ex.story.how}</div></div>
      </article>
    );
  }
  return (
    <article className="story">
      <StoryHead ex={ex} />
      {hasPlayer && (
        <div className="player card" ref={group.watch}>
          <div className={`player-media ${ex.primary && ex.grid.length ? "split" : ""} ${!ex.primary ? "grid-only" : ""}`}>
            {ex.primary && <VideoFrame v={ex.primary} group={group} k="primary" maxH={ex.grid.length ? undefined : ex.chart ? "38vh" : "60vh"} />}
            {ex.grid.length > 0 && (
              <div className="player-grid">
                {ex.grid.map((v) => (
                  <PolicyTile key={v.src} v={v} group={group} row={ex.policies?.find((p) => p.key === v.policy)} />
                ))}
                {ex.grid_caption && <p className="player-caption">{ex.grid_caption}</p>}
              </div>
            )}
          </div>
          {ex.chart && <ImportanceView chart={ex.chart} group={group} stages={ex.stages} />}
          <SyncControls group={group} stages={ex.stages} />
          {ex.extra.map((v) => (
            <div className="player-extra" key={v.src}>
              <div className="eyebrow">{v.label}</div>
              <VideoFrame v={v} group={group} k={v.src} />
            </div>
          ))}
        </div>
      )}

      <div className="story-strip">
        <StoryStep n="1" title="What it was shown" text={ex.story.shown} />
        <StoryStep n="2" title="What it learned" text={ex.story.learned} />
        <StoryStep n="3" title="How we know" text={ex.story.how} />
      </div>

      {ex.metrics.length > 0 && (
        <div className="ex-metrics">
          {ex.metrics.map((m) => (
            <div className={`card tile ${m.tone ?? ""}`} key={m.label}>
              <div className="k">{m.label}</div>
              <div className={`v ${m.display.length > 11 ? "long" : ""}`}>{m.display}{m.unit && <small>{m.unit}</small>}</div>
              {m.note && <div className="sub">{m.note}</div>}
            </div>
          ))}
        </div>
      )}

      {ex.policies && ex.columns && <PolicyTable rows={ex.policies} columns={ex.columns} />}
      {ex.compare && <CompareBars rows={ex.compare} labels={ex.compare_labels ?? { before: "Before", after: "After" }} />}
      {ex.robust && <RobustView table={ex.robust} />}

      <footer className="ex-sources">
        {ex.notes.map((n) => <p key={n}>{n}</p>)}
        <p>Source files: {ex.sources.map((s, i) => <span key={s}>{i > 0 && ", "}<code>{s}</code></span>)}</p>
      </footer>
    </article>
  );
}

function StoryHead({ ex }: { ex: Example }) {
  return (
    <header className="story-head">
      <div className="eyebrow">{ex.kicker}</div>
      <h1>{ex.title}</h1>
      <p className="lede">{ex.tagline}</p>
    </header>
  );
}

function StoryStep({ n, title, text }: { n: string; title: string; text: string }) {
  return (
    <div className="card story-step">
      <div className="n mono">{n}</div>
      <h3>{title}</h3>
      <p>{text}</p>
    </div>
  );
}

function VideoFrame({ v, group, k, maxH }: { v: ExVideo; group: SyncGroup; k: string; maxH?: string }) {
  // maxH caps the height (keeping the aspect) so a video plus its chart fit one screen.
  const width = maxH ? `min(100%, calc(${maxH} * ${(v.width / v.height).toFixed(4)}))` : undefined;
  return (
    <div className="ex-video" style={{ aspectRatio: `${v.width} / ${v.height}`, width, marginInline: maxH ? "auto" : undefined }}>
      <video ref={group.bind(k)} src={exUrl(v.src)} poster={exUrl(v.poster)} muted playsInline preload="auto" aria-label={v.label} />
      {v.panels?.map((p) => (
        <span key={p.label} className={`pill panel-tag ${p.tone === "good" ? "accent" : p.tone ?? ""}`}
          style={{ left: `calc(${(p.from * 100).toFixed(2)}% + 8px)` }}>{p.label}</span>
      ))}
    </div>
  );
}

function PolicyTile({ v, group, row }: { v: ExVideo; group: SyncGroup; row?: PolicyRow }) {
  const keyStats = row ? (["stages", "vjepa", "falls"] as const).filter((k) => row.values[k]) : [];
  const names: Record<string, string> = { stages: "stages", vjepa: "V-JEPA", falls: "falls" };
  return (
    <div className="policy-tile">
      <div className="ex-video" style={{ aspectRatio: `${v.width} / ${v.height}` }}>
        <video ref={group.bind(v.src)} src={exUrl(v.src)} poster={exUrl(v.poster)} muted playsInline preload="auto" aria-label={v.label} />
      </div>
      <div className="pt-body">
        <div className="pt-name"><i className="dot" style={{ background: v.color }} />{v.label}</div>
        {row && <p className="pt-reward">{row.reward}</p>}
        <div className="pt-stats">
          {keyStats.map((k) => (
            <span key={k} className={`mono ${row!.values[k].good ? "good" : ""}`}>
              <b>{row!.values[k].display}</b> {names[k]}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

function PolicyTable({ rows, columns }: { rows: PolicyRow[]; columns: { key: string; label: string; note: string }[] }) {
  return (
    <section className="ex-block">
      <h2>Per policy</h2>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Policy</th>
              {columns.map((c) => <th key={c.key} className="num" title={c.note}>{c.label}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.key}>
                <td><span style={{ display: "inline-flex", alignItems: "center", gap: 8, fontWeight: 600 }}><i className="dot" style={{ background: r.color }} />{r.label}</span>
                  <div className="faint ex-reward">{r.reward}</div></td>
                {columns.map((c) => {
                  const v = r.values[c.key];
                  return <td key={c.key} className={`num ${v?.good ? "good" : ""}`}>{v?.display ?? "—"}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <dl className="ex-legend">
        {columns.map((c) => <div key={c.key}><dt>{c.label}</dt><dd>{c.note}</dd></div>)}
      </dl>
    </section>
  );
}

function CompareBars({ rows, labels }: { rows: CompareRow[]; labels: { before: string; after: string } }) {
  const fmtv = (v: number, unit: string) =>
    unit === "%" ? `${v.toFixed(0)}%` : `${v.toFixed(v < 100 ? 1 : 0)}${unit === "°" ? "°" : ` ${unit}`}`;
  return (
    <section className="ex-block">
      <div className="ex-block-head">
        <h2>Before and after, mean of all clips</h2>
        <div className="ex-legend-inline">
          <span><i className="dot" style={{ background: "#6b7682" }} />{labels.before}</span>
          <span><i className="dot" style={{ background: "var(--accent)" }} />{labels.after}</span>
        </div>
      </div>
      <div className="card card-pad compare">
        {rows.map((r) => {
          const max = Math.max(r.before, r.after, 1e-9);
          return (
            <div className="cmp-row" key={r.label}>
              <div className="cmp-label">{r.label}</div>
              <div className="cmp-bars">
                <div className="cmp-bar before" style={{ width: `${(100 * r.before) / max}%` }}><span className="mono">{fmtv(r.before, r.unit)}</span></div>
                <div className="cmp-bar after" style={{ width: `max(${(100 * r.after) / max}%, 3px)` }}><span className="mono">{fmtv(r.after, r.unit)}</span></div>
              </div>
              <div className="cmp-ratio mono">{r.ratio ? `${r.ratio >= 10 ? r.ratio.toFixed(0) : r.ratio.toFixed(1)}× lower` : ""}</div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function heat(fr: number | null) {
  // Same scale as the results page: one hue (red), more opaque as more rollouts fall.
  return fr == null ? "transparent" : `rgba(230, 103, 103, ${0.06 + 0.5 * fr})`;
}

function RobustView({ table }: { table: RobustTable }) {
  const pct = (v: number | null) => (v == null ? "—" : `${Math.round(v * 100)}%`);
  const success = table.metric === "task_success";
  let lastSuite = "";
  return (
    <section className="ex-block">
      <div className="ex-block-head">
        <h2>{success ? "Task success under each condition" : "Fall rate under each condition"}</h2>
        <span className="faint" style={{ fontSize: 13 }}>
          {table.reps} rollouts per cell · small number: {success ? "fall rate" : "V-JEPA 2 similarity"}
        </span>
      </div>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Condition</th>
              {table.policies.map((p) => (
                <th key={p.key} className="num"><span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}><i className="dot" style={{ background: p.color }} />{p.label}</span></th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.conditions.flatMap((c, i) => {
              const rows = [];
              if (c.suite !== lastSuite) {
                lastSuite = c.suite;
                rows.push(<tr className="group" key={`g${i}`}><td colSpan={table.policies.length + 1}>{c.suite_label}</td></tr>);
              }
              rows.push(
                <tr key={i}>
                  <td>{c.level}</td>
                  {table.policies.map((p) => {
                    const cell = c.cells[p.key];
                    return (
                      <td key={p.key} className="num">
                        {success ? (
                          <span className="heat" style={{ background: heat(cell?.task_success == null ? null : 1 - cell.task_success) }}>
                            <span className="fr mono">{pct(cell?.task_success ?? null)}</span>
                            <span className="vj mono">fall {pct(cell?.fall_rate ?? null)}</span>
                          </span>
                        ) : (
                          <span className="heat" style={{ background: heat(cell?.fall_rate ?? null) }}>
                            <span className="fr mono">{pct(cell?.fall_rate ?? null)}</span>
                            <span className="vj mono">vj {cell?.vjepa == null ? "—" : cell.vjepa.toFixed(2)}</span>
                          </span>
                        )}
                      </td>
                    );
                  })}
                </tr>,
              );
              return rows;
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

// --- goal-relevance chart, synced to the video ---------------------------------------------------

function ImportanceView({ chart, group, stages }: { chart: ImportanceChart; group: SyncGroup; stages: Example["stages"] }) {
  const [box, W] = useWidth(900);
  const H = W < 560 ? 180 : 196;
  const m = { t: 38, r: 10, b: 22, l: 34 };
  const n = chart.weight.length;
  const dur = n / chart.fps;
  const sx = (t: number) => m.l + (t / dur) * (W - m.l - m.r);
  const sy = (y: number) => H - m.b - y * (H - m.t - m.b);
  const line = (ys: number[]) => ys.map((y, i) => `${i ? "L" : "M"}${sx(i / chart.fps).toFixed(1)},${sy(y).toFixed(1)}`).join("");
  const paths = useMemo(() => ({
    weight: line(chart.weight),
    area: `${line(chart.weight)}L${sx((n - 1) / chart.fps)},${sy(0)}L${sx(0)},${sy(0)}Z`,
    motion: line(chart.arm_motion),
  }), [chart, W, H]);
  const t = Math.min(group.t, dur);
  const i = Math.min(n - 1, Math.round(t * chart.fps));
  const onSeek = (e: PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    group.seek((((e.clientX - r.left) / r.width) * W - m.l) / (W - m.l - m.r) * dur);
  };
  const cur = stageAt(stages, t);
  const inHype = chart.hype && t >= chart.hype[0] && t <= chart.hype[1];
  return (
    <div className="importance" ref={box}>
      <div className="imp-head">
        <div className="ex-legend-inline">
          <span><i className="dot" style={{ background: "var(--accent)" }} />How much this moment matters (imitation weight)</span>
          <span><i className="dot" style={{ background: "var(--warn)" }} />Arm movement</span>
        </div>
        <div className="imp-readout mono">
          {cur >= 0 && stages && <span className="pill">{stages[cur].label}</span>}
          {inHype && <span className="pill warn">arm-waving</span>}
          <span>weight <b>{chart.weight[i].toFixed(2)}</b></span>
        </div>
      </div>
      <div className="chart">
        <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} onPointerDown={onSeek}
          onPointerMove={(e) => e.buttons === 1 && onSeek(e)} style={{ cursor: "ew-resize", touchAction: "none" }}>
          {stages?.map((s, k) => (
            <g key={k}>
              <rect x={sx(s.start_s)} y={2} width={Math.max(0, sx(stages[k + 1]?.start_s ?? dur) - sx(s.start_s) - 2)}
                height={H - m.b - 2} className={`imp-stage k-${s.kind} ${k === cur ? "on" : ""}`} />
              {sx(stages[k + 1]?.start_s ?? dur) - sx(s.start_s) > 34 && (
                <text x={sx(s.start_s) + 5} y={15} className="imp-stage-label">{s.label}</text>
              )}
            </g>
          ))}
          {chart.hype && (
            <g>
              <rect x={sx(chart.hype[0])} y={m.t} width={sx(chart.hype[1]) - sx(chart.hype[0])} height={H - m.t - m.b} className="imp-hype" />
              <text x={(sx(chart.hype[0]) + sx(chart.hype[1])) / 2} y={sy(0.62)} textAnchor="middle" className="imp-hype-label">arm-waving</text>
            </g>
          )}
          <g className="axis">
            {[0, 0.5, 1].map((y) => (
              <g key={y}>
                <line x1={m.l} x2={W - m.r} y1={sy(y)} y2={sy(y)} stroke="var(--border)" />
                <text x={m.l - 6} y={sy(y) + 4} textAnchor="end">{y}</text>
              </g>
            ))}
            {Array.from({ length: Math.floor(dur) + 1 }, (_, s) => s).filter((s) => W > 560 || s % 2 === 0).map((s) => (
              <text key={s} x={sx(s)} y={H - 6} textAnchor="middle">{s}s</text>
            ))}
          </g>
          <path d={paths.area} fill="var(--accent)" opacity={0.14} />
          <path d={paths.motion} fill="none" stroke="var(--warn)" strokeWidth={1.4} opacity={0.85} />
          <path d={paths.weight} fill="none" stroke="var(--accent)" strokeWidth={2.2} />
          {chart.events.map((e, k) => (
            <g key={k}>
              <circle cx={sx(e.t)} cy={sy(e.peak)} r={3.5} fill="var(--accent)" />
              {W > 560 && <text x={sx(e.t)} y={sy(e.peak) - 8} textAnchor="middle" className="imp-event">{e.label}</text>}
            </g>
          ))}
          <line x1={sx(t)} x2={sx(t)} y1={2} y2={H - m.b} stroke="var(--text)" strokeWidth={1.5} />
          <circle cx={sx(t)} cy={sy(chart.weight[i])} r={5} fill="var(--bg)" stroke="var(--text)" strokeWidth={2} />
        </svg>
      </div>
    </div>
  );
}
