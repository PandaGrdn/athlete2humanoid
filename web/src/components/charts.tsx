import { useEffect, useRef, useState, type MouseEvent } from "react";

/** The element's width in CSS pixels, so the SVG draws 1:1 and its text stays legible at any size. */
export function useWidth(fallback = 640) {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(160, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

type Series = { key: string; label: string; color: string; points: [number, number][] };

function niceTicks(min: number, max: number, count = 4) {
  if (max === min) return [min];
  const raw = (max - min) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

const short = (v: number) => (Math.abs(v) >= 1000 ? `${+(v / 1000).toFixed(1)}k` : `${+v.toFixed(2)}`);

/** Line chart with a hover crosshair. One y-axis; series share units. */
export function LineChart({ series, height = 200, xLabel, yLabel, xMax, compact = false }: {
  series: Series[]; height?: number; xLabel?: string; yLabel?: string; xMax?: number; compact?: boolean;
}) {
  const [box, W] = useWidth();
  const H = height, m = compact ? { t: 6, r: 8, b: 6, l: 4 } : { t: 10, r: 14, b: 26, l: 40 };
  const [hover, setHover] = useState<number | null>(null);
  const all = series.flatMap((s) => s.points);
  const x1 = xMax ?? Math.max(1, ...all.map((p) => p[0]));
  const ys = all.map((p) => p[1]);
  const y0 = Math.min(0, ...ys), y1 = Math.max(1e-6, ...ys) * 1.08;
  const sx = (x: number) => m.l + (x / x1) * (W - m.l - m.r);
  const sy = (y: number) => H - m.b - ((y - y0) / (y1 - y0)) * (H - m.t - m.b);
  const yt = niceTicks(y0, y1), xt = niceTicks(0, x1, 4);

  const onMove = (e: MouseEvent<SVGSVGElement>) => {
    const r = box.current!.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * W;
    setHover(Math.max(0, Math.min(x1, ((x - m.l) / (W - m.l - m.r)) * x1)));
  };
  const nearest = (pts: [number, number][], x: number) =>
    pts.reduce<[number, number] | null>((b, p) => (!b || Math.abs(p[0] - x) < Math.abs(b[0] - x) ? p : b), null);

  return (
    <div className="chart" ref={box}>
      {series.length > 1 && (
        <div className="legend">
          {series.map((s) => <span key={s.key}><i className="dot" style={{ background: s.color }} />{s.label}</span>)}
        </div>
      )}
      <svg viewBox={`0 0 ${W} ${H}`} onMouseMove={onMove} onMouseLeave={() => setHover(null)} role="img"
        aria-label={`${yLabel ?? "value"} by ${xLabel ?? "x"}`}>
        <g className="grid">{(compact ? [y0] : yt).map((t) => <line key={t} x1={m.l} x2={W - m.r} y1={sy(t)} y2={sy(t)} />)}</g>
        {!compact && <g className="axis">
          {yt.map((t) => <text key={t} x={m.l - 8} y={sy(t) + 4} textAnchor="end">{short(t)}</text>)}
          {xt.map((t) => <text key={t} x={sx(t)} y={H - 6} textAnchor="middle">{short(t)}</text>)}
        </g>}
        {series.map((s) => s.points.length > 1 && (
          <path key={s.key} d={s.points.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join("")}
            fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        ))}
        {series.map((s) => {
          const last = s.points[s.points.length - 1];
          return last && <circle key={s.key} cx={sx(last[0])} cy={sy(last[1])} r={4} fill={s.color} stroke="var(--surface-1)" strokeWidth={2} />;
        })}
        {hover != null && <line x1={sx(hover)} x2={sx(hover)} y1={m.t} y2={H - m.b} stroke="var(--text-3)" strokeDasharray="3 3" />}
        {hover != null && series.map((s) => {
          const p = nearest(s.points, hover);
          return p && <circle key={s.key} cx={sx(p[0])} cy={sy(p[1])} r={4} fill={s.color} stroke="var(--surface-1)" strokeWidth={2} />;
        })}
      </svg>
      {hover != null && all.length > 0 && (
        <div className="tooltip" style={{ left: `${(sx(hover) / W) * 100}%`, top: compact ? 0 : 24 + (series.length > 1 ? 28 : 0) }}>
          <div className="title">{xLabel ?? "x"} {Math.round(nearest(series[0].points, hover)?.[0] ?? hover)}</div>
          {series.map((s) => {
            const p = nearest(s.points, hover);
            return p && <div className="row" key={s.key}><i className="dot" style={{ background: s.color }} />{s.label} {p[1].toFixed(2)}</div>;
          })}
        </div>
      )}
    </div>
  );
}

type Bar = { label: string; value: number; color: string; note?: string };

/** Horizontal bars, labelled directly, with a per-bar hover tooltip. */
export function BarChart({ bars, max = 1, format = (v: number) => v.toFixed(3) }: {
  bars: Bar[]; max?: number; format?: (v: number) => string;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const [box, W] = useWidth();
  const rowH = 30, labelW = 110, valueW = 56;
  const H = bars.length * rowH;
  const scale = (v: number) => (Math.max(0, v) / max) * (W - labelW - valueW - 8);
  return (
    <div className="chart" ref={box}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="bar chart">
        {bars.map((b, i) => {
          const y = i * rowH, w = Math.max(4, scale(b.value));
          return (
            <g key={b.label} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
              <rect x={0} y={y} width={W} height={rowH} fill={hover === i ? "rgba(255,255,255,0.03)" : "transparent"} />
              <text x={0} y={y + rowH / 2 + 4} fill="var(--text-2)" fontSize={12.5}>{b.label}</text>
              <path d={`M${labelW},${y + 7} h${w - 4} a4,4 0 0 1 4,4 v${rowH - 22} a4,4 0 0 1 -4,4 h${-(w - 4)} z`} fill={b.color} />
              <text x={labelW + w + 8} y={y + rowH / 2 + 4} fill="var(--text)" fontSize={12} fontFamily="var(--mono)">{format(b.value)}</text>
            </g>
          );
        })}
      </svg>
      {hover != null && (
        <div className="tooltip" style={{ left: `${((labelW + scale(bars[hover].value) / 2) / W) * 100}%`, top: `${(hover * rowH / H) * 100}%` }}>
          <div className="title">{bars[hover].label}</div>
          <div className="row">{format(bars[hover].value)}{bars[hover].note ? ` · ${bars[hover].note}` : ""}</div>
        </div>
      )}
    </div>
  );
}
