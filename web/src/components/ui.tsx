import { useEffect, useRef } from "react";
import { url, type Credit } from "../api";
import { useUiMode } from "../theme";

export type IconName = "upload" | "play" | "pause" | "download" | "arrow" | "cube" | "chart" | "file" | "film" | "x" | "check" | "github";

const PATHS: Record<IconName, string> = {
  upload: "M12 16V4m0 0-4.5 4.5M12 4l4.5 4.5M4 16v2.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V16",
  play: "M7 5.5v13l11-6.5-11-6.5Z",
  pause: "M8.5 5v14m7-14v14",
  download: "M12 4v12m0 0-4.5-4.5M12 16l4.5-4.5M4 20h16",
  arrow: "M5 12h14m0 0-5-5m5 5-5 5",
  cube: "m12 3 8 4.5v9L12 21l-8-4.5v-9L12 3Zm0 0v18m8-13.5-8 4.5-8-4.5",
  chart: "M4 20V4m0 16h16M8 16l3.5-4 3 2.5L20 8",
  file: "M6 3h8l4 4v14H6V3Zm8 0v4h4",
  film: "M4 5h16v14H4V5Zm0 4h16M4 15h16M8 5v14m8-14v14",
  x: "M6 6l12 12M18 6 6 18",
  check: "m5 12.5 4.5 4.5L19 7",
  github: "M9 19c-4 1.5-4-2-6-2.5m12 5v-3.5a3 3 0 0 0-.9-2.4c2.9-.3 5.9-1.4 5.9-6.4a5 5 0 0 0-1.4-3.5 4.6 4.6 0 0 0-.1-3.5s-1.1-.3-3.6 1.4a12.3 12.3 0 0 0-6.4 0C6.2 2.8 5.1 3.1 5.1 3.1a4.6 4.6 0 0 0-.1 3.5A5 5 0 0 0 3.6 10c0 5 3 6.1 5.9 6.4a3 3 0 0 0-.9 2.4V22",
};

/** The pixel design's icons: 11 x 11 bitmaps, one row per string, "#" filled. */
const PIXELS: Record<IconName | "mark" | "menu", string[]> = {
  upload: [".....#.....", "....###....", "...#####...", ".....#.....", ".....#.....", ".....#.....", ".#...#...#.", ".#.......#.", ".#########."],
  play: ["...#.......", "...##......", "...###.....", "...####....", "...#####...", "...####....", "...###.....", "...##......", "...#......."],
  pause: ["..##...##..", "..##...##..", "..##...##..", "..##...##..", "..##...##..", "..##...##..", "..##...##.."],
  download: [".....#.....", ".....#.....", ".....#.....", "...#####...", "....###....", ".....#.....", ".#.......#.", ".#.......#.", ".#########."],
  arrow: ["......#....", ".......#...", "........#..", ".#########.", "........#..", ".......#...", "......#...."],
  cube: ["...#######.", "..#.....##.", ".#######.#.", ".#.....#.#.", ".#.....#.#.", ".#.....#.#.", ".#.....#.#.", ".#.....##..", ".#######..."],
  chart: [".#.........", ".#.......#.", ".#......#..", ".#.#...#...", ".#..#.#....", ".#...#.....", ".#.........", ".#.........", ".#########."],
  file: ["..#####....", "..#...##...", "..#...###..", "..#.....#..", "..#.....#..", "..#.....#..", "..#.....#..", "..#.....#..", "..#######.."],
  film: [".#########.", ".#.#...#.#.", ".###...###.", ".#.#...#.#.", ".###...###.", ".#.#...#.#.", ".###...###.", ".#.#...#.#.", ".#########."],
  x: ["..#.....#..", "...#...#...", "....#.#....", ".....#.....", "....#.#....", "...#...#...", "..#.....#.."],
  check: [".........#.", "........#..", ".......#...", ".#....#....", "..#..#.....", "...##......"],
  github: [".#.......#.", ".##.....##.", ".#########.", "###########", "###.###.###", "###########", ".#########.", "...#...#...", "...#...#..."],
  menu: [".#########.", "...........", "...........", ".#########.", "...........", "...........", ".#########."],
  // a person mid-jump, arms out: the wordmark's glyph
  mark: ["....###....", "....###....", "...........", ".#########.", ".....#.....", ".....#.....", "....###....", "...#...#...", "..#.....#..", "..#.....#.."],
};

/** Bitmap rows to one path of unit squares, vertically centred in the 11 x 11 box. */
function pixelPath(rows: string[]) {
  const top = Math.floor((11 - rows.length) / 2);
  let d = "";
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      if (row[x] !== "#") continue;
      let n = 1;
      while (row[x + n] === "#") n++;
      d += `M${x} ${y + top}h${n}v1h-${n}z`;
      x += n - 1;
    }
  });
  return d;
}
const PIXEL_PATHS = Object.fromEntries(Object.entries(PIXELS).map(([k, rows]) => [k, pixelPath(rows)])) as Record<IconName | "mark" | "menu", string>;

export function PixelIcon({ name, size = 18 }: { name: IconName | "mark" | "menu"; size?: number }) {
  return (
    <svg className="px-icon" width={size} height={size} viewBox="0 0 11 11" fill="currentColor" shapeRendering="crispEdges" aria-hidden="true">
      <path d={PIXEL_PATHS[name]} />
    </svg>
  );
}

export function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
  const ui = useUiMode();
  if (ui === "pixel") return <PixelIcon name={name} size={size} />;
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8}
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={PATHS[name]} />
    </svg>
  );
}

/** Muted looping video that only plays while on screen (keeps a page of videos cheap). */
export function LoopVideo({ src, poster, className, label }: { src: string; poster?: string; className?: string; label?: string }) {
  const ref = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    const v = ref.current;
    if (!v || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) v.play().catch(() => undefined);
      else v.pause();
    }, { threshold: 0.2 });
    io.observe(v);
    return () => io.disconnect();
  }, [src]);
  return (
    <video ref={ref} className={className} src={url(src)} poster={poster ? url(poster) : undefined}
      muted loop playsInline autoPlay preload="metadata" aria-label={label} />
  );
}

export function StatusPill({ status }: { status: string }) {
  const cls = status === "succeeded" || status === "done" ? "accent" : status === "failed" ? "bad" : status === "cancelled" ? "" : "warn";
  const live = status === "running" || status === "queued";
  return (
    <span className={`pill ${cls}`}>
      {live && <span className="dot live" />}
      {status}
    </span>
  );
}

/** The source video's attribution (a run's web.json "credit"): required on the page for CC BY clips. */
export function CreditLine({ credit }: { credit?: Credit | null }) {
  if (!credit) return null;
  return (
    <p className="muted credit" style={{ fontSize: 13, marginTop: 8, maxWidth: 820 }}>
      Source video: {credit.url ? <a href={credit.url} target="_blank" rel="noopener noreferrer">{credit.text}</a> : credit.text}
      {credit.licence && <> ({credit.licence})</>}
    </p>
  );
}
