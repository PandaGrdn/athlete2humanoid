import { useCallback, useEffect, useRef, useState, type PointerEvent } from "react";
import type { Stage } from "../examples";
import { Icon } from "./ui";

/** One clock for several videos: they play, pause, seek and loop together. The clock holds while any
 *  video is still buffering, and small drift is trimmed with playbackRate so nothing visibly jumps. */
export type SyncGroup = {
  t: number;
  duration: number;
  playing: boolean;
  rate: number;
  bind: (key: string) => (el: HTMLVideoElement | null) => void;
  watch: (el: HTMLElement | null) => void;
  seek: (t: number) => void;
  toggle: () => void;
  setRate: (r: number) => void;
};

const reducedMotion = () => typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

export function useSyncGroup(duration: number, autoplay = true): SyncGroup {
  const els = useRef(new Map<string, HTMLVideoElement>());
  const binders = useRef(new Map<string, (el: HTMLVideoElement | null) => void>());
  const clock = useRef({ t: 0, playing: false, rate: 1, wanted: autoplay && !reducedMotion(), visible: false, last: 0 });
  const [, setTick] = useState(0);
  const [box, setBox] = useState<HTMLElement | null>(null);

  const apply = useCallback((hard: boolean) => {
    const c = clock.current;
    els.current.forEach((v) => {
      const end = isFinite(v.duration) ? v.duration - 0.04 : Infinity;
      const target = Math.min(c.t, end);
      const drift = v.currentTime - target;
      if (hard || Math.abs(drift) > 0.2) {
        if (Math.abs(drift) > 0.01) v.currentTime = target;
        v.playbackRate = c.rate;
      } else if (c.playing) {
        v.playbackRate = c.rate * Math.max(0.9, Math.min(1.1, 1 - drift * 0.8));
      }
      const shouldPlay = c.playing && c.t < end;
      if (shouldPlay && v.paused) v.play().catch(() => undefined);
      if (!shouldPlay && !v.paused) v.pause();
    });
  }, []);

  useEffect(() => {
    let raf = 0;
    const step = (now: number) => {
      const c = clock.current;
      const dt = c.last ? Math.min(0.1, (now - c.last) / 1000) : 0;
      c.last = now;
      const ready = [...els.current.values()].every((v) => v.error || v.readyState >= 3 || v.currentTime >= v.duration - 0.05);
      if (c.playing && ready && duration > 0) {
        c.t += dt * c.rate;
        if (c.t >= duration) {
          c.t = 0;
          apply(true);
        }
        apply(false);
        setTick((n) => (n + 1) % 1e6);
      }
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [apply, duration]);

  const sync = useCallback(() => {
    const c = clock.current;
    c.playing = c.wanted && c.visible;
    apply(true);
    setTick((n) => (n + 1) % 1e6);
  }, [apply]);

  // Only play while on screen: a page of stories stays cheap, and each starts when you scroll to it.
  useEffect(() => {
    if (!box) return;
    if (typeof IntersectionObserver === "undefined") {
      clock.current.visible = true;
      sync();
      return;
    }
    const io = new IntersectionObserver(([e]) => {
      clock.current.visible = e.isIntersecting;
      sync();
    }, { threshold: 0.25 });
    io.observe(box);
    return () => io.disconnect();
  }, [box, sync]);

  const bind = useCallback((key: string) => {
    let b = binders.current.get(key);
    if (!b) {
      b = (el: HTMLVideoElement | null) => {
        if (el) {
          els.current.set(key, el);
          el.muted = true;
          el.currentTime = Math.min(clock.current.t, isFinite(el.duration) ? el.duration : Infinity);
        } else {
          els.current.delete(key);
        }
      };
      binders.current.set(key, b);
    }
    return b;
  }, []);

  const c = clock.current;
  return {
    t: c.t,
    duration,
    playing: c.playing,
    rate: c.rate,
    bind,
    watch: setBox,
    seek: (t: number) => {
      c.t = Math.max(0, Math.min(duration - 0.01, t));
      apply(true);
      setTick((n) => (n + 1) % 1e6);
    },
    toggle: () => {
      c.wanted = !c.playing;
      if (c.wanted) c.visible = true; // an explicit play wins over the visibility check
      sync();
    },
    setRate: (r: number) => {
      c.rate = r;
      apply(true);
      setTick((n) => (n + 1) % 1e6);
    },
  };
}

export const stageAt = (stages: Stage[] | null | undefined, t: number) =>
  stages?.findIndex((s, i) => t >= s.start_s && t < (stages[i + 1]?.start_s ?? Infinity)) ?? -1;

/** Play/pause, a scrubber banded by the action's stages, time and speed. */
export function SyncControls({ group, stages }: { group: SyncGroup; stages?: Stage[] | null }) {
  const track = useRef<HTMLDivElement>(null);
  const { t, duration } = group;
  const pct = (x: number) => `${(100 * Math.max(0, Math.min(1, x / duration))).toFixed(3)}%`;
  const seekTo = (e: PointerEvent<HTMLDivElement>) => {
    const r = track.current!.getBoundingClientRect();
    group.seek(((e.clientX - r.left) / r.width) * duration);
  };
  const cur = stageAt(stages, t);
  return (
    <div className="sync-controls">
      <button className="btn sm sync-play" onClick={group.toggle} aria-label={group.playing ? "Pause" : "Play"}>
        <Icon name={group.playing ? "pause" : "play"} size={15} />
      </button>
      <div className="sync-track-wrap">
        <div
          className="sync-track" ref={track} role="slider" tabIndex={0} aria-label="Seek"
          aria-valuemin={0} aria-valuemax={duration} aria-valuenow={+t.toFixed(1)}
          onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); seekTo(e); }}
          onPointerMove={(e) => e.buttons === 1 && seekTo(e)}
          onKeyDown={(e) => {
            if (e.key === "ArrowRight") group.seek(t + 0.5);
            if (e.key === "ArrowLeft") group.seek(t - 0.5);
            if (e.key === " ") { e.preventDefault(); group.toggle(); }
          }}
        >
          {stages?.map((s, i) => (
            <div key={i} className={`sync-stage k-${s.kind} ${i === cur ? "on" : ""}`}
              style={{ left: pct(s.start_s), width: `calc(${pct((stages[i + 1]?.start_s ?? duration) - s.start_s)} - 2px)` }}>
              <span>{s.label}</span>
            </div>
          ))}
          <div className="sync-fill" style={{ width: pct(t) }} />
          <div className="sync-head" style={{ left: pct(t) }} />
        </div>
      </div>
      <span className="sync-time mono">{t.toFixed(1)}<span className="faint"> / {duration.toFixed(1)} s</span></span>
      <div className="sync-rate" role="group" aria-label="Playback speed">
        {[0.5, 1].map((r) => (
          <button key={r} className={group.rate === r ? "on" : ""} onClick={() => group.setRate(r)}>{r}×</button>
        ))}
      </div>
    </div>
  );
}
