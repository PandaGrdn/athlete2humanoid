import type { AttemptMeta } from "../api";

// The fast replay of a recorded build (pages/BuildPage.tsx, the showcase's replay tab): the five build stages come in
// one after another, then stage 06 steps through the policy's attempts, every `stride`-th iteration, each for a
// couple of seconds (cut where it fell), and ends on the trained policy. The whole thing fits `seconds`. The pacing
// is pure arithmetic over the attempt list, so it is testable without a browser (replay.test.ts).

export type ReplaySettings = { seconds: number; attempt_stride: number };
export const DEFAULT_REPLAY: ReplaySettings = { seconds: 28, attempt_stride: 100 };
export const STAGE_COUNT = 5; // stages 01-05 (video, skeleton, retarget, moments, variants); 06 is the attempts
const STAGE_MAX_S = 2.6; // a stage never lingers longer than this before the next one comes
const STAGE_MIN_S = 0.8; // ... nor comes faster than this
const ATTEMPT_MAX_S = 3.0; // an attempt's slot at most (the clip's own length is usually longer)
const ATTEMPT_MIN_S = 0.9; // a slot at least, so a fall at 0.2 s still reads
const HOLD_S = 0.4; // after an attempt ends (or falls), it stays before the next one
const CLOSING_S = 1.5; // the end: the trained policy, counted inside `seconds`

export type ReplayStep = { index: number; iteration: number; start: number; end: number; seconds: number };
export type ReplayPlan = {
  seconds: number; // total
  stageStep: number; // seconds between stage reveals (stage i shows at i * stageStep)
  preSeconds: number; // when stage 06 begins
  steps: ReplayStep[]; // the attempts played, in order
  closingAt: number; // when the trained policy takes over (= end of the last step), or preSeconds if no attempts
};

/** Every `stride`-th iteration of the attempts (the last one always), by iteration; a stride of 0 keeps them all. */
export function pickAttempts(attempts: AttemptMeta[], stride: number): number[] {
  const sorted = [...attempts].map((a, i) => ({ i, it: a.iteration })).sort((a, b) => a.it - b.it);
  if (!sorted.length) return [];
  if (stride <= 0) return sorted.map((x) => x.i);
  const out: number[] = [];
  let next = stride;
  for (const x of sorted) {
    if (x.it >= next) { out.push(x.i); next = (Math.floor(x.it / stride) + 1) * stride; }
  }
  const last = sorted[sorted.length - 1].i;
  if (out[out.length - 1] !== last) out.push(last);
  return out;
}

/** How long attempt `a` is worth showing: its own length (cut at the fall) plus a hold, within the slot limits. */
export function attemptSeconds(a: AttemptMeta, fps = 50): number {
  const own = a.fell && a.fell_at_s != null ? a.fell_at_s : a.seconds ?? (a.frames != null ? a.frames / fps : ATTEMPT_MAX_S);
  return Math.max(ATTEMPT_MIN_S, Math.min(ATTEMPT_MAX_S, own + HOLD_S));
}

/** The plan for `attempts` in `settings.seconds`: the stages take what they need (at most 2.6 s each, less when the
 *  attempts need the time), the attempts share the rest in proportion to their own lengths, and the trained policy
 *  gets the last moment. With no attempts the stages pace themselves over the whole time. */
export function planReplay(attempts: AttemptMeta[], settings: Partial<ReplaySettings> = {}, fps = 50): ReplayPlan {
  const seconds = Math.max(5, settings.seconds ?? DEFAULT_REPLAY.seconds);
  const stride = settings.attempt_stride ?? DEFAULT_REPLAY.attempt_stride;
  const picked = pickAttempts(attempts, stride);
  if (!picked.length) {
    const stageStep = Math.max(STAGE_MIN_S, Math.min(STAGE_MAX_S, seconds / (STAGE_COUNT + 1)));
    return { seconds, stageStep, preSeconds: stageStep * STAGE_COUNT, steps: [], closingAt: stageStep * STAGE_COUNT };
  }
  const wants = picked.map((i) => attemptSeconds(attempts[i], fps));
  const wantTotal = wants.reduce((a, b) => a + b, 0);
  // the stages get their 2.6 s each when the attempts fit after them; else the two sides share the time 40/60
  let stageStep = Math.min(STAGE_MAX_S, (seconds - CLOSING_S - wantTotal) / STAGE_COUNT);
  stageStep = Math.max(STAGE_MIN_S, Math.min(STAGE_MAX_S, Math.max(stageStep, (0.4 * seconds) / STAGE_COUNT)));
  const preSeconds = stageStep * STAGE_COUNT;
  const budget = Math.max(ATTEMPT_MIN_S * picked.length, seconds - CLOSING_S - preSeconds);
  const slots = fitSlots(wants, budget);
  const steps: ReplayStep[] = [];
  let t = preSeconds;
  picked.forEach((i, k) => {
    steps.push({ index: i, iteration: attempts[i].iteration, start: t, end: t + slots[k], seconds: slots[k] });
    t += slots[k];
  });
  return { seconds: Math.max(seconds, t + CLOSING_S), stageStep, preSeconds, steps, closingAt: t };
}

/** `wants` scaled into `budget` seconds, no slot under the minimum: the short ones keep it, the rest share what is
 *  left in proportion. When even the minimum for each doesn't fit, the total runs over (every attempt stays readable). */
export function fitSlots(wants: number[], budget: number): number[] {
  if (wants.length === 0) return [];
  if (wants.reduce((a, b) => a + b, 0) <= budget) return [...wants];
  const floored = new Set<number>();
  for (;;) {
    const free = wants.map((w, k) => (floored.has(k) ? 0 : w)).reduce((a, b) => a + b, 0);
    const scale = free > 0 ? Math.max(0, budget - ATTEMPT_MIN_S * floored.size) / free : 0;
    const more = wants.map((w, k) => (!floored.has(k) && w * scale < ATTEMPT_MIN_S ? k : -1)).filter((k) => k >= 0);
    if (!more.length || floored.size === wants.length) return wants.map((w, k) => (floored.has(k) ? ATTEMPT_MIN_S : w * scale));
    more.forEach((k) => floored.add(k));
  }
}

export type ReplayPosition = {
  stagesReached: number; // 0..5 build stages shown so far (5 = all, stage 06 running)
  step: ReplayStep | null; // the attempt playing now
  closing: boolean; // the trained policy is on
  done: boolean; // past the end
};

/** Where the replay is at `elapsed` seconds. */
export function replayAt(plan: ReplayPlan, elapsed: number): ReplayPosition {
  if (elapsed < 0) return { stagesReached: 0, step: null, closing: false, done: false };
  const stagesReached = Math.min(STAGE_COUNT, Math.floor(elapsed / plan.stageStep));
  const step = plan.steps.find((s) => elapsed >= s.start && elapsed < s.end) ?? null;
  const closing = elapsed >= plan.closingAt;
  return { stagesReached, step, closing, done: elapsed >= plan.seconds };
}
