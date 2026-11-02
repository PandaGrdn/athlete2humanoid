import { describe, expect, it } from "vitest";
import type { AttemptMeta } from "../api";
import { attemptSeconds, fitSlots, pickAttempts, planReplay, replayAt, STAGE_COUNT } from "./replay";

const att = (iteration: number, seconds = 5.5, fell_at_s: number | null = null): AttemptMeta => ({
  iteration, reward: iteration / 100, success: fell_at_s == null, stages_completed: null, stages_total: null,
  fell: fell_at_s != null, fell_at_s, frames: Math.round(seconds * 50), seconds, url: `/x/iter_${iteration}.npz`,
});
const every50 = Array.from({ length: 20 }, (_, i) => att((i + 1) * 50, 5.5, i < 6 ? 0.2 + i * 0.4 : null));

describe("pickAttempts", () => {
  it("keeps every stride-th iteration and always the last one", () => {
    const idx = pickAttempts(every50, 100);
    expect(idx.map((i) => every50[i].iteration)).toEqual([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]);
    expect(pickAttempts(every50, 300).map((i) => every50[i].iteration)).toEqual([300, 600, 900, 1000]);
  });
  it("works when the recording is coarser than the stride, out of order, or empty", () => {
    const coarse = [att(300), att(100), att(200)];
    expect(pickAttempts(coarse, 50).map((i) => coarse[i].iteration)).toEqual([100, 200, 300]);
    expect(pickAttempts(coarse, 250).map((i) => coarse[i].iteration)).toEqual([300]);
    expect(pickAttempts([], 100)).toEqual([]);
    expect(pickAttempts(every50, 0)).toHaveLength(20);
  });
});

describe("attemptSeconds", () => {
  it("cuts at the fall, holds a moment, and stays inside the slot limits", () => {
    expect(attemptSeconds(att(50, 5.5, 0.2))).toBeCloseTo(0.9); // 0.2 + hold, lifted to the minimum
    expect(attemptSeconds(att(50, 5.5, 1.5))).toBeCloseTo(1.9);
    expect(attemptSeconds(att(1000, 5.5))).toBeCloseTo(3.0); // a full clip is capped
    expect(attemptSeconds({ ...att(1000), seconds: null, frames: 60 })).toBeCloseTo(1.6);
  });
});

describe("planReplay", () => {
  it("fits 10 attempts and the five stages into the default 28 s", () => {
    const plan = planReplay(every50);
    expect(plan.steps.map((s) => s.iteration)).toEqual([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]);
    expect(plan.seconds).toBeCloseTo(28);
    expect(plan.preSeconds).toBeCloseTo(plan.stageStep * STAGE_COUNT);
    expect(plan.stageStep).toBeGreaterThanOrEqual(0.8);
    expect(plan.stageStep).toBeLessThanOrEqual(2.6);
    const last = plan.steps[plan.steps.length - 1];
    expect(last.end).toBeCloseTo(plan.closingAt);
    expect(plan.closingAt).toBeLessThanOrEqual(plan.seconds - 1.5 + 1e-6);
    // slots are contiguous and ordered
    plan.steps.forEach((s, i) => { if (i) expect(s.start).toBeCloseTo(plan.steps[i - 1].end); });
    // the early falls get short slots, the full clips longer ones
    expect(plan.steps[0].seconds).toBeLessThan(plan.steps[9].seconds);
  });
  it("gives the stages their time when there are few attempts, and the attempts theirs when there are many", () => {
    const few = planReplay([att(1000)], { seconds: 28, attempt_stride: 100 });
    expect(few.stageStep).toBeCloseTo(2.6);
    expect(few.steps).toHaveLength(1);
    const many = planReplay(every50, { seconds: 25, attempt_stride: 50 });
    expect(many.steps).toHaveLength(20);
    expect(many.stageStep).toBeLessThan(2.6);
    expect(many.stageStep).toBeGreaterThanOrEqual(0.8);
    // every slot keeps a readable minimum even when the time is short: the plan then runs a little over
    many.steps.forEach((s) => expect(s.seconds).toBeGreaterThanOrEqual(0.9 - 1e-9));
    expect(many.seconds).toBeGreaterThanOrEqual(25);
    expect(many.seconds).toBeLessThan(32);
  });
  it("paces the stages alone when a build has no attempts", () => {
    const plan = planReplay([], { seconds: 15.6, attempt_stride: 100 });
    expect(plan.steps).toEqual([]);
    expect(plan.stageStep).toBeCloseTo(2.6);
    expect(plan.closingAt).toBeCloseTo(13);
  });
  it("respects the build's own settings", () => {
    const plan = planReplay(every50, { seconds: 30, attempt_stride: 250 });
    expect(plan.steps.map((s) => s.iteration)).toEqual([250, 500, 750, 1000]);
    expect(plan.seconds).toBeCloseTo(30);
  });
});

describe("fitSlots", () => {
  it("keeps the proportions when the wants fit, floors the short ones and shares the rest when they don't", () => {
    expect(fitSlots([1, 2, 3], 10)).toEqual([1, 2, 3]);
    const s = fitSlots([1.0, 3.0, 3.0], 5);
    expect(s[0]).toBeCloseTo(0.9); // floored
    expect(s[1]).toBeCloseTo(2.05); // (5 - 0.9) / 2
    expect(s.reduce((a, b) => a + b, 0)).toBeCloseTo(5);
    const over = fitSlots([3, 3, 3], 1);
    expect(over).toEqual([0.9, 0.9, 0.9]); // the minimum each: over budget, but readable
  });
});

describe("replayAt", () => {
  const plan = planReplay(every50);
  it("reveals the stages one by one, then steps through the attempts, then closes", () => {
    expect(replayAt(plan, -1).stagesReached).toBe(0);
    expect(replayAt(plan, plan.stageStep * 2 + 0.01).stagesReached).toBe(2);
    expect(replayAt(plan, plan.preSeconds + 0.01).stagesReached).toBe(STAGE_COUNT);
    const first = replayAt(plan, plan.steps[0].start + 0.01);
    expect(first.step?.iteration).toBe(100);
    expect(first.closing).toBe(false);
    const mid = replayAt(plan, plan.steps[4].start + 0.01);
    expect(mid.step?.iteration).toBe(500);
    const end = replayAt(plan, plan.closingAt + 0.01);
    expect(end.step).toBeNull();
    expect(end.closing).toBe(true);
    expect(end.done).toBe(false);
    expect(replayAt(plan, plan.seconds + 1).done).toBe(true);
  });
});
