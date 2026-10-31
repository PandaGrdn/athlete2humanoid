// Screen takes for the demo video kit (docs/demo_kit/EDIT_LIST.md), recorded from a running copy of the web app.
//
//   npm install --prefix /tmp/a2h-rec playwright-core@1.48.2
//   NODE_PATH=/tmp/a2h-rec/node_modules FF=$(python -c "import imageio_ffmpeg as f; print(f.get_ffmpeg_exe())") \
//     BASE=http://localhost:5173 node scripts/demo_kit/record_takes.mjs build_sj|build_kick|showcase|live clips/punches.mov
//
// Headless system Chrome on the real GPU (ANGLE Metal on a Mac), a 1280x720 page at 1.5x = 1920x1080. Frames come from
// CDP's screencast with their timestamps and are re-timed to constant 30 fps, so the take plays at wall-clock speed.
// Each take writes <OUT>/<name>.mp4 and <name>.marks.json (seconds from the take's start to each action), which
// scripts/demo_kit/build_kit.py cuts by.
import { createRequire } from "node:module";
import os from "node:os";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright-core");
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
const FF = process.env.FF ?? "ffmpeg";
async function startRec(page, dir, { quality = 92, w = 1920, h = 1080 } = {}) {
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
  const cdp = await page.context().newCDPSession(page);
  const frames = [];
  let n = 0;
  cdp.on("Page.screencastFrame", async (f) => {
    const file = path.join(dir, `f${String(n++).padStart(6, "0")}.jpg`);
    fs.writeFileSync(file, Buffer.from(f.data, "base64"));
    frames.push({ file, t: f.metadata.timestamp });
    try { await cdp.send("Page.screencastFrameAck", { sessionId: f.sessionId }); } catch {}
  });
  await cdp.send("Page.startScreencast", { format: "jpeg", quality, maxWidth: w, maxHeight: h, everyNthFrame: 1 });
  const t0 = Date.now() / 1000;
  return {
    t0,
    async stop(out, { fps = 30, speed = 1, trimStart = 0, dur = null } = {}) {
      await cdp.send("Page.stopScreencast");
      const t1 = Date.now() / 1000;
      if (!frames.length) throw new Error(`${out}: no frames captured (is the page rendering? is ${BASE} up?)`);
      // concat list with per-frame durations; last frame lasts until stop
      let lines = ["ffconcat version 1.0"];
      for (let i = 0; i < frames.length; i++) {
        const d = ((i + 1 < frames.length ? frames[i + 1].t : t1) - frames[i].t) / speed;
        lines.push(`file '${frames[i].file}'`, `duration ${Math.max(d, 0.001).toFixed(4)}`);
      }
      lines.push(`file '${frames[frames.length - 1].file}'`);
      const list = path.join(dir, "list.txt");
      fs.writeFileSync(list, lines.join("\n"));
      const span = (t1 - frames[0].t);
      console.log(`${out}: ${frames.length} frames over ${span.toFixed(1)}s = ${(frames.length / span).toFixed(1)} fps`);
      const args = ["-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", list,
        "-vf", `fps=${fps},scale=1920:1080:flags=lanczos:out_range=tv,format=yuv420p`, "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-movflags", "+faststart"];
      if (trimStart) args.splice(args.indexOf("-vf"), 0, "-ss", String(trimStart));
      if (dur) args.splice(args.indexOf("-vf"), 0, "-t", String(dur));
      execFileSync(FF, [...args, out], { stdio: "inherit" });
      return { frames: frames.length, span, firstT: frames[0].t };
    },
  };
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const BASE = process.env.BASE ?? "http://localhost:5173";
const OUT = process.env.OUT ?? "outputs/demo_kit/raw";
const FRAMES = process.env.FRAMES ?? path.join(os.tmpdir(), "a2h_demo_frames");
fs.mkdirSync(OUT, { recursive: true });

async function open() {
  const browser = await chromium.launch({ channel: "chrome", headless: true,
    args: ["--use-angle=metal", "--hide-scrollbars", "--autoplay-policy=no-user-gesture-required", "--disable-background-timer-throttling", "--disable-renderer-backgrounding"] });
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1.5, colorScheme: "dark" });
  const page = await ctx.newPage();
  page.on("response", (r) => r.status() >= 400 && console.log("http", r.status(), r.url()));
  page.on("console", (m) => m.type() === "error" && console.log("console:", m.text()));
  return { browser, page };
}
function marker(rec) { const marks = []; return { marks, mark: (k) => { const t = Date.now() / 1000 - rec.t0; marks.push([k, +t.toFixed(2)]); console.log(`  ${t.toFixed(1)}s ${k}`); } }; }
async function smoothScroll(page, to, ms = 1500) {
  await page.evaluate(async ([to, ms]) => {
    const from = scrollY, t0 = performance.now();
    await new Promise((res) => { const step = (now) => { const k = Math.min(1, (now - t0) / ms); const e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2; scrollTo(0, from + (to - from) * e); k < 1 ? requestAnimationFrame(step) : res(); }; requestAnimationFrame(step); });
  }, [to, ms]);
}
async function drag(page, x0, y0, x1, y1, ms) {
  await page.mouse.move(x0, y0); await page.mouse.down();
  const t0 = Date.now();
  for (;;) { const k = Math.min(1, (Date.now() - t0) / ms), e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2; await page.mouse.move(x0 + (x1 - x0) * e, y0 + (y1 - y0) * e); if (k >= 1) break; await sleep(8); }
  await page.mouse.up();
}
const click = (page, sel, i = 0) => page.evaluate(([s, i]) => document.querySelectorAll(s)[i].click(), [sel, i]);
const clickText = (page, sel, txt) => page.evaluate(([s, t]) => [...document.querySelectorAll(s)].find((e) => e.textContent.includes(t)).click(), [sel, txt]);
const top = (page, sel) => page.evaluate((s) => document.querySelector(s).getBoundingClientRect().top + scrollY, sel);

async function build(rid, name) {
  const { browser, page } = await open();
  await page.goto(`${BASE}/build/${rid}`);
  await page.waitForFunction(() => document.querySelectorAll(".bv-chip.ready").length === 6, null, { timeout: 60000 });
  await sleep(1500);
  const rail = await top(page, ".bv-rail");
  await page.evaluate((y) => scrollTo(0, y), rail - 59);
  await sleep(800);
  const rec = await startRec(page, path.join(FRAMES, name));
  const { marks, mark } = marker(rec);
  await sleep(1000);
  mark("replay");
  await clickText(page, "button", "Replay the build");
  await page.waitForFunction(() => document.querySelectorAll(".bv-chip.ready").length === 6, null, { timeout: 60000, polling: 100 });
  mark("all_ready");
  await sleep(2500);
  mark("chip03"); await click(page, ".bv-chip", 2); await sleep(2200);
  mark("orbit"); await drag(page, 560, 420, 780, 410, 3000); await sleep(400); await drag(page, 780, 410, 620, 415, 2200); await sleep(1000);
  mark("chip04"); await click(page, ".bv-chip", 3); await sleep(5000);
  mark("chip05"); await click(page, ".bv-chip", 4); await sleep(3500);
  mark("chip06"); await click(page, ".bv-chip", 5); await sleep(2500);
  await clickText(page, ".bv-seg button", "Shove");
  mark("push1"); await clickText(page, ".bv-policy button", "Push"); await sleep(4500);
  mark("push2"); await clickText(page, ".bv-policy button", "Push"); await sleep(4500);
  const below = await top(page, ".bv-below");
  mark("scroll_timeline"); await smoothScroll(page, below - 80, 1600);
  mark("timeline"); await sleep(7000);
  mark("end");
  const info = await rec.stop(`${OUT}/${name}.mp4`);
  fs.writeFileSync(`${OUT}/${name}.marks.json`, JSON.stringify({ marks, ...info }, null, 1));
  await browser.close();
}

async function live(clip, name) {
  const { browser, page } = await open();
  await page.goto(`${BASE}/live`);
  await sleep(2500);
  const rec = await startRec(page, path.join(FRAMES, name));
  const { marks, mark } = marker(rec);
  await sleep(2000);
  mark("upload");
  await page.setInputFiles("input[type=file]", clip);
  // follow the build until the policy stage offers a pre-trained policy or 150 s pass
  const t0 = Date.now();
  let last = "";
  while (Date.now() - t0 < 150000) {
    const st = await page.evaluate(() => [...document.querySelectorAll(".bv-chip")].map((c) => c.className.replace("bv-chip", "").trim()).join("|") + " " + location.pathname).catch(() => "");
    if (st !== last) { mark("state " + st); last = st; }
    if (/offer|ready$/.test(st.split(" ")[0].split("|")[5] ?? "")) break;
    await sleep(250);
  }
  mark("built"); await sleep(2500);
  const stage = await page.$(".bv-rail");
  if (stage) { const y = await top(page, ".bv-rail"); mark("scroll_stage"); await smoothScroll(page, y - 59, 1200); await sleep(1500); }
  const offer = await page.$(".bv-offer button");
  if (offer) { mark("offer"); await click(page, ".bv-offer button"); await sleep(4000); }
  if (await page.$(".bv-policy")) { await clickText(page, ".bv-seg button", "Shove"); mark("push"); await clickText(page, ".bv-policy button", "Push"); await sleep(5000); }
  mark("end");
  const info = await rec.stop(`${OUT}/${name}.mp4`);
  fs.writeFileSync(`${OUT}/${name}.marks.json`, JSON.stringify({ marks, ...info }, null, 1));
  await page.screenshot({ path: `${OUT}/${name}_last.png` });
  await browser.close();
}

async function showcase(name) {
  const { browser, page } = await open();
  await page.goto(`${BASE}/showcase`);
  await sleep(7000);
  const rec = await startRec(page, path.join(FRAMES, name));
  const { marks, mark } = marker(rec);
  await sleep(3000);
  await clickText(page, "button", "Shove");
  mark("push1"); await clickText(page, "button", "Push"); await sleep(5000);
  mark("orbit"); await drag(page, 800, 330, 520, 320, 3000); await sleep(1500);
  mark("push2"); await clickText(page, "button", "Push"); await sleep(5000);
  mark("kick"); await clickText(page, "button", "Front kick"); await sleep(6000);
  await clickText(page, "button", "Shove");
  mark("push3"); await clickText(page, "button", "Push"); await sleep(5500);
  mark("scroll_stats"); await smoothScroll(page, 560, 1500); await sleep(4000);
  mark("end");
  const info = await rec.stop(`${OUT}/${name}.mp4`);
  fs.writeFileSync(`${OUT}/${name}.marks.json`, JSON.stringify({ marks, ...info }, null, 1));
  await browser.close();
}

const [which, clip] = process.argv.slice(2);
const TAKES = {
  build_sj: () => build("latent_s3_sj", "build_sj"),
  build_kick: () => build("latent_s3_kick", "build_kick"),
  live: () => (clip && fs.existsSync(clip) ? live(clip, "live_upload") : Promise.reject(new Error("live needs a video file: live <clip>"))),
  showcase: () => showcase("showcase"),
};
if (!TAKES[which]) {
  console.error(`unknown take ${JSON.stringify(which)}; one of: ${Object.keys(TAKES).join(", ")} (live takes a clip path)`);
  process.exit(2);
}
await TAKES[which]();
