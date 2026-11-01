# Demo video kit: edit list

One clip per shot of the storyboard in [`docs/DEMO_SCRIPT.md`](../DEMO_SCRIPT.md), plus a rough cut. All clips are
1920×1080, 30 fps, H.264, no audio, and trimmed to the storyboard's length. They live in `outputs/demo_kit/`, which is
gitignored. The voice-over and the final cut are yours: the shot clips are clean (only the jog's source credit is burned
in), and only the rough cut has captions.

- **Rough cut:** `outputs/demo_kit/rough_cut.mp4`, **3:00.00**. No voice-over; numbers burned in as lower-third captions.
- **Rebuild:** `python scripts/demo_kit/build_kit.py` re-cuts every shot and the rough cut; pass shot numbers to rebuild
  only those. Trims, captions and card text are in its `SHOTS` table and `cards()`.
- **Policy renders:** `python scripts/demo_kit/render_policies.py` writes `outputs/demo_kit/src/{jog,discus,shot_put,agent}.mp4`
  from each run's `policy_qpos.npy`, at 1080p (the runs' own `policy.mp4` files are 320×240). The discus and shot-put runs
  live in the frontier worktrees' `outputs/`; copy each policy folder's `policy_qpos.npy` and `policy_metrics.json` to the
  same path under `outputs/` first:
  - `remote_adapt_v2_discus/policy_multi` (from the frontier-adapt worktree);
  - `remote_travel_c3_shot_put/policy_base` and `policy_all` (from the frontier-travel worktree).
- **Source video:** shot 1 reads the jog's original from `clips_pro/athletes/4065467-uhd_2560_1440_30fps.mp4` (gitignored;
  `docs/real_athletes/clips.json` has its page and trim).
- **Re-record a take:** `scripts/demo_kit/record_takes.mjs`; its header has the setup. `BASE=https://athlete2humanoid.pages.dev`
  works: the recorder bypasses the site's CSP for its own page scripts.

## Shots

Times are positions in the rough cut. "Source" gives each segment in order, as `file [in–out s]`. For the raw takes,
`build_kit.py` resolves these from the take's `.marks.json` (an action plus an offset), so a re-recorded take is cut at
the same moments. `raw/` and `src/` are under `outputs/demo_kit/`.

| # | Time | File | Length | Source |
|---|---|---|---|---|
| 1 | 0:00–0:12 | `shot01_hook.mp4` | 12.0 s | twice: the jog's source video [0–6], a 960 px crop with the credit, beside `src/jog.mp4` [0–6] |
| 2 | 0:12–0:27 | `shot02_problem.mp4` | 15.0 s | problem card, slow push-in |
| 3 | 0:27–1:00 | `shot03_pipeline.mp4` | 33.0 s | `raw/build_jog.mp4` [0.8–18.8]: the replay, 01→06 · `raw/build_jog.mp4` [21.3–27.3]: orbit on 03 · `raw/build_kick.mp4` [28.0–33.0]: 04 with Muse's names · `raw/build_jog.mp4` [33.0–37.0]: 05 variants |
| 4 | 1:00–1:20 | `shot04_speed.mp4` | 20.0 s | `docs/results/adaptation/adaptation_curves.png`, push-in (7 s) · `src/discus.mp4` [0–4.37] slowed to 9 s · "18 minutes" card (4 s) |
| 5 | 1:20–1:45 | `shot05_robust.mp4` | 25.0 s | `raw/build_jog.mp4` [39.0–43.5]: Shove + Push · `raw/build_kick.mp4` [39.0–43.5] and [43.5–48.0]: two pushes · `src/shot_put.mp4` [0–5.23] slowed to 11.5 s |
| 6 | 1:45–2:15 | `shot06_agent.mp4` | 30.0 s | tool-calls card (10 s) · `src/agent.mp4` [0–6.27] slowed to 14 s · numbers card (6 s) |
| 7 | 2:15–2:40 | `shot07_live_env.mp4` | 25.0 s | `raw/live_kick.mp4` [1.6–103.2] at 12.7× · `raw/env_jog.mp4` [0.6–9.8] in 8 s: 05 and the download · terminal card with `python -m a2h_env info` (9 s) |
| 8 | 2:40–3:00 | `shot08_close.mp4` | 20.0 s | "Not yet" card (8 s) · end card (12 s) |

## Captions (rough cut, seconds within the shot)

| # | Captions |
|---|---|
| 1 | 0.3–5.8 A stock video of a stranger jogging → a G1 policy, in physics · 6.2–11.8 **100%** of clean runs · **98%** under 1 m/s pushes · **0 falls** |
| 2 | none (the card) |
| 3 | 0.3–17.9 one line per stage as its chip lights: 01 SAM 3.1 · 02 SAM 3D Body · 03 mink IK · 04 key moments · 05 variants · 06 PPO in mjlab, V-JEPA 2 judges the likeness · 18.1–23.9 planted feet stay planted · 24.1–28.9 Muse Spark names each key moment: "kick right leg forward" · 29.1–32.9 **36 of 36** variants kept |
| 4 | 0.3–6.8 task success vs minutes: scratch, a specialist, the 5-clip generalist · 7.2–15.8 1900 Olympic discus: **100%** clean · **88%** under 1 m/s pushes. Card: **18 minutes**; from scratch 24 min; punches and jumping jacks not in 30 min |
| 5 | 0.3–13.3 shoved, it recovers · unseen variants: squat jump **82%**, kick **100%** · 13.6–24.8 shot put: success **8% → 100%**. Panel labels: 73% of the glide, 0.42 m short / 96%, 0.05 m off |
| 6 | 0.3–9.8 its tool calls, with its reasons, from the run's log · 10.2–23.8 same 900 PPO iterations, without (middle) and with the agent (right). Card: **0.410 ± 0.008 vs 0.388 ± 0.017**, success and falls level, gain from upweighting the likeness terms, one run per arm, $0.10 of API |
| 7 | 0.2–7.9 live: a stock kick clip, built on the API's CPU in **101 s** (12.7×) · 8.1–15.9 every build hands back a training environment: one **2 MB** zip |
| 8 | none (the cards) |

The numbers' sources are listed under "Numbers quoted, and where they come from" in `DEMO_SCRIPT.md`.

## Screen takes (`outputs/demo_kit/raw/`)

Recorded headless at 1280×720 CSS px × 1.5 = 1920×1080 on this Mac's GPU, re-timed to 30 fps. The 2026-11-01 takes were
recorded from the deployed site (athlete2humanoid.pages.dev and the Cloud Run API); the earlier ones from a local server
with the same data.

| Take | Recorded | Length | What happens (seconds) |
|---|---|---|---|
| `build_jog.mp4` | 2026-11-01 | 56.8 s | `/build/latent_ra_jog`: 1.0 "Replay the build" · 16.8 all six ready · 19.3 swing to 03 · 21.5 orbit · 28.2 04 · 33.2 05 · 36.7 06 · 39.2 Shove + Push · 43.7 Push · 48.3 scroll to the RL-environment panel · 56.9 end |
| `env_jog.mp4` | 2026-11-01 | 11.1 s | `/build/latent_ra_jog`: 0.8 05 Variants · 4.3 scroll to the button · 6.5 hover · 8.2 click "Download training environment" · 8.3 downloaded (saved as `latent_ra_jog.a2h-env.zip`) |
| `live_kick.mp4` | 2026-11-01 | 116.8 s | `/live`: 2.0 the Mixkit kick (`clips_pro/athletes/49634-1080.mp4`) chosen · 6.8 video · 25.2 skeleton · 52.1 retarget · 55.6 moments · 102.6 built, 06 offers the pre-trained front kick · 107.8 play it · 111.8 Shove + Push |
| `build_kick.mp4` | 2026-10-31 | 56.8 s | `/build/latent_s3_kick`, the same actions and times as `build_jog` |
| `build_sj.mp4`, `live_upload.mp4`, `showcase.mp4` | 2026-10-31 | | the first storyboard's takes; not used now |

## Notes for the final cut

- **Who appears:** the Pexels jogger (shot 1, credited on screen; shot 3's picture-in-picture) and the Mixkit kicker (shot
  7, small). Your own phone clip appears only in the picture-in-picture of the front-kick takes (shot 3 at 24–29 s, shot 5
  at 4.5–13.5 s). To leave it out, swap those segments for `raw/build_jog.mp4` [chip04] and a second jog push.
- **Stage 06 of a live upload is pre-trained** and labelled so on screen; shot 7 cuts away from the live take before it.
- **The model badges** on the stage chips are deployed; a skipped Meta model (SAM 3.1 on a CPU build) is struck through.
- **Upgrades if there's time:** the shot-put and agent renders are dim on the dark floor; a lighter floor or a closer
  camera in `render_policies.py` would read better on a phone.
