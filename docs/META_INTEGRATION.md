# Meta model integration plan

Research done 2026-10-26, before the Model API key is approved. **Nothing here has been called live.** Facts come from Meta's docs (links in [Sources](#sources)). Anything marked *(unverified)* is an inference or still needs a test call.

**Short version.**
1. Muse Spark 1.3 accepts **video directly** and supports **strict JSON-schema output** and **tool calling**.
2. That supports two high-value integrations:
   - a **skill reader** that turns the clip into a training spec (bank neighbours, phases, reward and termination settings);
   - a **reward-design agent** that reads eval metrics and watches rollout videos between training rounds.
3. For the open-weights prize, **Muse Glimmer** (30B, Apache-2.0, text + image) is the only counted open model we can realistically self-host and fine-tune. Muse Spark Open (816B MoE) is reachable only through hosted providers.

---

## 1. What is available

### 1.1 Access basics (Meta Model API)

- **Base URL:** `https://api.meta.ai/v1`. It is OpenAI-SDK compatible (Chat Completions and Responses) and also Anthropic-Messages compatible. The auth header is `Authorization: Bearer <key>`.
  - Meta's examples read the key from `MODEL_API_KEY`. `motion_brief.py` and `muse/` read `MUSE_API_KEY` (or `META_API_KEY`) through `config.get_secret`.
- **Status: public preview for developers in the US.** It is subject to the Geographic Use Policy, and Muse Wonder is not offered in South Korea.
- **Muse Spark always reasons.**
  - `reasoning_effort` accepts `minimal`, `low`, `medium`, `high` or `xhigh` (and `max` on `muse-spark-1.3`). `none` returns HTTP 400.
  - Reasoning tokens are billed as output.
  - **Chat Completions does not carry reasoning across turns.** For multi-step agent loops Meta recommends the Responses API with `previous_response_id`.
- **Prefer the `developer` role over `system`.** `system` is kept for compatibility and merged at the same level.
- **`tool_choice`:** only `"auto"` is supported. `"required"`, `"none"` and named choices return HTTP 400, so the code must handle a plain-text reply when it expected a tool call.
- **Unsupported Chat Completions parameters:** `stop`, `n>1`, `logit_bias`, `logprobs`, `frequency_penalty` and `presence_penalty`. Leave `temperature` and `top_p` at their defaults.
- **Files API:** `POST /v1/files` with `purpose="user_data"`.
  - Uploads can be up to 1 GiB; inline media is capped at 50 MB.
  - Files never expire by default (optional `expires_after` of 1 h to 30 d), and the team quota is 100 GiB.
  - `video/mp4` is accepted: the model reads the frames and transcribes the embedded audio.

### 1.2 Counted models

| Model | Access / model ID | In → out | Video in? | Tools / structured output | Price | Rate limit (per team) |
|---|---|---|---|---|---|---|
| **Muse Spark** | API: `muse-spark-1.3` (latest), `muse-spark-1.2`, `muse-spark-1.1`; `-contributor` variants of 1.3 and 1.2 | text, image, **video**, audio, PDF → text. Context 1,048,576 tokens | **Yes**: Chat Completions `video_url` part (URL or base64, optional `fps`), Responses `input_video` / `input_file` | **Both.** Function tools plus built-in `web_search`; `response_format: json_schema` with `strict` | Standard $1.25 in / $4.25 out / $0.15 cached per M tokens. Contributor $0.10 / $0.20 (Meta may train on your content) | 3,000 RPM, 4M TPM (contributor: 150 RPM, 3M TPM) |
| **Muse Wonder** | API: `muse-wonder-1.0` (Standard mode only) | text, image, **video**, audio, PDF → text. Context 1M | **Yes** | **Both**, plus computer use. Reasoning effort up to `max` | $4.00 / $20.00 / $0.20 per M | 3,000 RPM, 4M TPM |
| **Muse Image** | API: `muse-image-1.0` via `/v1/images/generations`, `/v1/images/edits` (1–10 input images, no mask) or Responses (multi-turn) | text + image → image (webp, png or jpeg). `size` sets the aspect ratio only | No | No (agentic internally: can search the web and run code before rendering) | $0.01 per image | 150 RPM |
| **Muse Voice Transcribe** | API: `muse-voice-transcribe-1.0`. `POST /v1/asr/transcribe` (multipart) or realtime `wss://api.meta.ai/v1/asr/realtime` | audio → text, with turn-level timestamps, diarization, `keywords` biasing and 25 languages | No | No | $0.18 per audio hour | 256 concurrent streams, 16k streams/h |
| **SAM 3.1** | API: `sam-3.1` via `POST /v1/responses` (streaming recommended). Also open weights on HF `facebook/sam3.1`, which is what `track.py` uses today | text noun phrase or box/point prompts + image/**video** → boxes + masks per frame, with stable object IDs | **Yes.** MP4/MOV (H.264, H.265, MPEG-4), up to 15,000 frames (~10 min), 16 objects | No (special-token output, parsed by `meta-sam-parser`) | Video $0.20 per 1,000 frames (50-frame minimum per request); images $2.50 per 1,000 | 3,000 RPM |
| **Muse Spark Open** | Open weights: `meta-models/Muse-Spark-Open-816B-A42B` (Muse Model License + AUP). Hosted by Baseten, Fireworks, Groq, Modal, Nebius, Together and OpenRouter | text, image, video, audio → text. Context 1M | Yes | Tool calling is covered in its prompting guide. Structured output is not documented | Provider pricing. Partner credits: Baseten $100, Modal $100, Fireworks $25 (to "select participants") | Provider limits |
| **Muse Glimmer** | Open weights: `meta-models/Muse-Glimmer-30B` (bf16, ~60 GB), `-GGUF` (Q4_K_M ~17 GB fits under 24 GB VRAM; Q4_K_XL ~20 GB for 32 GB), `-assistant` (spec-decode draft). **Apache-2.0** | text + image → text (no video). Dense 30B, 128K context. Runs on vLLM, SGLang, llama.cpp or Ollama | **No.** Send keyframes or contact sheets | **Tool calling yes**, one call per turn, no parallel calls. JSON output not documented; vLLM guided decoding can enforce it *(unverified for this model)* | Our GPU time | n/a |

**Post-training (Glimmer):** Meta documents Axolotl (QLoRA, bf16 LoRA and full SFT with FSDP2), TRL (SFT, DPO, GRPO) and TitanRL (GRPO with a verifiable reward).
- QLoRA has been verified on a single H100, peaking at about 27 GB per GPU, so a Colab A100 80 GB is plausible *(unverified)*.
- Data is chat JSONL with images inline in `messages[*].content`, with packing turned off.

### 1.3 What this means for our code

- **Model ID:** `config.MUSE_MODEL` defaults to `muse-spark-1.1`. That ID is valid but two releases old, and 1.3 costs the same. Set `MUSE_MODEL=muse-spark-1.3` (env only, no code change). Avoid `-contributor` for real footage of people: it lets Meta train on the content.
- **Gaps in `motion_brief.ask` today:**
  - it sends images only (no `video_url` part);
  - it uses `json_object` instead of a strict `json_schema`;
  - it uses the `system` role;
  - it has no Responses-API path for multi-turn tool use.

  The sketches below add all four.
- **Anti-project risk (fixed Oct 27):** `coach.py` said "biomechanics coach" and returned a `coaching` field, too close to the "sports analyzers or coaches" anti-project. It is now `motion_brief.py`, and every Muse prompt and output is phrased as a robot training spec, with nothing addressed to the person in the video.
- **Rough costs.** The video token rate is not published; check it with the token-counting endpoint once the key works.
  - A 4 s clip sampled at 4 fps is 16 frames, roughly 16 × 1.4k ≈ 22k input tokens at the image rate, about $0.03 per skill-reader call *(unverified)*.
  - An agent round (two short videos, metrics, about 5k reasoning tokens out) costs about $0.08.
  - SAM video costs about $0.02 per 4 s clip; Muse Image $0.01 per picture.
  - The $150 hackathon credit covers thousands of calls, so cost is not the constraint. GPU time per training round is.

### 1.4 Hackathon rules: what we could and couldn't verify

- **Public event page (verified):** names the Muse Spark series, Muse Wonder, Muse Image and SAM 3.1 through the Model API, plus "Meta open weights models". It also states the *Push the Frontier* theme ("Give a machine a new capability…"), the partner credits, and that submissions are due Nov 4, 9 PM ET.
- **Details page:** `/details` redirects to a Cerebral Valley login, so I did not read it.
- **From `PLAN.md`, not re-verified here:**
  - rubric weights: Impact 30, Meta tooling 25, Technical depth 20, Demo 25;
  - bonus prizes: Best Agentic $5K, Best Open Weights $10K, Most Viral $20K;
  - the anti-project list.

  A web search turned up third-party listings quoting the same rubric weights, but I couldn't trace them to an official page.
- **Still open from `DAY1_CHECKLIST.md`:** does SAM 3.1 run *locally from HF weights* count as "using Meta models", or only through the API? Integration 4 removes the doubt.

---

## 2. Proposed integrations, ranked

**How the scores are computed.**
- Each integration is scored 1–5 per rubric criterion and weighted 0.30 / 0.25 / 0.20 / 0.25.
- "Bonus" is the prize it most strengthens.
- Effort is for one person who already knows this repo.
- The ranking is by expected points. The **build order** in §4 differs because #2 needs #1's output.

| # | Integration | Impact | Meta tooling | Depth | Demo | **Score** | Bonus fit | Effort | Risk |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Reward-design agent loop (Muse Spark, Responses API, tools + video) | 4 | 5 | 5 | 5 | **4.70** | **Best Agentic** | 8–12 h | Med–High |
| 2 | Skill reader: video → training spec (Muse Spark, video + strict schema) | 4 | 5 | 3 | 4 | **4.05** | Agentic (feeds #1) | 4–6 h | Low–Med |
| 3 | Open-weights rollout critic (Muse Glimmer, zero-shot → QLoRA) | 3 | 4 | 5 | 2 | **3.40** | **Best Open Weights** | 5 h zero-shot; +10–14 h fine-tune | High |
| 4 | SAM 3.1 through the Model API (backend switch in `track.py`) | 3 | 4 | 2 | 3 | **3.05** | none | 3–4 h | Low |
| 5 | Spoken edits: "make it jump higher" (Voice Transcribe → #1's tools) | 2 | 3 | 2 | 4 | **2.75** | Most Viral, Agentic | 3–4 h (after #1) | Med |
| 6 | Muse Spark Open as the agent's brain via a hosted provider | 2 | 3 | 1 | 1 | **1.80** | Open Weights (weak) | 1–2 h | Med |
| 7 | Muse Image demo visuals (title cards, thumbnail, X post) | 1 | 2 | 1 | 3 | **1.75** | Most Viral | 1–2 h | Low |

### #1: Reward-design agent loop (Best Agentic)

- **What it does:**
  - Muse Spark gets the skill spec from #2, the eval metrics and the comparison video.
  - It calls a `train` tool with a reward/termination config. It may call it up to 3× in parallel: the A100 already runs three mjlab trainings at once in `experiments/latent_policy.py`.
  - It watches the new rollouts, and repeats or calls `finish`.
  - Every round is logged, which gives the demo its "agent fixes a failed skill" chart.
- **Where it plugs in:**
  - The loop lives in **`motion_brief.py`** (`reward_agent()`) and stays torch-free.
  - A new driver, `experiments/reward_agent.py`, supplies `run_round(cfg)`. That function wraps `train_motion.train_motion(..., task=A2H-Hybrid…, bundle=…)` and the step-5 evaluation from `experiments/latent_policy.py`, which needs a small refactor into an `evaluate(out, name)` function.
  - The config reaches mjlab through a JSON file whose path is in an env var (`A2H_SKILL`), read by `mjlab_tasks.latent_env_cfg`. Today `latent_env_cfg` hard-codes `tracking_scale`, `SIGMA`, the latent weight `3.0` and the termination thresholds.
- **Input:** `skill.json`, the `eval.json` fields (`aligned`, `dtw`, `dtw_rank_vs_bank`, `falls`, `mean_torso_position_error_m`) and `comparison.mp4` (video silhouette next to the robot silhouettes, the framing V-JEPA sees).
- **Output:** `agent_log.jsonl` (one line per round: config, metrics, Muse's stated reason), the best config, and its policy.
- **Why it scores:**
  - It is real tool use with a verifiable objective: the real V-JEPA score, not the distilled net, so the agent can't game its own reward model.
  - It uses video understanding, and it is exactly the "agent improves a failed skill live" beat in `PLAN.md`.
- **Risks:**
  - **GPU time per round.** Use short runs (e.g. `--iters 400`, then confirm the winner at 1500) and measure wall-clock before promising N rounds.
  - **No improvement.** The agent may thrash. Keep the best-so-far config, cap at 4 rounds, and show the log honestly.
  - **Video in tool turns.** `function_call_output` is a string, so the rollout video goes in a follow-up user message in the same `input` list. This should work per the Responses schema *(unverified)*. Fallback: a keyframe contact sheet as `input_image`.

### #2: Skill reader (closes the stage-1 Muse gap)

- **What it does:** a single Muse Spark call watches the raw clip and returns a strict-schema spec:
  - the action name;
  - the 3 nearest bank motions, chosen from an enum of `latent.bank.motion_bank()` names;
  - phases (in seconds) with a keyframe each, and whether that phase should be matched by `pose` (tracking), `style` (latent) or left `free`;
  - global reward weights and termination thresholds.
- **Where it plugs in:**
  - Add `read_skill()` alongside `motion_brief.segment_phases()`.
  - Call it in `experiments/latent_policy.py` right after step 1 (`humans()`), and write `out/skill.json`.
  - Uses, in order of value:
    1. Pass the reward and termination numbers to the A2H tasks (the same `A2H_SKILL` hook as #1).
    2. Weight the distillation dataset toward the named bank neighbours.
    3. v2: a phase gate in `mjlab_tasks` that scales tracking terms by phase via `cmd.time_steps`. This is the TRACC-style "fade tracking out after takeoff" from `RESEARCH.md`.
- **Input:** the clip MP4 (inline base64, ≤50 MB) at `fps=4`, plus the `retarget_report.json` summary and `meta.json` fps.
- **Output:** `skill.json`.
- **Why it scores:** it is Muse doing perception that pose estimation can't (intent, phases, which parts of the motion matter). It is cheap, it is the input #1 needs, and it is on screen in the first 20 s of the demo.
- **Risks:**
  - **Phase timing.** The sampled `fps` limits precision (±0.25 s at 4 fps), so snap phase boundaries to retarget extrema (e.g. pelvis-height minimum) in code.
  - **Wrong bank choice.** Validate against the V-JEPA DTW ranking we already compute (`_bank_top5_dtw`) and log disagreements; that comparison is itself a nice result.
  - **Schema bounds.** Numeric min/max may not be enforced under `strict`, so clamp in code.

### #3: Open-weights rollout critic (Best Open Weights)

- **What it does:** Muse Glimmer sees a contact sheet: 6 keyframes of the human silhouette above 6 of the robot rollout. It answers in fixed JSON: `{same_action: bool, failing_phase: str|null, failure: enum[fell, too_shallow, too_slow, wrong_limb, ok]}`.
- **Two stages:**
  1. **Zero-shot** behind vLLM or llama.cpp (Q4_K_M under 24 GB: a Colab L4 is enough, or Modal Endpoints with the partner credits).
  2. **QLoRA** on simulator-labelled data, which we get for free: the distillation dataset's bank motions and random variations have known identities and parameters, and rollouts have `falls` and tracking errors.
- **Where it plugs in:**
  - A new `latent/critic.py` (contact-sheet builder and client; same OpenAI SDK, different `base_url`).
  - Used as an extra tool in #1 (`critique(run)`): cheap and local, so it can run on every checkpoint, not just at the end of a round.
  - Evaluated against Muse Spark and real V-JEPA on held-out clips. This is "Critic fine-tuned and compared with Muse Spark", Day 7 of `PLAN.md`.
- **Input:** a PNG contact sheet.
- **Output:** JSON verdicts, and an accuracy table for Glimmer (zero-shot vs QLoRA) vs Muse Spark.
- **Risks:**
  - The fine-tune is the biggest time sink on the list, and QLoRA has been verified only on an H100.
  - The zero-shot model may already be "good enough", which makes the fine-tune an uninteresting result.
  - Glimmer can't take video, so we lose temporal detail.
  - **Cut rule:** if the zero-shot critic isn't serving by Oct 31, ship zero-shot only and fall back to #6 for the open-weights story.

### #4: SAM 3.1 through the Model API

- **What it does:** with `A2H_SAM_BACKEND=api`, `track.athlete_masks` uploads the clip once (Files API) and streams `sam-3.1` with `{"type":"input_text","text":"person"}` + `{"type":"input_video","file_id":…}` and `metadata.include_confidence="true"`. It then parses the output with `meta-sam-parser` (`decode_mask_to_raster`) into the same `{frame: {out_obj_ids, out_probs, out_binary_masks}}` dict that `track.main_track_masks` already consumes.
- **Where it plugs in:** `track.py` gets a second backend next to the in-process and `A2H_SAM_PYTHON` subprocess paths. `sam_track.py`'s athlete→person fallback logic is reused unchanged.
- **Why:**
  - It removes the separate numpy<2 / Python 3.12 environment and the gated HF download.
  - It makes `app.py` (upload a clip) run on a CPU host.
  - It puts SAM 3.1 unambiguously in the "via Model API" column.
- **Risks:**
  - The Python parser's stream API is not documented (only the mask decoders are named).
  - Mask rasters come back smaller than the frame, so upsample them with nearest-neighbour to `(H, W)`.
  - Frame lines can arrive out of order: key on the `<Nf>` marker.
  - About $0.02 per clip.

### #5: Spoken edits ("make it jump higher")

- **What it does:**
  1. A Gradio mic in `app.py` records WAV at 16 kHz mono.
  2. The audio goes to `POST /v1/asr/transcribe` (`mode: PUSH_TO_TALK`, `keywords: ["squat","jump","kick","lunge","higher","deeper","faster"]`).
  3. The transcript becomes a user turn in #1's agent, which edits the spec through the same `train` tool and retrains.
- **Input / output:** audio → transcript → `train` call.
- **Risks:**
  - A retrain takes minutes, so this can't be truly live on stage. Record it, and be honest in the video that it is time-lapsed.
  - Little technical depth on its own; it is mainly demo and virality value.

### #6: Muse Spark Open as the agent's model

- **What it does:** a pure `base_url` + `model` swap on #1's client. It points at a hosted provider (Baseten, Fireworks, Together or Modal) running `Muse-Spark-Open-816B-A42B`, and it keeps video input.
- **Why:** it is the cheapest way to show an open-weights model doing the agentic part.
- **Risks:**
  - Provider model IDs and Responses-API support vary (Chat Completions may be the only option, which loses reasoning carry-over).
  - Partner credits go only to "select participants".
  - "We called a hosted endpoint" is a weak open-weights story next to a fine-tuned Glimmer (#3).

### #7: Muse Image demo visuals

- **What it does:** `/v1/images/edits` with the athlete's frame and the robot's rollout frame produces stylised title cards, the YouTube thumbnail and X-post art.
- **Effort and cost:** $0.01 each, 1–2 h.
- **Rule:** label generated images as illustrations in the video; never pass one off as a simulation result.

**Considered and not ranked separately:**
- **Muse Wonder** as a drop-in "senior reviewer" for #1's final round. It is the same API and video input at 3–5× the price. Swapping the model ID gives a cheap Spark-vs-Wonder ablation if there's time.
- **Search grounding.** Nothing in this pipeline needs the web.

---

## 3. Code sketches for the top two

Both are written to drop into `motion_brief.py`. They reuse its lazy `client()`, `config.MUSE_MODEL` and the base64-inline style, and stay free of torch and mjlab imports. They are **sketches**: field names follow Meta's docs, but nothing has been run against the API.

### 3.1 Shared: video parts, strict schemas, `developer` role

```python
# motion_brief.py: replaces ask(). Same call sites keep working.

def _data_url(path, mime):
    return f"data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode()}"


def video_part(path, fps=4):
    """Chat Completions video part. Inline limit is 50 MB; above that, upload with
    client().files.create(file=..., purpose="user_data") and send {"type": "file", "file": {"file_id": id}}."""
    return {"type": "video_url", "video_url": {"url": _data_url(path, "video/mp4"), "fps": fps}}


def image_part(path):
    mime = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
    return {"type": "image_url", "image_url": {"url": _data_url(path, mime)}}


def ask(prompt, system=None, images=(), videos=(), json_mode=False, schema=None, effort="low", **kwargs):
    """One Muse Spark call. schema: a JSON Schema dict -> strict structured output, parsed."""
    content = [{"type": "text", "text": prompt}, *map(image_part, images), *map(video_part, videos)]
    messages = ([{"role": "developer", "content": system}] if system else []) + [
        {"role": "user", "content": content if images or videos else prompt}
    ]
    if schema:
        kwargs["response_format"] = {"type": "json_schema",
                                     "json_schema": {"name": "out", "schema": schema, "strict": True}}
    elif json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client().chat.completions.create(model=config.MUSE_MODEL, messages=messages,
                                            reasoning_effort=effort, **kwargs)
    text = resp.choices[0].message.content
    return json.loads(text) if schema or json_mode else text
```

### 3.2 #2 Skill reader

```python
# motion_brief.py

SKILL_SYSTEM = (
    "You turn a video of a person moving into a training spec for a Unitree G1 humanoid robot "
    "that will learn the same motion in a physics simulator. Describe the motion, not the person; "
    "never give the person advice. Times are seconds from the start of the clip."
)


def _obj(**props):  # strict mode: every property required, no extras
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def skill_schema(bank_names):
    num, txt = {"type": "number"}, {"type": "string"}
    phase = _obj(name=txt, start_s=num, end_s=num, keyframe_s=num,
                 match={"type": "string", "enum": ["pose", "style", "free"]},  # tracking / latent / neither
                 note=txt)
    return _obj(
        action=txt,
        nearest_bank={"type": "array", "items": {"type": "string", "enum": list(bank_names)}},
        phases={"type": "array", "items": phase},
        reward=_obj(tracking_scale=num, latent_weight=num, sigma=num),
        terminations=_obj(anchor_pos_m=num, anchor_ori_rad=num, ee_body_pos_m=num),
        invariants={"type": "array", "items": _obj(name=txt, description=txt)},
    )


SKILL_PROMPT = """Clip: {seconds:.1f} s at {fps:.0f} fps. Retarget error per body (cm, mean/max): {retarget}.
1. Name the action. Pick the 3 bank motions closest to it, most similar first.
2. Split it into phases with a keyframe each. Mark a phase "pose" if the robot should copy the exact pose
   (e.g. the bottom of a squat), "style" if it should only look like the video, "free" for set-up frames.
3. Reward: tracking_scale in [0, 1] (0 = latent reward only), latent_weight in [0.5, 5], sigma in [0.05, 0.3].
   Terminations: how far the torso (m, rad) and hands/feet (m) may drift from the retarget before the
   attempt counts as failed; allow more drift for dynamic moves.
4. Two to four physical invariants the simulator could check (e.g. "both feet leave the ground")."""


def read_skill(video_path, bank_names, meta, retarget_report, fps=4):
    """Muse Spark watches the clip -> training spec (see docs/META_INTEGRATION.md #2)."""
    prompt = SKILL_PROMPT.format(seconds=meta["frames"] / meta["fps"], fps=meta["fps"],
                                 retarget=json.dumps(retarget_report))
    spec = ask(prompt, system=SKILL_SYSTEM, videos=[video_path], schema=skill_schema(bank_names), effort="medium")
    # strict mode constrains shape, not ranges: clamp before anything reaches mjlab
    clip = lambda v, lo, hi: float(min(max(v, lo), hi))
    r, t = spec["reward"], spec["terminations"]
    r |= {"tracking_scale": clip(r["tracking_scale"], 0, 1), "latent_weight": clip(r["latent_weight"], 0.5, 5),
          "sigma": clip(r["sigma"], 0.05, 0.3)}
    t |= {"anchor_pos_m": clip(t["anchor_pos_m"], 0.25, 1.0), "anchor_ori_rad": clip(t["anchor_ori_rad"], 0.8, 2.0),
          "ee_body_pos_m": clip(t["ee_body_pos_m"], 0.25, 1.0)}
    return spec
```

Call site in `experiments/latent_policy.py`, after step 1:

```python
skill_f = out / "skill.json"
if not skill_f.exists():
    import motion_brief
    d = tgt["dir"]
    spec = motion_brief.read_skill(d / "source.mp4",  # the trimmed input clip (save it there, or pass the original path)
                            bank_motions(config.OUTPUT_DIR / "cache").keys(),
                            json.loads((d / "meta.json").read_text()),
                            json.loads((d / "retarget_report.json").read_text()))
    skill_f.write_text(json.dumps(spec, indent=1))
# train_motion(...) then runs with env A2H_SKILL=skill_f; mjlab_tasks.latent_env_cfg reads
# reward/terminations from it instead of the hard-coded 0.2 / 3.0 / 0.5 / 1.2 / 0.6.
```

### 3.3 #1 Reward-design agent (Responses API, tool calls, video)

```python
# motion_brief.py

AGENT_SYSTEM = (
    "You tune the reward of a reinforcement-learning run that teaches a Unitree G1 humanoid a motion "
    "from a human video. Each call to `train` runs PPO with the config you give and returns the "
    "evaluation; you then see a video: the human silhouette, then each robot rollout. Higher `aligned` "
    "and lower `dtw_rank_vs_bank` (1 = best) mean the robot looks more like the video; falls must be 0. "
    "Change one or two settings per run and say why. Call `finish` when no run improves on the best."
)

_num = {"type": "number"}
AGENT_TOOLS = [
    {"type": "function", "name": "train", "strict": True,
     "description": "Train and evaluate one policy (~10-15 min). You may call it up to 3 times in parallel.",
     "parameters": _obj(label={"type": "string"}, reason={"type": "string"},
                        reward=_obj(tracking_scale=_num, latent_weight=_num, sigma=_num),
                        terminations=_obj(anchor_pos_m=_num, anchor_ori_rad=_num, ee_body_pos_m=_num))},
    {"type": "function", "name": "finish", "strict": True,
     "description": "Stop tuning and name the best run.",
     "parameters": _obj(best_label={"type": "string"}, summary={"type": "string"})},
]


def _upload(path):
    """Responses takes uploaded videos by file id (up to 1 GiB, vs 50 MB inline)."""
    return client().files.create(file=open(path, "rb"), purpose="user_data",
                                 expires_after={"anchor": "created_at", "seconds": 86400}).id


def _watch(text, video):
    return {"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": text}, {"type": "input_video", "file_id": _upload(video)}]}


def reward_agent(skill, baseline, run_round, max_rounds=4, log_path=None):
    """skill: read_skill() output. baseline: {"metrics", "video"} of the default config.
    run_round(label, cfg) -> {"metrics": dict, "video": path}; the caller owns GPUs and mjlab.
    Returns (best_label, results). Each round is appended to log_path as one JSON line."""
    from concurrent.futures import ThreadPoolExecutor
    results = {"baseline": baseline}
    inp = [_watch(f"Training spec:\n{json.dumps(skill)}\n\nBaseline metrics:\n{json.dumps(baseline['metrics'])}",
                  baseline["video"])]
    prev = None
    for rnd in range(max_rounds):
        resp = client().responses.create(model=config.MUSE_MODEL, instructions=AGENT_SYSTEM, input=inp,
                                         tools=AGENT_TOOLS, previous_response_id=prev,
                                         reasoning={"effort": "medium"})  # carried across turns here
        prev = resp.id
        calls = [o for o in resp.output if o.type == "function_call"]
        if not calls:  # tool_choice can only be "auto": a text reply means it's done talking
            break
        args = {c.call_id: json.loads(c.arguments) for c in calls}
        done = next((args[c.call_id] for c in calls if c.name == "finish"), None)
        if done:
            _log(log_path, {"round": rnd, "finish": done})
            return done["best_label"], results
        trains = [c for c in calls if c.name == "train"][:3]
        with ThreadPoolExecutor(len(trains)) as pool:  # one GPU fits three mjlab trainings
            outs = list(pool.map(lambda c: run_round(args[c.call_id]["label"], args[c.call_id]), trains))
        inp = []
        for c, o in zip(trains, outs):
            label = args[c.call_id]["label"]
            results[label] = o
            _log(log_path, {"round": rnd, "label": label, "cfg": args[c.call_id], "metrics": o["metrics"]})
            inp.append({"type": "function_call_output", "call_id": c.call_id, "output": json.dumps(o["metrics"])})
            inp.append(_watch(f"Rollout video for run {label!r}.", o["video"]))
    best = max(results, key=lambda k: results[k]["metrics"].get("aligned", -1))
    return best, results


def _log(path, row):
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
```

The driver, `experiments/reward_agent.py` (new), keeps the GPU side out of `motion_brief.py`:

```python
def run_round(label, cfg):
    d = out / f"agent_{label}"
    (d / "skill.json").parent.mkdir(parents=True, exist_ok=True)
    (d / "skill.json").write_text(json.dumps(skill | cfg))       # reward + terminations override
    os.environ["A2H_SKILL"] = str(d / "skill.json")              # read by mjlab_tasks.latent_env_cfg
    train_motion(tgt["csv"], 30, d, iters=400, task="A2H-Hybrid-Flat-Unitree-G1", bundle=bundle)
    metrics = evaluate(d)                                         # step 5 of latent_policy.py, refactored
    return {"metrics": metrics, "video": d / "comparison.mp4"}
```

Two notes on the sketch:

- Setting `os.environ` from parallel threads is racy. In the real version, pass `env=` through `train_motion` the way `bundle` already is.
- The confirm step (re-train the winner at 1500 iterations) runs after `reward_agent` returns.

---

## 4. Recommended build order (Oct 26 – Nov 4)

Merged with the milestones in `PLAN.md`. Cut order if behind: #3 fine-tune → #5 → #6 → #7. **Never cut #2, #1 or the backup video.**

| Day | Date | Meta work | Done when |
|---|---|---|---|
| 1 | Mon Oct 26 | (Squat run continues.) As soon as the key lands: smoke-test `muse-spark-1.3`. Test calls: a text call, one `video_url` call on the squat clip, one strict `json_schema` call and one Responses tool call; also count the clip's video tokens. Set `MUSE_MODEL=muse-spark-1.3`. | Four test calls pass; video token cost recorded in `DAY1_CHECKLIST.md` |
| 2 | Tue Oct 27 | Build **#2 skill reader** and the new `ask()`. Remove the coach framing (done: `motion_brief.py`). Add the `A2H_SKILL` hook in `mjlab_tasks.latent_env_cfg` (global weights + terminations). Attend Agents 101. | `skill.json` for 3 clips; the squat retrains from a Muse-chosen config |
| 3 | Wed Oct 28 | **#1 agent loop**, first with a fake `run_round` (canned metrics) to debug tool turns, then real. Pick a move the default reward fails on (the bedroom jumps). | One real round end to end; `agent_log.jsonl` written |
| 4 | Thu Oct 29 | Run the agent for 3–4 rounds on the failing move; confirm the winner at full length. **Record the backup demo video.** | Logged improvement in `aligned`/rank over the baseline (or an honest "no gain" with the reasons) |
| 5 | Fri Oct 30 | **#4 SAM via API** (3–4 h). Serve **Glimmer zero-shot** (#3) on an L4 or Modal; build the contact sheets. Fireworks fine-tuning session. | `pipeline.py` runs with `A2H_SAM_BACKEND=api` on a CPU machine; Glimmer answers critic prompts |
| 6 | Sat Oct 31 | Label rollouts from the distillation set and agent runs; start the Glimmer **QLoRA** (Modal or A100). **Go/no-go at noon:** if not training by then, keep zero-shot and do #6 instead (1–2 h). | Fine-tune running, or #6 swapped in |
| 7 | Sun Nov 1 | Critic evaluation: Glimmer zero-shot vs QLoRA vs Muse Spark vs real V-JEPA. Add `critique` as a tool in #1. | Accuracy table in `RESEARCH.md` |
| 8 | Mon Nov 2 | `app.py`: upload → SAM API → retarget → skill reader → (cached) policy video. Add **#5 voice edits** if #1 is solid. **Feature freeze.** | App runs on a CPU host against cached policies |
| 9 | Tue Nov 3 | **#7 Muse Image** title cards and thumbnail; 3-min video; README section on the Meta models used (this table); X post. | Video exported |
| 10 | Wed Nov 4 | Buffer. Submit before 9 PM ET. | Submitted |

**Open questions to settle as soon as the key works or at office hours:**
1. Token cost per second of video, and whether `fps` above 4 is allowed.
2. Whether a Responses `function_call_output` can be followed by an `input_video` message in the same `input` list.
3. Whether self-hosted SAM 3.1 weights count as using the model.
4. Whether a fine-tuned Glimmer qualifies for "Best Build on Meta Open Weight Models" when the agent also uses the API.

---

## 5. Implemented: `muse/` (Oct 27)

The top two integrations are built as the `muse/` package. They differ from the §3 sketches in two ways:
- Each call is a single Chat Completions request with a strict `json_schema`.
- The agent is a CLI that loops over finished runs rather than a multi-turn tool loop, so no training happens inside the agent.

Everything runs offline against fixture responses until the key arrives.

| File | What it does |
|---|---|
| `muse/client.py` | `MuseClient`, with two modes. **Live:** OpenAI SDK pointed at `config.META_API_BASE`, model `config.MUSE_MODEL` (now `muse-spark-1.3`), key `MUSE_API_KEY` via `config.get_secret` (`META_API_KEY` still accepted). **Fixture:** no network and no key. Every response is validated against the schema and a semantic check; on failure Muse sees the errors and gets one retry. `record_dir` saves live exchanges in fixture format. |
| `muse/schema.py` | Strict-mode schema helpers, a stdlib validator, and `strict_errors()`, which flags schemas Muse would reject with HTTP 400. |
| `muse/media.py` | Clip → small silent H.264 MP4 (480 px, via ffmpeg / `imageio-ffmpeg`). Rollout video → PNG contact sheet (OpenCV). Both are optional imports. |
| `muse/skill_reader.py` | Video → `skill_spec.json` next to the clip's pipeline outputs. Contents: action label and nearest bank motions, tempo, phases (time, keyframe, `pose`/`style`/`free`, key bodies), weighted key bodies, and success criteria in simulator metrics. |
| `muse/reward_agent.py` | Run folder (`eval.json`, `distill_report.json`, contact sheets of `comparison.mp4` and `policy_<variant>/policy.mp4`, optional `skill_spec.json`) → `reward_proposal.json` (diagnosis + RFC 6902 patch) and `next_reward_config.json`. Patches are bounded per path, with at most 3 ops and no no-ops. Each proposal is appended to `agent_log.jsonl`, which the next proposal reads as history. |
| `muse/fixtures/*.json` | Hand-written stand-in responses in the API's shape. Replace them with `--record` output. |
| `tests/test_muse.py`, `tests/test_reward_config.py` | 28 stdlib `unittest` tests: fixture responses, real run data copied from `outputs/remote_latent` into `tests/data/`, and stub mjlab modules. |

```bash
python -m unittest discover -s tests -t .
```

**Now, without a key:**

```bash
python -m muse.skill_reader clips/squat.mov --out outputs/squat --dry-run
```

```bash
python -m muse.reward_agent outputs/remote_latent --dry-run --out-dir /tmp/agent
```

**When the key arrives** (save it as `a2h/env/MUSE_API_KEY` on Drive, or `MUSE_API_KEY=` in `.env`):

```bash
python -m muse.skill_reader clips/*.mov --out-root outputs/remote_latent/clips --record muse/fixtures/recorded
```

```bash
python -m muse.reward_agent outputs/latent_squat --record muse/fixtures/recorded
```

```bash
python -m muse.reward_agent outputs/ --watch 300 --log outputs/agent_log.jsonl
```

`pipeline.py ... --skill live` (or `--skill dry-run`) runs the skill reader as the last pipeline stage.

**Wired into training (Oct 27):**
- `train_motion.py --reward-config next_reward_config.json` copies the file to `<out>/reward_config.json`, points `A2H_REWARD_CONFIG` at the copy, and takes its `iterations` in place of `--iters`.
- `mjlab_tasks.reward_config()` then overrides `latent_weight`, `sigma` (passed to `LatentReward` as a term param), `tracking_scale` and the termination thresholds. The latent-only task keeps `tracking_scale` 0.
- `experiments/latent_policy.py --reward-config` passes the file through for the hybrid and latent policies; the tracking baseline is left unchanged.
- `reward_agent` reads `policy_<variant>/reward_config.json` back as the run's current config.
- `tests/test_reward_config.py` checks the round trip (proposal → env cfg) against stub mjlab modules, and that `reward_agent.DEFAULTS` equals `mjlab_tasks`' defaults.

**Still open:** the success-criteria metrics (`METRICS` in `skill_reader.py`) are defined, but nothing computes them from a rollout yet.

---

## Sources

Meta Model API docs (fetched 2026-10-26; `ai.developer.meta.com/docs/*` redirects to `dev.meta.ai/docs/*`):
- Overview and navigation: https://dev.meta.ai/docs/
- Models (IDs, modalities, features, open-weight entries): https://dev.meta.ai/docs/models
- Pricing and rate limits: https://dev.meta.ai/docs/pricing-rate-limits
- Chat Completions protocol (supported/unsupported params, `developer` role, no reasoning carry-over): https://dev.meta.ai/docs/protocols/chat-completions
- Chat Completions schemas (`video_url` with `fps`, `input_audio`, `file`, `response_format` variants): https://dev.meta.ai/docs/api-reference/chat-completions/schemas
- Video understanding: https://dev.meta.ai/docs/video-understanding
- Image understanding (50 images per request, 50 MB inline, ~1.3–1.5k tokens per 1280 px image): https://dev.meta.ai/docs/image-understanding
- Structured output (`json_schema`, strict subset, limits): https://dev.meta.ai/docs/structured-output
- Tool calling (`tool_choice` auto only, parallel calls, Responses vs Chat formats): https://dev.meta.ai/docs/tool-calling
- File handling (Files API, 1 GiB, expiry, MIME types): https://dev.meta.ai/docs/file-handling
- Media segmentation (SAM 3.1): https://dev.meta.ai/docs/media-segmentation
- SAM overview / segmenting / reading output / client libraries: https://dev.meta.ai/docs/sam/overview, https://dev.meta.ai/docs/sam/segmenting, https://dev.meta.ai/docs/sam/reading-segmentation, https://dev.meta.ai/docs/sam/client-libraries
- Speech to text (Muse Voice Transcribe): https://dev.meta.ai/docs/speech-to-text
- Image generation (Muse Image): https://dev.meta.ai/docs/image-generation
- Open-weight models: https://dev.meta.ai/docs/open-weight-models
- Muse Spark Open: https://dev.meta.ai/docs/muse-spark-open (license: https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B/blob/main/LICENSE.md)
- Muse Glimmer: https://dev.meta.ai/docs/muse-glimmer, https://dev.meta.ai/docs/muse-glimmer/get-the-model, https://dev.meta.ai/docs/muse-glimmer/prompting, https://dev.meta.ai/docs/muse-glimmer/post-training
- "Build with Muse Spark" blog (US-only preview, reasoning-effort range, pricing): https://dev.meta.ai/resources/blog/build-with-muse-spark/

Hackathon:
- Event page (models, themes, partner credits, dates): https://cerebralvalley.ai/e/meta-ai-developer-hackathon
- Details page (requires login, not read): https://cerebralvalley.ai/e/meta-ai-developer-hackathon/details
- Rubric, bonus prizes and anti-project list as recorded in this repo: `PLAN.md`
