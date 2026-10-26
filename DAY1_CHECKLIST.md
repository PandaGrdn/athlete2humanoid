# Day 1 checklist

## Tonight (Oct 25): things that take time to get approved

- [ ] **Confirm your hackathon registration went through.** One listing gave Oct 25 as the application deadline.
- [ ] **Request SAM 3.1 weights access** at https://huggingface.co/facebook/sam3.1 ("Agree and access"). The approval is manual and can take hours, so do this first.
- [ ] Create a Hugging Face **read token** (Settings → Access Tokens). You'll add it as the `HF_TOKEN` Colab secret.
- [ ] **Meta developer console:** sign up, create a Model API key, and claim the free credits plus any hackathon credits. The preview is US-only, so if you're outside the US, ask at kickoff how participants get access.
- [ ] **Colab:** check that you can get an **L4 or A100** GPU (usually needs Colab Pro or compute units). SAM 3.1 won't run on a T4.
- [ ] **GitHub:** create an empty repo called `athlete2humanoid` and push this folder:
  ```bash
  git init && git add . && git commit -m "Day 0 scaffold" && git branch -M main
  git remote add origin https://github.com/<you>/athlete2humanoid.git && git push -u origin main
  ```
  If the repo is private, Colab needs a token in the clone URL. Making it public is simpler, and judges will want to see it anyway.
- [ ] 30 min of tutorials:
  - MuJoCo basics: https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb
  - mink: read `retarget.py` here, then https://github.com/kevinzakka/mink/blob/main/examples/humanoid_g1.py
- [ ] Shortlist 3 candidate clips: 2–4 s, one athlete, full body in frame, side view, steady camera. Squat jump > lunge > kick. Filming yourself or a friend is the safest option for licensing.

## Day 1 (Oct 26)

1. **Kickoff.** Record the answers below.
2. Open `notebooks/day1_setup.ipynb` in Colab (File → Upload notebook, or open it from GitHub), set `REPO_URL`, and run it top to bottom.
3. Upload your chosen clip to Drive at `a2h/clips/raw.mp4`, set `START` and `DURATION` in step 6, and re-run step 6 until SAM tracks the right person in every frame.

### Kickoff answers

| Question | Answer |
|---|---|
| Muse Spark inputs: text only, or images/video too? (Notebook step 5 tests this.) | |
| Muse Spark model name and base URL (default here: `muse-spark-1.1`, `https://api.meta.ai/v1`) | |
| SAM 3.1: open weights, or the Meta Model API? Does either count for "using Meta models"? | |
| Must the demo run live, or is a recorded video OK? | |
| Submission format: repo, video length, write-up? | |
| Judging criteria and weighting | |
| Credit amount, and any rate limits | |

### Done when

- [ ] `g1_stand.mp4`, `g1_ik_squat.mp4`, and `sam_overlay.mp4` are in Drive at `a2h/outputs/`
- [ ] Muse Spark replied to a text call, and you know whether images work
- [ ] Clip is chosen and trimmed to `a2h/clips/clip.mp4`
