# What only you can do

These need your accounts or logins. Everything else is automated or already done.

## Before the deadline (Wed 2026-11-04, 9 PM ET)

1. **Meta Model API key, for Muse.** Today Muse runs on recorded stand-in answers (fixtures).
   - Once you have a key, put `MUSE_API_KEY=...` in `.env`. It's gitignored, so never commit it.
   - Then re-run `python -m muse.skill_reader clips/squat_jump.mov --record outputs/muse_rec`.
   - Also run `python -m muse.reward_agent outputs/remote_latent_final`, so the demo shows real Muse output (worth 25% for Meta tooling).
2. **Colab compute units.** About 3.4 were left on the evening of 2026-10-27, and a full training run costs 7–10. The next useful run is commanded targets (`A2H-TaskCmd`, about 5.5–7.5 units; the command is in `docs/PIPELINE_IMPROVEMENTS.md`).
   - Buy more at colab.research.google.com/signup if we need more runs. Pay-as-you-go: 100 units for about $10.
3. **Stanford FarmShare**, as the fallback GPU.
   - Do the first SSH login with Duo yourself.
   - Steps are in `docs/FARMSHARE.md`: add the SSH config block (with your SUNetID), run `ssh farmshare` once (password + Duo; the connection stays open for 12 h), and run the one-line GPU partition check in its step 4. The first job sets up the environments on FarmShare by itself.
   - After that, any job runs there with `--backend farmshare` (4 L40S GPUs per node).
4. **Deploy the web app.** Commands are in `docs/DEPLOY.md`.
   - Install Node 18+ (`brew install node`).
   - Log in to Hugging Face with a write token: `hf auth login`.
   - Log in to Cloudflare: `npx wrangler login`.
   - Run the deploy commands.
   - For CI/CD deploys on push, add the repo secrets `HF_TOKEN`, `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID`, and the variables `HF_SPACE` and `API_BASE`, on GitHub (table in `docs/DEPLOY.md`). CI itself already runs and is green.
5. **Open weights**, for the Best Open Weights prize.
   - `python scripts/weights.py push weights/<name> <you>/athlete2humanoid-g1`, after `hf auth login`.
   - This publishes the weights, so it's your call.
6. **Record the demo video.** The script is in `docs/DEMO_SCRIPT.md`. The final-run shots are already rendered in `outputs/demo/final/` (`python scripts/make_final_demo.py` remakes them), and the web app's `/examples` page is ready to screen-record.
7. **Submit.** The text is in `docs/SUBMISSION.md`.

## Housekeeping

- **Turn laptop sleep back on** after long runs: `sudo pmset -a disablesleep 0`. While a Colab run is going, the laptop must stay awake. It keeps the VM alive, downloads the results and stops the VM.
- **More clips:** URLs are listed in `clips/CANDIDATES.md`. Downloading them needs your OK. You can also record your own: one action per clip, named after it (`clips/<action>.mov`).
