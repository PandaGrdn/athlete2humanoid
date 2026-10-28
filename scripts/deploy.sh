#!/usr/bin/env bash
# One-command deploy: the backend to a Hugging Face Space, the frontend to Cloudflare Pages.
#
#   hf auth login            # once, with a write token (or: uvx --from huggingface_hub hf auth login)
#   npx wrangler login       # once
#   scripts/deploy.sh [hf-user]
#
# Steps: make sure the Pages project exists (its *.pages.dev origin becomes the API's CORS origin), build
# the Space bundle (scripts/build_space.py: the default runs + the live pipeline), refuse if anything
# secret-looking is in it, create the Space if needed, upload, wait for /readyz, build the frontend with
# API_BASE pointing at the Space and deploy it, then print both URLs. Re-run it to redeploy.
#
# hf-user defaults to `hf auth whoami`. Optional environment:
#   SPACE_NAME     default athlete2humanoid-api
#   PAGES_PROJECT  default athlete2humanoid
#   CORS           extra frontend origin(s), comma-separated (e.g. a custom domain), added to the Pages one
#   ONNX_PY        a Python with torch + onnx: ships policy.onnx next to each checkpoint
#   SKIP_WEB=1     backend only
#
# Never prints or copies tokens: hf and wrangler read their own stored logins.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
SPACE_NAME="${SPACE_NAME:-athlete2humanoid-api}"
PAGES_PROJECT="${PAGES_PROJECT:-athlete2humanoid}"
WRANGLER="wrangler@4.149.0"
HF_VERSION="huggingface_hub==2.2.0"
BUNDLE="$REPO/build/space"

say() { printf '\n==> %s\n' "$*"; }
die() { printf 'deploy: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "$1 not found. $2"; }

need uv "Install it: https://docs.astral.sh/uv/ (curl -LsSf https://astral.sh/uv/install.sh | sh)"
[ "${SKIP_WEB:-}" = 1 ] || { need node "Install Node 18+ (brew install node)."; need npx "It comes with Node."; }
hf() {  # always the pinned hf CLI (an hf on PATH may be older, without the flags used here); same stored login
  uvx --quiet --from "$HF_VERSION" hf "$@"
}
wrangler() { npx --yes "$WRANGLER" "$@"; }

# --- who: Hugging Face user, Cloudflare login ---------------------------------------------------------
HF_USER="${1:-}"
if [ -z "$HF_USER" ]; then
  # only the account name is read out of whoami's answer: JSON on newer hf ({"user": ...} or {"name": ...}),
  # "user=<name>" or the bare name on others
  HF_USER="$( { hf auth whoami --json 2>/dev/null || hf auth whoami 2>/dev/null; } | python3 -c '
import sys, json, re
t = sys.stdin.read().strip()
try:
    d = json.loads(t.splitlines()[0])
    u = d.get("name") or d.get("user") or ""
    print(u.get("name", "") if isinstance(u, dict) else u)
except (ValueError, IndexError, AttributeError):
    m = re.search(r"user[=:]\s*(\S+)", t) or re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)\s*$", t.splitlines()[0] if t else "")
    print(m.group(1) if m else "")' 2>/dev/null || true)"
  [ -n "$HF_USER" ] || die "not logged in to Hugging Face. Run: hf auth login (a token with write access), or pass your username."
fi
[[ "$HF_USER" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "odd Hugging Face username: $HF_USER"
SPACE="$HF_USER/$SPACE_NAME"
API_BASE="https://$(printf '%s-%s' "$HF_USER" "$SPACE_NAME" | tr '[:upper:]_.' '[:lower:]--').hf.space"

if [ "${SKIP_WEB:-}" != 1 ]; then
  say "Cloudflare Pages project $PAGES_PROJECT"
  who="$(wrangler whoami 2>&1 || true)"  # kept to ourselves: it names the account
  if grep -qi "not authenticated" <<<"$who"; then die "not logged in to Cloudflare. Run: npx wrangler login"; fi
  projects="$(wrangler pages project list 2>/dev/null || true)"
  row() { grep -E "(^|[│| ])${PAGES_PROJECT}[ ]*[│|]" <<<"$projects" | head -1 || true; }
  if [ -z "$(row)" ]; then
    wrangler pages project create "$PAGES_PROJECT" --production-branch main
    projects="$(wrangler pages project list 2>/dev/null || true)"
  fi
  # A taken name gets a suffixed subdomain (athlete2humanoid-abc.pages.dev): read the real one back.
  PAGES_HOST="$(row | grep -oE '[a-z0-9-]+\.pages\.dev' | head -1 || true)"
  PAGES_URL="https://${PAGES_HOST:-$PAGES_PROJECT.pages.dev}"
  ORIGINS="$PAGES_URL${CORS:+,$CORS}"
else
  PAGES_URL=""
  ORIGINS="${CORS:-https://$PAGES_PROJECT.pages.dev}"
fi

# --- backend ---------------------------------------------------------------------------------------
say "Space bundle (CORS $ORIGINS)"
# imageio-ffmpeg gives the gallery its poster frames; the script itself needs only the standard library
uv run --quiet --no-project --with imageio-ffmpeg python scripts/build_space.py --cors "$ORIGINS" \
  ${ONNX_PY:+--onnx-python "$ONNX_PY"}

# build_space.py never copies dotfiles, .env or logs; check anyway before anything leaves this machine
bad="$(find "$BUNDLE" \( -name '.env*' -o -name '*.pem' -o -name '*.key' -o -name 'id_*' -o -name '*.log' \
        -o -name 'token*' -o -name '.git' \) -print | head -5)"
[ -z "$bad" ] || die "refusing to upload, the bundle contains: $bad"
if grep -rIlE 'hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY' "$BUNDLE" >/dev/null 2>&1; then
  die "refusing to upload: a file in $BUNDLE looks like it contains a token (grep -rIlE ... to find it)"
fi

say "Space $SPACE"
hf repos create "$SPACE" --type space --space-sdk docker --exist-ok >/dev/null
hf upload "$SPACE" "$BUNDLE" . --repo-type space \
  --commit-message "deploy $(git rev-parse --short HEAD 2>/dev/null || echo local)"

say "Waiting for $API_BASE/readyz (the first image build takes ~5-10 min)"
ok=""
for _ in $(seq 1 80); do
  if curl -sf --max-time 10 "$API_BASE/readyz" >/dev/null 2>&1; then ok=1; break; fi
  sleep 15
done
[ -n "$ok" ] || die "$API_BASE/readyz not ready after 20 min. Build logs: hf spaces logs $SPACE"
curl -s --max-time 10 "$API_BASE/api/health" | python3 -c \
  'import sys, json; h = json.load(sys.stdin); print("runners:", ", ".join(h["runners"]), "| runs:", h.get("run_count"), "| demo run:", h.get("demo_run"))' || true

# --- frontend --------------------------------------------------------------------------------------
if [ "${SKIP_WEB:-}" != 1 ]; then
  say "Frontend (API_BASE=$API_BASE)"
  (cd web && npm ci --ignore-scripts --no-audit --no-fund && API_BASE="$API_BASE" npm run build)
  grep -q "connect-src 'self' $API_BASE" web/dist/_headers || die "web/dist/_headers doesn't allow $API_BASE"
  (cd web && wrangler pages deploy dist --project-name "$PAGES_PROJECT" --branch main --commit-dirty=true)
fi

say "Done"
echo "API:      $API_BASE   (Space: https://huggingface.co/spaces/$SPACE)"
[ -z "$PAGES_URL" ] || echo "Frontend: $PAGES_URL"
