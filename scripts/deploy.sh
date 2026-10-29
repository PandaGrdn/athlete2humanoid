#!/usr/bin/env bash
# One-command deploy. The frontend always goes to Cloudflare Pages; the API goes where the mode says.
#
#   scripts/deploy.sh static             # the website only, no API: the examples, and /live replaying recorded
#                                        #   builds (web/public/live-demo; REBUILD=1 re-records them and the
#                                        #   examples first, which needs the repo's Python environment)
#   scripts/deploy.sh web                # the website only (API_BASE=<url> to point it at an API that's already up)
#   scripts/deploy.sh tunnel             # the API on this machine (e.g. a Mac mini) in Docker, public through a
#                                        #   Cloudflare Tunnel, then the website pointed at it
#   scripts/deploy.sh [space] [hf-user]  # the API on a Hugging Face Space (Docker Spaces need HF PRO)
#   scripts/deploy.sh cloudrun [project] # the API on Google Cloud Run (free tier: scales to zero, one instance)
#
# Logins, once: `npx wrangler login` (all modes); `hf auth login` with a write token (space);
# `brew install cloudflared` + `cloudflared tunnel login` (tunnel with your own domain; see below);
# `gcloud auth login` and a project with billing linked (cloudrun; billing is required even at $0, see DEPLOY.md).
#
# Optional environment:
#   PAGES_PROJECT    default athlete2humanoid
#   CORS             extra frontend origin(s), comma-separated (e.g. a custom domain), added to the Pages one
#   SPACE_NAME       space mode: default athlete2humanoid-api
#   ONNX_PY          a Python with torch + onnx: ships policy.onnx next to each checkpoint
#   SKIP_WEB=1       space / tunnel / cloudrun: the API only
#   GCP_REGION       cloudrun mode: default us-central1
#   TUNNEL_HOSTNAME  tunnel mode: a stable https://<hostname> on a domain in your Cloudflare account (a named
#                    tunnel). Without it: a free quick tunnel (https://<random>.trycloudflare.com) that changes
#                    every time it restarts, so the website is redeployed with the new address each run.
#
# Never prints or copies tokens: hf, wrangler and cloudflared read their own stored logins.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
MODE=space
case "${1:-}" in static|web|tunnel|space|cloudrun) MODE="$1"; shift ;; esac
SPACE_NAME="${SPACE_NAME:-athlete2humanoid-api}"
PAGES_PROJECT="${PAGES_PROJECT:-athlete2humanoid}"
WRANGLER="wrangler@4.149.0"
HF_VERSION="huggingface_hub==2.2.0"
BUNDLE="$REPO/build/space"
IMAGE=a2h-api
LOCAL_PORT=7860

say() { printf '\n==> %s\n' "$*"; }
die() { printf 'deploy: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "$1 not found. $2"; }

[ "${SKIP_WEB:-}" = 1 ] || { need node "Install Node 18+ (brew install node)."; need npx "It comes with Node."; }
[ "$MODE" = web ] || [ "$MODE" = static ] || need uv "Install it: https://docs.astral.sh/uv/ (curl -LsSf https://astral.sh/uv/install.sh | sh)"
hf() {  # always the pinned hf CLI (an hf on PATH may be older, without the flags used here); same stored login
  uvx --quiet --from "$HF_VERSION" hf "$@"
}
wrangler() { npx --yes "$WRANGLER" "$@"; }

# --- the Pages project (its *.pages.dev origin is the API's CORS origin) ------------------------------
PAGES_URL=""
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
fi
ORIGINS="${PAGES_URL:-https://$PAGES_PROJECT.pages.dev}${CORS:+,$CORS}"

bundle() {
  say "API bundle (CORS $ORIGINS)"
  # imageio-ffmpeg gives the gallery its poster frames; the script itself needs only the standard library
  uv run --quiet --no-project --with imageio-ffmpeg python scripts/build_space.py --cors "$ORIGINS" \
    ${ONNX_PY:+--onnx-python "$ONNX_PY"}
  # build_space.py never copies dotfiles, .env or logs; check anyway before anything leaves this machine
  # (data/runs/*/job.log is allowed: build_space.py rewrites it to the classified, redacted lines the demo replays)
  local bad
  bad="$(find "$BUNDLE" \( -name '.env*' -o -name '*.pem' -o -name '*.key' -o -name 'id_*' \
          -o \( -name '*.log' ! -path "$BUNDLE/data/runs/*/job.log" \) -o -name 'token*' -o -name '.git' \) -print | head -5)"
  [ -z "$bad" ] || die "refusing to deploy, the bundle contains: $bad"
  if grep -rIlE 'hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY' "$BUNDLE" >/dev/null 2>&1; then
    die "refusing to deploy: a file in $BUNDLE looks like it contains a token (grep -rIlE ... to find it)"
  fi
}

wait_ready() {  # wait_ready <url> <tries> <what to look at if it never answers>
  local ok="" i
  for i in $(seq 1 "$2"); do
    if curl -sf --max-time 10 "$1/readyz" >/dev/null 2>&1; then ok=1; break; fi
    sleep 15
  done
  [ -n "$ok" ] || die "$1/readyz not ready. Look at: $3"
  curl -s --max-time 10 "$1/api/health" | python3 -c \
    'import sys, json; h = json.load(sys.stdin); print("runners:", ", ".join(h["runners"]), "| runs:", h.get("run_count"), "| demo run:", h.get("demo_run"))' || true
}

API_BASE="${API_BASE:-}"
case "$MODE" in
# --- no API: what the site shows without one --------------------------------------------------------
static)
  API_BASE=""
  if [ "${REBUILD:-}" = 1 ]; then
    PY="${PYTHON:-python3}"
    say "Re-recording the live builds and rebuilding the examples ($PY)"
    "$PY" scripts/build_live_demo.py ${MUSE:+--muse}
    "$PY" scripts/build_examples.py
  fi
  [ -f web/public/live-demo/index.json ] || die "no recorded builds in web/public/live-demo: run with REBUILD=1"
  ;;
# --- the API on a Hugging Face Space --------------------------------------------------------------
space)
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
  bundle
  say "Space $SPACE"
  hf repos create "$SPACE" --type space --space-sdk docker --exist-ok >/dev/null
  hf upload "$SPACE" "$BUNDLE" . --repo-type space \
    --commit-message "deploy $(git rev-parse --short HEAD 2>/dev/null || echo local)"
  say "Waiting for $API_BASE/readyz (the first image build takes ~5-10 min)"
  wait_ready "$API_BASE" 80 "hf spaces logs $SPACE"
  ;;

# --- the API on Google Cloud Run --------------------------------------------------------------------
cloudrun)
  GCLOUD="$(command -v gcloud || echo "$HOME/google-cloud-sdk/bin/gcloud")"
  [ -x "$GCLOUD" ] || die "gcloud not found. Install the Google Cloud CLI (see docs/DEPLOY.md)."
  gcloud() { "$GCLOUD" "$@"; }
  need docker "Install Docker Desktop and start it."
  docker info >/dev/null 2>&1 || die "Docker isn't running: start Docker Desktop."
  [ -n "$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null)" ] || die "not logged in. Run: gcloud auth login"
  PROJECT="${1:-$(gcloud config get-value project 2>/dev/null || true)}"
  [ -n "$PROJECT" ] || die "no project: pass one, or gcloud config set project <id>"
  [ "$(gcloud billing projects describe "$PROJECT" --format='value(billingEnabled)' 2>/dev/null)" = True ] \
    || die "project $PROJECT has no billing account linked (Cloud Run needs one even inside the free tier): console.cloud.google.com/billing"
  REGION="${GCP_REGION:-us-central1}"
  SERVICE="${SERVICE_NAME:-athlete2humanoid-api}"
  REPO_AR="$REGION-docker.pkg.dev/$PROJECT/a2h"
  TAG="$REPO_AR/api:$(git rev-parse --short HEAD 2>/dev/null || date +%s)"
  say "Google Cloud: project $PROJECT, $REGION"
  gcloud services enable run.googleapis.com artifactregistry.googleapis.com --project "$PROJECT"
  gcloud artifacts repositories describe a2h --location "$REGION" --project "$PROJECT" >/dev/null 2>&1 \
    || gcloud artifacts repositories create a2h --repository-format docker --location "$REGION" --project "$PROJECT"
  gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet >/dev/null
  bundle
  say "Docker image $TAG (linux/amd64)"
  docker build --platform linux/amd64 -t "$TAG" "$BUNDLE"
  docker push "$TAG"
  # Free tier: scales to zero, at most one instance (jobs and rate limits live in its memory), CPU kept on
  # between requests so a job's thread keeps running. /tmp is memory on Cloud Run: uploads capped to fit.
  say "Cloud Run service $SERVICE"
  gcloud run deploy "$SERVICE" --image "$TAG" --project "$PROJECT" --region "$REGION" --platform managed \
    --allow-unauthenticated --port 7860 --cpu 2 --memory 4Gi --min-instances 0 --max-instances 1 \
    --no-cpu-throttling --concurrency 80 --timeout 3600 --execution-environment gen2 \
    --set-env-vars A2H_MAX_UPLOAD_STORAGE_MB=400 --quiet  # CORS is baked into the bundle (build_space.py --cors)
  API_BASE="$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format 'value(status.url)')"
  say "Waiting for $API_BASE/readyz"
  wait_ready "$API_BASE" 20 "gcloud run services logs read $SERVICE --region $REGION --project $PROJECT"
  ;;

# --- the API on this machine, through a Cloudflare Tunnel ------------------------------------------
tunnel)
  need docker "Install Docker Desktop (https://www.docker.com/products/docker-desktop/) and start it."
  need cloudflared "Install it: brew install cloudflared"
  docker info >/dev/null 2>&1 || die "Docker isn't running: start Docker Desktop."
  bundle
  # The image's packages are locked for linux/amd64 (as on the Space); Apple silicon runs it under Rosetta.
  say "Docker image $IMAGE (linux/amd64)"
  docker build --platform linux/amd64 -t "$IMAGE" "$BUNDLE"
  docker rm -f "$IMAGE" >/dev/null 2>&1 || true
  # As hardened as on the Space: read-only root, no capabilities, only localhost can reach it (the tunnel does);
  # back after a reboot or a crash. /tmp holds uploads and live builds.
  docker run -d --name "$IMAGE" --platform linux/amd64 --restart unless-stopped \
    --read-only --tmpfs /tmp:rw,size=4g --cap-drop ALL --security-opt no-new-privileges \
    --memory 6g --pids-limit 512 -p "127.0.0.1:$LOCAL_PORT:7860" "$IMAGE" >/dev/null
  say "Waiting for the API on localhost:$LOCAL_PORT"
  wait_ready "http://127.0.0.1:$LOCAL_PORT" 40 "docker logs $IMAGE"

  pkill -f "cloudflared tunnel .*$LOCAL_PORT" 2>/dev/null || true  # this script's earlier tunnel, if any
  mkdir -p "$REPO/build"
  if [ -n "${TUNNEL_HOSTNAME:-}" ]; then
    # A named tunnel: a stable address on your own domain. `cloudflared tunnel login` once, picking that domain.
    [ -f "$HOME/.cloudflared/cert.pem" ] || die "run once: cloudflared tunnel login (pick the domain of $TUNNEL_HOSTNAME)"
    say "Named tunnel a2h -> https://$TUNNEL_HOSTNAME"
    cloudflared tunnel list 2>/dev/null | grep -qw a2h || cloudflared tunnel create a2h
    cloudflared tunnel route dns --overwrite-dns a2h "$TUNNEL_HOSTNAME"
    cat > "$REPO/build/tunnel.yml" <<YAML
tunnel: a2h
ingress:
  - hostname: $TUNNEL_HOSTNAME
    service: http://127.0.0.1:$LOCAL_PORT
  - service: http_status:404
YAML
    nohup cloudflared tunnel --no-autoupdate --config "$REPO/build/tunnel.yml" run a2h >"$REPO/build/tunnel.out" 2>&1 &
    API_BASE="https://$TUNNEL_HOSTNAME"
    echo "To keep it up across reboots: sudo cloudflared service install (with this config), see docs/DEPLOY.md"
  else
    # A quick tunnel: free, no domain or account needed, but a new random address on every start.
    say "Quick tunnel (https://<random>.trycloudflare.com)"
    nohup cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:$LOCAL_PORT" >"$REPO/build/tunnel.out" 2>&1 &
    for _ in $(seq 1 30); do
      API_BASE="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$REPO/build/tunnel.out" | head -1 || true)"
      [ -n "$API_BASE" ] && break
      sleep 2
    done
    [ -n "$API_BASE" ] || die "no tunnel address after 60 s; see build/tunnel.out"
    echo "Quick-tunnel addresses change on restart: re-run this script after a reboot."
  fi
  say "Waiting for $API_BASE through the tunnel"
  wait_ready "$API_BASE" 20 "build/tunnel.out"
  ;;
esac

# --- frontend --------------------------------------------------------------------------------------
if [ "${SKIP_WEB:-}" != 1 ]; then
  say "Frontend (API_BASE=${API_BASE:-none: the API shows as offline, the examples work})"
  (cd web && npm ci --ignore-scripts --no-audit --no-fund && API_BASE="$API_BASE" npm run build)
  [ -z "$API_BASE" ] || grep -q "connect-src 'self' $API_BASE" web/dist/_headers || die "web/dist/_headers doesn't allow $API_BASE"
  (cd web && wrangler pages deploy dist --project-name "$PAGES_PROJECT" --branch main --commit-dirty=true)
fi

say "Done"
[ -z "$API_BASE" ] || echo "API:      $API_BASE"
[ -z "$PAGES_URL" ] || echo "Frontend: $PAGES_URL"
