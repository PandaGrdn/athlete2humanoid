#!/usr/bin/env bash
# One-time: give the Cloud Run API its Modal token (from ~/.modal.toml, never printed) and turn on the GPU pose
# for live builds (A2H_LIVE_POSE=modal: SAM 3.1 + SAM 3D Body on the deployed a2h-live-pose app, docs/DEPLOY.md
# "The live runner's pose on a GPU"). Re-running it is harmless; scripts/deploy.sh keeps what it sets.
#     MODAL_PROFILE=a2h scripts/cloudrun_modal_pose.sh            # PROJECT, REGION, SERVICE as in deploy.sh
set -euo pipefail
PROJECT="${PROJECT:-athlete2humanoid}"; REGION="${REGION:-us-central1}"; SERVICE="${SERVICE:-athlete2humanoid-api}"
PROFILE="${MODAL_PROFILE:-a2h}"
TIMEOUT_S="${A2H_LIVE_TIMEOUT_S:-900}"  # a cold GPU container boots inside the build's wall clock
PY="${PYTHON:-python3}"

say() { printf '\n==> %s\n' "$*"; }
say "Secret Manager on $PROJECT"
gcloud services enable secretmanager.googleapis.com --project "$PROJECT" --quiet
SA="$(gcloud projects describe "$PROJECT" --format 'value(projectNumber)')-compute@developer.gserviceaccount.com"
for key in token_id token_secret; do
  name="a2h-modal-${key//_/-}"
  if ! gcloud secrets describe "$name" --project "$PROJECT" >/dev/null 2>&1; then
    say "creating secret $name from ~/.modal.toml [$PROFILE]"
    # the value goes from the toml straight into gcloud's stdin: nothing on a command line, nothing in the terminal
    "$PY" - "$PROFILE" "$key" <<'EOF' | gcloud secrets create "$name" --project "$PROJECT" --replication-policy automatic --data-file=- --quiet
import sys, tomllib
from pathlib import Path
profile, key = sys.argv[1:3]
sys.stdout.write(tomllib.loads((Path.home() / ".modal.toml").read_text())[profile][key])
EOF
  else
    say "secret $name exists (delete it to rotate: gcloud secrets delete $name --project $PROJECT)"
  fi
  gcloud secrets add-iam-policy-binding "$name" --project "$PROJECT" --member "serviceAccount:$SA" \
    --role roles/secretmanager.secretAccessor --quiet >/dev/null
done

say "Cloud Run service $SERVICE: A2H_LIVE_POSE=modal, A2H_LIVE_TIMEOUT_S=$TIMEOUT_S, the token from Secret Manager"
gcloud run services update "$SERVICE" --project "$PROJECT" --region "$REGION" --quiet \
  --update-secrets "MODAL_TOKEN_ID=a2h-modal-token-id:latest,MODAL_TOKEN_SECRET=a2h-modal-token-secret:latest" \
  --update-env-vars "A2H_LIVE_POSE=modal,A2H_LIVE_TIMEOUT_S=$TIMEOUT_S"
URL="$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format 'value(status.url)')"
say "Done. Check: curl -s $URL/api/health | grep -o '\"pose\":\"[a-z]*\"'   (expects \"modal\")"
