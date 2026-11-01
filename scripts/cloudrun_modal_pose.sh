#!/usr/bin/env bash
# One-time: give the Cloud Run API its Modal token (from ~/.modal.toml, never printed) and turn on the GPU pose
# for live builds (A2H_LIVE_POSE=modal: SAM 3.1 + SAM 3D Body on the deployed a2h-live-pose app, docs/DEPLOY.md
# "The live runner's pose on a GPU"). Re-running it is harmless; scripts/deploy.sh keeps what it sets.
#     MODAL_PROFILE=a2h scripts/cloudrun_modal_pose.sh            # PROJECT, REGION, SERVICE as in deploy.sh
#     FORCE=1 MODAL_PROFILE=a2h scripts/cloudrun_modal_pose.sh    # store the token again (rotation, or after a failed run)
set -euo pipefail
PROJECT="${PROJECT:-athlete2humanoid}"; REGION="${REGION:-us-central1}"; SERVICE="${SERVICE:-athlete2humanoid-api}"
PROFILE="${MODAL_PROFILE:-a2h}"
TIMEOUT_S="${A2H_LIVE_TIMEOUT_S:-900}"  # a cold GPU container boots inside the build's wall clock
PY="${PYTHON:-python3}"

say() { printf '\n==> %s\n' "$*"; }
say "Secret Manager on $PROJECT"
gcloud services enable secretmanager.googleapis.com --project "$PROJECT" --quiet
SA="$(gcloud projects describe "$PROJECT" --format 'value(projectNumber)')-compute@developer.gserviceaccount.com"
value() {  # value <profile> <key>: one field of ~/.modal.toml, on stdout only (any python3: no tomllib needed)
  "$PY" - "$1" "$2" <<'EOF'
import re, sys
from pathlib import Path
profile, key, section, found = sys.argv[1], sys.argv[2], None, None
for line in (Path.home() / ".modal.toml").read_text().splitlines():
    m = re.match(r"\s*\[\s*([^\]]+?)\s*\]", line)
    if m:
        section = m.group(1)
        continue
    m = re.match(r"\s*([\w-]+)\s*=\s*['\"]?([^'\"\s]+)", line)
    if section == profile and m and m.group(1) == key:
        found = m.group(2)
if not found:
    sys.exit(f"~/.modal.toml has no {key} under [{profile}] (python -m modal token new, or MODAL_PROFILE=<profile>)")
sys.stdout.write(found)
EOF
}
for key in token_id token_secret; do
  name="a2h-modal-${key//_/-}"
  gcloud secrets describe "$name" --project "$PROJECT" >/dev/null 2>&1 \
    || gcloud secrets create "$name" --project "$PROJECT" --replication-policy automatic --quiet
  # add the value as a new version when the secret has none yet, or on FORCE=1 (a rotation, or after a failed run
  # that left an empty version); the value goes from the toml straight into gcloud's stdin, never on a command line
  if [ "${FORCE:-0}" = "1" ] || [ -z "$(gcloud secrets versions list "$name" --project "$PROJECT" --filter 'state=enabled' --format 'value(name)' 2>/dev/null)" ]; then
    say "storing $key from ~/.modal.toml [$PROFILE] in secret $name"
    value "$PROFILE" "$key" | gcloud secrets versions add "$name" --project "$PROJECT" --data-file=- --quiet
  else
    say "secret $name already has a value (FORCE=1 to replace it)"
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
