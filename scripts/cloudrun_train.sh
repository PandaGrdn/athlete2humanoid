#!/usr/bin/env bash
# One-time: let visitors of the hosted site train a live build's environment on a Modal GPU (the modal runner,
# server/train.py; docs/DEPLOY.md "Live training on a GPU"). Needs scripts/cloudrun_modal_pose.sh to have run first
# (the Modal token on the service) and `python -m modal deploy scripts/modal_app.py` (the GPU app).
#
#     MODAL_PROFILE=a2h scripts/cloudrun_train.sh                 # asks for the invite code (typed, never echoed)
#     A2H_TRAIN_PER_DAY=2 scripts/cloudrun_train.sh               # ... and two free trainings a day without it
#     A2H_TRAIN_ITERS=1000 A2H_TRAIN_ATTEMPT_EVERY=50 scripts/cloudrun_train.sh
#
# The code goes to Secret Manager (a2h-train-code) and reaches the service as A2H_TRAIN_CODE; scripts/deploy.sh keeps
# the settings on later deploys (--update-env-vars). Re-running it is harmless; FORCE=1 stores a new code.
set -euo pipefail
PROJECT="${PROJECT:-athlete2humanoid}"; REGION="${REGION:-us-central1}"; SERVICE="${SERVICE:-athlete2humanoid-api}"
ITERS="${A2H_TRAIN_ITERS:-1000}"; EVERY="${A2H_TRAIN_ATTEMPT_EVERY:-50}"; PER_DAY="${A2H_TRAIN_PER_DAY:-0}"
TIMEOUT_H="${A2H_TRAIN_TIMEOUT_H:-3}"; BRANCH="${A2H_TRAIN_BRANCH:-main}"

say() { printf '\n==> %s\n' "$*"; }
say "Secret Manager on $PROJECT"
gcloud services enable secretmanager.googleapis.com --project "$PROJECT" --quiet
SA="$(gcloud projects describe "$PROJECT" --format 'value(projectNumber)')-compute@developer.gserviceaccount.com"
name="a2h-train-code"
gcloud secrets describe "$name" --project "$PROJECT" >/dev/null 2>&1 \
  || gcloud secrets create "$name" --project "$PROJECT" --replication-policy automatic --quiet
if [ "${FORCE:-0}" = "1" ] || [ -z "$(gcloud secrets versions list "$name" --project "$PROJECT" --filter 'state=enabled' --format 'value(name)' 2>/dev/null)" ]; then
  say "the invite code visitors type to start a training (letters and digits; not shown)"
  read -r -s -p "invite code: " CODE; echo
  [ -n "$CODE" ] || { echo "no code given"; exit 1; }
  printf '%s' "$CODE" | gcloud secrets versions add "$name" --project "$PROJECT" --data-file=- --quiet
  unset CODE
else
  say "secret $name already has a value (FORCE=1 to replace it)"
fi
gcloud secrets add-iam-policy-binding "$name" --project "$PROJECT" --member "serviceAccount:$SA" \
  --role roles/secretmanager.secretAccessor --quiet >/dev/null

say "Cloud Run service $SERVICE: A2H_RUNNERS=demo,modal, $ITERS iterations, an attempt every $EVERY, $PER_DAY free a day"
gcloud run services update "$SERVICE" --project "$PROJECT" --region "$REGION" --quiet \
  --update-secrets "A2H_TRAIN_CODE=$name:latest" \
  --update-env-vars "^:^A2H_RUNNERS=demo,modal:A2H_TRAIN_PER_DAY=$PER_DAY:A2H_TRAIN_ITERS=$ITERS:A2H_TRAIN_ATTEMPT_EVERY=$EVERY:A2H_TRAIN_TIMEOUT_H=$TIMEOUT_H:A2H_TRAIN_BRANCH=$BRANCH"
  # ^:^ makes ':' the separator (gcloud topic escaping): the runner list itself contains a comma
URL="$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format 'value(status.url)')"
say "Done. Check: curl -s $URL/api/health | grep -o '\"train\":{[^}]*}'   (expects \"backend\":\"modal\")"
