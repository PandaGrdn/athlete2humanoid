#!/usr/bin/env bash
# One-time: keep visitors' environments and trainings past a Cloud Run restart (docs/DEPLOY.md "Keeping environments
# between visits"): a Cloud Storage bucket the API writes finished builds and trainings to (server/store.py GcsStore),
# the service account's access to it, and A2H_STORE on the service. Re-running it is harmless.
#
#     PATH="$HOME/google-cloud-sdk/bin:$PATH" scripts/cloudrun_store.sh            # bucket athlete2humanoid-jobs
#     BUCKET=my-bucket scripts/cloudrun_store.sh                                     # another name
set -euo pipefail
PROJECT="${PROJECT:-athlete2humanoid}"; REGION="${REGION:-us-central1}"; SERVICE="${SERVICE:-athlete2humanoid-api}"
BUCKET="${BUCKET:-${PROJECT}-jobs}"

say() { printf '\n==> %s\n' "$*"; }
say "bucket gs://$BUCKET in $REGION ($PROJECT)"
if ! gcloud storage buckets describe "gs://$BUCKET" --project "$PROJECT" >/dev/null 2>&1; then
  # one region, uniform access (IAM only, no per-object ACLs), nothing public; versioning off: a job's files are written once
  gcloud storage buckets create "gs://$BUCKET" --project "$PROJECT" --location "$REGION" --uniform-bucket-level-access \
    --public-access-prevention --quiet
else
  say "bucket exists"
fi
DAYS="${A2H_STORE_TTL_D:-30}"
say "lifecycle: objects older than $DAYS days are deleted (a visitor's builds and trainings are kept that long)"
LIFECYCLE="$(mktemp)"; trap 'rm -f "$LIFECYCLE"' EXIT
printf '{"rule": [{"action": {"type": "Delete"}, "condition": {"age": %s}}]}\n' "$DAYS" > "$LIFECYCLE"
gcloud storage buckets update "gs://$BUCKET" --project "$PROJECT" --lifecycle-file "$LIFECYCLE" --quiet
SA="$(gcloud projects describe "$PROJECT" --format 'value(projectNumber)')-compute@developer.gserviceaccount.com"
say "service account $SA: objectAdmin on the bucket (objects only; /readyz lists one object, no bucket metadata needed)"
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" --project "$PROJECT" \
  --member "serviceAccount:$SA" --role roles/storage.objectAdmin --quiet >/dev/null

say "Cloud Run service $SERVICE: A2H_STORE=gcs://$BUCKET"
gcloud run services update "$SERVICE" --project "$PROJECT" --region "$REGION" --quiet \
  --update-env-vars "A2H_STORE=gcs://$BUCKET"
URL="$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format 'value(status.url)')"
say "Done. Check: curl -s $URL/readyz   (expects \"store\": true)"
