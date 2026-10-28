#!/usr/bin/env bash
# Fetch only the Unitree G1 folder from MuJoCo Menagerie (sparse checkout, a few MB).
set -euo pipefail
mkdir -p "$(dirname "$0")/../assets" && cd "$(dirname "$0")/../assets"
# Retried: a Colab VM's connection to GitHub occasionally times out.
for attempt in 1 2 3 4; do
  [ -f mujoco_menagerie/unitree_g1/scene.xml ] && break
  rm -rf mujoco_menagerie
  { git clone -q --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git &&
    git -C mujoco_menagerie sparse-checkout set unitree_g1; } || { echo "menagerie fetch failed (attempt $attempt)"; sleep 30; }
done
if [ ! -f pose_landmarker_heavy.task ]; then
  curl -sSL --retry 4 -o pose_landmarker_heavy.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
fi
# The pose model comes from a "latest" URL; the copy this repo was built and tested with is pinned. CI (and
# the Space image built there) refuses any other; elsewhere a changed model is only reported.
POSE_SHA256=64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b
got="$( (sha256sum pose_landmarker_heavy.task 2>/dev/null || shasum -a 256 pose_landmarker_heavy.task) | cut -d' ' -f1)"
if [ "$got" != "$POSE_SHA256" ]; then
  echo "pose_landmarker_heavy.task sha256 $got, expected $POSE_SHA256"
  [ "${A2H_VERIFY_ASSETS:-}" = 1 ] && exit 1
fi
ls mujoco_menagerie/unitree_g1/scene.xml pose_landmarker_heavy.task && echo "assets ready"
