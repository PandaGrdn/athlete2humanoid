#!/usr/bin/env bash
# Fetch only the Unitree G1 folder from MuJoCo Menagerie (sparse checkout, a few MB).
set -euo pipefail
mkdir -p "$(dirname "$0")/../assets" && cd "$(dirname "$0")/../assets"
if [ ! -d mujoco_menagerie ]; then
  git clone -q --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
  git -C mujoco_menagerie sparse-checkout set unitree_g1
fi
ls mujoco_menagerie/unitree_g1/scene.xml && echo "G1 ready"
