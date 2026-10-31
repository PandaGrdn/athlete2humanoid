#!/bin/bash
# Processed real-athlete clips -> one variants-trained tracker each, all at once on one GPU, then a short eval of each
# (scripts/remote_run.py athletes; docs/REAL_ATHLETES.md):
#   1. per clip in inputs/clips/, experiments/variant_policy.py: tracking_var trained on the clip's variants, warm-started
#      from inputs/policy_tracking_var (the squat-jump variants tracker); the clips train in parallel
#   2. per clip, experiments/robustness_eval.py: the clip clean, one 1 m/s push level, and its held-out variants
#      (--transfer), no V-JEPA scoring
#   bash scripts/real_athletes.sh <iters> [clip...]   (default: every clip in inputs/clips/)
set -u
ITERS=$1; shift
D=$A2H_JOB_DATA
IN=$D/inputs
OUT=$D/outputs/athletes
mkdir -p "$OUT"
CLIPS="$*"
[ -n "$CLIPS" ] || CLIPS=$(ls "$IN/clips")

for c in $CLIPS; do
  python3 experiments/variant_policy.py "$IN/clips/$c" --out "$OUT/$c" --policies tracking_var --init "$IN" \
    --iters "$ITERS" > "$OUT/$c.log" 2>&1 &
  echo "=== $c: training (log $OUT/$c.log)"
done
wait
for c in $CLIPS; do
  tail -3 "$OUT/$c.log"
  [ -f "$OUT/$c/results.tar.gz" ] || { echo "$c: no results"; continue; }
  mkdir -p "$OUT/$c/run" && tar xzf "$OUT/$c/results.tar.gz" -C "$OUT/$c/run"
  python3 experiments/robustness_eval.py --runs "$OUT/$c/run" --out "$OUT/$c/robust" --target "$c" --policies tracking_var \
    --push 1.0 --init --dynamics 0 --speed --tempo --phase-push --starts --no-noref --no-vjepa \
    --transfer "$OUT/$c/run"/held_out/*.npz || echo "$c: robustness_eval failed"
done

cd "$OUT" && tar czf "$OUT/results.tar.gz" $(for c in $CLIPS; do ls -d "$c/results.tar.gz" "$c/robust" "$c.log" 2>/dev/null; done)
echo "packed $OUT/results.tar.gz"
