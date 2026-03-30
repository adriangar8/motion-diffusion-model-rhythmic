#!/usr/bin/env bash
# Extract GT kinetic+manual features from all AIST++ joints_22 sequences.
# This is a CPU-only step (no GPU needed) and only needs to run once.
#
# Output: eval/aist_fid/cache/gt_features.npz
#
# Run from the project root:
#   bash scripts/aist_fid_gt_features.sh

set -euo pipefail
cd "$(dirname "$0")/.."

JOINTS_DIR="/Data/yash.bhardwaj/datasets/aist/processed/joints_22"
OUT_PATH="eval/aist_fid/cache/gt_features.npz"

python -m eval.aist_fid.gt_features \
    --joints_dir "$JOINTS_DIR" \
    --out_path   "$OUT_PATH"

echo ""
echo "Done. GT features saved to $OUT_PATH"
