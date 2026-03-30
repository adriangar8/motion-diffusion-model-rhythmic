#!/usr/bin/env bash
# Compute FID scores for all available cached feature files.
#
# Prerequisites: run the following scripts first (in any order):
#   bash scripts/aist_fid_gt_features.sh
#   bash scripts/aist_fid_edge_inference.sh
#   bash scripts/aist_fid_stage2_inference.sh beataware
#   bash scripts/aist_fid_stage2_inference.sh wav2clip
#   bash scripts/aist_fid_stage2_inference.sh mospa
#
# Run from project root:
#   bash scripts/aist_fid_compute.sh

set -euo pipefail
cd "$(dirname "$0")/.."

CACHE="eval/aist_fid/cache"
GT="$CACHE/gt_features.npz"

if [ ! -f "$GT" ]; then
    echo "GT features not found at $GT. Run aist_fid_gt_features.sh first."
    exit 1
fi

# Collect all available generated feature files
GEN_PATHS=()
LABELS=()

for F in \
    "$CACHE/edge_features.npz" \
    "$CACHE/stage2_wav2clip_beataware_features.npz" \
    "$CACHE/stage2_wav2clip_beataware_cfg_features.npz" \
    "$CACHE/stage2_wav2clip_beataware_cfg_ag2p5_features.npz" \
    "$CACHE/stage2_wav2clip_beataware_cfg_tg2_features.npz" \
    "$CACHE/stage2_wav2clip_features.npz" \
    "$CACHE/stage2_wav2clip_mospa_features.npz"; do
    if [ -f "$F" ]; then
        GEN_PATHS+=("$F")
        # derive a short label from filename
        BASE=$(basename "$F" _features.npz)
        LABELS+=("$BASE")
    fi
done

if [ ${#GEN_PATHS[@]} -eq 0 ]; then
    echo "No generated feature files found in $CACHE."
    echo "Run inference scripts first."
    exit 1
fi

echo "GT features: $GT"
echo "Models found: ${LABELS[*]}"
echo ""

python -m eval.aist_fid.compute_fid \
    --gt_path    "$GT" \
    --gen_paths  "${GEN_PATHS[@]}" \
    --labels     "${LABELS[@]}"
