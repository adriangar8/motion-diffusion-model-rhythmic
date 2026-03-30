#!/usr/bin/env bash
# Run EDGE inference on AIST++ test music, then extract kinetic+manual features.
#
# Prerequisites:
#   - EDGE checkpoint at /Data/yash.bhardwaj/models/EDGE/checkpoint.pt
#   - Test audio files at /Data/yash.bhardwaj/datasets/aist/audio/
#   - GPU available
#   - jukemirlib installed (for Jukebox features, the standard EDGE setup)
#     OR use --feature_type baseline for a fast CPU-compatible alternative
#       (note: baseline features differ from what the EDGE model was trained on)
#
# Run from the project root:
#   bash scripts/aist_fid_edge_inference.sh [baseline]
#
# Pass "baseline" as first argument to use 35-dim baseline audio features
# instead of Jukebox (4800-dim). Only useful if jukemirlib is unavailable.

set -euo pipefail
cd "$(dirname "$0")/.."

CHECKPOINT="/Data/yash.bhardwaj/models/EDGE/checkpoint.pt"
AUDIO_DIR="/Data/yash.bhardwaj/datasets/aist/audio"
MOTIONS_OUT="save/aist_fid_eval/edge_motions"
FEATURES_OUT="eval/aist_fid/cache/edge_features.npz"

FEATURE_TYPE="${1:-jukebox}"
# Number of times to re-run per music clip with different random audio window picks.
# 10 tracks × NUM_RUNS = total generated sequences.  Need >> 66 for reliable kinetic FID.
NUM_RUNS=20

mkdir -p "$MOTIONS_OUT"

# ---- Step 1: collect test music files ----
# Test music IDs from crossmodal_test.txt: mBR0 mHO5 mJB5 mJS3 mKR2 mLH4 mLO2 mMH3 mPO1 mWA0
TEST_MUSIC_DIR="save/aist_fid_eval/test_music"
mkdir -p "$TEST_MUSIC_DIR"

for MID in mBR0 mHO5 mJB5 mJS3 mKR2 mLH4 mLO2 mMH3 mPO1 mWA0; do
    WAV="${AUDIO_DIR}/${MID}.wav"
    if [ -f "$WAV" ]; then
        cp "$WAV" "$TEST_MUSIC_DIR/"
    else
        echo "Warning: $WAV not found, skipping."
    fi
done

echo "Test wav files in $TEST_MUSIC_DIR:"
ls "$TEST_MUSIC_DIR"

# ---- Step 2: run EDGE inference (NUM_RUNS times with different seeds) ----
# --out_length 25 allows mHO5 (32s) and mJB5 (29.5s) which were skipped with 30s.
# Each run picks a different random audio window → different motion sample.
echo ""
echo "Running EDGE inference x${NUM_RUNS} runs (feature_type=$FEATURE_TYPE)..."
export JUKEMIRLIB_CACHE_DIR="/Data/yash.bhardwaj/models/jukemirlib"
cd EDGE

# Clear previous motions
rm -f "../$MOTIONS_OUT"/*.pkl

for RUN in $(seq 1 $NUM_RUNS); do
    echo "  Run $RUN / $NUM_RUNS ..."
    python test.py \
        --checkpoint    "$CHECKPOINT" \
        --feature_type  "$FEATURE_TYPE" \
        --out_length    25 \
        --seed          "$RUN" \
        --music_dir     "../$TEST_MUSIC_DIR" \
        --save_motions \
        --motion_save_dir "../${MOTIONS_OUT}/run${RUN}" \
        --no_render
done
cd ..

# Flatten all run subdirs into MOTIONS_OUT
echo "Flattening run subdirs..."
for RUN in $(seq 1 $NUM_RUNS); do
    RUNDIR="$MOTIONS_OUT/run${RUN}"
    if [ -d "$RUNDIR" ]; then
        for PKL in "$RUNDIR"/*.pkl; do
            [ -f "$PKL" ] || continue
            BASE=$(basename "$PKL" .pkl)
            mv "$PKL" "${MOTIONS_OUT}/${BASE}_run${RUN}.pkl"
        done
        rmdir "$RUNDIR" 2>/dev/null || true
    fi
done

echo "Generated motions saved to $MOTIONS_OUT"

# ---- Step 3: extract features ----
echo ""
echo "Extracting features from EDGE motions..."
python -m eval.aist_fid.edge_features \
    --motions_dir "$MOTIONS_OUT" \
    --out_path    "$FEATURES_OUT"

echo ""
echo "Done. EDGE features saved to $FEATURES_OUT"
