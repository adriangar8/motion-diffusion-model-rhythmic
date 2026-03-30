#!/usr/bin/env bash
# Resume EDGE inference from seed START_RUN to END_RUN.
# Runs 1-2 already exist (20 pkl files). This adds runs 3-20 → 200 total.
#
# Run from project root:
#   nohup bash scripts/aist_fid_edge_inference_resume.sh > aist_fid_edge.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

CHECKPOINT="/Data/yash.bhardwaj/models/EDGE/checkpoint.pt"
MOTIONS_OUT="save/aist_fid_eval/edge_motions"
FEATURES_OUT="eval/aist_fid/cache/edge_features.npz"
TEST_MUSIC_DIR="save/aist_fid_eval/test_music"

START_RUN=3
END_RUN=20

export JUKEMIRLIB_CACHE_DIR="/Data/yash.bhardwaj/models/jukemirlib"
mkdir -p "$MOTIONS_OUT"

echo "Resuming EDGE inference: runs $START_RUN–$END_RUN (10 clips each = $((($END_RUN - $START_RUN + 1) * 10)) new files)"
echo "Existing files: $(ls "$MOTIONS_OUT"/*.pkl 2>/dev/null | wc -l)"

cd EDGE

for RUN in $(seq $START_RUN $END_RUN); do
    RUNDIR="../${MOTIONS_OUT}/run${RUN}"
    # Skip if already done
    if [ -d "$RUNDIR" ] && [ "$(ls "$RUNDIR"/*.pkl 2>/dev/null | wc -l)" -ge 8 ]; then
        echo "  Run $RUN already complete, skipping."
        continue
    fi
    echo "  Run $RUN / $END_RUN  ($(date +%H:%M:%S))"
    python test.py \
        --checkpoint    "$CHECKPOINT" \
        --feature_type  jukebox \
        --out_length    25 \
        --seed          "$RUN" \
        --music_dir     "../$TEST_MUSIC_DIR" \
        --save_motions \
        --motion_save_dir "$RUNDIR" \
        --no_render
done
cd ..

# Flatten new run subdirs into flat MOTIONS_OUT
echo "Flattening new run subdirs..."
for RUN in $(seq $START_RUN $END_RUN); do
    RUNDIR="$MOTIONS_OUT/run${RUN}"
    [ -d "$RUNDIR" ] || continue
    for PKL in "$RUNDIR"/*.pkl; do
        [ -f "$PKL" ] || continue
        BASE=$(basename "$PKL" .pkl)
        mv "$PKL" "${MOTIONS_OUT}/run${RUN}_${BASE}.pkl"
    done
    rmdir "$RUNDIR" 2>/dev/null || true
done

TOTAL=$(ls "$MOTIONS_OUT"/*.pkl 2>/dev/null | wc -l)
echo ""
echo "Total EDGE pkl files: $TOTAL"

# Re-extract features from all 200 files
echo "Extracting features from all $TOTAL EDGE motions..."
python -m eval.aist_fid.edge_features \
    --motions_dir "$MOTIONS_OUT" \
    --out_path    "$FEATURES_OUT"

echo ""
echo "Done. EDGE features saved to $FEATURES_OUT"
