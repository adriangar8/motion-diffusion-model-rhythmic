#!/usr/bin/env bash
# Generate EDGE motions on the AIST++ crossmodal test set (10 songs, 30s each).
# Saves .pkl files for BAS evaluation.
#
# Runtime estimate: ~50 min (18s model load + 22s Jukebox/5s clip x 11 clips x 10 songs + diffusion)
#
# Usage:
#   bash scripts/gen_edge_aist_test.sh
#
# Outputs:
#   save/eval_edge_bas/motions/    -- .pkl files, one per test song
#   save/eval_edge_bas/juke_feats/ -- cached Jukebox features (reused on re-run)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EDGE_DIR="$REPO_ROOT/EDGE"

AUDIO_DIR="/Data/yash.bhardwaj/datasets/aist/audio"
SPLIT="$EDGE_DIR/data/splits/crossmodal_test.txt"
CHECKPOINT="/Data/yash.bhardwaj/models/EDGE/checkpoint.pt"
JUKEBOX_CACHE="/Data/yash.bhardwaj/models/jukemirlib"

OUT_ROOT="/Data/yash.bhardwaj/eval/edge_bas"
MOTION_DIR="$OUT_ROOT/motions"
JUKE_CACHE_DIR="$OUT_ROOT/juke_feats"
RENDER_DIR="$OUT_ROOT/renders"

mkdir -p "$MOTION_DIR" "$JUKE_CACHE_DIR" "$RENDER_DIR"

# ── collect the 10 unique test audio files ────────────────────────────────────
TEST_AUDIO_DIR="/Data/yash.bhardwaj/datasets/aist/audio_test_10"
mkdir -p "$TEST_AUDIO_DIR"

echo "=== Collecting test audio files ==="
while IFS= read -r seq; do
    # sequence name: gBR_sBM_cAll_d04_mBR0_ch02 -> music id = mBR0
    music_id=$(echo "$seq" | sed 's/.*_\(m[A-Z0-9]*\)_ch.*/\1/')
    src="$AUDIO_DIR/${music_id}.wav"
    dst="$TEST_AUDIO_DIR/${music_id}.wav"
    if [ -f "$src" ] && [ ! -f "$dst" ]; then
        cp "$src" "$dst"
        echo "  copied $music_id.wav"
    fi
done < "$SPLIT"

N=$(ls "$TEST_AUDIO_DIR"/*.wav | wc -l)
echo "  $N unique audio files ready"

# ── run EDGE test ─────────────────────────────────────────────────────────────
echo ""
echo "=== Running EDGE on $N test songs (using cached Jukebox features, 30s output) ==="
echo "    Est. time: ~5 min  (diffusion only, Jukebox features loaded from cache)"
echo "    Feature cache: $JUKE_CACHE_DIR"
echo ""

cd "$EDGE_DIR"

# Redirect jukemirlib to use /Data cache (avoids home quota issues)
export JUKEMIRLIB_CACHE_DIR="$JUKEBOX_CACHE"

python test.py \
    --feature_type jukebox \
    --checkpoint "$CHECKPOINT" \
    --music_dir "$TEST_AUDIO_DIR" \
    --out_length 30 \
    --save_motions \
    --motion_save_dir "$MOTION_DIR" \
    --use_cached_features \
    --feature_cache_dir "$JUKE_CACHE_DIR" \
    --render_dir "$RENDER_DIR" \
    --no_render

cd "$REPO_ROOT"

echo ""
echo "=== EDGE generation done ==="
PKL_COUNT=$(ls "$MOTION_DIR"/*.pkl 2>/dev/null | wc -l)
echo "    $PKL_COUNT .pkl files saved to: $MOTION_DIR"

# ── compute BAS ──────────────────────────────────────────────────────────────
echo ""
echo "=== Computing Beat Alignment Score (EDGE, AIST++ test set) ==="

python -m eval.beat_align_score \
    --motion_dir "$MOTION_DIR" \
    --format edge \
    --fps 30 \
    --audio_dir "$TEST_AUDIO_DIR"

echo ""
echo "=== Done. ==="
