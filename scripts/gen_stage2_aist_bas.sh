#!/usr/bin/env bash
# Generate Stage-2 wav2clip_beataware motions on the 8 valid AIST++ test songs
# and compute Beat Alignment Score (σ=2 frames at 20 FPS ≈ 0.10 s, matching EDGE).
#
# Usage:
#   bash scripts/gen_stage2_aist_bas.sh
#
# Outputs:
#   ./eval_outputs/stage2_bas/motions/   -- .npy files, one per song
#   (override OUT_ROOT env var to change output location)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Auto-detect weights location
if [ -f "$REPO_ROOT/final_weights/stage2/audio_stage2_wav2clip_beataware/model_final.pt" ]; then
    _DEFAULT_MODEL="$REPO_ROOT/final_weights/stage2/audio_stage2_wav2clip_beataware/model_final.pt"
else
    _DEFAULT_MODEL="$REPO_ROOT/save/audio_stage2_wav2clip_beataware/model_final.pt"
fi
MODEL_PATH=${MODEL_PATH:-"$_DEFAULT_MODEL"}
AUDIO_DIR=${AUDIO_DIR:-"$REPO_ROOT/dataset/aist/audio_test_10"}
OUT_ROOT=${OUT_ROOT:-"$REPO_ROOT/eval_outputs/stage2_bas"}
MOTION_DIR="$OUT_ROOT/motions"

mkdir -p "$MOTION_DIR"

# The 8 valid test songs (mJB5 and mHO5 were too short for EDGE; skip for consistency)
SONGS=(mBR0 mJS3 mKR2 mLH4 mLO2 mMH3 mPO1 mWA0)

echo "=== Generating Stage-2 beataware motions on ${#SONGS[@]} AIST++ test songs ==="
echo "    Model : $MODEL_PATH"
echo "    Audio : $AUDIO_DIR"
echo "    Output: $MOTION_DIR"
echo ""

cd "$REPO_ROOT"

for song in "${SONGS[@]}"; do
    wav="$AUDIO_DIR/${song}.wav"
    out_npy="$MOTION_DIR/${song}_sample_00.npy"

    if [ -f "$out_npy" ]; then
        echo "  [$song] already exists, skipping"
        continue
    fi

    if [ ! -f "$wav" ]; then
        echo "  [$song] WAV not found at $wav, skipping"
        continue
    fi

    echo "  [$song] Generating..."
    TMPDIR=$(mktemp -d)
    python -m sample.generate_audio \
        --model_path "$MODEL_PATH" \
        --audio_path  "$wav" \
        --output_dir  "$TMPDIR" \
        --num_samples 1 \
        --seed        42

    # Rename the first output file to include the song ID
    FIRST_NPY=$(ls "$TMPDIR"/*.npy 2>/dev/null | head -1)
    if [ -n "$FIRST_NPY" ]; then
        mv "$FIRST_NPY" "$out_npy"
        echo "  [$song] Saved -> $out_npy  (shape: $(python3 -c "import numpy as np; a=np.load('$out_npy'); print(a.shape)"))"
    else
        echo "  [$song] WARNING: no .npy produced"
    fi
    rm -rf "$TMPDIR"
done

echo ""
echo "=== Generation done. Files in $MOTION_DIR ==="
ls "$MOTION_DIR"

# ── compute BAS (σ=2 frames at 20 FPS = 0.10 s, same time window as EDGE σ=3 at 30 FPS) ──
echo ""
echo "=== Computing Beat Alignment Score (Stage-2 beataware, σ=2 @ 20 FPS) ==="

python -m eval.beat_align_score \
    --motion_dir "$MOTION_DIR" \
    --format     humanml \
    --fps        20 \
    --sigma      2 \
    --audio_dir  "$AUDIO_DIR"

echo ""
echo "=== Done. ==="
