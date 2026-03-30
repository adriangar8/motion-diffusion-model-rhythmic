#!/usr/bin/env bash
# CFG sweep: find the audio_guidance_param that maximizes BAS
# for audio_stage2_wav2clip_beataware on AIST++ test songs.
#
# Sweep:
#   Legacy AudioCFG: audio_guidance_param in {1.5, 2.5, 5.0, 7.5}  (text fixed at 2.5)
#   GCDM default:    alpha=3.0, beta_text=1.0, beta_audio=1.5
#   GCDM strong:     alpha=5.0, beta_text=1.0, beta_audio=2.5
#
# Usage:
#   bash scripts/cfg_sweep_bas.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH=${MODEL_PATH:-"$REPO_ROOT/save/audio_stage2_wav2clip_beataware/model_final.pt"}
AUDIO_DIR=${AUDIO_DIR:-"$REPO_ROOT/dataset/aist/audio_test_10"}
SWEEP_ROOT=${SWEEP_ROOT:-"$REPO_ROOT/eval_outputs/cfg_sweep"}

SONGS=(mBR0 mJS3 mKR2 mLH4 mLO2 mMH3 mPO1 mWA0)

RESULTS_FILE="$SWEEP_ROOT/summary.txt"
mkdir -p "$SWEEP_ROOT"

echo "================================================================" | tee "$RESULTS_FILE"
echo "  CFG Sweep — audio_stage2_wav2clip_beataware BAS on AIST++"     | tee -a "$RESULTS_FILE"
echo "================================================================" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

generate_and_eval() {
    local TAG="$1"
    shift
    local EXTRA_ARGS=("$@")

    local MOTION_DIR="$SWEEP_ROOT/$TAG/motions"
    mkdir -p "$MOTION_DIR"

    echo "──────────────────────────────────────────────────────────────" | tee -a "$RESULTS_FILE"
    echo "  Config: $TAG" | tee -a "$RESULTS_FILE"
    echo "  Extra args: ${EXTRA_ARGS[*]}" | tee -a "$RESULTS_FILE"
    echo "──────────────────────────────────────────────────────────────" | tee -a "$RESULTS_FILE"

    for song in "${SONGS[@]}"; do
        wav="$AUDIO_DIR/${song}.wav"
        out_npy="$MOTION_DIR/${song}_sample_00.npy"

        if [ -f "$out_npy" ]; then
            echo "  [$song] exists, skip"
            continue
        fi
        if [ ! -f "$wav" ]; then
            echo "  [$song] WAV missing, skip"
            continue
        fi

        echo "  [$song] generating..."
        TMPDIR=$(mktemp -d)
        python -m sample.generate_audio \
            --model_path "$MODEL_PATH" \
            --audio_path  "$wav" \
            --output_dir  "$TMPDIR" \
            --num_samples 1 \
            --seed        42 \
            "${EXTRA_ARGS[@]}"

        FIRST_NPY=$(ls "$TMPDIR"/*.npy 2>/dev/null | head -1)
        if [ -n "$FIRST_NPY" ]; then
            mv "$FIRST_NPY" "$out_npy"
        else
            echo "  [$song] WARNING: no .npy produced"
        fi
        rm -rf "$TMPDIR"
    done

    echo "" | tee -a "$RESULTS_FILE"
    echo "  BAS for $TAG:" | tee -a "$RESULTS_FILE"
    python -m eval.beat_align_score \
        --motion_dir "$MOTION_DIR" \
        --format     humanml \
        --fps        20 \
        --audio_dir  "$AUDIO_DIR" 2>&1 | tee -a "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"
}

# ── Legacy AudioCFG: sweep audio_guidance_param ──
for ag in 1.5 2.5 5.0 7.5; do
    generate_and_eval "audioCFG_ag${ag}" \
        --guidance_param 2.5 \
        --audio_guidance_param "$ag"
done

# ── GCDM default ──
generate_and_eval "gcdm_a3.0_bt1.0_ba1.5" \
    --use_gcdm \
    --gcdm_alpha 3.0 \
    --gcdm_beta_text 1.0 \
    --gcdm_beta_audio 1.5

# ── GCDM strong ──
generate_and_eval "gcdm_a5.0_bt1.0_ba2.5" \
    --use_gcdm \
    --gcdm_alpha 5.0 \
    --gcdm_beta_text 1.0 \
    --gcdm_beta_audio 2.5

echo "================================================================" | tee -a "$RESULTS_FILE"
echo "  SWEEP COMPLETE — results in $RESULTS_FILE" | tee -a "$RESULTS_FILE"
echo "================================================================" | tee -a "$RESULTS_FILE"
