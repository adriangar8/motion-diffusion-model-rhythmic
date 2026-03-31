#!/usr/bin/env bash
# Run Stage-2 inference on AIST++ test music, then extract kinetic+manual features.
#
# Generates NUM_SAMPLES samples per music clip for all 10 test music IDs.
# Supports both wav2clip and wav2clip-beataware checkpoints.
#
# Run from project root:
#   bash scripts/aist_fid_stage2_inference.sh [wav2clip|beataware|mospa]
#
# Examples:
#   bash scripts/aist_fid_stage2_inference.sh beataware   # default
#   bash scripts/aist_fid_stage2_inference.sh wav2clip
#   bash scripts/aist_fid_stage2_inference.sh mospa

set -euo pipefail
cd "$(dirname "$0")/.."

VARIANT="${1:-beataware}"

case "$VARIANT" in
    beataware)
        MODEL_PATH="save/audio_stage2_wav2clip_beataware/model_final.pt"
        LABEL="stage2_wav2clip_beataware"
        ;;
    wav2clip)
        MODEL_PATH="save/audio_stage2_wav2clip/model_final.pt"
        LABEL="stage2_wav2clip"
        ;;
    mospa)
        MODEL_PATH="save/audio_stage2_wav2clip_mospa/model_final.pt"
        LABEL="stage2_wav2clip_mospa"
        ;;
    beataware_cfg)
        MODEL_PATH="save/audio_stage2_wav2clip_beataware/model_final.pt"
        LABEL="stage2_wav2clip_beataware_cfg"
        ;;
    beataware_cfg_ag2p5)
        MODEL_PATH="save/audio_stage2_wav2clip_beataware/model_final.pt"
        LABEL="stage2_wav2clip_beataware_cfg_ag2p5"
        ;;
    beataware_cfg_tg2)
        MODEL_PATH="save/audio_stage2_wav2clip_beataware/model_final.pt"
        LABEL="stage2_wav2clip_beataware_cfg_tg2"
        ;;
    *)
        echo "Unknown variant '$VARIANT'. Use: beataware | wav2clip | mospa | beataware_cfg"
        exit 1
        ;;
esac

AUDIO_DIR="${AUDIO_DIR:-./dataset/aist/audio}"
MOTIONS_OUT="save/aist_fid_eval/${LABEL}_motions"
FEATURES_OUT="eval/aist_fid/cache/${LABEL}_features.npz"
NUM_SAMPLES=20  # samples per music clip; 10 clips × 20 = 200 generated sequences (>> 98-dim, well-conditioned covariance)

mkdir -p "$MOTIONS_OUT"
mkdir -p "eval/aist_fid/cache"

echo "Model:   $MODEL_PATH"
echo "Output:  $MOTIONS_OUT"
echo "Samples: $NUM_SAMPLES per clip"
echo ""

# Test music IDs from crossmodal_test.txt
TEST_MUSIC_IDS=(mBR0 mHO5 mJB5 mJS3 mKR2 mLH4 mLO2 mMH3 mPO1 mWA0)

# ---- Step 1: generate samples for each test music clip ----
# Each clip goes to its own subdirectory to avoid filename collisions.
for MID in "${TEST_MUSIC_IDS[@]}"; do
    WAV="${AUDIO_DIR}/${MID}.wav"
    if [ ! -f "$WAV" ]; then
        echo "Warning: $WAV not found, skipping."
        continue
    fi

    echo "Generating $NUM_SAMPLES samples for $MID ..."
    if [ "$VARIANT" = "beataware_cfg" ]; then
        # Plain AudioCFG (no GCDM): text=2.5, audio=1.5
        python -m sample.generate_audio \
            --model_path   "$MODEL_PATH" \
            --audio_path   "$WAV" \
            --output_dir   "${MOTIONS_OUT}/${MID}" \
            --num_samples  "$NUM_SAMPLES" \
            --seed         42 \
            --guidance_param       2.5 \
            --audio_guidance_param 1.5 \
            --fps          20
    elif [ "$VARIANT" = "beataware_cfg_ag2p5" ]; then
        # Plain AudioCFG (no GCDM): text=2.5, audio=2.5
        python -m sample.generate_audio \
            --model_path   "$MODEL_PATH" \
            --audio_path   "$WAV" \
            --output_dir   "${MOTIONS_OUT}/${MID}" \
            --num_samples  "$NUM_SAMPLES" \
            --seed         42 \
            --guidance_param       2.5 \
            --audio_guidance_param 2.5 \
            --fps          20
    elif [ "$VARIANT" = "beataware_cfg_tg2" ]; then
        # Plain AudioCFG (no GCDM): text=2.0, audio=1.5
        python -m sample.generate_audio \
            --model_path   "$MODEL_PATH" \
            --audio_path   "$WAV" \
            --output_dir   "${MOTIONS_OUT}/${MID}" \
            --num_samples  "$NUM_SAMPLES" \
            --seed         42 \
            --guidance_param       2.0 \
            --audio_guidance_param 1.5 \
            --fps          20
    else
        # GCDM: alpha=3.0, beta_text=1.0, beta_audio=1.5
        python -m sample.generate_audio \
            --model_path   "$MODEL_PATH" \
            --audio_path   "$WAV" \
            --output_dir   "${MOTIONS_OUT}/${MID}" \
            --num_samples  "$NUM_SAMPLES" \
            --seed         42 \
            --use_gcdm \
            --gcdm_alpha        3.0 \
            --gcdm_beta_text    1.0 \
            --gcdm_beta_audio   1.5 \
            --fps          20
    fi
done

# Flatten per-clip subdirs into one directory with unique names
echo "Flattening per-clip subdirs..."
for MID in "${TEST_MUSIC_IDS[@]}"; do
    SUBDIR="${MOTIONS_OUT}/${MID}"
    [ -d "$SUBDIR" ] || continue
    for F in "$SUBDIR"/*.npy; do
        [ -f "$F" ] || continue
        BASE=$(basename "$F" .npy)
        mv "$F" "${MOTIONS_OUT}/${MID}_${BASE}.npy"
    done
    rm -rf "$SUBDIR"
done

echo ""
echo "All samples generated → $MOTIONS_OUT"
echo "Contents:"
ls "$MOTIONS_OUT" | head -20

# ---- Step 2: extract features ----
echo ""
echo "Extracting features from Stage-2 motions..."
python -m eval.aist_fid.stage2_features \
    --motions_dir "$MOTIONS_OUT" \
    --out_path    "$FEATURES_OUT"

echo ""
echo "Done. Stage-2 ($VARIANT) features saved to $FEATURES_OUT"
