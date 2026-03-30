#!/usr/bin/env bash
# Generate dance motion conditioned on audio using the trained Stage 2 checkpoint.
# Uses Wav2CLIP 519-d features (matches training). Output: .npy motion files + meta JSON.

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

CHECKPOINT="${1:-./save/audio_stage2_wav2clip/model_final.pt}"
AUDIO_PATH="${2:-/Data/yash.bhardwaj/datasets/aist/audio/mBR0.wav}"
OUTPUT_DIR="${3:-./save/audio_stage2_wav2clip/samples}"

python -m sample.generate_audio \
  --model_path "$CHECKPOINT" \
  --audio_path "$AUDIO_PATH" \
  --text_prompt "a person performs dance moves to music" \
  --output_dir "$OUTPUT_DIR" \
  --num_samples 3 \
  --guidance_param 2.5 \
  --audio_guidance_param 2.5 \
  --seed 42

echo ""
echo "Samples saved to: $OUTPUT_DIR"
