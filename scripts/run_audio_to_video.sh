#!/usr/bin/env bash
#
# End-to-end: given an AIST audio file and a text prompt, generate motion and
# render a video with audio so you can watch the result.
#
# Usage:
#   ./scripts/run_audio_to_video.sh <audio_path> [text_prompt]
#
# Example:
#   ./scripts/run_audio_to_video.sh /Data/yash.bhardwaj/datasets/aist/audio/mBR0.wav "a person performs breakdance to music"
#
# Output: ./save/audio_stage2_wav2clip/run_audio_to_video/videos/sample_audio_00.mp4 (and 01, 02)
#

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

AUDIO_PATH="${1:?Usage: $0 <audio_path> [text_prompt]}"
TEXT_PROMPT="${2:-a person performs dance moves to music}"
MODEL_PATH="${MODEL_PATH:-./save/audio_stage2_wav2clip/model_final.pt}"
HUMANML_DIR="${HUMANML_DIR:-/Data/yash.bhardwaj/datasets/HumanML3D}"
OUTPUT_BASE="${OUTPUT_BASE:-./save/audio_stage2_wav2clip/run_audio_to_video}"
SAMPLE_DIR="${OUTPUT_BASE}/samples"
VIDEO_DIR="${OUTPUT_BASE}/videos"

echo "=== Generate motion from audio + prompt ==="
echo "  Audio:       $AUDIO_PATH"
echo "  Prompt:      $TEXT_PROMPT"
echo "  Model:       $MODEL_PATH"
echo "  Samples →    $SAMPLE_DIR"
echo "  Videos →     $VIDEO_DIR"
echo ""

mkdir -p "$SAMPLE_DIR" "$VIDEO_DIR"

# Step 1: generate motion (.npy)
python -m sample.generate_audio \
  --model_path "$MODEL_PATH" \
  --audio_path "$AUDIO_PATH" \
  --text_prompt "$TEXT_PROMPT" \
  --output_dir "$SAMPLE_DIR" \
  --num_samples 3 \
  --guidance_param 2.5 \
  --audio_guidance_param 2.5 \
  --seed 42

# Step 2: render video(s) with audio (samples from generate_audio are already denormalized)
python -m sample.visualize_with_audio \
  --sample_dir "$SAMPLE_DIR" \
  --audio_path "$AUDIO_PATH" \
  --humanml_dir "$HUMANML_DIR" \
  --output_dir "$VIDEO_DIR" \
  --text_prompt "$TEXT_PROMPT" \
  --fps 20 \
  --max_samples 3 \
  --samples_denormalized

echo ""
echo "Done. Open a video to see the motion:"
ls -la "$VIDEO_DIR"/*.mp4 2>/dev/null || true
echo ""
echo "Example: xdg-open $VIDEO_DIR/sample_audio_00.mp4   # or open in your player"
