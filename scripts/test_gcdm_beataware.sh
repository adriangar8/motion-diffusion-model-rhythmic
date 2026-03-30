#!/usr/bin/env bash
# Test commands for GCDM + beataware Stage 2 defaults.
# Run from repo root. Produces .npy motion + MP4 videos (needs HumanML3D for rendering).

set -e
# Ballet jazz: mJB0–mJB5 (80–130 BPM). Breakdance: mBR0.wav
AUDIO="${AUDIO:-/Data/yash.bhardwaj/datasets/aist/audio/mJB0.wav}"
HUMANML_DIR="${HUMANML_DIR:-/Data/yash.bhardwaj/datasets/HumanML3D}"
PROMPT="${PROMPT:-a person performs ballet jazz dance moves to music}"

if [[ ! -d "$HUMANML_DIR" ]]; then
  echo "Warning: HUMANML_DIR=$HUMANML_DIR not found. Set HUMANML_DIR for MP4 rendering."
fi

echo "=== 1. Stage 2 only (beataware, legacy CFG) — 1 sample ==="
python -m sample.generate_audio \
  --audio_path "$AUDIO" \
  --text_prompt "$PROMPT" \
  --output_dir ./save/audio_stage2_wav2clip_beataware/samples_test_cfg \
  --num_samples 1

echo ""
echo "  Rendering MP4..."
python -m sample.visualize_with_audio \
  --sample_dir ./save/audio_stage2_wav2clip_beataware/samples_test_cfg \
  --audio_path "$AUDIO" \
  --humanml_dir "$HUMANML_DIR" \
  --output_dir ./save/audio_stage2_wav2clip_beataware/samples_test_cfg/videos \
  --max_samples 1 \
  --samples_denormalized

echo ""
echo "=== 2. Stage 2 only (beataware + GCDM) — 1 sample ==="
python -m sample.generate_audio \
  --audio_path "$AUDIO" \
  --text_prompt "$PROMPT" \
  --output_dir ./save/audio_stage2_wav2clip_beataware/samples_test_gcdm \
  --num_samples 1 \
  --use_gcdm --gcdm_alpha 3.0 --gcdm_beta_audio 1.5

echo ""
echo "  Rendering MP4..."
python -m sample.visualize_with_audio \
  --sample_dir ./save/audio_stage2_wav2clip_beataware/samples_test_gcdm \
  --audio_path "$AUDIO" \
  --humanml_dir "$HUMANML_DIR" \
  --output_dir ./save/audio_stage2_wav2clip_beataware/samples_test_gcdm/videos \
  --max_samples 1 \
  --samples_denormalized

echo ""
echo "=== 3. Stage 3 single style (old_elderly, beataware + GCDM) — 1 sample ==="
python -m sample.generate_stage3_style \
  --stage2_dir ./save/audio_stage2_wav2clip_beataware \
  --adapter_path ./save/audio_stage3_dora_beataware/old_elderly/adapter_final.pt \
  --audio_path "$AUDIO" \
  --text_prompt "$PROMPT" \
  --output_dir ./save/audio_stage3_dora_beataware/test_gcdm_old_elderly \
  --num_samples 1 \
  --use_gcdm

echo ""
echo "  Rendering MP4..."
python -m sample.visualize_with_audio \
  --sample_dir ./save/audio_stage3_dora_beataware/test_gcdm_old_elderly \
  --audio_path "$AUDIO" \
  --humanml_dir "$HUMANML_DIR" \
  --output_dir ./save/audio_stage3_dora_beataware/test_gcdm_old_elderly/videos \
  --max_samples 1 \
  --samples_denormalized

echo ""
echo "=== 4. Compare 4 DoRAs (beataware + GCDM) — generate + render + 2x2 grid ==="
python -m sample.compare_4_doras \
  --audio_path "$AUDIO" \
  --text_prompt "$PROMPT" \
  --output_root ./save/audio_stage3_dora_beataware/compare_4styles_test_gcdm \
  --humanml_dir "$HUMANML_DIR" \
  --num_samples 1 \
  --use_gcdm

echo ""
echo "Done. MP4s:"
echo "  1–2: ./save/audio_stage2_wav2clip_beataware/samples_test_cfg/videos/*.mp4"
echo "       ./save/audio_stage2_wav2clip_beataware/samples_test_gcdm/videos/*.mp4"
echo "  3:   ./save/audio_stage3_dora_beataware/test_gcdm_old_elderly/videos/*.mp4"
echo "  4:   ./save/audio_stage3_dora_beataware/compare_4styles_test_gcdm/grid_2x2_sample_00.mp4 (and videos_*/)"
