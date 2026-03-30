#!/bin/bash
# Re-run Stage 3 DoRA training for all 4 styles (Wav2CLIP + Librosa Stage 2)
# with new settings: scheduler_t_max_mult=2, style_upsample=3, num_workers=0

set -e
cd "$(dirname "$0")/.."

STAGE2_DIR="./save/audio_stage2_wav2clip"
STYLE_100_ROOT="/Data/yash.bhardwaj/datasets/Retargeted100Style/RETARGETED_100STYLE"
STYLE_SUBSETS="./outputs/style_subsets"
HUMANML_DIR="/Data/yash.bhardwaj/datasets/HumanML3D"
SAVE_DIR="./save/audio_stage3_dora"

OPTS="--stage2_dir $STAGE2_DIR --style_100_root $STYLE_100_ROOT --style_subsets_dir $STYLE_SUBSETS --humanml_dir $HUMANML_DIR --save_dir $SAVE_DIR --epochs 80 --batch_size 16 --num_workers 0 --scheduler_t_max_mult 2.0 --style_upsample 3"

for style in old_elderly angry_aggressive proud_confident robot_mechanical; do
  echo "=============================================="
  echo "Stage 3 DoRA: $style"
  echo "=============================================="
  python -m train.train_stage3_dora --style "$style" $OPTS
  echo "Done $style."
done

echo "All 4 DoRAs finished."
