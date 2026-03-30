#!/bin/bash
# Stage 3: DoRA style adapter training (one style at a time).
# Base model: Wav2CLIP+Librosa from save/audio_stage2_wav2clip (unchanged).
# Data: 100STYLE only (outputs/style_subsets/{style}/100style_motion_ids.txt).

STAGE2_DIR=./save/audio_stage2_wav2clip
STYLE_100_ROOT=/Data/yash.bhardwaj/datasets/Retargeted100Style/RETARGETED_100STYLE
STYLE_SUBSETS=./outputs/style_subsets
HUMANML_DIR=/Data/yash.bhardwaj/datasets/HumanML3D
SAVE_DIR=./save/audio_stage3_dora

# Train one style (choose: old_elderly, angry_aggressive, proud_confident, robot_mechanical)
STYLE=${1:-old_elderly}

python train/train_stage3_dora.py \
  --stage2_dir "$STAGE2_DIR" \
  --style "$STYLE" \
  --style_100_root "$STYLE_100_ROOT" \
  --style_subsets_dir "$STYLE_SUBSETS" \
  --humanml_dir "$HUMANML_DIR" \
  --save_dir "$SAVE_DIR" \
  --dora_rank 8 \
  --dora_alpha 16.0 \
  --epochs 80 \
  --lr 1e-4 \
  --batch_size 16 \
  --weight_decay 0.01 \
  --max_motion_length 196 \
  --save_interval_epochs 20 \
  --seed 42
