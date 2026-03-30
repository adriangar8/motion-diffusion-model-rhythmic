#!/usr/bin/env bash
# Stage 2: Wav2CLIP + 7-d librosa (519-d) with beat-aware cross-attention bias.
# Override paths via environment variables or edit defaults below.

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

PRETRAINED=${PRETRAINED:-./final_weights/pretrained/humanml_trans_enc_512/model000200000.pt}
AIST_DIR=${AIST_DIR:-./dataset/aist}
HUMANML_DIR=${HUMANML_DIR:-./dataset/HumanML3D}
SAVE_DIR="./save/audio_stage2_wav2clip_beataware"

python train/train_audio.py \
  --pretrained_path "$PRETRAINED" \
  --aist_dir "$AIST_DIR" \
  --humanml_dir "$HUMANML_DIR" \
  --save_dir "$SAVE_DIR" \
  --use_wav2clip \
  --batch_size 32 \
  --lr 1e-4 \
  --num_steps 100000 \
  --save_interval 10000 \
  --log_interval 100 \
  --num_workers 4 \
  --joint_cond_mask_prob 0.05 \
  --text_cond_mask_prob 0.15 \
  --audio_cond_mask_prob 0.15 \
  "$@"
