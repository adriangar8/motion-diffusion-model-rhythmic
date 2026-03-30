#!/usr/bin/env bash
# Stage 2 variant: Wav2CLIP + 7-d librosa (519-d) + MOSPA-style token concatenation.
# Override paths via environment variables or edit defaults below.

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

PRETRAINED=${PRETRAINED:-./final_weights/pretrained/humanml_trans_enc_512/model000200000.pt}
AIST_DIR=${AIST_DIR:-./dataset/aist}
HUMANML_DIR=${HUMANML_DIR:-./dataset/HumanML3D}
SAVE_DIR="./save/audio_stage2_wav2clip_mospa"

python train/train_audio.py \
  --pretrained_path "$PRETRAINED" \
  --aist_dir "$AIST_DIR" \
  --humanml_dir "$HUMANML_DIR" \
  --save_dir "$SAVE_DIR" \
  --use_wav2clip \
  --use_audio_token_concat \
  --batch_size 32 \
  --lr 1e-4 \
  --num_steps 100000 \
  --save_interval 10000 \
  --log_interval 100 \
  --num_workers 4 \
  "$@"
