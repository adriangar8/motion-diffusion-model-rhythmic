#!/usr/bin/env bash
# Stage 2 variant: Wav2CLIP + 7-d librosa (519-d) + MOSPA-style token concatenation.
# Checkpoints go to save/audio_stage2_wav2clip_mospa/ only (not under audio_stage2_wav2clip).

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

SAVE_DIR="./save/audio_stage2_wav2clip_mospa"

python train/train_audio.py \
  --pretrained_path /Data/yash.bhardwaj/pretrained/humanml_trans_enc_512/model000200000.pt \
  --aist_dir /Data/yash.bhardwaj/datasets/aist \
  --humanml_dir /Data/yash.bhardwaj/datasets/HumanML3D \
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
