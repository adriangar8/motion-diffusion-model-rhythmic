#!/usr/bin/env bash
# Resume Stage 2 training from latest checkpoint (50k steps).
# Same config as original run; new checkpoints will be saved to save_dir.

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

python train/train_audio.py \
  --pretrained_path /Data/yash.bhardwaj/pretrained/humanml_trans_enc_512/model000200000.pt \
  --aist_dir /Data/yash.bhardwaj/datasets/aist \
  --humanml_dir /Data/yash.bhardwaj/datasets/HumanML3D \
  --save_dir ./save/audio_stage2_wav2clip \
  --use_wav2clip \
  --resume_checkpoint ./save/audio_stage2_wav2clip/model000050000.pt \
  --batch_size 32 \
  --lr 1e-4 \
  --num_steps 100000 \
  --save_interval 10000 \
  --log_interval 100 \
  --num_workers 4 \
  "$@"
