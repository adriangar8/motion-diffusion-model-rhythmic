####################################################################################[start]
####################################################################################[start]

import sys
sys.path.insert(0, '.')

"""

Stage 2 Training v2: Audio Conditioning for MDM.

Changes from v1:
  - --unfreeze_top_n: Unfreeze top N backbone layers (default 2)
  - --audio_feat_dim: Supports 52-dim v2 features (default) or 145-dim v1
  - --audio_feats_subdir: Which preprocessed audio dir to use (default: audio_feats_v2)
  - Beat frames extracted on-the-fly and passed to model for beat-aware masking
  - Lower default learning rate for unfrozen layers (separate param groups)

Usage:
    python train_audio_v2.py \
        --pretrained_path ./save/humanml_trans_enc_512/model000200000.pt \
        --aist_dir ./dataset/aist \
        --humanml_dir ./dataset/HumanML3D \
        --save_dir ./save/audio_stage2_v2 \
        --batch_size 32 \
        --lr 1e-4 \
        --backbone_lr 1e-5 \
        --unfreeze_top_n 2 \
        --audio_feat_dim 52 \
        --num_steps 100000

"""

import os
import sys
import json
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from diffusion.resample import create_named_schedule_sampler

from data.aist_dataset import AISTDataset, aist_collate_fn
from torch.utils.data import DataLoader

def parse_args():
    
    parser = argparse.ArgumentParser(description='Train audio conditioning v2')

    # -- paths --

    parser.add_argument('--pretrained_path', type=str, required=True)
    parser.add_argument('--aist_dir', type=str, required=True)
    parser.add_argument('--humanml_dir', type=str, default='./dataset/HumanML3D')
    parser.add_argument('--save_dir', type=str, default='./save/audio_stage2_v2')

    # -- training --

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for audio modules')
    parser.add_argument('--backbone_lr', type=float, default=1e-5,
                        help='Learning rate for unfrozen backbone layers (lower)')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--save_interval', type=int, default=10000)
    parser.add_argument('--log_interval', type=int, default=100)

    # -- v2 changes --

    parser.add_argument('--unfreeze_top_n', type=int, default=2,
                        help='Unfreeze top N backbone transformer layers')
    parser.add_argument('--audio_feat_dim', type=int, default=52,
                        help='Audio feature dimension (52 for v2, 145 for v1)')
    parser.add_argument('--audio_feats_subdir', type=str, default='audio_feats_v2',
                        help='Subdirectory under processed/ for audio features')
    parser.add_argument('--temporal_sigma', type=float, default=4.0,
                        help='Temporal locality sigma for beat-aware masking')
    parser.add_argument('--beat_weight', type=float, default=2.0,
                        help='Extra attention weight at beat positions')

    # -- audio conditioning --

    parser.add_argument('--audio_cond_mask_prob', type=float, default=0.15)
    parser.add_argument('--text_cond_mask_prob', type=float, default=0.10)

    # -- model (must match pretrained) --

    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--ff_size', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--clip_version', type=str, default='ViT-B/32')

    # -- data --

    parser.add_argument('--max_motion_length', type=int, default=196)
    parser.add_argument('--num_workers', type=int, default=4)

    # -- misc --

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--resume_checkpoint', type=str, default='')

    return parser.parse_args()

def load_pretrained_mdm(args):

    """Create MDM with audio conditioning v2 and load pretrained weights."""

    pretrained_dir = os.path.dirname(args.pretrained_path)
    pretrained_args_path = os.path.join(pretrained_dir, 'args.json')

    if os.path.exists(pretrained_args_path):
        with open(pretrained_args_path, 'r') as f:
            pretrained_args = json.load(f)
        print(f"Loaded pretrained args from {pretrained_args_path}")
        args.latent_dim = pretrained_args.get('latent_dim', args.latent_dim)
        args.ff_size = pretrained_args.get('ff_size', args.ff_size)
        args.num_layers = pretrained_args.get('num_layers', args.num_layers)
        args.num_heads = pretrained_args.get('num_heads', args.num_heads)
        args.clip_version = pretrained_args.get('clip_version', args.clip_version)
        args.dropout = pretrained_args.get('dropout', args.dropout)
        args.activation = pretrained_args.get('activation', args.activation)
    else:
        print(f"Warning: No args.json found at {pretrained_args_path}")

    from model.mdm import MDM

    model = MDM(
        modeltype='',
        njoints=263,
        nfeats=1,
        num_actions=1,
        translation=True,
        pose_rep='rot6d',
        glob=True,
        glob_rot=[3.141592653589793, 0, 0],
        latent_dim=args.latent_dim,
        ff_size=args.ff_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        activation=args.activation,
        data_rep='hml_vec',
        dataset='humanml',
        clip_dim=512,
        arch='trans_enc',
        clip_version=args.clip_version,
        cond_mode='text',
        cond_mask_prob=args.text_cond_mask_prob,
        audio_conditioning=True,
        audio_feat_dim=args.audio_feat_dim,
        audio_cond_mask_prob=args.audio_cond_mask_prob,
        temporal_sigma=args.temporal_sigma,
        beat_weight=args.beat_weight,
    )

    # -- Load pretrained weights --
    
    print(f"Loading pretrained MDM from {args.pretrained_path}...")
    
    state_dict = torch.load(args.pretrained_path, map_location='cpu')
    
    if 'model_avg' in state_dict:
        state_dict = state_dict['model']

    model_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []

    for key, value in state_dict.items():
    
        if key.startswith('clip_model.'):
            continue
        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            loaded_keys.append(key)
        else:
            skipped_keys.append(key)

    model.load_state_dict(model_state, strict=False)
    
    print(f"Loaded {len(loaded_keys)} pretrained weight tensors")
    
    if skipped_keys:
        print(f"Skipped {len(skipped_keys)} keys (shape mismatch or new)")

    return model

def extract_beat_frames_from_audio(audio_features, audio_feat_dim=52):

    """

    Extract beat frame indices from audio features in a batch.

    For v2 features (52-dim): beat_soft is at index 34
    For v1 features (145-dim): beat_indicator is at index 129

    Returns list of beat frame indices (from first sample in batch).

    """

    if audio_feat_dim == 52:
        beat_channel = 34 # beat_soft in v2
    else:
        beat_channel = 129 # beat_indicator in v1

    beat_signal = audio_features[0, :, beat_channel].cpu().numpy()

    # -- find peaks above threshold --
    
    threshold = 0.3 if audio_feat_dim == 52 else 0.5
    beat_frames = list(np.where(beat_signal > threshold)[0])

    return beat_frames

def main():

    args = parse_args()
    fixseed(args.seed)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    # -- normalization stats --

    mean = np.load(os.path.join(args.humanml_dir, 'Mean.npy'))
    std = np.load(os.path.join(args.humanml_dir, 'Std.npy'))

    # -- create dataset with v2 audio features --

    print("Loading AIST++ dataset...")

    dataset = AISTDataset(
        aist_dir=args.aist_dir,
        split='train',
        max_motion_length=args.max_motion_length,
        humanml_mean=mean,
        humanml_std=std,
    )

    # -- override audio directory to use v2 features --
    
    dataset.audio_dir = os.path.join(
        args.aist_dir, 'processed', args.audio_feats_subdir
    )
    print(f"Using audio features from: {dataset.audio_dir}")

    # -- verify feature dimension --
    
    sample_audio = np.load(os.path.join(
        dataset.audio_dir,
        os.listdir(dataset.audio_dir)[0]
    ))
    
    actual_dim = sample_audio.shape[1]
    
    print(f"Audio feature dim: {actual_dim} (expected {args.audio_feat_dim})")
    
    assert actual_dim == args.audio_feat_dim, \
        f"Feature dim mismatch: got {actual_dim}, expected {args.audio_feat_dim}"

    # -- custom collate for v2 feature dim --
    
    def collate_v2(batch):
    
        batch_size = len(batch)
        max_len = max(item['length'] for item in batch)

        motion_padded = np.zeros((batch_size, max_len, 263), dtype=np.float32)
        audio_padded = np.zeros((batch_size, max_len, args.audio_feat_dim), dtype=np.float32)
        mask = np.zeros((batch_size, max_len), dtype=bool)
        lengths = []
        texts = []

        for i, item in enumerate(batch):
            L = item['length']
            motion_padded[i, :L] = item['motion']
            audio_padded[i, :L] = item['audio'][:L, :args.audio_feat_dim]
            mask[i, :L] = True
            lengths.append(L)
            texts.append(item['text'])

        motion_tensor = torch.from_numpy(motion_padded).permute(0, 2, 1).unsqueeze(2)
        audio_tensor = torch.from_numpy(audio_padded)
        mask_tensor = torch.from_numpy(mask).unsqueeze(1).unsqueeze(1)

        cond = {
            'text': texts,
            'audio_features': audio_tensor,
            'mask': mask_tensor,
            'lengths': torch.tensor(lengths),
        }
        return motion_tensor, cond

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_v2,
        drop_last=True,
        pin_memory=True,
    )
    
    print(f"Train loader: {len(train_loader)} batches")

    # -- model --
    
    model = load_pretrained_mdm(args)
    model.to(device)

    # -- freeze with partial unfreezing --
    
    model.freeze_non_audio(unfreeze_top_n=args.unfreeze_top_n)

    # -- diffusion --
    
    from utils.model_util import create_gaussian_diffusion
    from types import SimpleNamespace

    diff_args = SimpleNamespace(
        diffusion_steps=1000,
        noise_schedule='cosine',
        sigma_small=True,
        lambda_vel=0.0,
        lambda_rcxyz=0.0,
        lambda_fc=0.0,
        predict_xstart=True,
        rescale_timesteps=False,
        timestep_respacing='',
    )
    diffusion = create_gaussian_diffusion(diff_args)

    # -- optimizer with separate param groups --
    
    audio_params = model.audio_parameters(unfreeze_top_n=0) # audio-only params
    backbone_params = []
    
    if args.unfreeze_top_n > 0:
        
        backbone_params = model.audio_parameters(unfreeze_top_n=args.unfreeze_top_n)
        
        # -- remove audio params to get only backbone params --
        
        audio_param_ids = set(id(p) for p in audio_params)
        backbone_params = [p for p in backbone_params if id(p) not in audio_param_ids]

    param_groups = [
        {'params': audio_params, 'lr': args.lr},
    ]
    if backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': args.backbone_lr,
        })
        print(f"Optimizer: audio modules @ lr={args.lr}, "
              f"backbone top-{args.unfreeze_top_n} @ lr={args.backbone_lr}")
    else:
        print(f"Optimizer: audio modules @ lr={args.lr}")

    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_steps, eta_min=1e-6)
    schedule_sampler = create_named_schedule_sampler('uniform', diffusion)

    # -- training loop --
    
    print(f"\n=== Stage 2 Training v2 ===")
    print(f"Audio feat dim: {args.audio_feat_dim}")
    print(f"Unfrozen backbone layers: {args.unfreeze_top_n}")
    print(f"Beat-aware masking: sigma={args.temporal_sigma}, weight={args.beat_weight}")
    print(f"Steps: {args.num_steps}, Batch: {args.batch_size}")
    print()

    model.train()
    step = 0
    running_loss = 0.0
    start_time = time.time()

    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        step = ckpt['step']
        print(f"Resumed from step {step}")

    while step < args.num_steps:
        
        for batch_motion, batch_cond in train_loader:
        
            if step >= args.num_steps:
                break

            batch_motion = batch_motion.to(device)
            batch_cond['audio_features'] = batch_cond['audio_features'].to(device)
            batch_cond['mask'] = batch_cond['mask'].to(device)

            # -- extract beat frames for beat-aware masking --
            
            beat_frames = extract_beat_frames_from_audio(
                batch_cond['audio_features'],
                audio_feat_dim=args.audio_feat_dim
            )
            batch_cond['beat_frames'] = beat_frames

            t, weights = schedule_sampler.sample(
                batch_motion.shape[0], device
            )

            losses = diffusion.training_losses(
                model=model,
                x_start=batch_motion,
                t=t,
                model_kwargs={'y': batch_cond},
            )

            loss = (losses['loss'] * weights).mean()

            optimizer.zero_grad()
            loss.backward()

            # -- alip gradients for all trainable params --
            
            all_params = [p for g in param_groups for p in g['params']]
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            step += 1

            if step % args.log_interval == 0:
                
                avg_loss = running_loss / args.log_interval
                elapsed = time.time() - start_time
                lr_audio = optimizer.param_groups[0]['lr']
                lr_info = f"LR(audio): {lr_audio:.2e}"
                
                if len(optimizer.param_groups) > 1:
                
                    lr_backbone = optimizer.param_groups[1]['lr']
                    lr_info += f", LR(backbone): {lr_backbone:.2e}"

                # -- log gate values --
                
                gate_vals = [
                    f"{torch.tanh(layer.cross_attn_gate).item():.3f}"
                    for layer in model.seqTransEncoder.layers
                ]

                print(f"Step {step}/{args.num_steps} | "
                      f"Loss: {avg_loss:.4f} | {lr_info} | "
                      f"Gates: [{', '.join(gate_vals)}] | "
                      f"Time: {elapsed/60:.1f}min")

                running_loss = 0.0

            if step % args.save_interval == 0:
                save_path = os.path.join(args.save_dir, f'model{step:09d}.pt')
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                    'args': vars(args),
                }, save_path)
                print(f"Saved: {save_path}")

    # -- final save --
    
    save_path = os.path.join(args.save_dir, 'model_final.pt')
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'args': vars(args),
    }, save_path)

    print(f"\nTraining complete! {save_path}")
    print(f"Total time: {(time.time() - start_time) / 3600:.1f} hours")

if __name__ == '__main__':
    main()
    
####################################################################################[end]
####################################################################################[end]
