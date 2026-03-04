####################################################################################[start]
####################################################################################[start]

"""

Stage 2 Training: Audio Conditioning for MDM.

Loads a pretrained MDM checkpoint, freezes all existing parameters,
and trains only the audio cross-attention layers + audio encoder CNN
on the AIST++ dataset.

Usage:
    python train_audio.py \
        --pretrained_path ./save/humanml_trans_enc_512/model000200000.pt \
        --aist_dir /path/to/aist_plusplus_processed \
        --humanml_dir ./dataset/HumanML3D \
        --save_dir ./save/audio_conditioned \
        --batch_size 32 \
        --lr 1e-4 \
        --num_steps 100000

The script:
1. Creates an MDM model with audio_conditioning=True
2. Loads pretrained weights into the self-attention and FFN layers
3. Freezes everything except audio encoder + cross-attention layers
4. Trains on AIST++ with paired motion-audio data

"""

import os
import sys
import json
import time
import copy
import argparse
import functools
import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# -- MDM imports --

from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from diffusion.resample import create_named_schedule_sampler

# -- new imports --

from data.aist_dataset import get_aist_dataloader

def parse_args():

    parser = argparse.ArgumentParser(description='Train audio conditioning for MDM')

    # - paths --
    
    parser.add_argument('--pretrained_path', type=str, required=True,
                        help='Path to pretrained MDM checkpoint (.pt)')
    parser.add_argument('--aist_dir', type=str, required=True,
                        help='Path to preprocessed AIST++ directory')
    parser.add_argument('--humanml_dir', type=str, default='./dataset/HumanML3D',
                        help='Path to HumanML3D directory (for Mean/Std)')
    parser.add_argument('--save_dir', type=str, default='./save/audio_conditioned',
                        help='Directory to save checkpoints')

    # -- training hyperparameters --
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--num_steps', type=int, default=100000,
                        help='Total training steps')
    parser.add_argument('--save_interval', type=int, default=10000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log loss every N steps')

    # -- audio conditioning --
    
    parser.add_argument('--use_wav2clip', action='store_true',
                        help='Use 519-d audio (Wav2CLIP + 7-d librosa). Requires preprocessed audio_feats_519/.')
    parser.add_argument('--audio_feat_dim', type=int, default=None,
                        help='Audio feature dim (145 or 519). Default: 519 if --use_wav2clip else 145.')
    parser.add_argument('--audio_cond_mask_prob', type=float, default=0.15,
                        help='Probability of dropping audio condition (for CFG)')
    parser.add_argument('--text_cond_mask_prob', type=float, default=0.10,
                        help='Probability of dropping text condition (for CFG)')
    parser.add_argument('--use_audio_token_concat', action='store_true',
                        help='MOSPA-style: concatenate audio tokens with motion tokens at transformer input')
    parser.add_argument('--use_physics_losses', action='store_true',
                        help='Add auxiliary physics losses (L_contact, L_penetrate, L_skating)')
    parser.add_argument('--lambda_contact', type=float, default=0.5)
    parser.add_argument('--lambda_penetrate', type=float, default=1.0)
    parser.add_argument('--lambda_skating', type=float, default=0.5)

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

    args = parser.parse_args()
    if args.audio_feat_dim is None:
        args.audio_feat_dim = 519 if args.use_wav2clip else 145
    return args

def load_pretrained_mdm(args):
    
    """
    
    Create an MDM model with audio conditioning and load pretrained weights
    into the self-attention + FFN layers.
    
    """
    
    # -- load pretrained args to ensure architecture matches --
    
    pretrained_dir = os.path.dirname(args.pretrained_path)
    pretrained_args_path = os.path.join(pretrained_dir, 'args.json')

    if os.path.exists(pretrained_args_path):
    
        with open(pretrained_args_path, 'r') as f:
            pretrained_args = json.load(f)
    
        print(f"Loaded pretrained args from {pretrained_args_path}")
    
        # -- use pretrained architecture params --
        
        args.latent_dim = pretrained_args.get('latent_dim', args.latent_dim)
        args.ff_size = pretrained_args.get('ff_size', args.ff_size)
        args.num_layers = pretrained_args.get('num_layers', args.num_layers)
        args.num_heads = pretrained_args.get('num_heads', args.num_heads)
        args.clip_version = pretrained_args.get('clip_version', args.clip_version)
        args.dropout = pretrained_args.get('dropout', args.dropout)
        args.activation = pretrained_args.get('activation', args.activation)
    
    else:
        print(f"Warning: No args.json found at {pretrained_args_path}, using defaults")

    # -- build model with audio conditioning enabled --
    # -- we construct it manually since create_model_and_diffusion expects a specific args format from the original argparser --
    
    from model.mdm import MDM

    njoints = 263 # HumanML3D
    nfeats = 1

    model = MDM(
        modeltype='',
        njoints=njoints,
        nfeats=nfeats,
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
        
        # -- audio conditioning params (NEW) --
        
        audio_conditioning=True,
        audio_feat_dim=args.audio_feat_dim,
        audio_cond_mask_prob=args.audio_cond_mask_prob,
        use_audio_token_concat=args.use_audio_token_concat,
    )

    # -- load pretrained weights --
    
    print(f"Loading pretrained MDM from {args.pretrained_path}...")
    
    state_dict = torch.load(args.pretrained_path, map_location='cpu')

    # -- handle avg model checkpoints --
    
    if 'model_avg' in state_dict:
        state_dict = state_dict['model']

    """
    
    The pretrained model used nn.TransformerEncoder, but our model uses AudioCondTransformerEncoder. We need to map the weights.
    
    The pretrained keys look like:
    
      seqTransEncoder.layers.0.self_attn.in_proj_weight
      seqTransEncoder.layers.0.norm1.weight
      seqTransEncoder.layers.0.linear1.weight
      etc.
    
    Our new model has the same structure for self_attn, norm1, linear1, linear2, norm2

    """

    # -- load matching keys, skip cross-attention keys (they don't exist in pretrained) --
    
    model_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []

    for key, value in state_dict.items():
    
        if key.startswith('clip_model.'):
            continue # CLIP is loaded separately

        if key in model_state and model_state[key].shape == value.shape:
        
            model_state[key] = value
            loaded_keys.append(key)
        
        else:
            skipped_keys.append(key)

    model.load_state_dict(model_state, strict=False)

    print(f"Loaded {len(loaded_keys)} pretrained weight tensors")
    
    if skipped_keys:
    
        print(f"Skipped {len(skipped_keys)} keys (shape mismatch or new layers)")
    
        for k in skipped_keys[:10]:
            print(f"  - {k}")
    
        if len(skipped_keys) > 10:
            print(f"  ... and {len(skipped_keys) - 10} more")

    return model

def main():

    args = parse_args()
    fixseed(args.seed)

    # -- setup device --
    
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")

    # -- create save directory --
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    # -- load normalization stats from HumanML3D --
    
    mean = np.load(os.path.join(args.humanml_dir, 'Mean.npy'))
    std = np.load(os.path.join(args.humanml_dir, 'Std.npy'))
    
    print(f"Loaded HumanML3D normalization stats: mean {mean.shape}, std {std.shape}")

    # -- create AIST++ dataloader --
    
    print("Loading AIST++ dataset...")
    
    train_loader = get_aist_dataloader(
        aist_dir=args.aist_dir,
        split='train',
        batch_size=args.batch_size,
        max_motion_length=args.max_motion_length,
        num_workers=args.num_workers,
        humanml_mean=mean,
        humanml_std=std,
        use_wav2clip=args.use_wav2clip,
    )
    
    print(f"Train loader: {len(train_loader)} batches of {args.batch_size}")

    # -- create model with audio conditioning --
    
    model = load_pretrained_mdm(args)
    model.to(device)

    # -- freeze non-audio parameters --
    
    model.freeze_non_audio()

    # -- create diffusion --
    
    from utils.model_util import create_gaussian_diffusion
    
    # -- create_gaussian_diffusion expects an args object with specific attributes --
    
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

    # -- optimizer (only audio parameters) --
    
    audio_params = model.audio_parameters()
    optimizer = AdamW(audio_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_steps, eta_min=1e-6)

    # -- schedule sampler for diffusion timesteps --
    
    schedule_sampler = create_named_schedule_sampler('uniform', diffusion)

    # -- training loop --
    
    print("\n=== Starting Stage 2 Training ===")
    print(f"Training audio conditioning on AIST++")
    print(f"Total steps: {args.num_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print()

    model.train()
    step = 0
    running_loss = 0.0
    start_time = time.time()

    # -- resume if checkpoint exists --
    
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        step = ckpt['step']
        # Recreate scheduler so LR matches resumed step (next step will be step+1)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_steps, eta_min=1e-6, last_epoch=step - 1)
        
        print(f"Resumed from step {step}")

    while step < args.num_steps:
    
        for batch_motion, batch_cond in train_loader:
    
            if step >= args.num_steps:
                break

            # -- move to device --
            
            batch_motion = batch_motion.to(device)
            batch_cond['audio_features'] = batch_cond['audio_features'].to(device)
            batch_cond['mask'] = batch_cond['mask'].to(device)

            # -- sample diffusion timesteps --
            
            t, weights = schedule_sampler.sample(
                batch_motion.shape[0], device
            )

            # -- compute loss --
            
            losses = diffusion.training_losses(
                model=model,
                x_start=batch_motion,
                t=t,
                model_kwargs={'y': batch_cond},
            )

            loss = (losses['loss'] * weights).mean()

            if args.use_physics_losses and 'pred_xstart' in losses:
                from utils.physics_losses import physics_losses
                mean_t = torch.from_numpy(mean).float().to(device)
                std_t = torch.from_numpy(std).float().to(device)
                phys = physics_losses(
                    losses['pred_xstart'],
                    mean=mean_t, std=std_t,
                    lambda_contact=args.lambda_contact,
                    lambda_penetrate=args.lambda_penetrate,
                    lambda_skating=args.lambda_skating,
                )
                loss = loss + phys['physics_total']

            # -- backward --
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(audio_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # -- logging --
            
            running_loss += loss.item()
            step += 1

            if step % args.log_interval == 0:
            
                avg_loss = running_loss / args.log_interval
                elapsed = time.time() - start_time
                lr_current = optimizer.param_groups[0]['lr']
            
                print(f"Step {step}/{args.num_steps} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"LR: {lr_current:.2e} | "
                      f"Time: {elapsed/60:.1f}min")
            
                running_loss = 0.0

            # -- save checkpoint --
            
            if step % args.save_interval == 0:
            
                save_path = os.path.join(args.save_dir, f'model{step:09d}.pt')
            
                # -- save only audio parameters + full state for resuming --
            
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                    'args': vars(args),
                }, save_path)
            
                print(f"Saved checkpoint: {save_path}")

    # -- final save --
    
    save_path = os.path.join(args.save_dir, f'model_final.pt')
    
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'args': vars(args),
    }, save_path)
    
    print(f"\nTraining complete! Final model saved to {save_path}")
    print(f"Total time: {(time.time() - start_time) / 3600:.1f} hours")

if __name__ == '__main__':
    main()
    
####################################################################################[end]
####################################################################################[end]
