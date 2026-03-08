"""
Stage 3: DoRA style adapter training.

Loads the Wav2CLIP+Librosa Stage 2 model (save/audio_stage2_wav2clip), freezes all parameters,
injects DoRA adapters on Q and V of self-attention in each transformer layer, and trains
only the DoRA parameters on 100STYLE data for the given style.

Usage:
    python train/train_stage3_dora.py \\
        --stage2_dir ./save/audio_stage2_wav2clip \\
        --style old_elderly \\
        --style_100_root /Data/yash.bhardwaj/datasets/Retargeted100Style/RETARGETED_100STYLE \\
        --style_subsets_dir ./outputs/style_subsets \\
        --humanml_dir /Data/yash.bhardwaj/datasets/HumanML3D \\
        --save_dir ./save/audio_stage3_dora \\
        --dora_rank 8 \\
        --epochs 80 \\
        --lr 1e-4 \\
        --batch_size 16
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.fixseed import fixseed
from utils.model_util import create_gaussian_diffusion
from diffusion.resample import create_named_schedule_sampler
from types import SimpleNamespace

from data.style_100_dataset import (
    Style100Dataset, HumanML3DRetrievedDataset, CombinedStyleDataset, style_100_collate
)
from model.dora import DoRAMultiheadAttention


def parse_args():
    p = argparse.ArgumentParser(description='Stage 3: DoRA style adapter training')
    # paths
    p.add_argument('--stage2_dir', type=str, default='./save/audio_stage2_wav2clip',
                   help='Stage 2 checkpoint dir (Wav2CLIP model); do not modify this dir')
    p.add_argument('--style', type=str, required=True,
                   choices=['old_elderly', 'angry_aggressive', 'proud_confident', 'robot_mechanical'],
                   help='Style to train')
    p.add_argument('--style_100_root', type=str, default='/Data/yash.bhardwaj/datasets/Retargeted100Style/RETARGETED_100STYLE',
                   help='Path to RETARGETED_100STYLE (new_joint_vecs, texts)')
    p.add_argument('--style_subsets_dir', type=str, default='./outputs/style_subsets',
                   help='Dir containing {style}/100style_motion_ids.txt')
    p.add_argument('--humanml_dir', type=str, default='/Data/yash.bhardwaj/datasets/HumanML3D',
                   help='HumanML3D dir for Mean.npy, Std.npy')
    p.add_argument('--save_dir', type=str, default='./save/audio_stage3_dora',
                   help='Root dir for Stage 3; adapter saved to save_dir/style/')
    # DoRA
    p.add_argument('--dora_rank', type=int, default=8, help='DoRA low-rank r')
    p.add_argument('--dora_alpha', type=float, default=16.0, help='DoRA scale (alpha/rank in LoRA convention)')
    # training
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--max_motion_length', type=int, default=196)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--save_interval_epochs', type=int, default=20)
    p.add_argument('--log_interval', type=int, default=10)
    p.add_argument('--style_upsample', type=int, default=1)
    p.add_argument('--scheduler_t_max_mult', type=float, default=2.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='')
    p.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume (adapter only)')
    return p.parse_args()


def load_stage2_model(stage2_dir, device):
    """Load Stage 2 MDM (Wav2CLIP) and its args. Do not modify stage2_dir."""
    args_path = os.path.join(stage2_dir, 'args.json')
    ckpt_path = os.path.join(stage2_dir, 'model_final.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(stage2_dir, 'model000100000.pt')  # fallback
    assert os.path.exists(ckpt_path), f"Stage 2 checkpoint not found: {stage2_dir}"
    assert os.path.exists(args_path), f"Stage 2 args not found: {args_path}"

    with open(args_path, 'r') as f:
        ckpt_args = json.load(f)

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
        latent_dim=ckpt_args['latent_dim'],
        ff_size=ckpt_args['ff_size'],
        num_layers=ckpt_args['num_layers'],
        num_heads=ckpt_args['num_heads'],
        dropout=ckpt_args['dropout'],
        activation=ckpt_args['activation'],
        data_rep='hml_vec',
        dataset='humanml',
        clip_dim=512,
        arch='trans_enc',
        clip_version=ckpt_args.get('clip_version', 'ViT-B/32'),
        cond_mode='text',
        cond_mask_prob=ckpt_args.get('text_cond_mask_prob', 0.1),
        audio_conditioning=True,
        audio_feat_dim=ckpt_args.get('audio_feat_dim', 519),
        audio_cond_mask_prob=ckpt_args.get('audio_cond_mask_prob', 0.15),
        use_audio_token_concat=ckpt_args.get('use_audio_token_concat', False),
    )

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    return model, ckpt_args


def inject_dora(model, rank, alpha, device=None):
    """Replace self_attn in each transformer layer with DoRA version. Returns list of DoRA params."""
    if device is None:
        device = next(model.parameters()).device
    dora_params = []
    for layer in model.seqTransEncoder.layers:
        old_attn = layer.self_attn
        dora_attn = DoRAMultiheadAttention.from_mha(old_attn, rank=rank, alpha=alpha, device=device)
        layer.self_attn = dora_attn
        dora_params.extend(list(dora_attn.dora_q.parameters()))
        dora_params.extend(list(dora_attn.dora_v.parameters()))
    return dora_params


def freeze_all_except_dora(model):
    # Freeze everything first, then unfreeze only DoRA: magnitude m and low-rank lora_A, lora_B.
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if name.endswith('lora_A') or name.endswith('lora_B') or name.endswith('magnitude'):
            p.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen: {total - trainable:,} | Trainable (DoRA): {trainable:,} ({100*trainable/total:.2f}%)")


def main():
    args = parse_args()
    fixseed(args.seed)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    style_save_dir = os.path.join(args.save_dir, args.style)
    os.makedirs(style_save_dir, exist_ok=True)
    with open(os.path.join(style_save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    mean = np.load(os.path.join(args.humanml_dir, 'Mean.npy'))
    std = np.load(os.path.join(args.humanml_dir, 'Std.npy'))

    # 100STYLE motions
    style_ids_file = os.path.join(args.style_subsets_dir, args.style, '100style_motion_ids.txt')
    assert os.path.exists(style_ids_file), f"Missing {style_ids_file}"
    style_ds = Style100Dataset(
        motion_ids_file=style_ids_file,
        style_root=args.style_100_root,
        mean=mean,
        std=std,
        max_length=args.max_motion_length,
    )

    # CLIP-retrieved HumanML3D motions
    hml_ids_file = os.path.join(args.style_subsets_dir, args.style, 'retrieved_motion_ids.txt')
    if os.path.exists(hml_ids_file):
        hml_ds = HumanML3DRetrievedDataset(
            motion_ids_file=hml_ids_file,
            humanml_dir=args.humanml_dir,
            mean=mean,
            std=std,
            max_length=args.max_motion_length,
        )
        dataset = CombinedStyleDataset(style_ds, hml_ds, style_upsample=args.style_upsample)
    else:
        print(f"No retrieved_motion_ids.txt found for {args.style}, using 100STYLE only")
        dataset = style_ds

    if len(dataset) < args.batch_size:
        raise ValueError(
            f"Dataset has {len(dataset)} samples but batch_size is {args.batch_size}. "
            "Use a smaller --batch_size."
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=style_100_collate,
        drop_last=True,
    )

    print(f"Loading Stage 2 from {args.stage2_dir} (unchanged)...")
    model, ckpt_args = load_stage2_model(args.stage2_dir, device)
    inject_dora(model, rank=args.dora_rank, alpha=args.dora_alpha, device=device)
    model.to(device)
    freeze_all_except_dora(model)

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
    schedule_sampler = create_named_schedule_sampler('uniform', diffusion)

    dora_params = [p for p in model.parameters() if p.requires_grad]
    # foreach=False avoids multi-tensor device mismatch in some environments
    optimizer = AdamW(dora_params, lr=args.lr, weight_decay=args.weight_decay, foreach=False)
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    t_max = int(total_steps * args.scheduler_t_max_mult)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-6)
    print(f"LR schedule: CosineAnnealingLR T_max={t_max} (LR stays higher for longer)")

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        if 'dora_state' in ckpt:
            # Load only DoRA state into model
            model.load_state_dict(ckpt['dora_state'], strict=False)
        optimizer.load_state_dict(ckpt.get('optimizer', optimizer.state_dict()))
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resumed from epoch {start_epoch}")

    model.train()
    print(f"\n=== Stage 3 DoRA training: {args.style} ===")
    print(f"Samples: {len(dataset)} | Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")

    for epoch in range(start_epoch, args.epochs):
        running_loss = 0.0
        for batch_idx, (motion, cond) in enumerate(loader):
            motion = motion.to(device)
            for k in list(cond['y'].keys()):
                v = cond['y'][k]
                if torch.is_tensor(v):
                    cond['y'][k] = v.to(device)
            if 'text_embed' not in cond['y']:
                with torch.no_grad():
                    cond['y']['text_embed'] = model.encode_text(cond['y']['text']).to(device)
            # No audio for Stage 3
            t, weights = schedule_sampler.sample(motion.shape[0], device)
            losses = diffusion.training_losses(model=model, x_start=motion, t=t, model_kwargs={'y': cond['y']})
            loss = (losses['loss'] * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dora_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            if (batch_idx + 1) % args.log_interval == 0:
                print(f"Epoch {epoch+1}/{args.epochs} batch {batch_idx+1}/{len(loader)} loss {running_loss/(batch_idx+1):.4f}")
        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        if (epoch + 1) % args.save_interval_epochs == 0 or (epoch + 1) == args.epochs:
            dora_state = {k: v.cpu() for k, v in model.state_dict().items() if 'dora' in k or 'lora' in k}
            save_path = os.path.join(style_save_dir, f'adapter_epoch{epoch+1:03d}.pt')
            torch.save({
                'dora_state': dora_state,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'style': args.style,
                'args': vars(args),
            }, save_path)
            print(f"Saved {save_path}")

    final_path = os.path.join(style_save_dir, 'adapter_final.pt')
    dora_state = {k: v.cpu() for k, v in model.state_dict().items() if 'dora' in k or 'lora' in k}
    torch.save({
        'dora_state': dora_state,
        'style': args.style,
        'args': vars(args),
    }, final_path)
    print(f"Stage 3 ({args.style}) done. Adapter: {final_path}")


if __name__ == '__main__':
    main()
