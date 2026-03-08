"""
Generate motion with Stage 3 style adapter (DoRA) on top of Stage 2.

Supports both legacy AudioCFG (3 forward passes) and GCDM composite guidance
(4 forward passes with timestep-dependent condition weighting).
Also passes beat_frames for beat-aware cross-attention at inference.

Usage:
    python -m sample.generate_stage3_style \
        --stage2_dir ./save/audio_stage2_wav2clip_beataware \
        --adapter_path ./save/audio_stage3_dora/old_elderly/adapter_final.pt \
        --audio_path /Data/yash.bhardwaj/datasets/aist/audio/mBR0.wav \
        --text_prompt "a person performs breakdancing moves to music" \
        --output_dir ./save/audio_stage3_dora/old_elderly/samples_breakdance \
        --num_samples 3
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from copy import deepcopy
from types import SimpleNamespace

# Reuse CFG wrapper from generate_audio
from sample.generate_audio import AudioCFGSampleModel


def load_stage2_plus_adapter(stage2_dir, adapter_path, device, dora_rank=8, dora_alpha=16.0):
    """Load Stage 2 MDM, inject DoRA, load adapter weights."""
    from model.mdm import MDM
    from model.dora import DoRAMultiheadAttention

    args_path = os.path.join(stage2_dir, 'args.json')
    ckpt_path = os.path.join(stage2_dir, 'model_final.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(stage2_dir, 'model000100000.pt')
    assert os.path.exists(ckpt_path), f"Stage 2 not found: {stage2_dir}"
    with open(args_path, 'r') as f:
        ckpt_args = json.load(f)

    model = MDM(
        modeltype='', njoints=263, nfeats=1, num_actions=1, translation=True,
        pose_rep='rot6d', glob=True, glob_rot=[3.141592653589793, 0, 0],
        latent_dim=ckpt_args['latent_dim'], ff_size=ckpt_args['ff_size'],
        num_layers=ckpt_args['num_layers'], num_heads=ckpt_args['num_heads'],
        dropout=ckpt_args['dropout'], activation=ckpt_args['activation'],
        data_rep='hml_vec', dataset='humanml', clip_dim=512, arch='trans_enc',
        clip_version=ckpt_args.get('clip_version', 'ViT-B/32'),
        cond_mode='text', cond_mask_prob=ckpt_args.get('text_cond_mask_prob', 0.1),
        audio_conditioning=True, audio_feat_dim=ckpt_args.get('audio_feat_dim', 519),
        audio_cond_mask_prob=ckpt_args.get('audio_cond_mask_prob', 0.15),
        use_audio_token_concat=ckpt_args.get('use_audio_token_concat', False),
    )
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state, strict=False)

    # Inject DoRA
    for layer in model.seqTransEncoder.layers:
        old_attn = layer.self_attn
        layer.self_attn = DoRAMultiheadAttention.from_mha(
            old_attn, rank=dora_rank, alpha=dora_alpha, device=device
        )

    model.to(device)

    # Load adapter state (dora_state only)
    adapter_ckpt = torch.load(adapter_path, map_location=device)
    if 'dora_state' in adapter_ckpt:
        model.load_state_dict(adapter_ckpt['dora_state'], strict=False)
        print(f"Loaded DoRA adapter: {adapter_path}")
    else:
        print("Warning: no dora_state in adapter checkpoint")

    model.eval()
    return model, ckpt_args


def parse_args():
    p = argparse.ArgumentParser(description='Generate motion with Stage 3 style (DoRA)')
    p.add_argument('--stage2_dir', type=str, default='./save/audio_stage2_wav2clip_beataware',
                   help='Stage 2 checkpoint dir (default: beataware model)')
    p.add_argument('--adapter_path', type=str, required=True,
                    help='Path to adapter_final.pt (e.g. save/audio_stage3_dora/old_elderly/adapter_final.pt)')
    p.add_argument('--audio_path', type=str, default='/Data/yash.bhardwaj/datasets/aist/audio/mBR0.wav')
    p.add_argument('--text_prompt', type=str, default='a person performs breakdancing moves to music')
    p.add_argument('--output_dir', type=str, default='./save/audio_stage3_dora/old_elderly/samples_breakdance')
    p.add_argument('--num_samples', type=int, default=3)
    p.add_argument('--motion_length', type=float, default=0.0)
    p.add_argument('--guidance_param', type=float, default=2.5)
    p.add_argument('--audio_guidance_param', type=float, default=2.5)
    p.add_argument('--dora_rank', type=int, default=8)
    p.add_argument('--dora_alpha', type=float, default=16.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='')
    p.add_argument('--fps', type=int, default=20)
    p.add_argument('--use_gcdm', action='store_true',
                    help='Use GCDM composite guidance instead of legacy AudioCFG')
    p.add_argument('--gcdm_alpha', type=float, default=3.0,
                    help='GCDM overall guidance strength')
    p.add_argument('--gcdm_beta_text', type=float, default=1.0,
                    help='GCDM per-condition weight for text')
    p.add_argument('--gcdm_beta_audio', type=float, default=1.5,
                    help='GCDM per-condition weight for audio')
    p.add_argument('--gcdm_lambda_start', type=float, default=0.8,
                    help='GCDM lambda at t=T (early, joint dominates)')
    p.add_argument('--gcdm_lambda_end', type=float, default=0.2,
                    help='GCDM lambda at t=0 (late, independent dominates)')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    # Load Stage 2 + DoRA adapter
    model, ckpt_args = load_stage2_plus_adapter(
        args.stage2_dir, args.adapter_path, device,
        dora_rank=args.dora_rank, dora_alpha=args.dora_alpha,
    )

    from utils.model_util import create_gaussian_diffusion
    diff_args = SimpleNamespace(
        diffusion_steps=1000, noise_schedule='cosine', sigma_small=True,
        lambda_vel=0.0, lambda_rcxyz=0.0, lambda_fc=0.0, lambda_target_loc=0.0,
    )
    diffusion = create_gaussian_diffusion(diff_args)
    if args.use_gcdm:
        from sample.gcdm import GCDMSampleModel
        cfg_model = GCDMSampleModel(
            model, alpha=args.gcdm_alpha,
            beta_text=args.gcdm_beta_text, beta_audio=args.gcdm_beta_audio,
            lambda_start=args.gcdm_lambda_start, lambda_end=args.gcdm_lambda_end,
            diffusion_steps=diffusion.num_timesteps,
        )
        print(f"GCDM guidance: alpha={args.gcdm_alpha}, "
              f"beta_text={args.gcdm_beta_text}, beta_audio={args.gcdm_beta_audio}, "
              f"lambda=[{args.gcdm_lambda_start} -> {args.gcdm_lambda_end}]")
    else:
        cfg_model = AudioCFGSampleModel(
            model, text_scale=args.guidance_param,
            audio_scale=args.audio_guidance_param,
        )

    humanml_dir = ckpt_args.get('humanml_dir', '/Data/yash.bhardwaj/datasets/HumanML3D')
    mean = np.load(os.path.join(humanml_dir, 'Mean.npy'))
    std = np.load(os.path.join(humanml_dir, 'Std.npy'))

    # Audio features
    audio_features = None
    n_frames = int(args.motion_length * args.fps) if args.motion_length > 0 else 196

    if args.audio_path and os.path.exists(args.audio_path):
        from model.audio_features_wav2clip import extract_wav2clip_plus_librosa
        duration = args.motion_length if args.motion_length > 0 else None
        audio_feat = extract_wav2clip_plus_librosa(
            args.audio_path, target_fps=args.fps, duration=duration, device=device,
        )
        if args.motion_length <= 0:
            n_frames = min(audio_feat.shape[0], 196)
        audio_feat = audio_feat[:n_frames]
        audio_features = torch.from_numpy(audio_feat).float().unsqueeze(0)
        audio_features = audio_features.repeat(args.num_samples, 1, 1).to(device)
        print(f"Audio: {args.audio_path} → {n_frames} frames ({n_frames/args.fps:.1f}s)")
    elif args.audio_path:
        print(f"Warning: audio not found {args.audio_path}")

    if args.motion_length > 0:
        n_frames = int(args.motion_length * args.fps)

    print(f"Generating {args.num_samples} samples, {n_frames} frames | Text: '{args.text_prompt}'")

    model_kwargs = {
        'y': {
            'text': [args.text_prompt] * args.num_samples,
            'mask': torch.ones(args.num_samples, 1, 1, n_frames, dtype=torch.bool).to(device),
            'lengths': torch.tensor([n_frames] * args.num_samples).to(device),
        }
    }
    if audio_features is not None:
        model_kwargs['y']['audio_features'] = audio_features
        feat_dim = audio_features.shape[-1]
        _bi = 513 if feat_dim >= 519 else 129
        _bs = audio_features[0, :, _bi].cpu().numpy()
        model_kwargs['y']['beat_frames'] = list(np.where(_bs > 0.5)[0])
        print(f"Beat frames: {len(model_kwargs['y']['beat_frames'])} beats for cross-attn bias")

    sample_shape = (args.num_samples, 263, 1, n_frames)
    with torch.no_grad():
        sample = diffusion.p_sample_loop(
            cfg_model, sample_shape, clip_denoised=False,
            model_kwargs=model_kwargs, skip_timesteps=0, init_image=None,
            progress=True, dump_steps=None, noise=None, const_noise=False,
        )

    sample = sample.squeeze(2).permute(0, 2, 1).cpu().numpy()
    sample = sample * std + mean
    from scipy.ndimage import gaussian_filter1d
    for i in range(sample.shape[0]):
        sample[i] = gaussian_filter1d(sample[i], sigma=1.0, axis=0)

    os.makedirs(args.output_dir, exist_ok=True)
    mode = 'audio' if audio_features is not None else 'text'
    for i in range(args.num_samples):
        out_path = os.path.join(args.output_dir, f'sample_{mode}_{i:02d}.npy')
        np.save(out_path, sample[i].astype(np.float32))
        print(f"Saved: {out_path}")
    meta = {
        'stage2_dir': args.stage2_dir, 'adapter_path': args.adapter_path,
        'audio_path': args.audio_path, 'text_prompt': args.text_prompt,
        'n_frames': n_frames, 'num_samples': args.num_samples, 'seed': args.seed,
    }
    with open(os.path.join(args.output_dir, f'meta_{mode}.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Done. Samples in {args.output_dir}")


if __name__ == '__main__':
    main()
