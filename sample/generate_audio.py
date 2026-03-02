####################################################################################[start]
####################################################################################[start]

"""

Generate dance motion conditioned on audio (and optionally text).

Supports three modes:

  1. Audio + text:  conditioned on both music and text prompt
  2. Audio only:    conditioned on music, generic text
  3. Text only:     no audio, text prompt only (tests pretrained ability)

Usage:

    # Audio-conditioned generation
    python -m sample.generate_audio \
        --model_path ./save/audio_stage2/model_final.pt \
        --audio_path ./dataset/aist/audio/mBR0.wav \
        --text_prompt "a person performs breakdancing moves to music" \
        --output_dir ./save/audio_stage2/samples \
        --num_samples 3

    # Text-only generation (verify pretrained ability)
    
    python -m sample.generate_audio \
        --model_path ./save/audio_stage2/model_final.pt \
        --text_prompt "a person walks forward" \
        --motion_length 6.0 \
        --output_dir ./save/audio_stage2/samples_text \
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

def parse_args():

    parser = argparse.ArgumentParser(description='Generate motion with audio conditioning')

    # -- paths --

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to audio-conditioned checkpoint (.pt)')
    parser.add_argument('--audio_path', type=str, default='',
                        help='Path to .wav file (leave empty for text-only)')
    parser.add_argument('--output_dir', type=str, default='./save/audio_stage2/samples',
                        help='Directory to save generated motions')

    # -- generation params --

    parser.add_argument('--text_prompt', type=str, default='a person dances to music',
                        help='Text prompt for generation')
    parser.add_argument('--motion_length', type=float, default=0.0,
                        help='Motion length in seconds (0 = infer from audio)')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=42)

    # -- guidance --

    parser.add_argument('--guidance_param', type=float, default=2.5,
                        help='CFG scale for text')
    parser.add_argument('--audio_guidance_param', type=float, default=2.5,
                        help='CFG scale for audio')

    # -- model (overridden by checkpoint args.json if available) --

    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--fps', type=int, default=20)

    return parser.parse_args()

class AudioCFGSampleModel(torch.nn.Module):

    """

    Classifier-free guidance wrapper for audio-conditioned MDM.

    Computes: output = uncond + s_text * (text_cond - uncond) + s_audio * (full_cond - text_cond)

    This decomposes the guidance into text guidance and audio guidance, allowing independent control of each modality.

    """

    def __init__(self, model, text_scale=2.5, audio_scale=2.5):

        super().__init__()

        self.model = model
        self.text_scale = text_scale
        self.audio_scale = audio_scale

        # -- pointers for diffusion compatibility --

        self.rot2xyz = self.model.rot2xyz
        self.translation = self.model.translation
        self.njoints = self.model.njoints
        self.nfeats = self.model.nfeats
        self.data_rep = self.model.data_rep
        self.cond_mode = self.model.cond_mode
        self.encode_text = self.model.encode_text

    def forward(self, x, timesteps, y=None):

        has_audio = (self.model.audio_conditioning and
                     y is not None and
                     'audio_features' in y and
                     y['audio_features'] is not None)

        if has_audio:

            # -- 1. fully conditioned (text + audio) --

            out_full = self.model(x, timesteps, y)

            # -- 2. text only (audio masked) --

            y_no_audio = deepcopy(y)
            y_no_audio['uncond_audio'] = True

            out_text = self.model(x, timesteps, y_no_audio)

            # -- 3. fully unconditional --

            y_uncond = deepcopy(y)
            y_uncond['uncond'] = True
            y_uncond['uncond_audio'] = True

            out_uncond = self.model(x, timesteps, y_uncond)

            # -- compose: uncond + text_scale*(text-uncond) + audio_scale*(full-text) --

            return (out_uncond
                    + self.text_scale * (out_text - out_uncond)
                    + self.audio_scale * (out_full - out_text))

        else:

            # -- text-only CFG (same as original MDM) --

            y_uncond = deepcopy(y)
            y_uncond['uncond'] = True

            out = self.model(x, timesteps, y)
            out_uncond = self.model(x, timesteps, y_uncond)

            return out_uncond + self.text_scale * (out - out_uncond)

def load_model(model_path, device):

    """Load the audio-conditioned MDM model from checkpoint."""

    # -- load checkpoint --

    ckpt = torch.load(model_path, map_location='cpu')

    if 'args' in ckpt:
        ckpt_args = ckpt['args']
    
    else:
        
        # -- try args.json in same directory --
        
        args_path = os.path.join(os.path.dirname(model_path), 'args.json')
        
        with open(args_path, 'r') as f:
            ckpt_args = json.load(f)

    # -- build model --

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
        latent_dim=ckpt_args.get('latent_dim', 512),
        ff_size=ckpt_args.get('ff_size', 1024),
        num_layers=ckpt_args.get('num_layers', 8),
        num_heads=ckpt_args.get('num_heads', 4),
        dropout=ckpt_args.get('dropout', 0.1),
        activation=ckpt_args.get('activation', 'gelu'),
        data_rep='hml_vec',
        dataset='humanml',
        clip_dim=512,
        arch='trans_enc',
        clip_version=ckpt_args.get('clip_version', 'ViT-B/32'),
        cond_mode='text',
        cond_mask_prob=ckpt_args.get('text_cond_mask_prob', 0.1),
        audio_conditioning=True,
        audio_feat_dim=ckpt_args.get('audio_feat_dim', 145),
        audio_cond_mask_prob=ckpt_args.get('audio_cond_mask_prob', 0.15),
    )

    # -- load weights --

    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    print(f"Model loaded from {model_path}")

    return model, ckpt_args

def main():

    args = parse_args()

    # -- seed --

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -- device --

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # -- load model --

    model, ckpt_args = load_model(args.model_path, device)

    # -- create diffusion --

    from utils.model_util import create_gaussian_diffusion

    diff_args = SimpleNamespace(
        diffusion_steps=1000,
        noise_schedule='cosine',
        sigma_small=True,
        lambda_vel=0.0,
        lambda_rcxyz=0.0,
        lambda_fc=0.0,
        lambda_target_loc=0.0,
    )

    diffusion = create_gaussian_diffusion(diff_args)

    # -- wrap model with CFG --

    cfg_model = AudioCFGSampleModel(
        model,
        text_scale=args.guidance_param,
        audio_scale=args.audio_guidance_param,
    )

    # -- load normalization stats --

    humanml_dir = ckpt_args.get('humanml_dir', './dataset/HumanML3D')
    mean = np.load(os.path.join(humanml_dir, 'Mean.npy'))
    std = np.load(os.path.join(humanml_dir, 'Std.npy'))

    # -- prepare audio features (if provided) --

    audio_features = None
    n_frames = int(args.motion_length * args.fps) if args.motion_length > 0 else 196

    if args.audio_path and os.path.exists(args.audio_path):

        from model.audio_features import extract_audio_features

        duration = args.motion_length if args.motion_length > 0 else None
        audio_feat = extract_audio_features(
            args.audio_path,
            target_fps=args.fps,
            duration=duration
        )

        # -- if no explicit length, use audio length --

        if args.motion_length <= 0:
            n_frames = min(audio_feat.shape[0], 196) # cap at max_motion_length

        audio_feat = audio_feat[:n_frames] # (T, 145)
        audio_features = torch.from_numpy(audio_feat).float().unsqueeze(0) # (1, T, 145)
        audio_features = audio_features.repeat(args.num_samples, 1, 1).to(device)

        print(f"Audio: {args.audio_path} → {audio_feat.shape[0]} frames ({audio_feat.shape[0]/args.fps:.1f}s)")

    elif args.audio_path:
        print(f"Warning: audio file not found: {args.audio_path}")

    if args.motion_length > 0:
        n_frames = int(args.motion_length * args.fps)

    print(f"Generating {args.num_samples} samples, {n_frames} frames ({n_frames/args.fps:.1f}s)")
    print(f"Text: '{args.text_prompt}'")
    print(f"Audio: {'yes' if audio_features is not None else 'no'}")
    print(f"CFG: text={args.guidance_param}, audio={args.audio_guidance_param}")

    # -- build conditioning dict --

    model_kwargs = {
        'y': {
            'text': [args.text_prompt] * args.num_samples,
            'mask': torch.ones(args.num_samples, 1, 1, n_frames, dtype=torch.bool).to(device),
            'lengths': torch.tensor([n_frames] * args.num_samples).to(device),
            'scale': torch.tensor([args.guidance_param] * args.num_samples).to(device),
        }
    }

    if audio_features is not None:
        model_kwargs['y']['audio_features'] = audio_features

    # -- sample --

    print("\nSampling...")

    sample_shape = (args.num_samples, 263, 1, n_frames)

    with torch.no_grad():

        sample = diffusion.p_sample_loop(
            cfg_model,
            sample_shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

    # -- denormalize --

    # sample: (B, 263, 1, T)
    
    sample = sample.squeeze(2).permute(0, 2, 1).cpu().numpy() # (B, T, 263)
    sample = sample * std + mean # denormalize
    
    from scipy.ndimage import gaussian_filter1d

    # -- smooth along time axis, sigma=1 frame = 50ms at 20fps --
    
    for i in range(sample.shape[0]):
        sample[i] = gaussian_filter1d(sample[i], sigma=1.0, axis=0)

    # -- save --

    os.makedirs(args.output_dir, exist_ok=True)

    mode = 'audio' if audio_features is not None else 'text'

    for i in range(args.num_samples):

        out_path = os.path.join(args.output_dir, f'sample_{mode}_{i:02d}.npy')
        np.save(out_path, sample[i].astype(np.float32))
        print(f"Saved: {out_path}  shape={sample[i].shape}")

    # -- also save metadata --

    meta = {
        'model_path': args.model_path,
        'audio_path': args.audio_path,
        'text_prompt': args.text_prompt,
        'motion_length': n_frames / args.fps,
        'n_frames': n_frames,
        'guidance_param': args.guidance_param,
        'audio_guidance_param': args.audio_guidance_param,
        'seed': args.seed,
        'num_samples': args.num_samples,
    }

    meta_path = os.path.join(args.output_dir, f'meta_{mode}.json')

    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! {args.num_samples} samples saved to {args.output_dir}")

    # -- quick stats --

    print(f"\nSample stats:")

    for i in range(args.num_samples):

        s = sample[i]
        print(f"  [{i}] range=[{s.min():.2f}, {s.max():.2f}], "
              f"std={s.std():.4f}, "
              f"nan={np.isnan(s).any()}")

if __name__ == '__main__':
    main()

####################################################################################[end]
####################################################################################[end]