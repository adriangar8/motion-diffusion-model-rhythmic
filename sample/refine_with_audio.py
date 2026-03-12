####################################################################################[start]
####################################################################################[start]


"""

Audio-conditioned motion refinement via SDEdit.

Instead of generating rhythmic motion from scratch (which requires the model to
compose text and audio distributions it has never seen together), this script:

  1. Generates clean text-only motion: "walk forward and wave" → smooth walking
  2. Partially noises this motion to timestep T_start
  3. Denoises from T_start → 0 with audio conditioning active

The text defines WHAT happens; the audio refinement adjusts WHEN emphasis occurs.

This is more principled because:
  - The text-only motion is already a valid trajectory (walking, waving, etc.)
  - The partial noising preserves gross structure (pose, trajectory, action type)
  - The audio-conditioned denoising only modifies timing/dynamics
  - No need for the model to generate rhythmic walking from scratch

The noise_level parameter controls the trade-off:
  - Low noise (skip=800): subtle rhythmic nudges, text fully preserved
  - Medium noise (skip=500): clear rhythmic modulation, text mostly preserved
  - High noise (skip=200): strong audio influence, text structure may degrade

Usage:

    # Basic refinement
    python -m sample.refine_with_audio \
        --model_path ./save/audio_stage2/model_final.pt \
        --audio_path ./dataset/aist/audio/mBR0.wav \
        --text_prompt "a person walks forward and waves" \
        --output_dir ./save/audio_stage2/samples_refined \
        --skip_timesteps 500 \
        --num_samples 3

    # Noise level sweep (key experiment)
    python -m sample.refine_with_audio \
        --model_path ./save/audio_stage2/model_final.pt \
        --audio_path ./dataset/aist/audio/mBR0.wav \
        --text_prompt "a person walks forward and waves" \
        --output_dir ./save/audio_stage2/samples_sweep \
        --sweep_skip 200 300 400 500 600 700 800 \
        --num_samples 3

    # Compare: text-only vs refined vs full audio generation
    python -m sample.refine_with_audio \
        --model_path ./save/audio_stage2/model_final.pt \
        --audio_path ./dataset/aist/audio/mBR0.wav \
        --text_prompt "a person walks forward and waves" \
        --output_dir ./save/audio_stage2/samples_compare \
        --skip_timesteps 500 \
        --save_text_only \
        --save_full_audio \
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

sys.path.insert(0, '.')

def parse_args():

    parser = argparse.ArgumentParser(description='Refine text motion with audio')

    # -- paths --
    
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./save/audio_stage2/samples_refined')
    parser.add_argument('--humanml_dir', type=str, default='')

    # -- generation --
    
    parser.add_argument('--text_prompt', type=str, default='a person walks forward and waves')
    parser.add_argument('--motion_length', type=float, default=0.0,
                        help='Seconds (0 = infer from audio)')
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fps', type=int, default=20)

    # -- refinement --
    
    parser.add_argument('--skip_timesteps', type=int, default=500,
                        help='Skip this many timesteps (start denoising from T-skip). '
                             '500 = start from 50%% noise. Higher = less modification.')
    parser.add_argument('--sweep_skip', type=int, nargs='+', default=None,
                        help='Sweep over multiple skip values (overrides --skip_timesteps)')

    # -- guidance --
    
    parser.add_argument('--text_guidance', type=float, default=2.5)
    parser.add_argument('--audio_guidance', type=float, default=2.5)
    parser.add_argument('--text_only_guidance', type=float, default=2.5,
                        help='CFG scale for the initial text-only generation')

    # -- comparison flags --
    
    parser.add_argument('--save_text_only', action='store_true',
                        help='Also save the text-only motion (before refinement)')
    parser.add_argument('--save_full_audio', action='store_true',
                        help='Also save full audio generation (no refinement, for comparison)')

    # -- smoothing --
    
    parser.add_argument('--smooth_sigma', type=float, default=1.0,
                        help='Gaussian smoothing sigma (0 = no smoothing)')

    # -- device --
    
    parser.add_argument('--device', type=str, default='')

    return parser.parse_args()

class TextOnlyCFGModel(torch.nn.Module):

    """Standard text-only CFG wrapper (no audio)."""

    def __init__(self, model, text_scale=2.5):

        super().__init__()

        self.model = model
        self.text_scale = text_scale
        
        # -- diffusion compatibility --
        
        self.rot2xyz = model.rot2xyz
        self.translation = model.translation
        self.njoints = model.njoints
        self.nfeats = model.nfeats
        self.data_rep = model.data_rep
        self.cond_mode = model.cond_mode
        self.encode_text = model.encode_text

    def forward(self, x, timesteps, y=None):
        
        out = self.model(x, timesteps, y)

        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        
        if self.model.audio_conditioning:
            y_uncond['uncond_audio'] = True
        
        out_uncond = self.model(x, timesteps, y_uncond)

        return out_uncond + self.text_scale * (out - out_uncond)

class AudioCFGModel(torch.nn.Module):

    """Decomposed text+audio CFG wrapper."""

    def __init__(self, model, text_scale=2.5, audio_scale=2.5):

        super().__init__()

        self.model = model
        self.text_scale = text_scale
        self.audio_scale = audio_scale

        # -- diffusion compatibility --

        self.rot2xyz = model.rot2xyz
        self.translation = model.translation
        self.njoints = model.njoints
        self.nfeats = model.nfeats
        self.data_rep = model.data_rep
        self.cond_mode = model.cond_mode
        self.encode_text = model.encode_text

    def forward(self, x, timesteps, y=None):
        
        # -- 1. full conditioning (text + audio) --
        
        out_full = self.model(x, timesteps, y)

        # -- 2. Text only (audio masked) --
        
        y_no_audio = deepcopy(y)
        y_no_audio['uncond_audio'] = True
        
        out_text = self.model(x, timesteps, y_no_audio)

        # -- 3. fully unconditional --
        
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        y_uncond['uncond_audio'] = True
        
        out_uncond = self.model(x, timesteps, y_uncond)

        return (out_uncond
                + self.text_scale * (out_text - out_uncond)
                + self.audio_scale * (out_full - out_text))


def load_model(model_path, device):
    
    """Load audio-conditioned MDM from checkpoint."""

    ckpt = torch.load(model_path, map_location='cpu')

    if 'args' in ckpt:
        ckpt_args = ckpt['args']
    else:
        args_path = os.path.join(os.path.dirname(model_path), 'args.json')
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
        use_audio_token_concat=ckpt_args.get('use_audio_token_concat', False),
        temporal_sigma=ckpt_args.get('temporal_sigma', 4.0),
        beat_weight=ckpt_args.get('beat_weight', 2.0),
    )

    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    print(f"Model loaded from {model_path}")
    
    return model, ckpt_args

def generate_text_only(diffusion, model, text_prompt, n_frames, num_samples, device,
                       text_guidance=2.5, seed=42):

    """Step 1: Generate clean text-only motion."""

    torch.manual_seed(seed)

    cfg_model = TextOnlyCFGModel(model, text_scale=text_guidance)

    model_kwargs = {
        'y': {
            'text': [text_prompt] * num_samples,
            'mask': torch.ones(num_samples, 1, 1, n_frames, dtype=torch.bool).to(device),
            'lengths': torch.tensor([n_frames] * num_samples).to(device),
        }
    }

    # -- remove audio from conditioning for text-only pass (audio_features absent = no audio conditioning) --

    sample_shape = (num_samples, 263, 1, n_frames)

    print("  Step 1: Generating text-only motion...")
    
    with torch.no_grad():
        
        text_motion = diffusion.p_sample_loop(
            cfg_model,
            sample_shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
        )

    return text_motion  # (B, 263, 1, T): normalized, in model space

def refine_with_audio(diffusion, model, text_motion, audio_features,
                      text_prompt, n_frames, num_samples, device,
                      skip_timesteps=500, text_guidance=2.5, audio_guidance=2.5,
                      seed=42, beat_frames=None):
    
    """Step 2: Refine text motion with audio via SDEdit."""

    torch.manual_seed(seed + 1) # different seed for refinement noise

    cfg_model = AudioCFGModel(
        model,
        text_scale=text_guidance,
        audio_scale=audio_guidance,
    )

    model_kwargs = {
        'y': {
            'text': [text_prompt] * num_samples,
            'audio_features': audio_features,
            'mask': torch.ones(num_samples, 1, 1, n_frames, dtype=torch.bool).to(device),
            'lengths': torch.tensor([n_frames] * num_samples).to(device),
        }
    }

    if beat_frames is not None:
        model_kwargs['y']['beat_frames'] = beat_frames

    sample_shape = (num_samples, 263, 1, n_frames)

    noise_pct = 100 * (1000 - skip_timesteps) / 1000
    
    print(f"  Step 2: Refining with audio (skip={skip_timesteps}, "
          f"noise={noise_pct:.0f}%, denoise {1000-skip_timesteps} steps)...")

    with torch.no_grad():
        refined = diffusion.p_sample_loop(
            cfg_model,
            sample_shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=skip_timesteps,
            init_image=text_motion, # start from text-only motion
            progress=True,
        )

    return refined # (B, 263, 1, T)

def generate_full_audio(diffusion, model, audio_features, text_prompt,
                        n_frames, num_samples, device,
                        text_guidance=2.5, audio_guidance=2.5, seed=42,
                        beat_frames=None):
    
    """Full audio generation from scratch (for comparison)."""

    torch.manual_seed(seed)

    cfg_model = AudioCFGModel(
        model,
        text_scale=text_guidance,
        audio_scale=audio_guidance,
    )

    model_kwargs = {
        'y': {
            'text': [text_prompt] * num_samples,
            'audio_features': audio_features,
            'mask': torch.ones(num_samples, 1, 1, n_frames, dtype=torch.bool).to(device),
            'lengths': torch.tensor([n_frames] * num_samples).to(device),
        }
    }

    if beat_frames is not None:
        model_kwargs['y']['beat_frames'] = beat_frames

    sample_shape = (num_samples, 263, 1, n_frames)

    print("  Generating full audio-conditioned motion (from scratch)...")
    
    with torch.no_grad():
    
        sample = diffusion.p_sample_loop(
            cfg_model,
            sample_shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
        )

    return sample

def denormalize_and_smooth(sample, mean, std, sigma=1.0):
    
    """Convert from model space to joint space, optionally smooth."""

    # -- sample: (B, 263, 1, T) to (B, T, 263) --
    
    out = sample.squeeze(2).permute(0, 2, 1).cpu().numpy()
    out = out * std + mean

    if sigma > 0:
        
        from scipy.ndimage import gaussian_filter1d
        
        for i in range(out.shape[0]):
            out[i] = gaussian_filter1d(out[i], sigma=sigma, axis=0)

    return out

def main():
    
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # -- load model --
    
    model, ckpt_args = load_model(args.model_path, device)
    audio_feat_dim = ckpt_args.get('audio_feat_dim', 145)

    # -- diffusion --
    
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

    # -- normalization stats --
    
    # humanml_dir = args.humanml_dir or ckpt_args.get('humanml_dir', './dataset/HumanML3D')
    humanml_dir = "/Data/adrian.garcia/motion-diffusion-model-rhythmic/dataset/HumanML3D"
    mean = np.load(os.path.join(humanml_dir, 'Mean.npy'))
    std = np.load(os.path.join(humanml_dir, 'Std.npy'))

    # -- load audio --
    
    use_wav2clip = ckpt_args.get('use_wav2clip', False) or audio_feat_dim == 519
    duration = args.motion_length if args.motion_length > 0 else None

    if use_wav2clip:
        from model.audio_features_wav2clip import extract_wav2clip_plus_librosa
        audio_feat = extract_wav2clip_plus_librosa(
            args.audio_path,
            target_fps=args.fps,
            duration=duration,
            device=device,
        )
    elif audio_feat_dim == 52:
        from model.audio_features_v2 import extract_audio_features_v2 as extract_fn
        audio_feat = extract_fn(args.audio_path, target_fps=args.fps, duration=duration)
    else:
        from model.audio_features import extract_audio_features as extract_fn
        audio_feat = extract_fn(args.audio_path, target_fps=args.fps, duration=duration)

    if args.motion_length <= 0:
        n_frames = min(audio_feat.shape[0], 196)
    else:
        n_frames = int(args.motion_length * args.fps)

    audio_feat = audio_feat[:n_frames]
    audio_features = torch.from_numpy(audio_feat).float().unsqueeze(0)
    audio_features = audio_features.repeat(args.num_samples, 1, 1).to(device)

    # -- extract beat frames for beat-aware masking --
    
    beat_frames = None
    
    if audio_feat_dim == 519:
        # 7d librosa: index 1 is beat_indicator (after 512-d Wav2CLIP)
        beat_signal = audio_feat[:, 512 + 1]
        beat_frames = list(np.where(beat_signal > 0.5)[0])
    elif audio_feat_dim == 52:
        beat_signal = audio_feat[:, 34]  # beat_soft channel
        beat_frames = list(np.where(beat_signal > 0.3)[0])
    elif audio_feat_dim == 145:
        beat_signal = audio_feat[:, 129]  # beat_indicator channel
        beat_frames = list(np.where(beat_signal > 0.5)[0])

    print(f"\nAudio: {args.audio_path}")
    print(f"  {audio_feat.shape[0]} frames ({audio_feat.shape[0]/args.fps:.1f}s), "
          f"{len(beat_frames) if beat_frames else 0} beats detected")
    print(f"Text: '{args.text_prompt}'")
    print(f"Samples: {args.num_samples}, Frames: {n_frames}")
    print(f"Guidance: text={args.text_guidance}, audio={args.audio_guidance}")

    os.makedirs(args.output_dir, exist_ok=True)

    # -- determine skip values --
    
    skip_values = args.sweep_skip if args.sweep_skip else [args.skip_timesteps]

    # -- step 1: Generate text-only motion (shared across all skip values) --

    print(f"\n{'='*60}")
    print("PHASE 1: Text-only generation")
    print(f"{'='*60}")

    text_motion = generate_text_only(
        diffusion, model,
        text_prompt=args.text_prompt,
        n_frames=n_frames,
        num_samples=args.num_samples,
        device=device,
        text_guidance=args.text_only_guidance,
        seed=args.seed,
    )

    if args.save_text_only:
        text_np = denormalize_and_smooth(text_motion, mean, std, args.smooth_sigma)
        for i in range(args.num_samples):
            path = os.path.join(args.output_dir, f'sample_text_only_{i:02d}.npy')
            np.save(path, text_np[i].astype(np.float32))
            print(f"  Saved text-only: {path}")

    # -- step 2: Refine with audio at each skip level --

    results = {}

    for skip in skip_values:
        print(f"\n{'='*60}")
        print(f"PHASE 2: Audio refinement (skip_timesteps={skip})")
        print(f"{'='*60}")

        refined = refine_with_audio(
            diffusion, model, text_motion,
            audio_features=audio_features,
            text_prompt=args.text_prompt,
            n_frames=n_frames,
            num_samples=args.num_samples,
            device=device,
            skip_timesteps=skip,
            text_guidance=args.text_guidance,
            audio_guidance=args.audio_guidance,
            seed=args.seed,
            beat_frames=beat_frames,
        )

        refined_np = denormalize_and_smooth(refined, mean, std, args.smooth_sigma)

        for i in range(args.num_samples):
            path = os.path.join(args.output_dir, f'sample_refined_skip{skip}_{i:02d}.npy')
            np.save(path, refined_np[i].astype(np.float32))

        # -- compute stats --
        
        text_np = denormalize_and_smooth(text_motion, mean, std, args.smooth_sigma)
        residual = refined_np - text_np
        residual_rms = np.sqrt(np.mean(residual ** 2))

        # -- root height variation (index 3 in HumanML3D representation) --
        
        root_height_std = np.mean([refined_np[i, :, 3].std() for i in range(args.num_samples)])

        results[skip] = {
            'residual_rms': float(residual_rms),
            'root_height_std': float(root_height_std),
        }

        print(f"  Residual RMS: {residual_rms:.4f}")
        print(f"  Root height std: {root_height_std:.4f}")
        print(f"  Saved {args.num_samples} samples to {args.output_dir}")

    # -- optional: Full audio generation for comparison --

    if args.save_full_audio:
        
        print(f"\n{'='*60}")
        print("COMPARISON: Full audio generation (from scratch)")
        print(f"{'='*60}")

        full_audio = generate_full_audio(
            diffusion, model,
            audio_features=audio_features,
            text_prompt=args.text_prompt,
            n_frames=n_frames,
            num_samples=args.num_samples,
            device=device,
            text_guidance=args.text_guidance,
            audio_guidance=args.audio_guidance,
            seed=args.seed,
            beat_frames=beat_frames,
        )

        full_np = denormalize_and_smooth(full_audio, mean, std, args.smooth_sigma)
        for i in range(args.num_samples):
            path = os.path.join(args.output_dir, f'sample_full_audio_{i:02d}.npy')
            np.save(path, full_np[i].astype(np.float32))
            print(f"  Saved full audio: {path}")

    # -- save metadata --

    meta = {
        'model_path': args.model_path,
        'audio_path': args.audio_path,
        'text_prompt': args.text_prompt,
        'n_frames': n_frames,
        'motion_length_sec': n_frames / args.fps,
        'num_samples': args.num_samples,
        'seed': args.seed,
        'text_guidance': args.text_guidance,
        'audio_guidance': args.audio_guidance,
        'text_only_guidance': args.text_only_guidance,
        'skip_values': skip_values,
        'results': results,
        'n_beats': len(beat_frames) if beat_frames else 0,
        'audio_feat_dim': audio_feat_dim,
        'smooth_sigma': args.smooth_sigma,
    }

    meta_path = os.path.join(args.output_dir, 'meta_refinement.json')
    
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Text prompt: '{args.text_prompt}'")
    print(f"Audio: {os.path.basename(args.audio_path)} ({n_frames/args.fps:.1f}s, "
          f"{len(beat_frames) if beat_frames else 0} beats)")

    if len(skip_values) > 1:
        
        print(f"\nSkip sweep results:")
        print(f"  {'Skip':>6}  {'Noise%':>6}  {'Residual RMS':>14}  {'Root H. Std':>12}")
        
        for skip in skip_values:
            r = results[skip]
            noise_pct = 100 * (1000 - skip) / 1000
            print(f"  {skip:>6}  {noise_pct:>5.0f}%  {r['residual_rms']:>14.4f}  "
                  f"{r['root_height_std']:>12.4f}")

    print(f"\nAll outputs saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
    
####################################################################################[start]
####################################################################################[start]
