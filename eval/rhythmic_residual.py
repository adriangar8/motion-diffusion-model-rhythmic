####################################################################################[start]
####################################################################################[start]

"""

Rhythmic Residual Beat Alignment (RRBA) metric.

Measures how much of the motion ADDED by audio conditioning is rhythmically
aligned with the music beats. This isolates the audio contribution from
the base text-conditioned motion.

Concept:
    residual(t) = motion_audio(t) - motion_text(t)

    The residual represents what audio conditioning added to the base motion.
    We compute kinetic energy of this residual and check if its peaks
    align with musical beats.

    RRBA = fraction of beats where residual kinetic energy has a local peak

This metric is complementary to BAS:
    - BAS measures: does the FULL motion hit beats?
    - RRBA measures: does the AUDIO-ADDED motion hit beats?

RRBA is particularly useful when comparing different audio guidance scales:
    - Too low guidance → RRBA ≈ random (audio adds noise, not rhythm)
    - Good guidance → RRBA > BAS baseline (audio specifically adds beat-aligned motion)
    - Too high guidance → RRBA may stay high but motion quality degrades

Usage:
    # Quick evaluation on existing paired samples
    python -m eval.rhythmic_residual \
        --audio_dir ./save/audio_stage2/samples_fair_audio \
        --text_dir ./save/audio_stage2/samples_fair_noaudio \
        --audio_path ./dataset/aist/audio/mBR0.wav \
        --humanml_dir ./dataset/HumanML3D

    # Full sweep over audio guidance scales
    python -m eval.rhythmic_residual \
        --model_path ./save/audio_stage2/model_final.pt \
        --audio_path ./dataset/aist/audio/mBR0.wav \
        --text_prompt "a person walks forward and waves" \
        --humanml_dir ./dataset/HumanML3D \
        --sweep_scales 0.0 0.5 1.0 1.5 2.0 2.5 3.0 4.0 \
        --num_samples 5 \
        --seed 42 \
        --output_path ./save/audio_stage2/rrba_sweep.json

"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from glob import glob
from scipy.signal import find_peaks

sys.path.insert(0, '.')

# -- core RRBA metric --

def compute_residual_kinetic_energy(joints_audio, joints_text):
    
    """
    
    Compute kinetic energy of the motion residual (what audio conditioning added).

    Args:
        joints_audio: (T, 22, 3) — audio+text conditioned motion
        joints_text:  (T, 22, 3) — text-only motion (same seed)

    Returns:
        residual_ke: (T-1,) — kinetic energy of the residual per frame
        residual: (T, 22, 3) — the raw residual motion
    
    """
    
    T = min(joints_audio.shape[0], joints_text.shape[0])
    residual = joints_audio[:T] - joints_text[:T] # (T, 22, 3)

    # -- velocity of the residual --
    
    vel = np.diff(residual, axis=0)  # (T-1, 22, 3)
    ke = 0.5 * np.sum(vel ** 2, axis=(1, 2))  # (T-1,)

    return ke, residual

def extract_music_beats(wav_path, fps=20):
    
    """Extract beat frame indices from audio."""
    
    import librosa
    
    y, sr = librosa.load(wav_path, sr=None)
    
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beat_indices = (beat_times * fps).astype(int)
    
    return beat_indices, beat_times

def rhythmic_residual_beat_alignment(joints_audio, joints_text, beat_indices,
                                      fps=20, window=2):
    
    """
    
    Rhythmic Residual Beat Alignment (RRBA).

    Measures what fraction of music beats coincide with peaks in the
    residual kinetic energy (the motion added by audio conditioning).

    Args:
        joints_audio: (T, 22, 3) — audio+text conditioned joints
        joints_text:  (T, 22, 3) — text-only joints (same prompt, same seed)
        beat_indices: array of beat frame indices
        fps: frame rate
        window: tolerance in frames (±window)

    Returns:
        rrba: float in [0, 1] — fraction of beats aligned with residual peaks
        details: dict with additional analysis
    
    """
    
    T = min(joints_audio.shape[0], joints_text.shape[0])
    residual_ke, residual = compute_residual_kinetic_energy(joints_audio, joints_text)

    if len(residual_ke) == 0 or len(beat_indices) == 0:
        return 0.0, {}

    # -- find peaks in residual kinetic energy --
    
    peaks, peak_props = find_peaks(residual_ke, distance=int(fps * 0.15))
    peak_set = set(peaks)

    # -- count aligned beats --
    
    aligned = 0
    valid_beats = 0

    for b in beat_indices:
        
        if b >= T - 1:
            continue
        
        valid_beats += 1
        
        for offset in range(-window, window + 1):
            if (b + offset) in peak_set:
                aligned += 1
                break

    rrba = aligned / valid_beats if valid_beats > 0 else 0.0

    # -- also compute BAS on full audio motion for reference --
    
    full_ke = 0.5 * np.sum(np.diff(joints_audio[:T], axis=0) ** 2, axis=(1, 2))
    full_peaks, _ = find_peaks(full_ke, distance=int(fps * 0.15))
    full_peak_set = set(full_peaks)

    full_aligned = 0
    
    for b in beat_indices:
    
        if b >= T - 1:
            continue
    
        for offset in range(-window, window + 1):
    
            if (b + offset) in full_peak_set:
                full_aligned += 1
                break

    bas = full_aligned / valid_beats if valid_beats > 0 else 0.0

    # -- residual magnitude statistics --
    
    residual_rms = np.sqrt(np.mean(residual ** 2))
    residual_max = np.max(np.abs(residual))

    details = {
        'rrba': rrba,
        'bas': bas,
        'n_beats': valid_beats,
        'n_residual_peaks': len(peaks),
        'n_aligned': aligned,
        'residual_rms': float(residual_rms),
        'residual_max': float(residual_max),
        'residual_ke_mean': float(residual_ke.mean()),
        'residual_ke_std': float(residual_ke.std()),
    }

    return rrba, details

# -- evaluation on pre-generated samples --

def recover_joints(motion_263, mean, std, joints_num=22):
    
    """Convert (T, 263) normalized features → (T, 22, 3) joints."""
    
    from data_loaders.humanml.scripts.motion_process import recover_from_ric
    
    motion = motion_263 * std + mean
    motion_t = torch.from_numpy(motion).float().unsqueeze(0)
    
    joints = recover_from_ric(motion_t, joints_num)
    
    return joints.squeeze(0).numpy()

def evaluate_paired_samples(audio_dir, text_dir, audio_path, mean, std, fps=20):

    """Evaluate RRBA on paired (audio, text) sample directories."""

    audio_files = sorted(glob(os.path.join(audio_dir, '*.npy')))
    text_files = sorted(glob(os.path.join(text_dir, '*.npy')))

    n_pairs = min(len(audio_files), len(text_files))

    if n_pairs == 0:
        print("No paired samples found!")
        return {}

    beat_indices, _ = extract_music_beats(audio_path, fps=fps)

    results = []

    for i in range(n_pairs):

        motion_a = np.load(audio_files[i])
        motion_t = np.load(text_files[i])

        joints_a = recover_joints(motion_a, mean, std)
        joints_t = recover_joints(motion_t, mean, std)

        rrba, details = rhythmic_residual_beat_alignment(
            joints_a, joints_t, beat_indices, fps=fps
        )
        results.append(details)

        print(f"  Pair {i}: RRBA={rrba:.4f}, BAS={details['bas']:.4f}, "
              f"residual_rms={details['residual_rms']:.6f}")

    # -- aggregate --
    
    agg = {
        'rrba_mean': np.mean([r['rrba'] for r in results]),
        'rrba_std': np.std([r['rrba'] for r in results]),
        'bas_mean': np.mean([r['bas'] for r in results]),
        'bas_std': np.std([r['bas'] for r in results]),
        'residual_rms_mean': np.mean([r['residual_rms'] for r in results]),
        'n_pairs': n_pairs,
        'per_sample': results,
    }

    return agg

# -- guidance scale sweep --

def generate_paired_sample(model_path, audio_path, text_prompt, audio_scale,
                            humanml_dir, seed=42, device='cuda', fps=20):
    
    """
    
    Generate a paired (audio+text, text-only) sample at given audio guidance scale.
    Uses the same seed for both so the only difference is audio conditioning.
    
    """
    
    from utils.fixseed import fixseed
    from sample.generate_audio import load_model, AudioCFGSampleModel
    from utils.model_util import create_gaussian_diffusion
    from model.audio_features import extract_audio_features
    from types import SimpleNamespace

    # -- load model --
    
    model, _ = load_model(model_path, device)

    diff_args = SimpleNamespace(
        diffusion_steps=1000, noise_schedule='cosine', sigma_small=True,
        lambda_vel=0.0, lambda_rcxyz=0.0, lambda_fc=0.0, lambda_target_loc=0.0,
    )
    diffusion = create_gaussian_diffusion(diff_args)

    mean = np.load(os.path.join(humanml_dir, 'Mean.npy'))
    std = np.load(os.path.join(humanml_dir, 'Std.npy'))

    # -- audio features --
    
    audio_feat = extract_audio_features(audio_path, target_fps=fps)
    n_frames = min(audio_feat.shape[0], 196)
    audio_feat = audio_feat[:n_frames]
    audio_tensor = torch.from_numpy(audio_feat).float().unsqueeze(0).to(device)

    # -- generate WITH audio --
    
    fixseed(seed)
    cfg_model = AudioCFGSampleModel(model, text_scale=2.5, audio_scale=audio_scale)

    model_kwargs_audio = {
        'y': {
            'text': [text_prompt],
            'mask': torch.ones(1, 1, 1, n_frames, dtype=torch.bool).to(device),
            'lengths': torch.tensor([n_frames]).to(device),
            'scale': torch.tensor([2.5]).to(device),
            'audio_features': audio_tensor,
        }
    }

    with torch.no_grad():
        sample_audio = diffusion.p_sample_loop(
            cfg_model,
            (1, 263, 1, n_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs_audio,
            progress=False,
        )
    sample_audio = sample_audio.squeeze(2).permute(0, 2, 1).cpu().numpy()[0] # (T, 263)

    # -- generate WITHOUT audio (same seed) --
    
    fixseed(seed)
    cfg_model_text = AudioCFGSampleModel(model, text_scale=2.5, audio_scale=0.0)

    model_kwargs_text = {
        'y': {
            'text': [text_prompt],
            'mask': torch.ones(1, 1, 1, n_frames, dtype=torch.bool).to(device),
            'lengths': torch.tensor([n_frames]).to(device),
            'scale': torch.tensor([2.5]).to(device),
            'audio_features': audio_tensor, # present but guidance=0
        }
    }

    with torch.no_grad():
        
        sample_text = diffusion.p_sample_loop(
            cfg_model_text,
            (1, 263, 1, n_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs_text,
            progress=False,
        )
    
    sample_text = sample_text.squeeze(2).permute(0, 2, 1).cpu().numpy()[0]

    # -- recover joints --
    
    joints_audio = recover_joints(sample_audio, mean, std)
    joints_text = recover_joints(sample_text, mean, std)

    return joints_audio, joints_text

def sweep_guidance_scales(model_path, audio_path, text_prompt, scales,
                           humanml_dir, num_samples=5, seed=42, device='cuda'):
    
    """
    
    Sweep over audio guidance scales and compute RRBA + BAS for each.
    Returns results suitable for plotting.
    
    """
    
    beat_indices, _ = extract_music_beats(audio_path)

    mean = np.load(os.path.join(humanml_dir, 'Mean.npy'))
    std = np.load(os.path.join(humanml_dir, 'Std.npy'))

    all_results = {}

    for scale in scales:
    
        print(f"\n=== Audio guidance scale = {scale} ===")

        scale_rrba = []
        scale_bas = []
        scale_residual_rms = []

        for s in range(num_samples):
            current_seed = seed + s

            joints_audio, joints_text = generate_paired_sample(
                model_path, audio_path, text_prompt,
                audio_scale=scale,
                humanml_dir=humanml_dir,
                seed=current_seed,
                device=device,
            )

            rrba, details = rhythmic_residual_beat_alignment(
                joints_audio, joints_text, beat_indices
            )

            scale_rrba.append(rrba)
            scale_bas.append(details['bas'])
            scale_residual_rms.append(details['residual_rms'])

        result = {
            'audio_scale': scale,
            'rrba_mean': float(np.mean(scale_rrba)),
            'rrba_std': float(np.std(scale_rrba)),
            'bas_mean': float(np.mean(scale_bas)),
            'bas_std': float(np.std(scale_bas)),
            'residual_rms_mean': float(np.mean(scale_residual_rms)),
            'residual_rms_std': float(np.std(scale_residual_rms)),
        }

        all_results[str(scale)] = result
        print(f"  RRBA: {result['rrba_mean']:.4f} ± {result['rrba_std']:.4f}")
        print(f"  BAS:  {result['bas_mean']:.4f} ± {result['bas_std']:.4f}")
        print(f"  Residual RMS: {result['residual_rms_mean']:.6f}")

    return all_results


def plot_sweep_results(results, output_path):
    
    """Create a plot of RRBA and BAS vs audio guidance scale."""
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    scales = []
    rrba_means = []
    rrba_stds = []
    bas_means = []
    bas_stds = []
    rms_means = []

    for key in sorted(results.keys(), key=float):
    
        r = results[key]
        scales.append(r['audio_scale'])
        rrba_means.append(r['rrba_mean'])
        rrba_stds.append(r['rrba_std'])
        bas_means.append(r['bas_mean'])
        bas_stds.append(r['bas_std'])
        rms_means.append(r['residual_rms_mean'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # -- left: RRBA and BAS vs scale --
    
    ax1.errorbar(scales, rrba_means, yerr=rrba_stds, marker='o', capsize=4,
                  label='RRBA (residual alignment)', color='tab:blue')
    ax1.errorbar(scales, bas_means, yerr=bas_stds, marker='s', capsize=4,
                  label='BAS (full motion alignment)', color='tab:orange')
    ax1.set_xlabel('Audio Guidance Scale', fontsize=12)
    ax1.set_ylabel('Beat Alignment Score', fontsize=12)
    ax1.set_title('Rhythmic Alignment vs Audio Guidance', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1)

    # -- right: residual magnitude vs scale --
    
    ax2.plot(scales, rms_means, marker='D', color='tab:green')
    ax2.set_xlabel('Audio Guidance Scale', fontsize=12)
    ax2.set_ylabel('Residual RMS', fontsize=12)
    ax2.set_title('Audio-Added Motion Magnitude', fontsize=13)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sweep plot: {output_path}")

# -- main --

def main():
    
    parser = argparse.ArgumentParser(description='Rhythmic Residual Beat Alignment')

    # -- mode 1: evaluate existing paired samples --
    
    parser.add_argument('--audio_dir', type=str, default='',
                        help='Dir with audio-conditioned .npy samples')
    parser.add_argument('--text_dir', type=str, default='',
                        help='Dir with text-only .npy samples')

    # -- mode 2: generate and sweep --
    
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--text_prompt', type=str, default='a person walks forward and waves')
    parser.add_argument('--sweep_scales', nargs='+', type=float, default=[])

    # -- shared --
    
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--humanml_dir', type=str, default='./dataset/HumanML3D')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_path', type=str, default='./save/audio_stage2/rrba_results.json')

    args = parser.parse_args()

    mean = np.load(os.path.join(args.humanml_dir, 'Mean.npy'))
    std = np.load(os.path.join(args.humanml_dir, 'Std.npy'))

    results = {}

    # -- mode 1: evaluate existing paired samples --
    
    if args.audio_dir and args.text_dir:
    
        print("=== Evaluating paired samples ===")
        results = evaluate_paired_samples(
            args.audio_dir, args.text_dir, args.audio_path, mean, std
        )
        print(f"\nRRBA: {results['rrba_mean']:.4f} ± {results['rrba_std']:.4f}")
        print(f"BAS:  {results['bas_mean']:.4f} ± {results['bas_std']:.4f}")

    # -- mode 2: guidance scale sweep --
    
    if args.sweep_scales and args.model_path:
    
        print("=== Guidance scale sweep ===")
        sweep = sweep_guidance_scales(
            model_path=args.model_path,
            audio_path=args.audio_path,
            text_prompt=args.text_prompt,
            scales=args.sweep_scales,
            humanml_dir=args.humanml_dir,
            num_samples=args.num_samples,
            seed=args.seed,
            device=args.device,
        )
        results['sweep'] = sweep

        plot_path = args.output_path.replace('.json', '_sweep.png')
        plot_sweep_results(sweep, plot_path)

    # -- save results --
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {args.output_path}")

if __name__ == '__main__':
    main()
    
####################################################################################[end]
####################################################################################[end]