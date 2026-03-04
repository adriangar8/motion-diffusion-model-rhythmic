####################################################################################[start]
####################################################################################[start]

"""

Evaluate audio-conditioned dance generation quality.

Metrics:

  1. Beat Alignment Score (BAS) — does the motion hit beats in the music?
  2. Motion Quality (FID-like) — distribution distance to real AIST++ dances
  3. Diversity — variance across generated samples
  4. Physical plausibility — foot sliding, penetration, smoothness

Usage:

    python -m eval.evaluate_audio \
        --model_path ./save/audio_stage2/model_final.pt \
        --aist_dir ./dataset/aist \
        --humanml_dir ./dataset/HumanML3D \
        --num_samples 50 \
        --output_path ./save/audio_stage2/eval_results.json

"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from glob import glob
from types import SimpleNamespace

sys.path.insert(0, '.')

# -- beat alignment score --

def extract_music_beats(wav_path, fps=20):

    """
    
    Extract beat times from music using librosa.
    Returns beat frame indices at the given fps.
    
    """

    import librosa

    y, sr = librosa.load(wav_path, sr=None)
    tempo, beat_frames_audio = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames_audio, sr=sr)

    # -- convert to motion frame indices --

    beat_frame_indices = (beat_times * fps).astype(int)

    return beat_frame_indices, beat_times

def compute_kinetic_energy(joints):

    """

    Compute per-frame kinetic energy from joint positions.

    joints: (T, 22, 3)

    Returns: (T-1,) kinetic energy per frame

    """

    vel = np.diff(joints, axis=0) # (T-1, 22, 3)
    kinetic = 0.5 * np.sum(vel ** 2, axis=(1, 2)) # (T-1,)

    return kinetic

def beat_alignment_score(joints, beat_indices, fps=20, window=2):

    """

    Beat Alignment Score: measures if motion kinetic peaks align with beats.

    For each beat, we check if there's a kinetic energy peak within ±window frames.
    Returns the fraction of beats that are aligned with motion peaks.

    Args:
        joints: (T, 22, 3) joint positions
        beat_indices: array of beat frame indices
        fps: frame rate
        window: tolerance window in frames (±window)

    """

    if len(beat_indices) == 0:
        return 0.0

    T = joints.shape[0]
    kinetic = compute_kinetic_energy(joints) # (T-1,)

    # -- find local peaks in kinetic energy --

    from scipy.signal import find_peaks

    peaks, _ = find_peaks(kinetic, distance=int(fps * 0.2)) # min 0.2s between peaks
    peak_set = set(peaks)

    # -- count aligned beats --

    aligned = 0

    for b in beat_indices:
        
        if b >= T - 1:
            continue

        # -- check if any peak within window --

        for offset in range(-window, window + 1):
        
            if (b + offset) in peak_set:
        
                aligned += 1
                break

    valid_beats = sum(1 for b in beat_indices if b < T - 1)

    if valid_beats == 0:
        return 0.0

    return aligned / valid_beats

# -- motion quality metrics --

def compute_fid_features(joints_list):

    """
    
    Compute simple motion features for FID-like comparison.
    Uses joint velocities and positions as features.

    Args:
        joints_list: list of (T, 22, 3) arrays

    Returns:
        features: (N, feat_dim) array
        
    """

    features = []

    for joints in joints_list:

        T = joints.shape[0]

        # -- velocity statistics --

        vel = np.diff(joints, axis=0) # (T-1, 22, 3)
        vel_flat = vel.reshape(T - 1, -1) # (T-1, 66)

        vel_mean = vel_flat.mean(axis=0) # (66,)
        vel_std = vel_flat.std(axis=0) # (66,)

        # -- position statistics --

        pos_flat = joints.reshape(T, -1) # (T, 66)
        pos_mean = pos_flat.mean(axis=0)
        pos_std = pos_flat.std(axis=0)

        # -- kinetic energy stats --

        ke = compute_kinetic_energy(joints)
        ke_stats = np.array([ke.mean(), ke.std(), ke.max(), np.median(ke)])

        feat = np.concatenate([vel_mean, vel_std, pos_mean, pos_std, ke_stats])
        features.append(feat)

    return np.array(features)


def compute_fid(feat_real, feat_gen):

    """
    
    Compute FID between real and generated feature distributions.
    Uses simple Gaussian assumption.
    
    """

    mu_real = feat_real.mean(axis=0)
    mu_gen = feat_gen.mean(axis=0)

    sigma_real = np.cov(feat_real, rowvar=False) + np.eye(feat_real.shape[1]) * 1e-6
    sigma_gen = np.cov(feat_gen, rowvar=False) + np.eye(feat_gen.shape[1]) * 1e-6

    from scipy.linalg import sqrtm

    diff = mu_real - mu_gen
    covmean = sqrtm(sigma_real @ sigma_gen)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_real + sigma_gen - 2 * covmean)

    return float(fid)


def compute_diversity(joints_list):

    """
    
    Compute diversity as average pairwise distance between generated samples.
    
    """

    if len(joints_list) < 2:
        return 0.0

    features = compute_fid_features(joints_list)
    n = features.shape[0]

    dists = []

    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(features[i] - features[j]))

    return float(np.mean(dists))

# -- physical plausibility --

def compute_foot_sliding(joints, floor_height=0.0, contact_threshold=0.05, fps=20):

    """
    
    Measure foot sliding: when foot is on ground but moving horizontally.

    Args:
        joints: (T, 22, 3)
        floor_height: y coordinate of floor
        contact_threshold: height below which foot is considered grounded

    Returns:
        avg_sliding: average horizontal displacement per ground-contact frame
        
    """

    # -- foot joint indices --

    foot_joints = [7, 10, 8, 11] # left ankle, left toe, right ankle, right toe

    total_slide = 0.0
    contact_frames = 0

    for fj in foot_joints:

        foot_pos = joints[:, fj] # (T, 3)
        foot_height = foot_pos[:, 1] # Y axis

        # -- ground contact frames --

        contact = foot_height < (floor_height + contact_threshold)

        # -- horizontal displacement during contact --

        for t in range(1, len(foot_pos)):
            
            if contact[t] and contact[t - 1]:
                
                dx = foot_pos[t, 0] - foot_pos[t - 1, 0]
                dz = foot_pos[t, 2] - foot_pos[t - 1, 2]
                
                slide = np.sqrt(dx**2 + dz**2)
                total_slide += slide
                contact_frames += 1

    if contact_frames == 0:
        return 0.0

    return total_slide / contact_frames

def compute_smoothness(joints):

    """
    
    Measure motion smoothness via average jerk (3rd derivative of position).
    Lower = smoother.
    
    """

    vel = np.diff(joints, axis=0)
    acc = np.diff(vel, axis=0)
    jerk = np.diff(acc, axis=0)

    avg_jerk = np.mean(np.linalg.norm(jerk.reshape(jerk.shape[0], -1), axis=-1))

    return float(avg_jerk)

# -- full evaluation pipeline --

def recover_joints(motion_263, mean, std, joints_num=22):

    """Convert 263-dim features to (T, 22, 3) joints."""

    from data_loaders.humanml.scripts.motion_process import recover_from_ric

    motion = motion_263 * std + mean
    motion_t = torch.from_numpy(motion).float().unsqueeze(0)
    joints = recover_from_ric(motion_t, joints_num)
    joints = joints.squeeze(0).numpy()

    return joints


def generate_samples(model_path, audio_paths, text_prompts, n_per_audio=5,
                      humanml_dir='./dataset/HumanML3D', device='cuda', fps=20):

    """Generate samples for evaluation."""

    from sample.generate_audio import load_model, AudioCFGSampleModel
    from utils.model_util import create_gaussian_diffusion
    from model.audio_features import extract_audio_features

    model, ckpt_args = load_model(model_path, device)

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
    cfg_model = AudioCFGSampleModel(model, text_scale=2.5, audio_scale=2.5)

    mean = np.load(os.path.join(humanml_dir, 'Mean.npy'))
    std = np.load(os.path.join(humanml_dir, 'Std.npy'))

    generated = []

    for audio_path, text in zip(audio_paths, text_prompts):

        audio_feat = extract_audio_features(audio_path, target_fps=fps)
        n_frames = min(audio_feat.shape[0], 196)
        audio_feat = audio_feat[:n_frames]
        audio_tensor = torch.from_numpy(audio_feat).float().unsqueeze(0)
        audio_tensor = audio_tensor.repeat(n_per_audio, 1, 1).to(device)

        model_kwargs = {
            'y': {
                'text': [text] * n_per_audio,
                'mask': torch.ones(n_per_audio, 1, 1, n_frames, dtype=torch.bool).to(device),
                'lengths': torch.tensor([n_frames] * n_per_audio).to(device),
                'scale': torch.tensor([2.5] * n_per_audio).to(device),
                'audio_features': audio_tensor,
            }
        }

        with torch.no_grad():
            sample = diffusion.p_sample_loop(
                cfg_model,
                (n_per_audio, 263, 1, n_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
            )

        sample = sample.squeeze(2).permute(0, 2, 1).cpu().numpy()

        for i in range(n_per_audio):

            joints = recover_joints(sample[i], mean, std)
            generated.append({
                'joints': joints,
                'motion_263': sample[i],
                'audio_path': audio_path,
                'text': text,
                'n_frames': n_frames,
            })

    return generated


def load_real_samples(aist_dir, humanml_dir, split='test', max_samples=100):

    """Load real AIST++ samples for comparison."""

    mean = np.load(os.path.join(humanml_dir, 'Mean.npy'))
    std = np.load(os.path.join(humanml_dir, 'Std.npy'))

    motion_dir = os.path.join(aist_dir, 'processed', 'motions_263')
    joints_dir = os.path.join(aist_dir, 'processed', 'joints_22')

    split_file = os.path.join(aist_dir, 'splits', f'crossmodal_{split}.txt')

    with open(split_file, 'r') as f:
        names = [l.strip() for l in f if l.strip()]

    real_joints = []

    for name in names[:max_samples]:

        j_path = os.path.join(joints_dir, f'{name}.npy')
        m_path = os.path.join(motion_dir, f'{name}.npy')

        if os.path.exists(j_path):
            
            joints = np.load(j_path)
            
            # -- downsample to match 20fps if needed --
            
            real_joints.append(joints[:196]) # cap length

        elif os.path.exists(m_path):
            
            motion = np.load(m_path)
            joints = recover_joints(motion, mean, std)
            real_joints.append(joints[:196])

    print(f"Loaded {len(real_joints)} real samples from {split} split")

    return real_joints

def main():

    parser = argparse.ArgumentParser(description='Evaluate audio-conditioned generation')

    parser.add_argument('--model_path', type=str, default='./save/audio_stage2/model_final.pt')
    parser.add_argument('--aist_dir', type=str, default='./dataset/aist')
    parser.add_argument('--humanml_dir', type=str, default='./dataset/HumanML3D')
    parser.add_argument('--num_audio_tracks', type=int, default=10,
                        help='Number of audio tracks to evaluate on')
    parser.add_argument('--samples_per_track', type=int, default=5,
                        help='Number of samples per audio track')
    parser.add_argument('--output_path', type=str, default='./save/audio_stage2/eval_results.json')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--skip_generation', action='store_true',
                        help='Skip generation, only evaluate existing samples')
    parser.add_argument('--sample_dir', type=str, default='',
                        help='Directory with pre-generated .npy samples (for --skip_generation)')

    args = parser.parse_args()

    mean = np.load(os.path.join(args.humanml_dir, 'Mean.npy'))
    std = np.load(os.path.join(args.humanml_dir, 'Std.npy'))

    # -- genre text map --

    genre_texts = {
        'BR': 'a person performs breakdancing moves to music',
        'PO': 'a person performs popping dance moves to music',
        'LO': 'a person performs locking dance moves to music',
        'HO': 'a person performs house dance moves to music',
        'WA': 'a person performs waacking dance moves to music',
        'KR': 'a person performs krumping dance moves to music',
        'JS': 'a person performs jazz dance moves to music',
        'JB': 'a person performs ballet jazz dance moves to music',
        'MH': 'a person performs hip hop dance moves to music',
        'LH': 'a person performs hip hop dance moves to music',
    }

    # -- generate or load samples --

    if args.skip_generation and args.sample_dir:

        print("Loading pre-generated samples...")

        sample_files = sorted(glob(os.path.join(args.sample_dir, '*.npy')))
        generated_joints = []

        for f in sample_files:
            
            motion = np.load(f)
            joints = recover_joints(motion, mean, std)
            generated_joints.append(joints)

        print(f"Loaded {len(generated_joints)} samples")

    else:

        print("Generating samples for evaluation...")

        audio_dir = os.path.join(args.aist_dir, 'audio')
        audio_files = sorted(glob(os.path.join(audio_dir, '*.wav')))[:args.num_audio_tracks]

        audio_paths = []
        text_prompts = []

        for af in audio_files:
            
            name = os.path.splitext(os.path.basename(af))[0]
            genre = name[1:3] # e.g. mBR0 → BR
            text = genre_texts.get(genre, 'a person dances to music')
            audio_paths.append(af)
            text_prompts.append(text)

        generated = generate_samples(
            model_path=args.model_path,
            audio_paths=audio_paths,
            text_prompts=text_prompts,
            n_per_audio=args.samples_per_track,
            humanml_dir=args.humanml_dir,
            device=args.device,
        )

        generated_joints = [g['joints'] for g in generated]

    # -- load real samples --

    real_joints = load_real_samples(args.aist_dir, args.humanml_dir,
                                     split='test', max_samples=100)

    # -- compute metrics --

    print("\n=== Computing Metrics ===\n")

    results = {}

    # -- 1. beat alignment score --

    if not args.skip_generation:

        print("Beat Alignment Score...")

        bas_scores = []

        for g in generated:

            beats, _ = extract_music_beats(g['audio_path'])
            bas = beat_alignment_score(g['joints'], beats)
            bas_scores.append(bas)

        results['beat_alignment'] = {
            'mean': float(np.mean(bas_scores)),
            'std': float(np.std(bas_scores)),
            'min': float(np.min(bas_scores)),
            'max': float(np.max(bas_scores)),
        }

        print(f"  BAS: {results['beat_alignment']['mean']:.4f} ± {results['beat_alignment']['std']:.4f}")

        # -- BAS for real data (baseline) --

        real_bas = []
        audio_dir = os.path.join(args.aist_dir, 'audio')

        # -- load test split names to match audio --

        test_file = os.path.join(args.aist_dir, 'splits', 'crossmodal_test.txt')

        with open(test_file) as f:
            test_names = [l.strip() for l in f if l.strip()]

        for name in test_names[:50]:

            j_path = os.path.join(args.aist_dir, 'processed', 'joints_22', f'{name}.npy')

            if not os.path.exists(j_path):
                continue

            # -- extract music ID --

            parts = name.split('_')
            music_id = None

            for p in parts:
                if p.startswith('m') and len(p) >= 3 and p[1:3].isalpha():
                    music_id = p
                    break

            if music_id is None:
                continue

            wav_path = os.path.join(audio_dir, f'{music_id}.wav')

            if not os.path.exists(wav_path):
                continue

            joints = np.load(j_path)[:196]
            beats, _ = extract_music_beats(wav_path)
            bas = beat_alignment_score(joints, beats)
            real_bas.append(bas)

        if real_bas:

            results['beat_alignment_real'] = {
                'mean': float(np.mean(real_bas)),
                'std': float(np.std(real_bas)),
            }

            print(f"  BAS (real): {results['beat_alignment_real']['mean']:.4f} ± {results['beat_alignment_real']['std']:.4f}")

    # -- 2. FID --

    print("FID...")

    feat_real = compute_fid_features(real_joints)
    feat_gen = compute_fid_features(generated_joints)

    fid = compute_fid(feat_real, feat_gen)
    results['fid'] = float(fid)
    print(f"  FID: {fid:.2f}")

    # -- 3. Diversity --

    print("Diversity...")

    div_gen = compute_diversity(generated_joints)
    div_real = compute_diversity(real_joints)

    results['diversity_gen'] = float(div_gen)
    results['diversity_real'] = float(div_real)
    print(f"  Diversity (gen): {div_gen:.4f}")
    print(f"  Diversity (real): {div_real:.4f}")

    # -- 4. Physical plausibility --

    print("Physical plausibility...")

    foot_slides = [compute_foot_sliding(j) for j in generated_joints]
    smoothness = [compute_smoothness(j) for j in generated_joints]

    foot_slides_real = [compute_foot_sliding(j) for j in real_joints]
    smoothness_real = [compute_smoothness(j) for j in real_joints]

    results['foot_sliding'] = {
        'gen_mean': float(np.mean(foot_slides)),
        'gen_std': float(np.std(foot_slides)),
        'real_mean': float(np.mean(foot_slides_real)),
        'real_std': float(np.std(foot_slides_real)),
    }

    results['smoothness'] = {
        'gen_mean': float(np.mean(smoothness)),
        'gen_std': float(np.std(smoothness)),
        'real_mean': float(np.mean(smoothness_real)),
        'real_std': float(np.std(smoothness_real)),
    }

    print(f"  Foot sliding (gen): {np.mean(foot_slides):.6f} ± {np.std(foot_slides):.6f}")
    print(f"  Foot sliding (real): {np.mean(foot_slides_real):.6f} ± {np.std(foot_slides_real):.6f}")
    print(f"  Smoothness (gen): {np.mean(smoothness):.6f} ± {np.std(smoothness):.6f}")
    print(f"  Smoothness (real): {np.mean(smoothness_real):.6f} ± {np.std(smoothness_real):.6f}")

    # -- summary --

    print("\n=== Summary ===\n")

    for k, v in results.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for kk, vv in v.items():
                print(f"  {kk}: {vv:.4f}")
        else:
            print(f"{k}: {v:.4f}")

    # -- save --

    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_path}")

if __name__ == '__main__':
    main()

####################################################################################[end]
####################################################################################[end]