####################################################################################[start]
####################################################################################[start]

"""

Comprehensive Evaluation for Audio-Conditioned Motion Generation.

=== PRIMARY METRIC ===

  RhythmScore = RDS_norm × (1 - TF_norm)

  Where:
    RDS  = Residual-Dance Similarity: Pearson correlation between the kinetic
           energy profile of the audio-added residual and the kinetic energy
           profile of real AIST++ dance to the same music.
    TF   = Text Fidelity: RMS joint-space deviation from text-only motion.

  RhythmScore is 0 if either component fails:
    - Perfect rhythm but destroyed text → 0
    - Perfect text but no rhythm added → 0
    - Good rhythm AND preserved text → high score

=== SECONDARY METRICS (diagnostic) ===

  BAS   — Beat Alignment Score: fraction of beats with nearby KE peak (AIST++)
  RRBA  — Rhythmic Residual Beat Alignment: BAS on the audio-added residual
  BDS   — Beat Distance Score: avg distance (seconds) from beats to nearest KE peak
  DEC   — Dynamic Energy Correlation: Pearson(onset_envelope, motion_KE)
  OMA   — Onset-Motion Alignment: fraction of audio onsets aligned with motion peaks
  Gap   — RRBA - BAS: how targeted the audio effect is

=== WHAT IT NEEDS ===

  1. Generated motion files (.npy, shape (T, 263), HumanML3D-normalized):
     - Text-only samples (baseline, no audio)
     - Audio-conditioned samples (refined or from-scratch)

  2. Audio file (.wav) used for conditioning

  3. Ground truth AIST++ dance motions for the SAME music track (.npy, shape (T, 263))
     Found in: dataset/aist/processed/motions_263/
     Filter by music ID: e.g., all sequences containing "mBR0" in their name

  4. HumanML3D normalization stats: Mean.npy, Std.npy

=== HOW IT WORKS ===

  Step 1: Load all .npy motion files and convert from (T, 263) normalized features
          to (T, 22, 3) joint positions using HumanML3D's recover_from_ric().

  Step 2: Extract audio analysis from the .wav file:
          - Beat positions (librosa beat tracker)
          - Onset envelope (continuous energy)
          - Onset peaks (discrete events)

  Step 3: Load ground truth AIST++ dance motions for the same music and compute
          their kinetic energy profiles. Average across multiple choreographies
          if available (different dancers to same music).

  Step 4: For each (audio_sample, text_sample) pair:
          a. Compute residual = audio_joints - text_joints
          b. Compute KE of residual, KE of full audio motion, KE of ground truth dance
          c. RDS = pearson(residual_KE, avg_dance_KE)
          d. TF = RMS joint deviation (audio vs text)
          e. BAS, RRBA, BDS, DEC, OMA on appropriate signals

  Step 5: Aggregate across samples, compute RhythmScore.

=== USAGE ===

  # 1. Single model evaluation
  python -m eval.evaluate_all \
      --audio_dir ./save/model/refined_skip500 \
      --text_dir ./save/model/text_only \
      --gt_dance_dir ./dataset/aist/processed/motions_263 \
      --music_id mBR0 \
      --audio_path ./dataset/aist/audio/mBR0.wav \
      --humanml_dir ./dataset/HumanML3D

  # 2. Skip timestep sweep (single model)
  python -m eval.evaluate_all \
      --sweep_mode skip \
      --sweep_dir ./save/model/samples_refined \
      --sweep_values 200 300 400 500 600 700 800 \
      --text_dir ./save/model/samples_refined \
      --text_pattern "sample_text_only_*.npy" \
      --audio_pattern "sample_refined_skip{}_*.npy" \
      --gt_dance_dir ./dataset/aist/processed/motions_263 \
      --music_id mBR0 \
      --audio_path ./dataset/aist/audio/mBR0.wav \
      --humanml_dir ./dataset/HumanML3D \
      --output_path ./save/model/eval_skip_sweep.json

  # 3. Compare two models (Adrian vs Yash) at the same skip value
  python -m eval.evaluate_all \
      --compare_mode \
      --model_dirs ./save/comparison/adrian_v2/refined_skip500 \
                   ./save/comparison/yash/refined_skip500 \
      --model_names "Adrian_v2" "Yash_wav2clip" \
      --text_dirs ./save/comparison/adrian_v2/text_only \
                  ./save/comparison/yash/text_only \
      --gt_dance_dir ./dataset/aist/processed/motions_263 \
      --music_id mBR0 \
      --audio_path ./dataset/aist/audio/mBR0.wav \
      --humanml_dir ./dataset/HumanML3D \
      --output_path ./save/comparison/eval_comparison.json

"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from glob import glob
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, '.')

# -- step 1: joint recovery from HumanML3D 263-dim representation --

def recover_joints(motion_263, mean, std, joints_num=22):
    
    """
    Convert (T, 263) HumanML3D-normalized features to (T, 22, 3) joint positions.
    """
    
    from data_loaders.humanml.scripts.motion_process import recover_from_ric
    
    motion = motion_263 * std + mean
    motion_t = torch.from_numpy(motion).float().unsqueeze(0)
    joints = recover_from_ric(motion_t, joints_num)
    
    return joints.squeeze(0).numpy()

def compute_kinetic_energy(joints):

    """Per-frame kinetic energy: KE(t) = 0.5 * sum_j ||v_j(t)||^2. Returns (T-1,)."""

    vel = np.diff(joints, axis=0)

    return 0.5 * np.sum(vel ** 2, axis=(1, 2))

def compute_velocity_magnitude(joints):

    """Per-frame total velocity magnitude across all joints. Returns (T-1,)."""

    vel = np.diff(joints, axis=0)
    speed_per_joint = np.sqrt(np.sum(vel ** 2, axis=2))

    return np.sum(speed_per_joint, axis=1)

# -- step 2: audio analysis

def analyze_audio(wav_path, fps=20):

    """Extract beats, onset envelope, and onset peaks from audio."""

    import librosa

    y, sr = librosa.load(wav_path, sr=22050)
    hop_length = sr // fps

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    beat_indices = (beat_times * fps).astype(int)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_peaks, _ = find_peaks(onset_env, distance=int(fps * 0.1), height=np.mean(onset_env))

    if isinstance(tempo, np.ndarray):

        tempo = float(tempo[0])

    return {
        'beat_indices': beat_indices,
        'onset_env': onset_env,
        'onset_peaks': onset_peaks,
        'tempo': float(tempo),
    }

# -- step 3: Load ground truth AIST++ dance --

def load_ground_truth_dances(gt_dance_dir, music_id, mean, std):

    """Load all AIST++ dances for a specific music track, return their KE profiles."""

    all_files = sorted(glob(os.path.join(gt_dance_dir, '*.npy')))
    dance_ke_profiles = []
    names = []

    for f in all_files:

        name = os.path.splitext(os.path.basename(f))[0]
        parts = name.split('_')

        if len(parts) >= 5 and parts[4] == music_id:

            motion = np.load(f)
            joints = recover_joints(motion, mean, std)
            ke = compute_kinetic_energy(joints)
            dance_ke_profiles.append(ke)
            names.append(name)

    print(f"  Loaded {len(dance_ke_profiles)} ground truth dances for '{music_id}'")

    for n in names[:5]:
        print(f"    - {n}")

    if len(names) > 5:
        print(f"    ... and {len(names) - 5} more")

    return dance_ke_profiles

def compute_average_dance_ke(dance_ke_profiles, target_length):

    """Average KE profiles across choreographies, truncating/padding to target_length."""

    if not dance_ke_profiles:
        return np.zeros(target_length)

    aligned = []

    for ke in dance_ke_profiles:

        if len(ke) >= target_length:
            aligned.append(ke[:target_length])

        else:
            padded = np.full(target_length, np.mean(ke))
            padded[:len(ke)] = ke
            aligned.append(padded)

    return np.mean(aligned, axis=0)

# -- step 4: individual metrics --

def metric_bas(joints, beat_indices, fps=20, window=2):
    
    """Beat Alignment Score. Range: [0, 1], higher = better."""
    
    ke = compute_kinetic_energy(joints)
    peaks, _ = find_peaks(ke, distance=int(fps * 0.15))
    peak_set = set(peaks)
    aligned, valid = 0, 0
    
    for b in beat_indices:
    
        if b >= len(ke):
            continue
    
        valid += 1
    
        if any((b + off) in peak_set for off in range(-window, window + 1)):
            aligned += 1
    
    return aligned / valid if valid > 0 else 0.0

def metric_rrba(joints_audio, joints_text, beat_indices, fps=20, window=2):

    """Rhythmic Residual Beat Alignment. Range: [0, 1], higher = better."""

    T = min(joints_audio.shape[0], joints_text.shape[0])

    residual = joints_audio[:T] - joints_text[:T]
    vel = np.diff(residual, axis=0)
    res_ke = 0.5 * np.sum(vel ** 2, axis=(1, 2))

    if len(res_ke) == 0:
        return 0.0

    peaks, _ = find_peaks(res_ke, distance=int(fps * 0.15))
    peak_set = set(peaks)
    aligned, valid = 0, 0

    for b in beat_indices:

        if b >= len(res_ke):
            continue

        valid += 1

        if any((b + off) in peak_set for off in range(-window, window + 1)):
            aligned += 1

    return aligned / valid if valid > 0 else 0.0

def metric_rds(joints_audio, joints_text, avg_dance_ke, fps=20):

    """

    Residual-Dance Similarity. Pearson correlation between residual KE and
    ground truth dance KE. Range: [-1, 1], higher = better.

    """

    T = min(joints_audio.shape[0], joints_text.shape[0])
    residual = joints_audio[:T] - joints_text[:T]
    vel = np.diff(residual, axis=0)
    res_ke = 0.5 * np.sum(vel ** 2, axis=(1, 2))

    L = min(len(res_ke), len(avg_dance_ke))

    if L < 10:
        return 0.0

    r_signal = gaussian_filter1d(res_ke[:L], sigma=2.0)
    d_signal = gaussian_filter1d(avg_dance_ke[:L], sigma=2.0)

    if np.std(r_signal) < 1e-10 or np.std(d_signal) < 1e-10:
        return 0.0

    corr, _ = pearsonr(r_signal, d_signal)
    return float(corr)

def metric_bds(joints, beat_indices, fps=20):

    """Beat Distance Score (seconds). Range: [0, inf), lower = better."""

    ke = compute_kinetic_energy(joints)
    peaks, _ = find_peaks(ke, distance=int(fps * 0.15))

    if len(peaks) == 0:
        return float('inf')

    distances = []

    for b in beat_indices:

        if b >= len(ke):
            continue

        distances.append(np.min(np.abs(peaks - b)) / fps)

    return float(np.mean(distances)) if distances else float('inf')

def metric_dec(joints, onset_env, fps=20):

    """Dynamic Energy Correlation. Range: [-1, 1], higher = better."""

    ke = compute_kinetic_energy(joints)
    T = min(len(ke), len(onset_env) - 1)

    if T < 10:
        return 0.0

    onset_s = gaussian_filter1d(onset_env[1:T + 1], sigma=2.0)
    ke_s = gaussian_filter1d(ke[:T], sigma=2.0)

    if np.std(onset_s) < 1e-10 or np.std(ke_s) < 1e-10:
        return 0.0

    corr, _ = pearsonr(onset_s, ke_s)

    return float(corr)

def metric_oma(joints, onset_peaks, fps=20, window=2):

    """Onset-Motion Alignment. Range: [0, 1], higher = better."""

    vel_mag = compute_velocity_magnitude(joints)
    motion_peaks, _ = find_peaks(vel_mag, distance=int(fps * 0.1))
    peak_set = set(motion_peaks)
    aligned, valid = 0, 0

    for o in onset_peaks:

        if o >= len(vel_mag):
            continue

        valid += 1

        if any((o + off) in peak_set for off in range(-window, window + 1)):
            aligned += 1

    return aligned / valid if valid > 0 else 0.0

def metric_tf(joints_audio, joints_text):

    """Text Fidelity: RMS joint deviation. Range: [0, inf), lower = better."""

    T = min(joints_audio.shape[0], joints_text.shape[0])
    diff = joints_audio[:T] - joints_text[:T]

    return float(np.sqrt(np.mean(diff ** 2)))

# -- step 5: combined evaluation --

def evaluate_single_pair(joints_audio, joints_text, audio_info, avg_dance_ke, fps=20):
    
    """Run all metrics on one (audio, text) pair. Returns dict."""
    
    bi = audio_info['beat_indices']
    oe = audio_info['onset_env']
    op = audio_info['onset_peaks']

    return {
        'RDS': metric_rds(joints_audio, joints_text, avg_dance_ke, fps),
        'TF': metric_tf(joints_audio, joints_text),
        'BAS': metric_bas(joints_audio, bi, fps),
        'RRBA': metric_rrba(joints_audio, joints_text, bi, fps),
        'BDS': metric_bds(joints_audio, bi, fps),
        'DEC': metric_dec(joints_audio, oe, fps),
        'OMA': metric_oma(joints_audio, op, fps),
        'BAS_text': metric_bas(joints_text, bi, fps),
        'DEC_text': metric_dec(joints_text, oe, fps),
    }

def evaluate_sample_set(audio_files, text_files, audio_info, avg_dance_ke,
                        mean, std, fps=20, tf_max=None):

    """Evaluate all metrics over paired samples. Returns aggregated dict."""

    n = min(len(audio_files), len(text_files))

    if n == 0:
        print("  WARNING: no sample pairs found!")
        return {}

    all_r = []

    for i in range(n):
        ja = recover_joints(np.load(audio_files[i]), mean, std)
        jt = recover_joints(np.load(text_files[i]), mean, std)
        r = evaluate_single_pair(ja, jt, audio_info, avg_dance_ke, fps)
        r['RRBA_BAS_gap'] = r['RRBA'] - r['BAS']
        r['BAS_improvement'] = r['BAS'] - r['BAS_text']
        all_r.append(r)
        print(f"  [{i}] RDS={r['RDS']:.3f} BAS={r['BAS']:.3f} RRBA={r['RRBA']:.3f} "
              f"BDS={r['BDS']:.3f}s DEC={r['DEC']:.3f} OMA={r['OMA']:.3f} TF={r['TF']:.4f}")

    # -- aggregate mean/std --
    
    agg = {}
    
    for key in all_r[0].keys():
    
        vals = [r[key] for r in all_r]
        agg[f'{key}_mean'] = float(np.mean(vals))
        agg[f'{key}_std'] = float(np.std(vals))

    # -- TF normalization for RhythmScore --
    
    if tf_max is None:
        tf_max = max(r['TF'] for r in all_r) if all_r else 1.0
    
    if tf_max < 1e-8:
        tf_max = 1.0
    
    agg['TF_max'] = tf_max

    # -- RhythmScore per sample --
    
    rhythm_scores = []
    
    for r in all_r:
    
        rds_norm = (r['RDS'] + 1.0) / 2.0
        tf_norm = min(r['TF'] / tf_max, 1.0)
        rhythm_scores.append(rds_norm * (1.0 - tf_norm))

    agg['RhythmScore_mean'] = float(np.mean(rhythm_scores))
    agg['RhythmScore_std'] = float(np.std(rhythm_scores))
    agg['n_samples'] = n
    agg['n_beats'] = len(audio_info['beat_indices'])
    agg['tempo'] = audio_info['tempo']
    agg['per_sample'] = all_r

    return agg

# -- output formatting --

def print_results(agg, label=""):
    
    prefix = f"[{label}] " if label else ""
    
    print(f"\n{prefix}RhythmScore = {agg['RhythmScore_mean']:.4f} +/- {agg['RhythmScore_std']:.4f}")
    print(f"  RDS:  {agg['RDS_mean']:.3f} +/- {agg['RDS_std']:.3f}")
    print(f"  TF:   {agg['TF_mean']:.4f} +/- {agg['TF_std']:.4f}  (lower=better)")
    print(f"  BAS:  {agg['BAS_mean']:.3f}  RRBA: {agg['RRBA_mean']:.3f}  "
          f"Gap: {agg['RRBA_BAS_gap_mean']:.3f}")
    print(f"  BDS:  {agg['BDS_mean']:.3f}s  DEC: {agg['DEC_mean']:.3f}  "
          f"OMA: {agg['OMA_mean']:.3f}")

def print_sweep_table(sweep_results, sweep_name="Value"):

    print(f"\n{'='*112}")
    print(f"{sweep_name:>8} {'RScore':>8} {'RDS':>7} {'BAS':>7} {'RRBA':>7} "
          f"{'Gap':>7} {'BDS(s)':>7} {'DEC':>7} {'OMA':>7} {'TF':>8}")
    print(f"{'='*112}")

    for val in sorted(sweep_results.keys(), key=lambda x: float(x)):

        r = sweep_results[val]

        print(f"{val:>8} {r.get('RhythmScore_mean',0):>8.4f} {r['RDS_mean']:>7.3f} "
              f"{r['BAS_mean']:>7.3f} {r['RRBA_mean']:>7.3f} "
              f"{r['RRBA_BAS_gap_mean']:>7.3f} {r['BDS_mean']:>7.3f} "
              f"{r['DEC_mean']:>7.3f} {r['OMA_mean']:>7.3f} {r['TF_mean']:>8.4f}")
    print(f"{'='*112}")

def print_comparison_table(results_by_model):

    print(f"\n{'='*117}")
    print(f"{'Model':>20} {'RScore':>8} {'RDS':>7} {'BAS':>7} {'RRBA':>7} "
          f"{'Gap':>7} {'BDS(s)':>7} {'DEC':>7} {'OMA':>7} {'TF':>8}")
    print(f"{'='*117}")

    for name, r in results_by_model.items():
        print(f"{name:>20} {r.get('RhythmScore_mean',0):>8.4f} {r['RDS_mean']:>7.3f} "
              f"{r['BAS_mean']:>7.3f} {r['RRBA_mean']:>7.3f} "
              f"{r['RRBA_BAS_gap_mean']:>7.3f} {r['BDS_mean']:>7.3f} "
              f"{r['DEC_mean']:>7.3f} {r['OMA_mean']:>7.3f} {r['TF_mean']:>8.4f}")
    print(f"{'='*117}")

# -- main --

def main():
    
    parser = argparse.ArgumentParser(description='Comprehensive motion evaluation')

    # -- mode 1: single --
    
    parser.add_argument('--audio_dir', type=str, default='')
    parser.add_argument('--text_dir', type=str, default='')

    # -- mode 2: sweep --
    
    parser.add_argument('--sweep_mode', type=str, default='',
                        choices=['', 'skip', 'audio_guidance', 'text_guidance'])
    parser.add_argument('--sweep_dir', type=str, default='')
    parser.add_argument('--sweep_values', nargs='+', default=[])
    parser.add_argument('--audio_pattern', type=str, default='sample_refined_skip{}_*.npy')
    parser.add_argument('--text_pattern', type=str, default='sample_text_only_*.npy')

    # -- mode 3: compare --
    
    parser.add_argument('--compare_mode', action='store_true')
    parser.add_argument('--model_dirs', nargs='+', default=[])
    parser.add_argument('--model_names', nargs='+', default=[])
    parser.add_argument('--text_dirs', nargs='+', default=[])

    # -- common args --
    
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--humanml_dir', type=str, default='./dataset/HumanML3D')
    parser.add_argument('--gt_dance_dir', type=str, default='')
    parser.add_argument('--music_id', type=str, default='mBR0')
    parser.add_argument('--output_path', type=str, default='./eval_results.json')
    parser.add_argument('--fps', type=int, default=20)

    args = parser.parse_args()

    # -- load shared resources --
    
    print("Loading normalization stats...")
    
    mean = np.load(os.path.join(args.humanml_dir, 'Mean.npy'))
    std = np.load(os.path.join(args.humanml_dir, 'Std.npy'))

    print("Analyzing audio...")

    audio_info = analyze_audio(args.audio_path, args.fps)

    print(f"  Tempo: {audio_info['tempo']:.1f} BPM, "
          f"{len(audio_info['beat_indices'])} beats, "
          f"{len(audio_info['onset_peaks'])} onsets")

    # -- ground truth dance --
    
    avg_dance_ke = np.array([])
    
    if args.gt_dance_dir and os.path.isdir(args.gt_dance_dir):
    
        print(f"Loading ground truth dances for '{args.music_id}'...")
        profiles = load_ground_truth_dances(args.gt_dance_dir, args.music_id, mean, std)
    
        if profiles:
    
            target_len = max(len(ke) for ke in profiles)
            avg_dance_ke = compute_average_dance_ke(profiles, target_len)
    
    if len(avg_dance_ke) == 0:
        print("  WARNING: No ground truth dances. RDS will be 0.")

    results = {}

    # -- mode 1: single --
    
    if args.audio_dir and args.text_dir:
    
        print(f"\n=== Single evaluation ===")
        af = sorted(glob(os.path.join(args.audio_dir, '*.npy')))
        tf = sorted(glob(os.path.join(args.text_dir, '*.npy')))
        results = evaluate_sample_set(af, tf, audio_info, avg_dance_ke, mean, std, args.fps)
        print_results(results)

    # -- mode 2: sweep --
    
    elif args.sweep_mode and args.sweep_dir and args.sweep_values:
    
        print(f"\n=== Sweep evaluation ({args.sweep_mode}) ===")
    
        text_src = args.text_dir if args.text_dir else args.sweep_dir
        tf = sorted(glob(os.path.join(text_src, args.text_pattern)))
    
        print(f"  Text samples: {len(tf)} files")

        # -- TF normalization pass --
        
        all_tfs = []
        
        for val in args.sweep_values:
        
            pat = args.audio_pattern.format(val)
        
            for af_path, tf_path in zip(sorted(glob(os.path.join(args.sweep_dir, pat))), tf):
        
                ja = recover_joints(np.load(af_path), mean, std)
                jt = recover_joints(np.load(tf_path), mean, std)
                all_tfs.append(metric_tf(ja, jt))
        
        tf_max = max(all_tfs) if all_tfs else 1.0
        
        print(f"  TF_max = {tf_max:.4f}")

        sweep_results = {}
        
        for val in args.sweep_values:
        
            pat = args.audio_pattern.format(val)
            af = sorted(glob(os.path.join(args.sweep_dir, pat)))
        
            if not af:
                continue
        
            print(f"\n--- {args.sweep_mode}={val} ---")
        
            sweep_results[str(val)] = evaluate_sample_set(
                af, tf, audio_info, avg_dance_ke, mean, std, args.fps, tf_max)

        results = {'sweep': sweep_results, 'tf_max': tf_max}
        print_sweep_table(sweep_results, args.sweep_mode.capitalize())

    # -- mode 3: compare --
    
    elif args.compare_mode and args.model_dirs:
    
        print(f"\n=== Model comparison ===")
    
        if not args.model_names:
            args.model_names = [f"Model_{i}" for i in range(len(args.model_dirs))]
        if not args.text_dirs:
            args.text_dirs = args.model_dirs

        # -- TF normalization --
        
        all_tfs = []
        
        for ad, td in zip(args.model_dirs, args.text_dirs):
        
            for af_p, tf_p in zip(sorted(glob(os.path.join(ad, '*.npy'))),
                                  sorted(glob(os.path.join(td, '*.npy')))):
                ja = recover_joints(np.load(af_p), mean, std)
                jt = recover_joints(np.load(tf_p), mean, std)
                all_tfs.append(metric_tf(ja, jt))
        
        tf_max = max(all_tfs) if all_tfs else 1.0

        comp = {}
        
        for name, ad, td in zip(args.model_names, args.model_dirs, args.text_dirs):
        
            print(f"\n--- {name} ---")
        
            af = sorted(glob(os.path.join(ad, '*.npy')))
            tff = sorted(glob(os.path.join(td, '*.npy')))
            comp[name] = evaluate_sample_set(
                af, tff, audio_info, avg_dance_ke, mean, std, args.fps, tf_max)

        results = {'comparison': comp, 'tf_max': tf_max}
        print_comparison_table(comp)

    else:
        
        print("Specify: --audio_dir+--text_dir, --sweep_mode+--sweep_dir+--sweep_values, "
              "or --compare_mode+--model_dirs")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or '.', exist_ok=True)
    
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_path}")

if __name__ == '__main__':
    main()
    
####################################################################################[end]
####################################################################################[end]