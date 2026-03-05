# [start]
# [start]

"""

Music-Dance Alignment Metrics.

  1. Beat Distance Score (BDS) — average time distance from kinematic beats
     to nearest music beat. Lower = better. (AIST++ formulation)

  2. Dynamic Energy Correlation (DEC) — Pearson correlation between
     music RMS energy and motion kinetic energy over time. Higher = better.

  3. Onset-Motion Alignment (OMA) — fraction of audio onset peaks with
     a nearby kinetic energy peak. Higher = better.

Usage:

    # single pair
    python -m eval.music_motion_alignment --mode single \
        --joints_path ./path/joints.npy --audio_path ./path/audio.wav

    # evaluate dataset directory
    python -m eval.music_motion_alignment --mode dataset --dataset aist \
        --joints_dir ./dataset/aist_raw/processed/joints_22 \
        --audio_dir ./dataset/aist_raw/audio --max_samples 100

    # ground-truth baselines (both AIST++ and FineDance)
    python -m eval.music_motion_alignment --mode gt_baseline --plot

"""

import os
import sys
import json
import argparse
import numpy as np
from glob import glob

sys.path.insert(0, '.')

# ---------------------------------------------------------------------------
# -- shared utilities --
# ---------------------------------------------------------------------------


def extract_music_beats(wav_path, fps=20):

    import librosa

    y, sr = librosa.load(wav_path, sr=22050)
    tempo, beat_frames_audio = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames_audio, sr=sr)
    beat_indices = (beat_times * fps).astype(int)

    return beat_indices, beat_times


def compute_kinetic_energy(joints):

    # -- joints: (T, 22, 3) → ke: (T-1,) --

    vel = np.diff(joints, axis=0)
    ke = 0.5 * np.sum(vel ** 2, axis=(1, 2))

    return ke


def compute_velocity_norm(joints):

    # -- frobenius norm of joint velocity per frame --

    vel = np.diff(joints, axis=0)  # (T-1, 22, 3)
    vel_flat = vel.reshape(vel.shape[0], -1)  # (T-1, 66)

    return np.linalg.norm(vel_flat, axis=-1)  # (T-1,)


def detect_kinematic_beats(joints, fps=20, method='velocity_minima'):

    if method == 'velocity_minima':

        # -- local minima of velocity norm (standard in AIST++ literature) --

        from scipy.signal import argrelmin

        vel_norm = compute_velocity_norm(joints)
        order = max(int(fps * 0.25), 1)  # -- 250ms minimum spacing --
        minima = argrelmin(vel_norm, order=order)[0]

        return minima

    elif method == 'kinetic_peaks':

        # -- local maxima of kinetic energy (used in existing BAS) --

        from scipy.signal import find_peaks

        ke = compute_kinetic_energy(joints)
        peaks, _ = find_peaks(ke, distance=int(fps * 0.2))

        return peaks

    else:
        raise ValueError(f"unknown kinematic method: {method}")


def recover_joints(motion_263, mean, std, joints_num=22):

    import torch
    from data_loaders.humanml.scripts.motion_process import recover_from_ric

    motion = motion_263 * std + mean
    motion_t = torch.from_numpy(motion).float().unsqueeze(0)
    joints = recover_from_ric(motion_t, joints_num)

    return joints.squeeze(0).numpy()


def get_music_id_aist(sequence_name):

    # -- extract music ID from AIST++ sequence name --
    # -- e.g. gBR_sBM_cAll_d04_mBR0_ch01 → mBR0 --

    for part in sequence_name.split('_'):
        if part.startswith('m') and len(part) >= 3 and part[1:3].isalpha():
            return part

    return None


# ---------------------------------------------------------------------------
# -- metric 1: beat distance score (bds) --
# ---------------------------------------------------------------------------

def beat_distance_score(joints, beat_indices, fps=20, method='velocity_minima'):
    """
    Average time distance (seconds) from each kinematic beat to the nearest
    music beat. Lower = better alignment.
    """

    kin_beats = detect_kinematic_beats(joints, fps=fps, method=method)

    if len(kin_beats) == 0 or len(beat_indices) == 0:
        return {'bds': float('nan'), 'bds_std': float('nan'),
                'n_kinematic_beats': 0, 'n_music_beats': len(beat_indices)}

    # -- for each kinematic beat, find nearest music beat --

    beat_sorted = np.sort(beat_indices)
    distances = []

    for k in kin_beats:
        idx = np.searchsorted(beat_sorted, k)

        # -- check left and right neighbours --

        candidates = []
        if idx > 0:
            candidates.append(abs(k - beat_sorted[idx - 1]))
        if idx < len(beat_sorted):
            candidates.append(abs(k - beat_sorted[idx]))

        if candidates:
            distances.append(min(candidates) / fps)

    distances = np.array(distances)

    return {
        'bds': float(np.mean(distances)),
        'bds_std': float(np.std(distances)),
        'bds_median': float(np.median(distances)),
        'n_kinematic_beats': len(kin_beats),
        'n_music_beats': len(beat_indices),
    }


# ---------------------------------------------------------------------------
# -- metric 2: dynamic energy correlation (dec) --
# ---------------------------------------------------------------------------

def dynamic_energy_correlation(joints, audio_path, fps=20, smooth_sigma=5):
    """
    Pearson correlation between smoothed music RMS energy and motion kinetic
    energy. Higher = better.
    """

    import librosa
    from scipy.ndimage import gaussian_filter1d
    from scipy.stats import pearsonr

    # -- motion kinetic energy --

    ke = compute_kinetic_energy(joints)  # (T-1,)

    # -- audio RMS energy at target fps --

    sr = 22050
    y, _ = librosa.load(audio_path, sr=sr)
    hop_length = sr // fps
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]  # (N,)

    # -- align lengths --

    L = min(len(ke), len(rms))
    ke = ke[:L]
    rms = rms[:L]

    if L < 10:
        return {'dec': float('nan'), 'dec_pvalue': float('nan'),
                'length_frames': L}

    # -- smooth both signals --

    ke_smooth = gaussian_filter1d(ke, sigma=smooth_sigma)
    rms_smooth = gaussian_filter1d(rms, sigma=smooth_sigma)

    # -- pearson correlation --

    r, p = pearsonr(ke_smooth, rms_smooth)

    return {
        'dec': float(r),
        'dec_pvalue': float(p),
        'motion_energy_mean': float(np.mean(ke)),
        'audio_rms_mean': float(np.mean(rms)),
        'length_frames': L,
    }


# ---------------------------------------------------------------------------
# -- metric 3: onset-motion alignment (oma) --
# ---------------------------------------------------------------------------

def onset_motion_alignment(joints, audio_path, fps=20, window=2,
                           onset_threshold=0.5):
    """
    Fraction of audio onset peaks with a nearby kinetic energy peak.
    Higher = better.
    """

    import librosa
    from scipy.signal import find_peaks

    # -- audio onset strength at target fps --

    sr = 22050
    y, _ = librosa.load(audio_path, sr=sr)
    hop_length = sr // fps
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # -- detect onset peaks --

    height = onset_threshold * np.max(onset_env) if len(onset_env) > 0 else 0
    onset_peaks, _ = find_peaks(onset_env, height=height,
                                distance=int(fps * 0.15))

    # -- motion kinetic energy peaks --
    # -- use 0.5s min spacing to avoid saturation on long sequences --

    ke = compute_kinetic_energy(joints)
    motion_peaks, _ = find_peaks(ke, distance=int(fps * 0.5))
    motion_peak_set = set(motion_peaks)

    T = len(ke)

    # -- count aligned onsets --

    aligned = 0
    valid = 0

    for op in onset_peaks:

        if op >= T:
            continue

        valid += 1

        for offset in range(-window, window + 1):
            if (op + offset) in motion_peak_set:
                aligned += 1
                break

    oma = aligned / valid if valid > 0 else 0.0

    return {
        'oma': float(oma),
        'n_onset_peaks': int(valid),
        'n_motion_peaks': int(len(motion_peaks)),
        'n_aligned': int(aligned),
    }


# ---------------------------------------------------------------------------
# -- combined evaluation --
# ---------------------------------------------------------------------------

def evaluate_alignment(joints, audio_path, fps=20, method='velocity_minima',
                       smooth_sigma=5, window=2):

    beat_indices, _ = extract_music_beats(audio_path, fps=fps)

    # -- filter beats to motion length --

    T = joints.shape[0]
    beat_indices = beat_indices[beat_indices < T - 1]

    bds = beat_distance_score(joints, beat_indices, fps=fps, method=method)
    dec = dynamic_energy_correlation(joints, audio_path, fps=fps,
                                     smooth_sigma=smooth_sigma)
    oma = onset_motion_alignment(joints, audio_path, fps=fps, window=window)

    return {'bds': bds, 'dec': dec, 'oma': oma}


# ---------------------------------------------------------------------------
# -- dataset evaluation --
# ---------------------------------------------------------------------------

def resolve_audio_path(joints_filename, audio_dir, dataset='aist'):

    stem = os.path.splitext(os.path.basename(joints_filename))[0]

    if dataset == 'aist':
        music_id = get_music_id_aist(stem)
        if music_id is None:
            return None
        wav = os.path.join(audio_dir, f'{music_id}.wav')
    elif dataset == 'finedance':
        wav = os.path.join(audio_dir, f'{stem}.wav')
    else:
        wav = os.path.join(audio_dir, f'{stem}.wav')

    return wav if os.path.exists(wav) else None


def evaluate_dataset(joints_dir, audio_dir, dataset='aist', fps=20,
                     max_samples=0, method='velocity_minima',
                     smooth_sigma=5, window=2):

    joint_files = sorted(glob(os.path.join(joints_dir, '*.npy')))

    if max_samples > 0:
        joint_files = joint_files[:max_samples]

    all_bds, all_dec, all_oma = [], [], []
    skipped = 0

    for jf in joint_files:

        wav = resolve_audio_path(jf, audio_dir, dataset=dataset)
        if wav is None:
            skipped += 1
            continue

        joints = np.load(jf)
        result = evaluate_alignment(joints, wav, fps=fps, method=method,
                                    smooth_sigma=smooth_sigma, window=window)

        if not np.isnan(result['bds']['bds']):
            all_bds.append(result['bds']['bds'])
        if not np.isnan(result['dec']['dec']):
            all_dec.append(result['dec']['dec'])
        all_oma.append(result['oma']['oma'])

    n = len(all_oma)
    print(f"  evaluated {n} sequences, skipped {skipped}")

    return {
        'n_sequences': n,
        'skipped': skipped,
        'bds_mean': float(np.mean(all_bds)) if all_bds else float('nan'),
        'bds_std': float(np.std(all_bds)) if all_bds else float('nan'),
        'dec_mean': float(np.mean(all_dec)) if all_dec else float('nan'),
        'dec_std': float(np.std(all_dec)) if all_dec else float('nan'),
        'oma_mean': float(np.mean(all_oma)) if all_oma else float('nan'),
        'oma_std': float(np.std(all_oma)) if all_oma else float('nan'),
    }


def evaluate_ground_truth(fps=20, method='velocity_minima', smooth_sigma=5,
                          window=2, max_samples=0):

    results = {}

    # -- aist++ --

    aist_joints = './dataset/aist_raw/processed/joints_22'
    aist_audio = './dataset/aist_raw/audio'

    if os.path.isdir(aist_joints) and os.path.isdir(aist_audio):
        print("evaluating AIST++ ground truth...")
        results['aist'] = evaluate_dataset(
            aist_joints, aist_audio, dataset='aist', fps=fps,
            max_samples=max_samples, method=method,
            smooth_sigma=smooth_sigma, window=window,
        )
    else:
        print(f"AIST++ not found at {aist_joints}")

    # -- finedance --

    fd_joints = './dataset/finedance_raw/processed/joints_22'
    fd_audio = './dataset/finedance_raw/music_wav'

    if os.path.isdir(fd_joints) and os.path.isdir(fd_audio):
        print("evaluating FineDance ground truth...")
        results['finedance'] = evaluate_dataset(
            fd_joints, fd_audio, dataset='finedance', fps=fps,
            max_samples=max_samples, method=method,
            smooth_sigma=smooth_sigma, window=window,
        )
    else:
        print(f"FineDance not found at {fd_joints}")

    return results


# ---------------------------------------------------------------------------
# -- plotting --
# ---------------------------------------------------------------------------

def plot_comparison(results_dict, output_path):

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = list(results_dict.keys())
    n = len(names)

    if n == 0:
        print("nothing to plot")
        return

    bds_means = [results_dict[k].get('bds_mean', float('nan')) for k in names]
    bds_stds = [results_dict[k].get('bds_std', 0) for k in names]
    dec_means = [results_dict[k].get('dec_mean', float('nan')) for k in names]
    dec_stds = [results_dict[k].get('dec_std', 0) for k in names]
    oma_means = [results_dict[k].get('oma_mean', float('nan')) for k in names]
    oma_stds = [results_dict[k].get('oma_std', 0) for k in names]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(n)
    w = 0.5

    # -- bds (lower = better) --

    axes[0].bar(x, bds_means, w, yerr=bds_stds, capsize=5, color='tab:blue',
                alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=11)
    axes[0].set_ylabel('seconds', fontsize=11)
    axes[0].set_title('Beat Distance Score (lower=better)', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)

    # -- dec (higher = better) --

    axes[1].bar(x, dec_means, w, yerr=dec_stds, capsize=5, color='tab:orange',
                alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, fontsize=11)
    axes[1].set_ylabel('Pearson r', fontsize=11)
    axes[1].set_title(
        'Dynamic Energy Correlation (higher=better)', fontsize=12)
    axes[1].grid(axis='y', alpha=0.3)

    # -- oma (higher = better) --

    axes[2].bar(x, oma_means, w, yerr=oma_stds, capsize=5, color='tab:green',
                alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, fontsize=11)
    axes[2].set_ylabel('fraction', fontsize=11)
    axes[2].set_title('Onset-Motion Alignment (higher=better)', fontsize=12)
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"saved comparison plot: {output_path}")


def plot_timeline(joints, audio_path, output_path, fps=20):

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import librosa

    sr = 22050
    y, _ = librosa.load(audio_path, sr=sr)
    hop_length = sr // fps

    # -- audio features --

    beat_indices, _ = extract_music_beats(audio_path, fps=fps)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # -- motion features --

    ke = compute_kinetic_energy(joints)
    kin_beats = detect_kinematic_beats(
        joints, fps=fps, method='velocity_minima')

    L = min(len(ke), len(rms), len(onset_env))
    time = np.arange(L) / fps

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # -- crop everything to the overlapping time range --

    max_time = L / fps
    max_audio_sample = int(max_time * sr)

    # -- panel 1: waveform with beats --

    t_audio = np.arange(min(len(y), max_audio_sample)) / sr
    axes[0].plot(t_audio, y[:len(t_audio)], color='gray', alpha=0.5,
                 linewidth=0.5)
    for b in beat_indices:
        if b < L:
            axes[0].axvline(b / fps, color='red', alpha=0.6, linewidth=0.8)
    axes[0].set_ylabel('amplitude')
    axes[0].set_title('audio waveform + beats (red)')
    axes[0].set_xlim(0, max_time)

    # -- panel 2: audio RMS energy --

    axes[1].plot(time, rms[:L], color='tab:orange')
    axes[1].set_ylabel('RMS energy')
    axes[1].set_title('audio RMS energy')

    # -- panel 3: motion kinetic energy + kinematic beats --

    axes[2].plot(time, ke[:L], color='tab:blue')
    for kb in kin_beats:
        if kb < L:
            axes[2].axvline(kb / fps, color='green', alpha=0.5, linewidth=0.8)
    axes[2].set_ylabel('kinetic energy')
    axes[2].set_title('motion kinetic energy + kinematic beats (green)')

    # -- panel 4: onset strength --

    axes[3].plot(time, onset_env[:L], color='tab:purple')
    axes[3].set_ylabel('onset strength')
    axes[3].set_xlabel('time (s)')
    axes[3].set_title('audio onset strength')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"saved timeline plot: {output_path}")


# ---------------------------------------------------------------------------
# -- main --
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description='Music-Dance Alignment Metrics (BDS, DEC, OMA)')

    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'dataset', 'gt_baseline'])

    # -- single mode --

    parser.add_argument('--joints_path', type=str, default='')
    parser.add_argument('--audio_path', type=str, default='')

    # -- dataset mode --

    parser.add_argument('--joints_dir', type=str, default='')
    parser.add_argument('--audio_dir', type=str, default='')
    parser.add_argument('--dataset', type=str, default='aist',
                        choices=['aist', 'finedance'])
    parser.add_argument('--max_samples', type=int, default=0)

    # -- shared --

    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--kinematic_method', type=str, default='velocity_minima',
                        choices=['velocity_minima', 'kinetic_peaks'])
    parser.add_argument('--smooth_sigma', type=float, default=5.0)
    parser.add_argument('--window', type=int, default=2)
    parser.add_argument('--output_path', type=str,
                        default='./eval_alignment.json')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--plot_timeline', action='store_true')
    parser.add_argument('--humanml_dir', type=str,
                        default='./dataset/HumanML3D')

    args = parser.parse_args()

    results = {}

    # -- single mode --

    if args.mode == 'single':

        if not args.joints_path or not args.audio_path:
            parser.error(
                "--joints_path and --audio_path required for single mode")

        joints = np.load(args.joints_path)
        print(f"loaded joints: {joints.shape}")

        result = evaluate_alignment(
            joints, args.audio_path, fps=args.fps,
            method=args.kinematic_method, smooth_sigma=args.smooth_sigma,
            window=args.window,
        )

        results = result

        print(f"\n=== Results ===")
        print(f"  BDS: {result['bds']['bds']:.4f}s (lower=better)")
        print(f"  DEC: {result['dec']['dec']:.4f} (higher=better)")
        print(f"  OMA: {result['oma']['oma']:.4f} (higher=better)")

        if args.plot_timeline:
            tl_path = args.output_path.replace('.json', '_timeline.png')
            plot_timeline(joints, args.audio_path, tl_path, fps=args.fps)

    # -- dataset mode --

    elif args.mode == 'dataset':

        if not args.joints_dir or not args.audio_dir:
            parser.error(
                "--joints_dir and --audio_dir required for dataset mode")

        print(f"evaluating {args.dataset} dataset...")
        results = evaluate_dataset(
            args.joints_dir, args.audio_dir, dataset=args.dataset,
            fps=args.fps, max_samples=args.max_samples,
            method=args.kinematic_method, smooth_sigma=args.smooth_sigma,
            window=args.window,
        )

        print(f"\n=== Results ({args.dataset}) ===")
        print(f"  BDS: {results['bds_mean']:.4f} ± {results['bds_std']:.4f}s")
        print(f"  DEC: {results['dec_mean']:.4f} ± {results['dec_std']:.4f}")
        print(f"  OMA: {results['oma_mean']:.4f} ± {results['oma_std']:.4f}")

    # -- gt_baseline mode --

    elif args.mode == 'gt_baseline':

        results = evaluate_ground_truth(
            fps=args.fps, method=args.kinematic_method,
            smooth_sigma=args.smooth_sigma, window=args.window,
            max_samples=args.max_samples,
        )

        print(f"\n=== Ground Truth Baselines ===")
        for name, r in results.items():
            print(f"\n  {name} ({r['n_sequences']} sequences):")
            print(f"    BDS: {r['bds_mean']:.4f} ± {r['bds_std']:.4f}s")
            print(f"    DEC: {r['dec_mean']:.4f} ± {r['dec_std']:.4f}")
            print(f"    OMA: {r['oma_mean']:.4f} ± {r['oma_std']:.4f}")

        if args.plot and results:
            plot_path = args.output_path.replace('.json', '_comparison.png')
            plot_comparison(results, plot_path)

    # -- save --

    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)

    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nresults saved to {args.output_path}")


if __name__ == '__main__':
    main()

# [end]
# [end]
