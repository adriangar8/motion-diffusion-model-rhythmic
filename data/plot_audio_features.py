# [start]
# [start]

import os
import sys
# -- ensure project root is on sys.path --
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import numpy as np
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model.audio_features_wav2clip import extract_librosa_rhythmic_7

# -- reproducibility --
np.random.seed(42)

# -- audio tracks (copied from plot_generated_pca.py) --
AUDIO_TRACKS = [
    ("mBR0.wav", "a person performs breakdancing moves to music"),
    ("mHO0.wav", "a person dances house style to music"),
    ("mJB0.wav", "a person dances jazz ballet to music"),
    ("mJS0.wav", "a person dances street jazz to music"),
    ("mKR0.wav", "a person performs krumping moves to music"),
    ("mLH0.wav", "a person dances locking and hip-hop to music"),
    ("mLO0.wav", "a person dances locking to music"),
    ("mMH0.wav", "a person dances middle hip-hop to music"),
    ("mPO0.wav", "a person performs popping moves to music"),
    ("mWA0.wav", "a person dances waacking to music"),
]

# -- 7 librosa rhythmic feature names --
FEATURE_NAMES = [
    "onset_strength",
    "beat_indicator",
    "beat_dist_past",
    "beat_dist_future",
    "rms",
    "spectral_centroid",
    "tempo",
]

# -- style labels and colours --
STYLE_NAMES = {
    "mBR0": "break",
    "mHO0": "house",
    "mJB0": "jazz-ballet",
    "mJS0": "street-jazz",
    "mKR0": "krump",
    "mLH0": "lock-hiphop",
    "mLO0": "locking",
    "mMH0": "mid-hiphop",
    "mPO0": "popping",
    "mWA0": "waacking",
}

STYLE_COLORS = {
    "mBR0": "#e6194b",
    "mHO0": "#3cb44b",
    "mJB0": "#4363d8",
    "mJS0": "#f58231",
    "mKR0": "#911eb4",
    "mLH0": "#42d4f4",
    "mLO0": "#f032e6",
    "mMH0": "#bfef45",
    "mPO0": "#fabed4",
    "mWA0": "#dcbeff",
}

TRACK_IDS = [wav.replace('.wav', '') for wav, _ in AUDIO_TRACKS]


# ============================================================
# feature extraction
# ============================================================

def extract_all_features(audio_dir, target_fps):
    all_feats = {}
    for wav_name, _ in AUDIO_TRACKS:
        track_id = wav_name.replace('.wav', '')
        audio_path = os.path.join(audio_dir, wav_name)
        feats = extract_librosa_rhythmic_7(audio_path, target_fps=target_fps)
        all_feats[track_id] = feats
        print(f"  {track_id}: {feats.shape[0]} frames")
    return all_feats


# ============================================================
# figure 1: timeseries per track (10x7 grid)
# ============================================================

def plot_timeseries_per_track(all_feats, output_dir):
    n_rows, n_cols = len(TRACK_IDS), len(FEATURE_NAMES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, 20), sharex=False)
    fig.suptitle("audio features — timeseries per track", fontsize=16, y=0.995)

    for r, tid in enumerate(TRACK_IDS):
        feats = all_feats[tid]
        T = feats.shape[0]
        t_axis = np.arange(T)
        color = STYLE_COLORS[tid]

        # -- beat frames for vertical lines --
        beat_frames = np.where(feats[:, 1] > 0.5)[0]

        for c, fname in enumerate(FEATURE_NAMES):
            ax = axes[r, c]
            ax.plot(t_axis, feats[:, c], color=color, linewidth=0.6, alpha=0.9)

            # -- overlay beat markers --
            for bf in beat_frames:
                ax.axvline(bf, color='gray', linestyle='--', linewidth=0.3, alpha=0.5)

            ax.set_xlim(0, T)
            ax.tick_params(labelsize=5)

            if r == 0:
                ax.set_title(fname, fontsize=8)
            if c == 0:
                ax.set_ylabel(STYLE_NAMES[tid], fontsize=7, rotation=0,
                              labelpad=50, va='center')

            if r < n_rows - 1:
                ax.set_xticklabels([])

    plt.tight_layout(rect=[0.05, 0, 1, 0.98])
    path = os.path.join(output_dir, "timeseries_per_track.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved → {path}")


# ============================================================
# figure 2: distribution per feature (1x7 grid)
# ============================================================

def plot_distribution_per_feature(all_feats, output_dir):
    fig, axes = plt.subplots(1, len(FEATURE_NAMES), figsize=(28, 4))
    fig.suptitle("audio feature distributions by dance style", fontsize=14, y=1.02)

    for c, fname in enumerate(FEATURE_NAMES):
        ax = axes[c]

        if fname == "beat_indicator":
            # -- bar chart: beat density per style --
            densities = []
            labels = []
            colors = []
            for tid in TRACK_IDS:
                feats = all_feats[tid]
                density = (feats[:, c] > 0.5).mean()
                densities.append(density)
                labels.append(STYLE_NAMES[tid])
                colors.append(STYLE_COLORS[tid])

            bars = ax.bar(range(len(densities)), densities, color=colors, alpha=0.8)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
            ax.set_ylabel("beat density", fontsize=7)

            # -- annotate mean ± std --
            m, s = np.mean(densities), np.std(densities)
            ax.set_title(f"{fname}\n(mean={m:.3f} ± {s:.3f})", fontsize=8)
        else:
            # -- kde curves per style --
            for tid in TRACK_IDS:
                vals = all_feats[tid][:, c]
                if vals.std() < 1e-8:
                    # -- constant feature (e.g. tempo), skip kde --
                    ax.axvline(vals[0], color=STYLE_COLORS[tid],
                               label=STYLE_NAMES[tid], linewidth=1.5)
                    continue
                try:
                    kde = gaussian_kde(vals)
                    x_grid = np.linspace(vals.min(), vals.max(), 200)
                    ax.plot(x_grid, kde(x_grid), color=STYLE_COLORS[tid],
                            label=STYLE_NAMES[tid], linewidth=1.0, alpha=0.8)
                except np.linalg.LinAlgError:
                    continue

            # -- annotate mean ± std across all styles --
            all_vals = np.concatenate([all_feats[tid][:, c] for tid in TRACK_IDS])
            m, s = all_vals.mean(), all_vals.std()
            ax.set_title(f"{fname}\n(mean={m:.3f} ± {s:.3f})", fontsize=8)

        ax.tick_params(labelsize=6)

    # -- shared legend --
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=5,
                   fontsize=6, bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    path = os.path.join(output_dir, "distribution_per_feature.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved → {path}")


# ============================================================
# figure 3: per-style radar chart
# ============================================================

def plot_per_style_radar(all_feats, output_dir):
    # -- compute 5 summary stats per style --
    axis_names = ["mean onset", "beat density", "mean RMS",
                  "mean centroid", "tempo"]
    raw_stats = {}
    for tid in TRACK_IDS:
        feats = all_feats[tid]
        raw_stats[tid] = np.array([
            feats[:, 0].mean(),                    # mean onset strength
            (feats[:, 1] > 0.5).mean(),            # beat density
            feats[:, 4].mean(),                     # mean rms
            feats[:, 5].mean(),                     # mean spectral centroid
            feats[:, 6][0],                         # tempo (constant per track)
        ])

    # -- min-max normalise each axis to [0, 1] --
    all_raw = np.array([raw_stats[tid] for tid in TRACK_IDS])
    mins = all_raw.min(axis=0)
    maxs = all_raw.max(axis=0)
    rng = maxs - mins
    rng[rng < 1e-8] = 1.0

    normed = {tid: (raw_stats[tid] - mins) / rng for tid in TRACK_IDS}

    # -- radar plot --
    n_axes = len(axis_names)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]  # -- close the polygon --

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_names, fontsize=9)

    for tid in TRACK_IDS:
        vals = normed[tid].tolist() + [normed[tid][0]]
        ax.plot(angles, vals, color=STYLE_COLORS[tid],
                label=STYLE_NAMES[tid], linewidth=1.5, alpha=0.8)
        ax.fill(angles, vals, color=STYLE_COLORS[tid], alpha=0.08)

    ax.set_ylim(0, 1.05)
    ax.set_title("per-style audio feature radar", fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

    path = os.path.join(output_dir, "per_style_radar.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved → {path}")


# ============================================================
# cli
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='visualise librosa rhythmic features for AIST++ audio tracks')
    parser.add_argument('--audio_dir', type=str, default='dataset/aist/audio',
                        help='directory containing .wav files')
    parser.add_argument('--output_dir', type=str, default='figures/audio_features',
                        help='directory for output figures')
    parser.add_argument('--target_fps', type=int, default=20,
                        help='frames per second for feature extraction')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("extracting audio features...")
    all_feats = extract_all_features(args.audio_dir, args.target_fps)

    print("\nplotting timeseries grid...")
    plot_timeseries_per_track(all_feats, args.output_dir)

    print("plotting feature distributions...")
    plot_distribution_per_feature(all_feats, args.output_dir)

    print("plotting radar chart...")
    plot_per_style_radar(all_feats, args.output_dir)

    print("\ndone.")


if __name__ == '__main__':
    main()

# [end]
# [end]
