# [start]
# [start]

import os
import sys
# -- ensure project root is on sys.path --
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import numpy as np
from glob import glob
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model.audio_features_wav2clip import extract_librosa_rhythmic_7

# -- reproducibility --
np.random.seed(42)

# -- paths --
AIST_DIR = "dataset/aist/processed/motions_263"
MEAN_PATH = "dataset/HumanML3D/humanml/Mean.npy"
STD_PATH = "dataset/HumanML3D/humanml/Std.npy"

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

# -- cfg sweep configs (copied from plot_generated_pca.py) --
CFG_CONFIGS = [
    ("balanced_2.5_2.5",    2.5, 2.5),
    ("text_heavy_5.0_1.0",  5.0, 1.0),
    ("audio_heavy_1.0_5.0", 1.0, 5.0),
    ("audio_only_0.0_5.0",  0.0, 5.0),
    ("high_both_5.0_5.0",   5.0, 5.0),
    ("best_bas_2.5_1.5",    2.5, 1.5),
    ("text_only_2.5_0.0",   2.5, 0.0),
]

CFG_COLORS = {
    "balanced_2.5_2.5":    "#9C27B0",
    "text_heavy_5.0_1.0":  "#FF9800",
    "audio_heavy_1.0_5.0": "#4CAF50",
    "audio_only_0.0_5.0":  "#00BCD4",
    "high_both_5.0_5.0":   "#795548",
    "best_bas_2.5_1.5":    "#E91E63",
    "text_only_2.5_0.0":   "#9E9E9E",
}

CFG_LABELS = {
    "balanced_2.5_2.5":    "balanced (2.5/2.5)",
    "text_heavy_5.0_1.0":  "text-heavy (5.0/1.0)",
    "audio_heavy_1.0_5.0": "audio-heavy (1.0/5.0)",
    "audio_only_0.0_5.0":  "audio-only (0.0/5.0)",
    "high_both_5.0_5.0":   "high-both (5.0/5.0)",
    "best_bas_2.5_1.5":    "best-BAS (2.5/1.5)",
    "text_only_2.5_0.0":   "text-only (2.5/0.0)",
}

# -- audio feature names --
FEATURE_NAMES = [
    "onset_strength",
    "beat_indicator",
    "beat_dist_past",
    "beat_dist_future",
    "rms",
    "spectral_centroid",
    "tempo",
]

GT_COLOR = "#F44336"


# ============================================================
# helpers
# ============================================================

def load_normalization():
    mean = np.load(MEAN_PATH).astype(np.float32)
    std = np.load(STD_PATH).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


def compute_kinetic_energy(motion):
    # -- l2 norm of joint velocity vector (193:259) per frame --
    return np.linalg.norm(motion[:, 193:259], axis=1)


def compute_foot_contact(motion):
    # -- mean of foot contact channels (259:263) per frame --
    return motion[:, 259:263].mean(axis=1)


def compute_bas(beat_indicator, kinetic_energy, window=2):
    beat_frames = np.where(beat_indicator > 0.5)[0]
    if len(beat_frames) == 0:
        return 0.0
    aligned = 0
    T = len(kinetic_energy)
    for b in beat_frames:
        lo = max(0, b - window)
        hi = min(T, b + window + 1)
        if kinetic_energy[b] >= kinetic_energy[lo:hi].max():
            aligned += 1
    return aligned / len(beat_frames)


def safe_pearsonr(x, y):
    # -- handle constant or near-constant arrays gracefully --
    if x.std() < 1e-6 or y.std() < 1e-6:
        return 0.0
    r, _ = pearsonr(x, y)
    if np.isnan(r):
        return 0.0
    return float(r)


def max_lag_xcorr(x, y, max_lag=10, smooth_sigma=2):
    # -- smoothed cross-correlation with lag sweep --
    x_s = gaussian_filter1d(x.astype(np.float64), sigma=smooth_sigma)
    y_s = gaussian_filter1d(y.astype(np.float64), sigma=smooth_sigma)
    best_r, best_lag = 0.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            r = safe_pearsonr(x_s[lag:], y_s[:len(y_s) - lag]) if lag < len(x_s) else 0.0
        else:
            r = safe_pearsonr(x_s[:len(x_s) + lag], y_s[-lag:]) if -lag < len(y_s) else 0.0
        if abs(r) > abs(best_r):
            best_r, best_lag = r, lag
    return best_r, best_lag


def align_and_trim(*arrays):
    min_len = min(a.shape[0] for a in arrays)
    return tuple(a[:min_len] for a in arrays)


def load_generated_for_track(output_dir, cfg_name, track_id):
    # -- try individual sample files first --
    cfg_dir = os.path.join(output_dir, "generated", cfg_name)
    pattern = os.path.join(cfg_dir, f"sample_{track_id}_*.npy")
    files = sorted(glob(pattern))
    if files:
        return [np.load(f) for f in files]

    # -- fall back to splitting concatenated all_frames file --
    all_frames_path = os.path.join(output_dir, "all_frames", f"{cfg_name}.npy")
    if not os.path.exists(all_frames_path):
        return []

    manifest_path = os.path.join(output_dir, "all_frames", "manifest.json")
    import json
    with open(manifest_path) as f:
        manifest = json.load(f)

    n_samples = manifest.get("num_samples_per_track", 6)
    track_ids = sorted([w.replace('.wav', '') for w in manifest["audio_tracks"]])
    if track_id not in track_ids:
        return []

    all_frames = np.load(all_frames_path)
    frames_per_sample = all_frames.shape[0] // (len(track_ids) * n_samples)
    track_idx = track_ids.index(track_id)

    samples = []
    for s in range(n_samples):
        offset = (track_idx * n_samples + s) * frames_per_sample
        samples.append(all_frames[offset:offset + frames_per_sample])
    return samples


def load_gt_for_track(track_id):
    pattern = os.path.join(AIST_DIR, f"*_{track_id}_*.npy")
    files = sorted(glob(pattern))
    return [np.load(f) for f in files]


# ============================================================
# analysis
# ============================================================

def run_analysis(args):
    mean, std = load_normalization()

    # -- results containers --
    bas_per_cfg = {cfg: [] for cfg, _, _ in CFG_CONFIGS}
    bas_gt = []
    onset_ke_per_cfg = {cfg: [] for cfg, _, _ in CFG_CONFIGS}
    onset_ke_gt = []
    rms_fc_per_cfg = {cfg: [] for cfg, _, _ in CFG_CONFIGS}
    rms_fc_gt = []
    # -- smoothed cross-correlation containers --
    onset_ke_xcorr_per_cfg = {cfg: [] for cfg, _, _ in CFG_CONFIGS}
    onset_ke_xcorr_gt = []
    rms_fc_xcorr_per_cfg = {cfg: [] for cfg, _, _ in CFG_CONFIGS}
    rms_fc_xcorr_gt = []
    # -- for heatmap: per-feature correlations --
    heatmap_best_bas = {f: {"ke": [], "fc": []} for f in FEATURE_NAMES}
    heatmap_gt = {f: {"ke": [], "fc": []} for f in FEATURE_NAMES}

    for wav_name, _ in AUDIO_TRACKS:
        track_id = wav_name.replace('.wav', '')
        audio_path = os.path.join(args.audio_dir, wav_name)
        audio_feats = extract_librosa_rhythmic_7(audio_path, target_fps=args.target_fps)
        print(f"  {track_id}: audio {audio_feats.shape[0]} frames")

        # -- gt motions --
        gt_motions = load_gt_for_track(track_id)
        for mot in gt_motions:
            mot_norm = (mot - mean) / std
            af, mn = align_and_trim(audio_feats, mot_norm)
            ke = compute_kinetic_energy(mn)
            fc = compute_foot_contact(mn)

            bas_gt.append(compute_bas(af[:, 1], ke))
            onset_ke_gt.append(safe_pearsonr(af[:, 0], ke))
            rms_fc_gt.append(safe_pearsonr(af[:, 4], fc))
            onset_ke_xcorr_gt.append(max_lag_xcorr(af[:, 0], ke)[0])
            rms_fc_xcorr_gt.append(max_lag_xcorr(af[:, 4], fc)[0])

            # -- heatmap gt --
            for fi, fname in enumerate(FEATURE_NAMES):
                heatmap_gt[fname]["ke"].append(safe_pearsonr(af[:, fi], ke))
                heatmap_gt[fname]["fc"].append(safe_pearsonr(af[:, fi], fc))

        # -- generated motions per cfg --
        for cfg_name, _, _ in CFG_CONFIGS:
            samples = load_generated_for_track(args.output_dir, cfg_name, track_id)
            for mot in samples:
                mot_norm = (mot - mean) / std
                af, mn = align_and_trim(audio_feats, mot_norm)
                ke = compute_kinetic_energy(mn)
                fc = compute_foot_contact(mn)

                bas_per_cfg[cfg_name].append(compute_bas(af[:, 1], ke))
                onset_ke_per_cfg[cfg_name].append(safe_pearsonr(af[:, 0], ke))
                rms_fc_per_cfg[cfg_name].append(safe_pearsonr(af[:, 4], fc))
                onset_ke_xcorr_per_cfg[cfg_name].append(max_lag_xcorr(af[:, 0], ke)[0])
                rms_fc_xcorr_per_cfg[cfg_name].append(max_lag_xcorr(af[:, 4], fc)[0])

                # -- heatmap for best_bas only --
                if cfg_name == "best_bas_2.5_1.5":
                    for fi, fname in enumerate(FEATURE_NAMES):
                        heatmap_best_bas[fname]["ke"].append(safe_pearsonr(af[:, fi], ke))
                        heatmap_best_bas[fname]["fc"].append(safe_pearsonr(af[:, fi], fc))

    return {
        "bas_per_cfg": {k: np.array(v) for k, v in bas_per_cfg.items()},
        "bas_gt": np.array(bas_gt),
        "onset_ke_per_cfg": {k: np.array(v) for k, v in onset_ke_per_cfg.items()},
        "onset_ke_gt": np.array(onset_ke_gt),
        "rms_fc_per_cfg": {k: np.array(v) for k, v in rms_fc_per_cfg.items()},
        "rms_fc_gt": np.array(rms_fc_gt),
        "onset_ke_xcorr_per_cfg": {k: np.array(v) for k, v in onset_ke_xcorr_per_cfg.items()},
        "onset_ke_xcorr_gt": np.array(onset_ke_xcorr_gt),
        "rms_fc_xcorr_per_cfg": {k: np.array(v) for k, v in rms_fc_xcorr_per_cfg.items()},
        "rms_fc_xcorr_gt": np.array(rms_fc_xcorr_gt),
        "heatmap_best_bas": heatmap_best_bas,
        "heatmap_gt": heatmap_gt,
    }


# ============================================================
# figure 1: beat alignment by cfg
# ============================================================

def plot_beat_alignment_by_cfg(results, fig_dir):
    fig, ax = plt.subplots(figsize=(10, 5))

    cfg_names = [c for c, _, _ in CFG_CONFIGS]
    means = [results["bas_per_cfg"][c].mean() for c in cfg_names]
    stds = [results["bas_per_cfg"][c].std() for c in cfg_names]
    colors = [CFG_COLORS[c] for c in cfg_names]
    labels = [CFG_LABELS[c] for c in cfg_names]

    bars = ax.bar(range(len(cfg_names)), means, yerr=stds, color=colors,
                  alpha=0.85, capsize=4, edgecolor='white', linewidth=0.5)

    # -- gt reference line --
    gt_mean = results["bas_gt"].mean()
    ax.axhline(gt_mean, color=GT_COLOR, linestyle='--', linewidth=2,
               label=f"AIST++ GT ({gt_mean:.3f})")

    # -- value annotations --
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.01, f"{m:.3f}", ha='center', fontsize=7)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel("beat alignment score (BAS)", fontsize=10)
    ax.set_title("beat alignment score by CFG configuration", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, None)

    plt.tight_layout()
    path = os.path.join(fig_dir, "beat_alignment_by_cfg.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved → {path}")


# ============================================================
# figure 2: onset-kinetic correlation
# ============================================================

def plot_onset_kinetic_correlation(results, fig_dir):
    fig, ax = plt.subplots(figsize=(10, 5))

    cfg_names = [c for c, _, _ in CFG_CONFIGS]
    means = [results["onset_ke_per_cfg"][c].mean() for c in cfg_names]
    stds = [results["onset_ke_per_cfg"][c].std() for c in cfg_names]
    colors = [CFG_COLORS[c] for c in cfg_names]
    labels = [CFG_LABELS[c] for c in cfg_names]

    # -- cfg bars --
    x = list(range(len(cfg_names)))
    ax.bar(x, means, yerr=stds, color=colors, alpha=0.85,
           capsize=4, edgecolor='white', linewidth=0.5)

    # -- gt bar --
    gt_m = results["onset_ke_gt"].mean()
    gt_s = results["onset_ke_gt"].std()
    ax.bar(len(x), gt_m, yerr=gt_s, color=GT_COLOR, alpha=0.85,
           capsize=4, edgecolor='white', linewidth=0.5)

    all_labels = labels + ["AIST++ GT"]
    for i, m in enumerate(means + [gt_m]):
        ax.text(i, m + (stds + [gt_s])[i] + 0.01, f"{m:.3f}",
                ha='center', fontsize=7)

    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel("Pearson r (onset × kinetic energy)", fontsize=10)
    ax.set_title("onset strength – kinetic energy correlation", fontsize=12)

    plt.tight_layout()
    path = os.path.join(fig_dir, "onset_kinetic_correlation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved → {path}")


# ============================================================
# figure 3: rms-footcontact correlation
# ============================================================

def plot_rms_footcontact_correlation(results, fig_dir):
    fig, ax = plt.subplots(figsize=(10, 5))

    cfg_names = [c for c, _, _ in CFG_CONFIGS]
    means = [results["rms_fc_per_cfg"][c].mean() for c in cfg_names]
    stds = [results["rms_fc_per_cfg"][c].std() for c in cfg_names]
    colors = [CFG_COLORS[c] for c in cfg_names]
    labels = [CFG_LABELS[c] for c in cfg_names]

    x = list(range(len(cfg_names)))
    ax.bar(x, means, yerr=stds, color=colors, alpha=0.85,
           capsize=4, edgecolor='white', linewidth=0.5)

    gt_m = results["rms_fc_gt"].mean()
    gt_s = results["rms_fc_gt"].std()
    ax.bar(len(x), gt_m, yerr=gt_s, color=GT_COLOR, alpha=0.85,
           capsize=4, edgecolor='white', linewidth=0.5)

    all_labels = labels + ["AIST++ GT"]
    for i, m in enumerate(means + [gt_m]):
        ax.text(i, m + (stds + [gt_s])[i] + 0.01, f"{m:.3f}",
                ha='center', fontsize=7)

    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel("Pearson r (RMS × foot contact)", fontsize=10)
    ax.set_title("RMS energy – foot contact correlation", fontsize=12)

    plt.tight_layout()
    path = os.path.join(fig_dir, "rms_footcontact_correlation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved → {path}")


# ============================================================
# figure 4: audio-motion heatmap
# ============================================================

def plot_audio_motion_heatmap(results, fig_dir):
    # -- build correlation matrices (7 x 2) --
    mat_best = np.zeros((len(FEATURE_NAMES), 2))
    mat_gt = np.zeros((len(FEATURE_NAMES), 2))

    for fi, fname in enumerate(FEATURE_NAMES):
        bb = results["heatmap_best_bas"][fname]
        gt = results["heatmap_gt"][fname]
        mat_best[fi, 0] = np.mean(bb["ke"]) if bb["ke"] else 0.0
        mat_best[fi, 1] = np.mean(bb["fc"]) if bb["fc"] else 0.0
        mat_gt[fi, 0] = np.mean(gt["ke"]) if gt["ke"] else 0.0
        mat_gt[fi, 1] = np.mean(gt["fc"]) if gt["fc"] else 0.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    col_labels = ["kinetic energy", "foot contact"]
    vmin, vmax = -1, 1

    for ax, mat, title in [
        (ax1, mat_best, "best-BAS (2.5/1.5)"),
        (ax2, mat_gt, "AIST++ GT"),
    ]:
        im = ax.imshow(mat, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(2))
        ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_yticks(range(len(FEATURE_NAMES)))
        ax.set_yticklabels(FEATURE_NAMES, fontsize=8)
        ax.set_title(title, fontsize=11)

        # -- annotate cells --
        for i in range(len(FEATURE_NAMES)):
            for j in range(2):
                val = mat[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                        fontsize=9, color=color, fontweight='bold')

    fig.colorbar(im, ax=[ax1, ax2], shrink=0.8, label="Pearson r")
    fig.suptitle("audio feature × motion feature correlations", fontsize=13, y=1.01)

    plt.tight_layout()
    path = os.path.join(fig_dir, "audio_motion_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved → {path}")


# ============================================================
# figure 5: onset-kinetic smoothed cross-correlation
# ============================================================

def plot_onset_kinetic_xcorr(results, fig_dir):
    fig, ax = plt.subplots(figsize=(10, 5))

    cfg_names = [c for c, _, _ in CFG_CONFIGS]
    means = [results["onset_ke_xcorr_per_cfg"][c].mean() for c in cfg_names]
    stds = [results["onset_ke_xcorr_per_cfg"][c].std() for c in cfg_names]
    colors = [CFG_COLORS[c] for c in cfg_names]
    labels = [CFG_LABELS[c] for c in cfg_names]

    x = list(range(len(cfg_names)))
    ax.bar(x, means, yerr=stds, color=colors, alpha=0.85,
           capsize=4, edgecolor='white', linewidth=0.5)

    gt_m = results["onset_ke_xcorr_gt"].mean()
    gt_s = results["onset_ke_xcorr_gt"].std()
    ax.bar(len(x), gt_m, yerr=gt_s, color=GT_COLOR, alpha=0.85,
           capsize=4, edgecolor='white', linewidth=0.5)

    all_labels = labels + ["AIST++ GT"]
    for i, m in enumerate(means + [gt_m]):
        ax.text(i, m + (stds + [gt_s])[i] + 0.005, f"{m:.3f}",
                ha='center', fontsize=7)

    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel("max cross-corr (smoothed, ±10 lag)", fontsize=10)
    ax.set_title("onset strength – kinetic energy (smoothed cross-correlation)", fontsize=12)

    plt.tight_layout()
    path = os.path.join(fig_dir, "onset_kinetic_xcorr.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved → {path}")


# ============================================================
# figure 6: rms-footcontact smoothed cross-correlation
# ============================================================

def plot_rms_footcontact_xcorr(results, fig_dir):
    fig, ax = plt.subplots(figsize=(10, 5))

    cfg_names = [c for c, _, _ in CFG_CONFIGS]
    means = [results["rms_fc_xcorr_per_cfg"][c].mean() for c in cfg_names]
    stds = [results["rms_fc_xcorr_per_cfg"][c].std() for c in cfg_names]
    colors = [CFG_COLORS[c] for c in cfg_names]
    labels = [CFG_LABELS[c] for c in cfg_names]

    x = list(range(len(cfg_names)))
    ax.bar(x, means, yerr=stds, color=colors, alpha=0.85,
           capsize=4, edgecolor='white', linewidth=0.5)

    gt_m = results["rms_fc_xcorr_gt"].mean()
    gt_s = results["rms_fc_xcorr_gt"].std()
    ax.bar(len(x), gt_m, yerr=gt_s, color=GT_COLOR, alpha=0.85,
           capsize=4, edgecolor='white', linewidth=0.5)

    all_labels = labels + ["AIST++ GT"]
    for i, m in enumerate(means + [gt_m]):
        ax.text(i, m + (stds + [gt_s])[i] + 0.005, f"{m:.3f}",
                ha='center', fontsize=7)

    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel("max cross-corr (smoothed, ±10 lag)", fontsize=10)
    ax.set_title("RMS energy – foot contact (smoothed cross-correlation)", fontsize=12)

    plt.tight_layout()
    path = os.path.join(fig_dir, "rms_footcontact_xcorr.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved → {path}")


# ============================================================
# figure 7: coupling migration across cfg configs
# ============================================================

# -- cfg ordering by increasing audio guidance strength --
CFG_ORDER = [
    "text_only_2.5_0.0",
    "best_bas_2.5_1.5",
    "balanced_2.5_2.5",
    "text_heavy_5.0_1.0",
    "audio_heavy_1.0_5.0",
    "high_both_5.0_5.0",
    "audio_only_0.0_5.0",
]


def plot_coupling_migration(results, fig_dir):
    metrics = [
        ("BAS", "bas_per_cfg", "bas_gt"),
        ("onset–KE xcorr", "onset_ke_xcorr_per_cfg", "onset_ke_xcorr_gt"),
        ("RMS–FC xcorr", "rms_fc_xcorr_per_cfg", "rms_fc_xcorr_gt"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("audio-motion coupling migration across CFG configs", fontsize=14, y=1.02)

    for ax, (title, cfg_key, gt_key) in zip(axes, metrics):
        xs, ys, errs, colors = [], [], [], []
        for i, cfg in enumerate(CFG_ORDER):
            arr = results[cfg_key][cfg]
            xs.append(i)
            ys.append(arr.mean())
            errs.append(arr.std())
            colors.append(CFG_COLORS[cfg])

        # -- gt at the end --
        gt_arr = results[gt_key]
        gt_x = len(CFG_ORDER)
        gt_m = gt_arr.mean()
        gt_s = gt_arr.std()

        # -- connected line through configs --
        ax.plot(xs, ys, color='#555555', linewidth=1, alpha=0.5, zorder=1)

        # -- scatter with error bars for each config --
        for i in range(len(xs)):
            ax.errorbar(xs[i], ys[i], yerr=errs[i], fmt='o', color=colors[i],
                        markersize=8, capsize=4, capthick=1.5, zorder=2,
                        markeredgecolor='white', markeredgewidth=0.5)

        # -- gt star --
        ax.errorbar(gt_x, gt_m, yerr=gt_s, fmt='*', color=GT_COLOR,
                    markersize=14, capsize=4, capthick=1.5, zorder=3)
        ax.axhline(gt_m, color=GT_COLOR, linestyle='--', linewidth=1, alpha=0.5)

        # -- labels --
        tick_labels = [CFG_LABELS[c].split(' (')[0] for c in CFG_ORDER] + ["GT"]
        ax.set_xticks(list(range(len(CFG_ORDER) + 1)))
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=7)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(title, fontsize=9)

    plt.tight_layout()
    path = os.path.join(fig_dir, "coupling_migration.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved → {path}")


# ============================================================
# cli
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='audio-motion bridge analysis across CFG configurations')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='base output dir with generated/ subfolder')
    parser.add_argument('--model_tag', type=str, required=True,
                        help='model tag (for display)')
    parser.add_argument('--audio_dir', type=str, default='dataset/aist/audio',
                        help='directory containing .wav files')
    parser.add_argument('--fig_dir', type=str, default='figures/audio_motion_bridge',
                        help='directory for output figures')
    parser.add_argument('--target_fps', type=int, default=20,
                        help='frames per second for audio feature extraction')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    print("running audio-motion bridge analysis...")
    results = run_analysis(args)

    print("\nplotting beat alignment...")
    plot_beat_alignment_by_cfg(results, args.fig_dir)

    print("plotting onset-kinetic correlation...")
    plot_onset_kinetic_correlation(results, args.fig_dir)

    print("plotting rms-footcontact correlation...")
    plot_rms_footcontact_correlation(results, args.fig_dir)

    print("plotting audio-motion heatmap...")
    plot_audio_motion_heatmap(results, args.fig_dir)

    print("plotting onset-kinetic cross-correlation...")
    plot_onset_kinetic_xcorr(results, args.fig_dir)

    print("plotting rms-footcontact cross-correlation...")
    plot_rms_footcontact_xcorr(results, args.fig_dir)

    print("plotting coupling migration...")
    plot_coupling_migration(results, args.fig_dir)

    print("\ndone.")


if __name__ == '__main__':
    main()

# [end]
# [end]
