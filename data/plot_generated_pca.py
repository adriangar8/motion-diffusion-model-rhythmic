# [start]
# [start]

import os
import sys
# -- ensure project root is on sys.path --
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import argparse
import numpy as np
import torch
from glob import glob
from copy import deepcopy
from types import SimpleNamespace
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.spatial.distance import pdist

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -- reproducibility --
RNG = np.random.default_rng(42)
N_SUBSAMPLE = 50_000

# -- paths --
HUMANML_DIR = "dataset/HumanML3D/humanml/new_joint_vecs"
AIST_DIR = "dataset/aist/processed/motions_263"
MEAN_PATH = "dataset/HumanML3D/humanml/Mean.npy"
STD_PATH = "dataset/HumanML3D/humanml/Std.npy"
AUDIO_DIR = "dataset/aist/audio"
FIG_DIR = "figures"

# -- 263-dim hml-vec feature groups --
FEATURE_GROUPS = {
    "Root (4d)":             slice(0, 4),
    "Joint pos (63d)":       slice(4, 67),
    "Joint rot (126d)":      slice(67, 193),
    "Joint vel (66d)":       slice(193, 259),
    "Foot contact (4d)":     slice(259, 263),
}

# -- short names for feature groups (used in filenames) --
GROUP_SHORT_NAMES = {
    "Root (4d)":         "root",
    "Joint pos (63d)":   "joint_pos",
    "Joint rot (126d)":  "joint_rot",
    "Joint vel (66d)":   "joint_vel",
    "Foot contact (4d)": "foot_contact",
}

# -- cfg sweep configs --
CFG_CONFIGS = [
    ("balanced_2.5_2.5",    2.5, 2.5),
    ("text_heavy_5.0_1.0",  5.0, 1.0),
    ("audio_heavy_1.0_5.0", 1.0, 5.0),
    ("audio_only_0.0_5.0",  0.0, 5.0),
    ("high_both_5.0_5.0",   5.0, 5.0),
    ("best_bas_2.5_1.5",    2.5, 1.5),
    ("text_only_2.5_0.0",   2.5, 0.0),
]

# -- audio tracks: (wav filename, text prompt) --
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

# -- colors --
GT_COLORS = {
    "HumanML3D (GT)": "#2196F3",
    "AIST++ (GT)":    "#F44336",
}

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


# ============================================================
# helpers (adapted from data/plot_dataset_distributions.py)
# ============================================================

def load_frames(motion_dir):
    # -- stack all (T, 263) arrays from directory --
    files = sorted(glob(os.path.join(motion_dir, "*.npy")))
    if not files:
        raise FileNotFoundError(f"no .npy files found in {motion_dir}")
    arrays = [np.load(f).astype(np.float32) for f in files]
    arrays = [a if a.ndim == 2 else a[None] for a in arrays]
    return np.concatenate(arrays, axis=0)


def subsample(frames, n):
    if len(frames) <= n:
        return frames
    idx = RNG.choice(len(frames), size=n, replace=False)
    return frames[idx]


def add_kde_contours(ax, xy, color, levels=5):
    x, y = xy[:, 0], xy[:, 1]
    h, xedges, yedges = np.histogram2d(x, y, bins=100)
    h = h.T
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    h_smooth = gaussian_filter(h, sigma=2)
    ax.contour(xc, yc, h_smooth, levels=levels, colors=[color],
               linewidths=0.8, alpha=0.7)


def symmetric_kl(mu1, cov1, mu2, cov2):
    d = mu1.shape[0]
    cov1_inv = np.linalg.inv(cov1)
    cov2_inv = np.linalg.inv(cov2)
    diff = mu1 - mu2
    kl_12 = 0.5 * (np.trace(cov2_inv @ cov1) + diff @ cov2_inv @ diff
                    - d + np.log(np.linalg.det(cov2) / np.linalg.det(cov1)))
    kl_21 = 0.5 * (np.trace(cov1_inv @ cov2) + diff @ cov1_inv @ diff
                    - d + np.log(np.linalg.det(cov1) / np.linalg.det(cov2)))
    return 0.5 * (kl_12 + kl_21)


def rbf_mmd(X, Y, sigma):
    def rbf(A, B):
        dists_sq = np.sum(A**2, axis=1, keepdims=True) + np.sum(B**2, axis=1) - 2 * A @ B.T
        return np.exp(-dists_sq / (2 * sigma**2))
    xx = rbf(X, X).mean()
    yy = rbf(Y, Y).mean()
    xy = rbf(X, Y).mean()
    return xx + yy - 2 * xy


def get_beat_index(audio_feat_dim):
    # -- beat indicator index depends on feature dim --
    if audio_feat_dim >= 519:
        return 513
    elif audio_feat_dim == 52:
        return 34
    else:
        return 129


def extract_audio_for_model(audio_path, ckpt_args, device, duration=None):
    # -- auto-detect feature extractor from model config --
    feat_dim = ckpt_args.get('audio_feat_dim', 145)
    use_wav2clip = ckpt_args.get('use_wav2clip', False) or feat_dim == 519

    if use_wav2clip:
        from model.audio_features_wav2clip import extract_wav2clip_plus_librosa
        return extract_wav2clip_plus_librosa(audio_path, target_fps=20,
                                             duration=duration, device=device)
    elif feat_dim == 52:
        from model.audio_features_v2 import extract_audio_features_v2
        return extract_audio_features_v2(audio_path, target_fps=20,
                                          duration=duration)
    else:
        from model.audio_features import extract_audio_features
        return extract_audio_features(audio_path, target_fps=20,
                                       duration=duration)


# ============================================================
# generation
# ============================================================

def run_generate(args):

    from sample.generate_audio import load_model, AudioCFGSampleModel
    from utils.model_util import create_gaussian_diffusion

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # pyright: ignore[reportAttributeAccessIssue]

    # -- seed --
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -- load model once --
    print(f"loading model: {args.model_path}")
    model, ckpt_args = load_model(args.model_path, device)

    # -- override humanml_dir for normalization stats --
    ckpt_args['humanml_dir'] = './dataset/HumanML3D'

    feat_dim = ckpt_args.get('audio_feat_dim', 145)
    beat_idx = get_beat_index(feat_dim)
    print(f"audio feature dim: {feat_dim}, beat index: {beat_idx}")

    # -- create diffusion --
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

    # -- load normalization stats --
    mean = np.load(MEAN_PATH).astype(np.float32)
    std = np.load(STD_PATH).astype(np.float32)

    n_frames = 196

    # -- sweep configs --
    for cfg_name, text_scale, audio_scale in CFG_CONFIGS:

        cfg_dir = os.path.join(args.output_dir, "generated", cfg_name)
        os.makedirs(cfg_dir, exist_ok=True)

        # -- check if already done --
        expected = len(AUDIO_TRACKS) * args.num_samples_per_track
        existing = len(glob(os.path.join(cfg_dir, "sample_*.npy")))
        if existing >= expected and not args.force:
            print(f"\n[{cfg_name}] already has {existing}/{expected} samples, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"config: {cfg_name}  (text={text_scale}, audio={audio_scale})")
        print(f"{'='*60}")

        cfg_model = AudioCFGSampleModel(model, text_scale=text_scale,
                                         audio_scale=audio_scale)

        all_samples = []

        for wav_name, text_prompt in AUDIO_TRACKS:

            audio_path = os.path.join(AUDIO_DIR, wav_name)
            if not os.path.exists(audio_path):
                print(f"  skipping {wav_name}: file not found")
                continue

            # -- extract audio features --
            audio_feat = extract_audio_for_model(audio_path, ckpt_args,
                                                  str(device))
            audio_feat = audio_feat[:n_frames]
            T = audio_feat.shape[0]

            audio_tensor = torch.from_numpy(audio_feat).float().unsqueeze(0)
            audio_tensor = audio_tensor.repeat(args.num_samples_per_track, 1, 1).to(device)

            # -- beat frames --
            beat_frames = list(np.where(audio_feat[:, beat_idx] > 0.5)[0])

            # -- build model_kwargs --
            model_kwargs = {
                'y': {
                    'text': [text_prompt] * args.num_samples_per_track,
                    'mask': torch.ones(args.num_samples_per_track, 1, 1, T,
                                       dtype=torch.bool).to(device),
                    'lengths': torch.tensor([T] * args.num_samples_per_track).to(device),
                    'scale': torch.tensor([text_scale] * args.num_samples_per_track).to(device),
                    'audio_features': audio_tensor,
                    'beat_frames': beat_frames,
                }
            }

            sample_shape = (args.num_samples_per_track, 263, 1, T)

            with torch.no_grad():
                sample = diffusion.p_sample_loop(
                    cfg_model, sample_shape,
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
            sample = sample.squeeze(2).permute(0, 2, 1).cpu().numpy()  # (B, T, 263)  # pyright: ignore[reportAttributeAccessIssue]
            sample = sample * std + mean

            # -- smooth --
            for i in range(sample.shape[0]):
                sample[i] = gaussian_filter1d(sample[i], sigma=1.0, axis=0)

            # -- save per-sample files --
            track_id = wav_name.replace('.wav', '')
            for i in range(sample.shape[0]):
                out_path = os.path.join(cfg_dir, f"sample_{track_id}_{i:02d}.npy")
                np.save(out_path, sample[i].astype(np.float32))

            all_samples.append(sample)

            print(f"  {track_id}: {sample.shape[0]} samples × {sample.shape[1]} frames")

        # -- save meta --
        meta = {
            'cfg_name': cfg_name,
            'text_scale': text_scale,
            'audio_scale': audio_scale,
            'model_path': args.model_path,
            'model_tag': args.model_tag,
            'num_samples_per_track': args.num_samples_per_track,
            'audio_feat_dim': feat_dim,
            'n_frames': n_frames,
            'seed': args.seed,
        }
        with open(os.path.join(cfg_dir, "meta.json"), 'w') as f:
            json.dump(meta, f, indent=2)

    # -- save concatenated frames for each config --
    all_frames_dir = os.path.join(args.output_dir, "all_frames")
    os.makedirs(all_frames_dir, exist_ok=True)

    for cfg_name, _, _ in CFG_CONFIGS:
        cfg_dir = os.path.join(args.output_dir, "generated", cfg_name)
        files = sorted(glob(os.path.join(cfg_dir, "sample_*.npy")))
        if files:
            frames = np.concatenate([np.load(f) for f in files], axis=0)
            np.save(os.path.join(all_frames_dir, f"{cfg_name}.npy"), frames)
            print(f"saved all_frames/{cfg_name}.npy: {frames.shape}")

    # -- save gt subsampled frames --
    print("\nsaving GT frames...")
    mean_arr = np.load(MEAN_PATH).astype(np.float32)
    std_arr = np.load(STD_PATH).astype(np.float32)

    humanml_frames = load_frames(HUMANML_DIR)
    aist_frames = load_frames(AIST_DIR)

    humanml_sub = subsample(humanml_frames, N_SUBSAMPLE)
    aist_sub = subsample(aist_frames, N_SUBSAMPLE)

    np.save(os.path.join(all_frames_dir, "humanml3d_gt.npy"), humanml_sub)
    np.save(os.path.join(all_frames_dir, "aist_gt.npy"), aist_sub)
    print(f"  humanml3d_gt: {humanml_sub.shape}, aist_gt: {aist_sub.shape}")

    # -- manifest --
    manifest = {
        'model_tag': args.model_tag,
        'model_path': args.model_path,
        'audio_feat_dim': feat_dim,
        'configs': {name: {'text_scale': ts, 'audio_scale': as_}
                    for name, ts, as_ in CFG_CONFIGS},
        'audio_tracks': [t[0] for t in AUDIO_TRACKS],
        'num_samples_per_track': args.num_samples_per_track,
    }
    with open(os.path.join(all_frames_dir, "manifest.json"), 'w') as f:
        json.dump(manifest, f, indent=2)

    print("\ngeneration complete!")


# ============================================================
# plotting
# ============================================================

def run_plot(args):

    all_frames_dir = os.path.join(args.output_dir, "all_frames")

    # -- per-model figure directory --
    fig_dir = os.path.join(FIG_DIR, args.model_tag)
    os.makedirs(fig_dir, exist_ok=True)

    # -- load normalization stats --
    mean = np.load(MEAN_PATH).astype(np.float32)
    std = np.load(STD_PATH).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)

    # -- load gt --
    print("loading ground-truth frames...")
    humanml_raw = load_frames(HUMANML_DIR)
    aist_raw = load_frames(AIST_DIR)

    gt_normed = {
        "HumanML3D (GT)": (subsample(humanml_raw, N_SUBSAMPLE) - mean) / std,
        "AIST++ (GT)":    (subsample(aist_raw, N_SUBSAMPLE) - mean) / std,
    }

    for name, f in gt_normed.items():
        print(f"  {name}: {len(f):,} frames")

    # -- load generated --
    gen_normed = {}
    for cfg_name, _, _ in CFG_CONFIGS:
        path = os.path.join(all_frames_dir, f"{cfg_name}.npy")
        if os.path.exists(path):
            raw = np.load(path).astype(np.float32)
            gen_normed[cfg_name] = (raw - mean) / std
            print(f"  {CFG_LABELS[cfg_name]}: {len(raw):,} frames")
        else:
            print(f"  {cfg_name}: NOT FOUND, skipping")

    if not gen_normed:
        print("no generated data found! run --mode generate first.")
        return

    # -- 1. main pca overlay --
    plot_pca_overlay(gt_normed, gen_normed, args.model_tag, fig_dir)

    # -- 2. per-config side-by-side --
    plot_pca_per_config(gt_normed, gen_normed, args.model_tag, fig_dir)

    # -- 3. feature-group pca (1×5 row) --
    plot_feature_group_pca(gt_normed, gen_normed, args.model_tag, fig_dir)

    # -- 4. domain gap heatmaps --
    if not args.skip_domain_gap:
        plot_domain_gap(gt_normed, gen_normed, args.model_tag, fig_dir)

    # -- 5. feature group × config grid (5×5) --
    plot_feature_group_config_grid(gt_normed, gen_normed, args.model_tag, fig_dir)

    # -- 6. individual feature group × config figures --
    plot_feature_group_individual(gt_normed, gen_normed, args.model_tag, fig_dir)

    # -- 7. centroid distances bar chart --
    plot_centroid_distances(gt_normed, gen_normed, args.model_tag, fig_dir)

    # -- 8. variance ratio bar chart --
    plot_variance_ratio(gt_normed, gen_normed, args.model_tag, fig_dir)

    # -- 9. pc1/pc2 histograms --
    plot_pc_histograms(gt_normed, gen_normed, args.model_tag, fig_dir)

    # -- 10. marginal violin plots --
    plot_marginal_violins(gt_normed, gen_normed, args.model_tag, fig_dir)

    # -- 11. centroid migration arrows --
    plot_centroid_migration(gt_normed, gen_normed, args.model_tag, fig_dir)

    # -- 12. scatter + marginal kde (all configs) --
    plot_scatter_marginal_kde(gt_normed, gen_normed, args.model_tag, fig_dir)

    # -- 13. scatter + marginal kde (per config, side by side) --
    plot_scatter_marginal_per_config(gt_normed, gen_normed, args.model_tag, fig_dir)

    # -- 14-15. audio conditioning impact (needs text_only baseline) --
    plot_audio_conditioning_impact(gt_normed, gen_normed, args.model_tag, fig_dir)

    # -- 16-17. original-dimension metrics per feature group --
    plot_feature_group_metrics_original_dims(gt_normed, gen_normed, args.model_tag, fig_dir)

    print("\nplotting complete!")


# ============================================================
# existing plots (updated: fig_dir param, overlap fixes)
# ============================================================

def plot_pca_overlay(gt, gen, model_tag, fig_dir):

    # -- fit pca on gt only --
    gt_combined = np.concatenate(list(gt.values()), axis=0)
    print(f"\nfitting PCA on {len(gt_combined):,} GT frames...")
    pca = PCA(n_components=2).fit(gt_combined)
    var = pca.explained_variance_ratio_ * 100
    print(f"  PC1: {var[0]:.1f}%, PC2: {var[1]:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 8))

    # -- plot gt as background --
    for name, frames in gt.items():
        xy = pca.transform(frames)
        c = GT_COLORS[name]
        ax.scatter(xy[:, 0], xy[:, 1], c=c, s=1, alpha=0.05, rasterized=True)
        add_kde_contours(ax, xy, color=c, levels=5)

    # -- plot generated as foreground --
    for cfg_name, frames in gen.items():
        xy = pca.transform(frames)
        c = CFG_COLORS[cfg_name]
        ax.scatter(xy[:, 0], xy[:, 1], c=c, s=2, alpha=0.2, rasterized=True)
        add_kde_contours(ax, xy, color=c, levels=4)

    ax.set_xlabel(f"PC1 ({var[0]:.1f}% var)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% var)", fontsize=12)
    ax.set_title(f"Generated Motion PCA — {model_tag}", fontsize=14, pad=15)

    # -- legend --
    handles = []
    for name in gt:
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=GT_COLORS[name], markersize=8, label=name))
    for cfg_name in gen:
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=CFG_COLORS[cfg_name], markersize=8,
                       label=CFG_LABELS[cfg_name]))
    ax.legend(handles=handles, fontsize=9, loc='upper right')

    plt.tight_layout()
    path = os.path.join(fig_dir, "pca_overlay.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved → {path}")


def plot_pca_per_config(gt, gen, model_tag, fig_dir):

    # -- fit pca on gt only (same basis as overlay) --
    gt_combined = np.concatenate(list(gt.values()), axis=0)
    pca = PCA(n_components=2).fit(gt_combined)
    var = pca.explained_variance_ratio_ * 100

    # -- pre-project gt --
    gt_xy = {name: pca.transform(frames) for name, frames in gt.items()}

    # -- shared axis limits from gt --
    all_gt = np.concatenate(list(gt_xy.values()), axis=0)
    pad = 5
    xlim = (np.percentile(all_gt[:, 0], 1) - pad, np.percentile(all_gt[:, 0], 99) + pad)
    ylim = (np.percentile(all_gt[:, 1], 1) - pad, np.percentile(all_gt[:, 1], 99) + pad)

    n_cfgs = len(gen)
    fig, axes = plt.subplots(1, n_cfgs, figsize=(5 * n_cfgs, 6))
    if n_cfgs == 1:
        axes = [axes]

    for ax, (cfg_name, frames) in zip(axes, gen.items()):

        # -- gt background --
        for name, xy in gt_xy.items():
            c = GT_COLORS[name]
            ax.scatter(xy[:, 0], xy[:, 1], c=c, s=0.5, alpha=0.03, rasterized=True)
            add_kde_contours(ax, xy, color=c, levels=4)

        # -- generated foreground --
        xy = pca.transform(frames)
        c = CFG_COLORS[cfg_name]
        ax.scatter(xy[:, 0], xy[:, 1], c=c, s=3, alpha=0.3, rasterized=True)
        add_kde_contours(ax, xy, color=c, levels=4)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(CFG_LABELS[cfg_name], fontsize=11, fontweight='bold')
        ax.set_xlabel(f"PC1 ({var[0]:.1f}%)", fontsize=9)
        ax.set_ylabel(f"PC2 ({var[1]:.1f}%)", fontsize=9)
        ax.tick_params(labelsize=7)

    # -- legend on first axis --
    handles = []
    for name in gt:
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=GT_COLORS[name], markersize=6, label=name))
    handles.append(plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#333333', markersize=6, label='generated'))
    axes[0].legend(handles=handles, fontsize=7, loc='upper right')

    fig.suptitle(f"PCA per CFG Config — {model_tag}", fontsize=14, y=0.98)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    path = os.path.join(fig_dir, "pca_per_config.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved → {path}")


def plot_feature_group_pca(gt, gen, model_tag, fig_dir):

    n_groups = len(FEATURE_GROUPS)
    fig, axes = plt.subplots(1, n_groups, figsize=(4 * n_groups, 5))

    all_data = {**gt, **{CFG_LABELS[k]: v for k, v in gen.items()}}
    all_colors = {**GT_COLORS, **{CFG_LABELS[k]: v for k, v in CFG_COLORS.items()}}

    for ax, (gname, sl) in zip(axes, FEATURE_GROUPS.items()):
        # -- slice feature group, fit pca on gt only --
        gt_sliced = np.concatenate([f[:, sl] for f in gt.values()], axis=0)
        pca = PCA(n_components=2).fit(gt_sliced)
        var = pca.explained_variance_ratio_ * 100

        for name, frames in all_data.items():
            xy = pca.transform(frames[:, sl])
            c = all_colors[name]
            is_gt = name in GT_COLORS
            ax.scatter(xy[:, 0], xy[:, 1], c=c, s=0.5,
                       alpha=0.05 if is_gt else 0.15, rasterized=True)
            add_kde_contours(ax, xy, color=c, levels=4)

        ax.set_title(gname, fontsize=10)
        ax.set_xlabel(f"PC1 ({var[0]:.1f}%)", fontsize=8)
        ax.set_ylabel(f"PC2 ({var[1]:.1f}%)", fontsize=8)
        ax.tick_params(labelsize=7)

    # -- legend on last axis --
    handles = [plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=all_colors[n], markersize=5, label=n)
               for n in all_data]
    axes[-1].legend(handles=handles, fontsize=5, loc='upper right')

    fig.suptitle(f"PCA by Feature Group — {model_tag}", fontsize=13, y=0.98)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    path = os.path.join(fig_dir, "pca_by_feature_group.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved → {path}")


def plot_domain_gap(gt, gen, model_tag, fig_dir):

    all_data = {**gt, **{CFG_LABELS[k]: v for k, v in gen.items()}}
    names = list(all_data.keys())
    n = len(names)

    # -- reduce to pca-50 --
    combined = np.concatenate(list(all_data.values()), axis=0)
    n_comp = min(50, combined.shape[1])
    pca = PCA(n_components=n_comp).fit(combined)
    reduced = {name: pca.transform(frames) for name, frames in all_data.items()}
    print(f"\ndomain gap (PCA-{n_comp}, {pca.explained_variance_ratio_.sum()*100:.1f}% var):")

    # -- fit gaussians --
    stats = {}
    for name, X in reduced.items():
        stats[name] = (X.mean(axis=0), np.cov(X, rowvar=False))

    # -- rbf bandwidth --
    sample = combined[RNG.choice(len(combined), size=min(5000, len(combined)), replace=False)]
    sample_r = pca.transform(sample)
    sigma = np.median(pdist(sample_r))

    # -- pairwise metrics --
    kl_mat = np.zeros((n, n))
    mmd_mat = np.zeros((n, n))
    mmd_n = 5000

    for i in range(n):
        for j in range(i + 1, n):
            kl_val = symmetric_kl(stats[names[i]][0], stats[names[i]][1],
                                   stats[names[j]][0], stats[names[j]][1])
            kl_mat[i, j] = kl_mat[j, i] = kl_val

            Xi = reduced[names[i]]
            Xj = reduced[names[j]]
            Xi_s = Xi[RNG.choice(len(Xi), size=min(mmd_n, len(Xi)), replace=False)]
            Xj_s = Xj[RNG.choice(len(Xj), size=min(mmd_n, len(Xj)), replace=False)]
            mmd_val = rbf_mmd(Xi_s, Xj_s, sigma)
            mmd_mat[i, j] = mmd_mat[j, i] = mmd_val

    # -- print table --
    print("\n  symmetric KL divergence:")
    print(f"  {'':>25s}  {'  '.join(f'{n:>25s}' for n in names)}")
    for i, ni in enumerate(names):
        row = "  ".join(f"{kl_mat[i, j]:25.1f}" for j in range(n))
        print(f"  {ni:>25s}  {row}")

    print(f"\n  MMD² (RBF, σ={sigma:.1f}):")
    print(f"  {'':>25s}  {'  '.join(f'{n:>25s}' for n in names)}")
    for i, ni in enumerate(names):
        row = "  ".join(f"{mmd_mat[i, j]:25.6f}" for j in range(n))
        print(f"  {ni:>25s}  {row}")

    # -- heatmap figure --
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    for ax, mat, title, fmt in [
        (ax1, kl_mat, "Symmetric KL Divergence", ".1f"),
        (ax2, mmd_mat, "MMD² (RBF)", ".4f"),
    ]:
        im = ax.imshow(mat, cmap="YlOrRd")
        ax.set_xticks(range(n))
        ax.set_xticklabels(names, fontsize=7, rotation=45, ha='right')
        ax.set_yticks(range(n))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_title(title, fontsize=11)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{mat[i, j]:{fmt}}", ha="center",
                        va="center", fontsize=6)
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Domain Gap — {model_tag}", fontsize=13, y=0.98)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    path = os.path.join(fig_dir, "domain_gap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nsaved → {path}")


# ============================================================
# new plots
# ============================================================

def plot_feature_group_config_grid(gt, gen, model_tag, fig_dir):

    n_groups = len(FEATURE_GROUPS)
    n_cfgs = len(gen)

    # -- subsample gt for memory (25 scatter subplots) --
    gt_sub = {name: subsample(frames, 10_000) for name, frames in gt.items()}

    fig, axes = plt.subplots(n_groups, n_cfgs, figsize=(4 * n_cfgs, 4 * n_groups))

    for row, (gname, sl) in enumerate(FEATURE_GROUPS.items()):

        # -- fit pca for this feature group on gt --
        gt_sliced = np.concatenate([f[:, sl] for f in gt_sub.values()], axis=0)
        pca = PCA(n_components=2).fit(gt_sliced)
        var = pca.explained_variance_ratio_ * 100

        # -- shared axis limits for row --
        gt_xy_all = pca.transform(gt_sliced)
        pad = 3
        xlim = (np.percentile(gt_xy_all[:, 0], 1) - pad,
                np.percentile(gt_xy_all[:, 0], 99) + pad)
        ylim = (np.percentile(gt_xy_all[:, 1], 1) - pad,
                np.percentile(gt_xy_all[:, 1], 99) + pad)

        for col, (cfg_name, gen_frames) in enumerate(gen.items()):
            ax = axes[row, col]

            # -- gt background --
            for name, frames in gt_sub.items():
                xy = pca.transform(frames[:, sl])
                c = GT_COLORS[name]
                ax.scatter(xy[:, 0], xy[:, 1], c=c, s=0.3, alpha=0.03, rasterized=True)
                add_kde_contours(ax, xy, color=c, levels=3)

            # -- generated foreground --
            xy = pca.transform(gen_frames[:, sl])
            c = CFG_COLORS[cfg_name]
            ax.scatter(xy[:, 0], xy[:, 1], c=c, s=1.5, alpha=0.3, rasterized=True)
            add_kde_contours(ax, xy, color=c, levels=3)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.tick_params(labelsize=5)

            # -- row labels on left column --
            if col == 0:
                ax.set_ylabel(gname, fontsize=9, fontweight='bold')
            else:
                ax.set_ylabel("")

            # -- column headers on top row --
            if row == 0:
                ax.set_title(CFG_LABELS[cfg_name], fontsize=9, fontweight='bold')

            # -- axis labels only on edges --
            if row == n_groups - 1:
                ax.set_xlabel(f"PC1 ({var[0]:.1f}%)", fontsize=7)
            else:
                ax.set_xlabel("")

    fig.suptitle(f"Feature Group × CFG Config — {model_tag}", fontsize=14)
    plt.tight_layout()
    fig.subplots_adjust(top=0.95)
    path = os.path.join(fig_dir, "feature_group_config_grid.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved → {path}")


def plot_feature_group_individual(gt, gen, model_tag, fig_dir):

    for cfg_name, gen_frames in gen.items():
        cfg_subdir = os.path.join(fig_dir, "feature_groups", cfg_name)
        os.makedirs(cfg_subdir, exist_ok=True)

        for gname, sl in FEATURE_GROUPS.items():
            short = GROUP_SHORT_NAMES[gname]

            # -- fit pca on gt --
            gt_sliced = np.concatenate([f[:, sl] for f in gt.values()], axis=0)
            pca = PCA(n_components=2).fit(gt_sliced)
            var = pca.explained_variance_ratio_ * 100

            fig, ax = plt.subplots(figsize=(8, 7))

            # -- gt background --
            for name, frames in gt.items():
                xy = pca.transform(frames[:, sl])
                c = GT_COLORS[name]
                ax.scatter(xy[:, 0], xy[:, 1], c=c, s=1, alpha=0.05, rasterized=True)
                add_kde_contours(ax, xy, color=c, levels=5)

            # -- generated foreground --
            xy = pca.transform(gen_frames[:, sl])
            c = CFG_COLORS[cfg_name]
            ax.scatter(xy[:, 0], xy[:, 1], c=c, s=3, alpha=0.3, rasterized=True)
            add_kde_contours(ax, xy, color=c, levels=4)

            ax.set_xlabel(f"PC1 ({var[0]:.1f}% var)", fontsize=11)
            ax.set_ylabel(f"PC2 ({var[1]:.1f}% var)", fontsize=11)
            ax.set_title(f"{gname} — {CFG_LABELS[cfg_name]}", fontsize=13)

            # -- legend --
            handles = []
            for name in gt:
                handles.append(plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=GT_COLORS[name], markersize=7, label=name))
            handles.append(plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=c, markersize=7,
                           label=CFG_LABELS[cfg_name]))
            ax.legend(handles=handles, fontsize=8, loc='upper right')

            plt.tight_layout()
            path = os.path.join(cfg_subdir, f"{short}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)

    print(f"saved → {fig_dir}/feature_groups/ (25 figures)")


def plot_centroid_distances(gt, gen, model_tag, fig_dir):

    # -- reduce all to pca-50 --
    all_data = {**gt, **{CFG_LABELS[k]: v for k, v in gen.items()}}
    combined = np.concatenate(list(all_data.values()), axis=0)
    n_comp = min(50, combined.shape[1])
    pca = PCA(n_components=n_comp).fit(combined)
    reduced = {name: pca.transform(frames) for name, frames in all_data.items()}

    # -- compute centroids --
    centroids = {name: X.mean(axis=0) for name, X in reduced.items()}
    humanml_c = centroids["HumanML3D (GT)"]
    aist_c = centroids["AIST++ (GT)"]

    # -- distances for each config --
    cfg_names = list(gen.keys())
    cfg_labels = [CFG_LABELS[k] for k in cfg_names]
    dist_humanml = [np.linalg.norm(centroids[CFG_LABELS[k]] - humanml_c) for k in cfg_names]
    dist_aist = [np.linalg.norm(centroids[CFG_LABELS[k]] - aist_c) for k in cfg_names]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(cfg_names))
    w = 0.35
    bars1 = ax.bar(x - w/2, dist_humanml, w, label='dist to HumanML3D',
                   color=GT_COLORS["HumanML3D (GT)"], alpha=0.8)
    bars2 = ax.bar(x + w/2, dist_aist, w, label='dist to AIST++',
                   color=GT_COLORS["AIST++ (GT)"], alpha=0.8)

    # -- value annotations --
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(cfg_labels, fontsize=9, rotation=15, ha='right')
    ax.set_ylabel("L2 distance (PCA-50 space)", fontsize=11)
    ax.set_title(f"Centroid Distance to GT — {model_tag}", fontsize=13)
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(fig_dir, "centroid_distances.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved → {path}")


def plot_variance_ratio(gt, gen, model_tag, fig_dir):

    aist_frames = gt["AIST++ (GT)"]

    # -- compute variance ratios per feature group + overall --
    groups_plus = list(FEATURE_GROUPS.items()) + [("Overall", slice(0, 263))]
    group_labels = [g[0] for g in groups_plus]

    cfg_names = list(gen.keys())
    n_groups = len(groups_plus)
    n_cfgs = len(cfg_names)

    ratios = np.zeros((n_cfgs, n_groups))

    for gi, (gname, sl) in enumerate(groups_plus):
        aist_var = np.var(aist_frames[:, sl], axis=0).sum()
        for ci, cfg_name in enumerate(cfg_names):
            gen_var = np.var(gen[cfg_name][:, sl], axis=0).sum()
            ratios[ci, gi] = gen_var / aist_var if aist_var > 1e-8 else 0.0

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(n_groups)
    w = 0.8 / n_cfgs

    for ci, cfg_name in enumerate(cfg_names):
        offset = (ci - n_cfgs / 2 + 0.5) * w
        ax.bar(x + offset, ratios[ci], w, label=CFG_LABELS[cfg_name],
               color=CFG_COLORS[cfg_name], alpha=0.85)

    ax.axhline(y=1.0, color='#F44336', linestyle='--', linewidth=1.5,
               alpha=0.7, label='AIST++ GT (=1.0)')
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_ylabel("variance ratio (gen / AIST++ GT)", fontsize=11)
    ax.set_title(f"Variance Ratio by Feature Group — {model_tag}", fontsize=13)
    ax.legend(fontsize=7, ncol=3, loc='upper right')

    plt.tight_layout()
    path = os.path.join(fig_dir, "variance_ratio.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved → {path}")


def plot_pc_histograms(gt, gen, model_tag, fig_dir):

    # -- fit pca on gt --
    gt_combined = np.concatenate(list(gt.values()), axis=0)
    pca = PCA(n_components=2).fit(gt_combined)
    var = pca.explained_variance_ratio_ * 100

    # -- project everything --
    gt_proj = {name: pca.transform(frames) for name, frames in gt.items()}
    gen_proj = {cfg_name: pca.transform(frames) for cfg_name, frames in gen.items()}

    n_cfgs = len(gen)
    fig, axes = plt.subplots(2, n_cfgs, figsize=(5 * n_cfgs, 8))
    if n_cfgs == 1:
        axes = axes.reshape(2, 1)

    pc_labels = [f"PC1 ({var[0]:.1f}%)", f"PC2 ({var[1]:.1f}%)"]

    for col, (cfg_name, gen_xy) in enumerate(gen_proj.items()):
        for row in range(2):
            ax = axes[row, col]

            # -- gt histograms --
            for name, gt_xy in gt_proj.items():
                ax.hist(gt_xy[:, row], bins=80, density=True, alpha=0.4,
                        color=GT_COLORS[name], label=name)

            # -- generated histogram --
            ax.hist(gen_xy[:, row], bins=50, density=True, alpha=0.6,
                    color=CFG_COLORS[cfg_name], label=CFG_LABELS[cfg_name])

            ax.set_xlabel(pc_labels[row], fontsize=9)
            ax.set_ylabel("density", fontsize=9)
            ax.tick_params(labelsize=7)

            if row == 0:
                ax.set_title(CFG_LABELS[cfg_name], fontsize=10, fontweight='bold')
            if col == 0:
                ax.legend(fontsize=6, loc='upper right')

    fig.suptitle(f"PC1/PC2 Histograms — {model_tag}", fontsize=14, y=0.98)
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    path = os.path.join(fig_dir, "pc_histograms.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved → {path}")


def plot_marginal_violins(gt, gen, model_tag, fig_dir):

    aist_frames = gt["AIST++ (GT)"]
    n_groups = len(FEATURE_GROUPS)

    # -- collect per-dim std for each dataset per feature group --
    datasets = {"AIST++ (GT)": aist_frames}
    datasets.update({CFG_LABELS[k]: v for k, v in gen.items()})
    ds_names = list(datasets.keys())
    ds_colors = {**GT_COLORS, **{CFG_LABELS[k]: v for k, v in CFG_COLORS.items()}}

    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 6))

    for ax, (gname, sl) in zip(axes, FEATURE_GROUPS.items()):
        # -- compute per-dim std for each dataset --
        box_data = []
        box_labels = []
        box_colors = []

        for ds_name, frames in datasets.items():
            per_dim_std = np.std(frames[:, sl], axis=0)
            box_data.append(per_dim_std)
            box_labels.append(ds_name)
            box_colors.append(ds_colors[ds_name])

        bp = ax.boxplot(box_data, patch_artist=True, widths=0.6)
        for patch, c in zip(bp['boxes'], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(1.5)

        ax.set_xticklabels(box_labels, fontsize=6, rotation=45, ha='right')
        ax.set_ylabel("per-dim std", fontsize=9)
        ax.set_title(gname, fontsize=10)
        ax.tick_params(labelsize=7)

    fig.suptitle(f"Per-Dimension Std by Feature Group — {model_tag}", fontsize=13, y=0.98)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    path = os.path.join(fig_dir, "marginal_violins.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved → {path}")


def _kde_curve(data, bw=0.5, n_pts=300):
    # -- simple 1d kde using gaussian smoothing of histogram --
    lo, hi = np.percentile(data, [0.5, 99.5])
    pad = (hi - lo) * 0.1
    edges = np.linspace(lo - pad, hi + pad, n_pts + 1)
    counts, _ = np.histogram(data, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    smoothed = gaussian_filter1d(counts, sigma=bw * n_pts / (hi - lo + 2 * pad))
    return centers, smoothed


def plot_scatter_marginal_kde(gt, gen, model_tag, fig_dir):
    # -- scatter + marginal kde on axes, all configs in one figure --

    from matplotlib.gridspec import GridSpec

    gt_combined = np.concatenate(list(gt.values()), axis=0)
    pca = PCA(n_components=2).fit(gt_combined)
    var = pca.explained_variance_ratio_ * 100

    # -- project everything --
    gt_proj = {name: pca.transform(frames) for name, frames in gt.items()}
    gen_proj = {cfg_name: pca.transform(frames) for cfg_name, frames in gen.items()}

    all_colors = {**GT_COLORS, **CFG_COLORS}
    all_labels = {**{k: k for k in GT_COLORS}, **CFG_LABELS}

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 4, figure=fig, hspace=0.05, wspace=0.05)

    ax_main = fig.add_subplot(gs[1:4, 0:3])
    ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

    # -- scatter + marginal kde for gt --
    for name, xy in gt_proj.items():
        c = GT_COLORS[name]
        ax_main.scatter(xy[:, 0], xy[:, 1], c=c, s=0.5, alpha=0.03, rasterized=True)
        add_kde_contours(ax_main, xy, color=c, levels=4)

        xc, xd = _kde_curve(xy[:, 0])
        ax_top.fill_between(xc, xd, alpha=0.15, color=c)
        ax_top.plot(xc, xd, color=c, linewidth=1.5, alpha=0.8)

        yc, yd = _kde_curve(xy[:, 1])
        ax_right.fill_between(yd, yc, alpha=0.15, color=c)
        ax_right.plot(yd, yc, color=c, linewidth=1.5, alpha=0.8)

    # -- scatter + marginal kde for generated --
    for cfg_name, xy in gen_proj.items():
        c = CFG_COLORS[cfg_name]
        ax_main.scatter(xy[:, 0], xy[:, 1], c=c, s=1.5, alpha=0.15, rasterized=True)
        add_kde_contours(ax_main, xy, color=c, levels=3)

        xc, xd = _kde_curve(xy[:, 0])
        ax_top.plot(xc, xd, color=c, linewidth=1.5, alpha=0.8, linestyle='--')

        yc, yd = _kde_curve(xy[:, 1])
        ax_right.plot(yd, yc, color=c, linewidth=1.5, alpha=0.8, linestyle='--')

    ax_main.set_xlabel(f"PC1 ({var[0]:.1f}% var)", fontsize=12)
    ax_main.set_ylabel(f"PC2 ({var[1]:.1f}% var)", fontsize=12)

    ax_top.set_ylabel("density", fontsize=9)
    ax_top.tick_params(labelbottom=False, labelsize=7)
    ax_right.set_xlabel("density", fontsize=9)
    ax_right.tick_params(labelleft=False, labelsize=7)

    # -- legend on main --
    handles = []
    for name in gt_proj:
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=GT_COLORS[name], markersize=7, label=name))
    for cfg_name in gen_proj:
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=CFG_COLORS[cfg_name], markersize=7,
                       label=CFG_LABELS[cfg_name]))
    ax_main.legend(handles=handles, fontsize=8, loc='upper right')

    fig.suptitle(f"PCA Scatter + Marginal KDE — {model_tag}", fontsize=14, y=0.95)
    path = os.path.join(fig_dir, "scatter_marginal_kde.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved → {path}")


def plot_scatter_marginal_per_config(gt, gen, model_tag, fig_dir):
    # -- one scatter+marginal figure per config, side by side --

    from matplotlib.gridspec import GridSpec

    gt_combined = np.concatenate(list(gt.values()), axis=0)
    pca = PCA(n_components=2).fit(gt_combined)
    var = pca.explained_variance_ratio_ * 100

    gt_proj = {name: pca.transform(frames) for name, frames in gt.items()}

    # -- shared axis limits --
    all_gt = np.concatenate(list(gt_proj.values()), axis=0)
    pad = 5
    xlim = (np.percentile(all_gt[:, 0], 1) - pad, np.percentile(all_gt[:, 0], 99) + pad)
    ylim = (np.percentile(all_gt[:, 1], 1) - pad, np.percentile(all_gt[:, 1], 99) + pad)

    n_cfgs = len(gen)
    fig = plt.figure(figsize=(5.5 * n_cfgs, 6.5))

    for idx, (cfg_name, gen_frames) in enumerate(gen.items()):
        gen_xy = pca.transform(gen_frames)

        # -- create mini gridspec for this config panel --
        gs = GridSpec(4, 4, figure=fig,
                      left=idx / n_cfgs + 0.02, right=(idx + 1) / n_cfgs - 0.02,
                      bottom=0.08, top=0.88,
                      hspace=0.05, wspace=0.05)

        ax_main = fig.add_subplot(gs[1:4, 0:3])
        ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

        # -- gt background --
        for name, xy in gt_proj.items():
            c = GT_COLORS[name]
            ax_main.scatter(xy[:, 0], xy[:, 1], c=c, s=0.3, alpha=0.02, rasterized=True)
            add_kde_contours(ax_main, xy, color=c, levels=3)

            xc, xd = _kde_curve(xy[:, 0])
            ax_top.fill_between(xc, xd, alpha=0.1, color=c)
            ax_top.plot(xc, xd, color=c, linewidth=1.0, alpha=0.6)

            yc, yd = _kde_curve(xy[:, 1])
            ax_right.fill_between(yd, yc, alpha=0.1, color=c)
            ax_right.plot(yd, yc, color=c, linewidth=1.0, alpha=0.6)

        # -- generated foreground --
        c = CFG_COLORS[cfg_name]
        ax_main.scatter(gen_xy[:, 0], gen_xy[:, 1], c=c, s=2, alpha=0.25, rasterized=True)
        add_kde_contours(ax_main, gen_xy, color=c, levels=4)

        xc, xd = _kde_curve(gen_xy[:, 0])
        ax_top.fill_between(xc, xd, alpha=0.2, color=c)
        ax_top.plot(xc, xd, color=c, linewidth=1.5, linestyle='--')

        yc, yd = _kde_curve(gen_xy[:, 1])
        ax_right.fill_between(yd, yc, alpha=0.2, color=c)
        ax_right.plot(yd, yc, color=c, linewidth=1.5, linestyle='--')

        ax_main.set_xlim(xlim)
        ax_main.set_ylim(ylim)
        ax_main.set_xlabel(f"PC1 ({var[0]:.1f}%)", fontsize=8)
        ax_main.tick_params(labelsize=6)
        ax_top.tick_params(labelbottom=False, labelleft=False, labelsize=5)
        ax_right.tick_params(labelleft=False, labelbottom=False, labelsize=5)

        if idx == 0:
            ax_main.set_ylabel(f"PC2 ({var[1]:.1f}%)", fontsize=8)
        else:
            ax_main.set_ylabel("")

        ax_top.set_title(CFG_LABELS[cfg_name], fontsize=10, fontweight='bold')

    fig.suptitle(f"PCA + Marginal KDE per Config — {model_tag}", fontsize=14, y=0.96)
    path = os.path.join(fig_dir, "scatter_marginal_per_config.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved → {path}")


def plot_centroid_migration(gt, gen, model_tag, fig_dir):
    # -- shows where each cfg centroid lands between humanml3d and aist++ in pca-2 --

    gt_combined = np.concatenate(list(gt.values()), axis=0)
    pca = PCA(n_components=2).fit(gt_combined)

    gt_centroids = {name: pca.transform(frames).mean(axis=0) for name, frames in gt.items()}
    gen_centroids = {cfg_name: pca.transform(frames).mean(axis=0) for cfg_name, frames in gen.items()}

    hc = gt_centroids["HumanML3D (GT)"]
    ac = gt_centroids["AIST++ (GT)"]

    fig, ax = plt.subplots(figsize=(10, 8))

    # -- gt scatter as faint background --
    for name, frames in gt.items():
        xy = pca.transform(frames)
        c = GT_COLORS[name]
        ax.scatter(xy[:, 0], xy[:, 1], c=c, s=0.5, alpha=0.02, rasterized=True)
        add_kde_contours(ax, xy, color=c, levels=4)

    # -- gt centroids as large markers --
    ax.scatter(*hc, c=GT_COLORS["HumanML3D (GT)"], s=200, marker='*',
               edgecolors='black', linewidths=1.0, zorder=10, label='HumanML3D centroid')
    ax.scatter(*ac, c=GT_COLORS["AIST++ (GT)"], s=200, marker='*',
               edgecolors='black', linewidths=1.0, zorder=10, label='AIST++ centroid')

    # -- dashed line between gt centroids --
    ax.plot([hc[0], ac[0]], [hc[1], ac[1]], 'k--', alpha=0.3, linewidth=1)

    # -- generated centroids with arrows from origin midpoint --
    mid = (hc + ac) / 2
    for cfg_name, centroid in gen_centroids.items():
        c = CFG_COLORS[cfg_name]
        ax.scatter(*centroid, c=c, s=120, marker='D', edgecolors='black',
                   linewidths=0.8, zorder=10)
        ax.annotate(CFG_LABELS[cfg_name], xy=centroid, fontsize=8,
                    xytext=(8, 8), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=c, alpha=0.8))

        # -- arrow from midpoint to centroid --
        ax.annotate('', xy=centroid, xytext=mid,
                    arrowprops=dict(arrowstyle='->', color=c, lw=1.5, alpha=0.6))

    var = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({var[0]:.1f}% var)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% var)", fontsize=12)
    ax.set_title(f"Centroid Migration — {model_tag}", fontsize=14, pad=15)
    ax.legend(fontsize=9, loc='upper right')

    plt.tight_layout()
    path = os.path.join(fig_dir, "centroid_migration.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved → {path}")


# ============================================================
# audio conditioning impact analysis
# ============================================================

def plot_audio_conditioning_impact(gt, gen, model_tag, fig_dir):

    from scipy.stats import wasserstein_distance, ks_2samp

    # -- need text_only baseline --
    text_only_key = "text_only_2.5_0.0"
    if text_only_key not in gen:
        print("skipping audio_conditioning_impact: text_only_2.5_0.0 not found")
        return

    text_only = gen[text_only_key]
    audio_cfgs = {k: v for k, v in gen.items() if k != text_only_key}

    if not audio_cfgs:
        print("skipping audio_conditioning_impact: no audio configs found")
        return

    group_names = list(FEATURE_GROUPS.keys())
    group_slices = list(FEATURE_GROUPS.values())
    cfg_names = list(audio_cfgs.keys())
    n_groups = len(group_names)
    n_cfgs = len(cfg_names)

    # -- compute metrics: mean_shift, wasserstein, ks_stat, cohens_d --
    metrics = {
        "Mean shift (L2)": np.zeros((n_cfgs, n_groups)),
        "Wasserstein-1": np.zeros((n_cfgs, n_groups)),
        "KS statistic": np.zeros((n_cfgs, n_groups)),
        "Cohen's d": np.zeros((n_cfgs, n_groups)),
    }

    for ci, cfg_name in enumerate(cfg_names):
        audio_frames = audio_cfgs[cfg_name]
        for gi, (gname, sl) in enumerate(FEATURE_GROUPS.items()):
            t = text_only[:, sl]
            a = audio_frames[:, sl]
            ndims = t.shape[1]

            # -- mean shift --
            metrics["Mean shift (L2)"][ci, gi] = np.linalg.norm(a.mean(0) - t.mean(0))

            # -- per-dim wasserstein, ks, cohen's d --
            w_vals, ks_vals, d_vals = [], [], []
            for d in range(ndims):
                w_vals.append(wasserstein_distance(t[:, d], a[:, d]))
                ks_stat, _ = ks_2samp(t[:, d], a[:, d])
                ks_vals.append(ks_stat)
                # -- cohen's d --
                pooled_std = np.sqrt((np.var(t[:, d]) + np.var(a[:, d])) / 2)
                if pooled_std > 1e-8:
                    d_vals.append(abs(a[:, d].mean() - t[:, d].mean()) / pooled_std)
                else:
                    d_vals.append(0.0)

            metrics["Wasserstein-1"][ci, gi] = np.mean(w_vals)
            metrics["KS statistic"][ci, gi] = np.mean(ks_vals)
            metrics["Cohen's d"][ci, gi] = np.mean(d_vals)

    # -- plot 1: 4-panel bar chart --
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Audio Conditioning Impact vs Text-Only Baseline — {model_tag}",
                 fontsize=14, y=0.98)

    x = np.arange(n_groups)
    bar_w = 0.8 / n_cfgs

    for ax, (metric_name, vals) in zip(axes.flat, metrics.items()):
        for ci, cfg_name in enumerate(cfg_names):
            color = CFG_COLORS.get(cfg_name, "#333333")
            label = CFG_LABELS.get(cfg_name, cfg_name)
            ax.bar(x + ci * bar_w - 0.4 + bar_w / 2, vals[ci],
                   bar_w * 0.9, color=color, label=label, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([g.split(" (")[0] for g in group_names], fontsize=9, rotation=15)
        ax.set_title(metric_name, fontsize=12)
        ax.grid(axis='y', alpha=0.3)

    axes[0, 0].legend(fontsize=7, loc='upper left')
    fig.subplots_adjust(top=0.92, hspace=0.3, wspace=0.25)
    path = os.path.join(fig_dir, "audio_impact_bars.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved → {path}")

    # -- plot 2: heatmap (configs × feature groups, wasserstein) --
    fig, ax = plt.subplots(figsize=(10, max(4, n_cfgs * 0.6 + 2)))
    im = ax.imshow(metrics["Wasserstein-1"], aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels([g.split(" (")[0] for g in group_names], fontsize=10)
    ax.set_yticks(range(n_cfgs))
    ax.set_yticklabels([CFG_LABELS.get(c, c) for c in cfg_names], fontsize=10)

    # -- annotate cells --
    for ci in range(n_cfgs):
        for gi in range(n_groups):
            val = metrics["Wasserstein-1"][ci, gi]
            ax.text(gi, ci, f"{val:.3f}", ha='center', va='center', fontsize=9,
                    color='white' if val > metrics["Wasserstein-1"].max() * 0.6 else 'black')

    fig.colorbar(im, ax=ax, label="Wasserstein-1 distance")
    ax.set_title(f"Audio vs Text-Only: Wasserstein-1 per Feature Group — {model_tag}",
                 fontsize=12, pad=10)
    plt.tight_layout()
    path = os.path.join(fig_dir, "audio_impact_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved → {path}")

    # -- print table --
    print("\n  Audio conditioning impact (Wasserstein-1 vs text-only):")
    header = f"  {'Config':<30s}" + "".join(f"{g.split(' (')[0]:>15s}" for g in group_names)
    print(header)
    for ci, cfg_name in enumerate(cfg_names):
        row = f"  {CFG_LABELS.get(cfg_name, cfg_name):<30s}"
        row += "".join(f"{metrics['Wasserstein-1'][ci, gi]:>15.4f}" for gi in range(n_groups))
        print(row)


# ============================================================
# original-dimension metrics per feature group
# ============================================================

def plot_feature_group_metrics_original_dims(gt, gen, model_tag, fig_dir):

    from scipy.stats import wasserstein_distance

    group_names = list(FEATURE_GROUPS.keys())
    group_slices = list(FEATURE_GROUPS.values())
    n_groups = len(group_names)

    # -- all datasets --
    all_data = {}
    all_data.update(gt)
    for k, v in gen.items():
        all_data[CFG_LABELS.get(k, k)] = v

    ds_names = list(all_data.keys())
    n_ds = len(ds_names)

    # -- compute pairwise wasserstein per feature group --
    wasserstein_maps = {}  # group_name -> (n_ds, n_ds) matrix

    for gname, sl in FEATURE_GROUPS.items():
        mat = np.zeros((n_ds, n_ds))
        for i in range(n_ds):
            for j in range(i + 1, n_ds):
                a = all_data[ds_names[i]][:, sl]
                b = all_data[ds_names[j]][:, sl]
                # -- per-dim wasserstein, averaged --
                ndims = a.shape[1]
                w = np.mean([wasserstein_distance(a[:, d], b[:, d]) for d in range(ndims)])
                mat[i, j] = w
                mat[j, i] = w
        wasserstein_maps[gname] = mat

    # -- plot 1: 1×5 heatmaps --
    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, max(5, n_ds * 0.4 + 2)))
    fig.suptitle(f"Original-Dim Pairwise Wasserstein-1 — {model_tag}", fontsize=14, y=1.02)

    short_names = []
    for name in ds_names:
        if "(GT)" in name:
            short_names.append(name.replace(" (GT)", ""))
        else:
            short_names.append(name[:12])

    for ax, (gname, mat) in zip(axes, wasserstein_maps.items()):
        im = ax.imshow(mat, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(n_ds))
        ax.set_xticklabels(short_names, fontsize=7, rotation=45, ha='right')
        ax.set_yticks(range(n_ds))
        ax.set_yticklabels(short_names, fontsize=7)
        ax.set_title(gname, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.6)

        # -- annotate --
        vmax = mat.max()
        for i in range(n_ds):
            for j in range(n_ds):
                if i != j:
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha='center', va='center',
                            fontsize=5, color='white' if mat[i, j] > vmax * 0.6 else 'black')

    fig.subplots_adjust(top=0.88, wspace=0.35)
    path = os.path.join(fig_dir, "original_dim_wasserstein_heatmaps.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved → {path}")

    # -- plot 2: distance to AIST++ GT per feature group --
    aist_idx = ds_names.index("AIST++ (GT)")
    cfg_ds_names = [n for n in ds_names if "(GT)" not in n]
    cfg_indices = [ds_names.index(n) for n in cfg_ds_names]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_groups)
    n_bars = len(cfg_ds_names)
    bar_w = 0.8 / n_bars

    # -- map display names back to cfg keys for colors --
    label_to_cfg = {v: k for k, v in CFG_LABELS.items()}

    for bi, (cfg_label, ci) in enumerate(zip(cfg_ds_names, cfg_indices)):
        cfg_key = label_to_cfg.get(cfg_label, "")
        color = CFG_COLORS.get(cfg_key, "#333333")
        distances = [wasserstein_maps[gname][ci, aist_idx] for gname in group_names]
        ax.bar(x + bi * bar_w - 0.4 + bar_w / 2, distances,
               bar_w * 0.9, color=color, label=cfg_label, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([g.split(" (")[0] for g in group_names], fontsize=11)
    ax.set_ylabel("Wasserstein-1 distance to AIST++ GT", fontsize=11)
    ax.set_title(f"Distance to AIST++ GT per Feature Group (Original Dims) — {model_tag}",
                 fontsize=13, pad=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(fig_dir, "original_dim_distance_to_aist.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved → {path}")

    # -- print pairwise table per group --
    for gname in group_names:
        mat = wasserstein_maps[gname]
        print(f"\n  Wasserstein-1 ({gname}):")
        header = f"  {'':>20s}" + "".join(f"{n[:12]:>14s}" for n in ds_names)
        print(header)
        for i, name in enumerate(ds_names):
            row = f"  {name[:20]:>20s}"
            row += "".join(f"{mat[i, j]:>14.4f}" for j in range(n_ds))
            print(row)


# ============================================================
# cli
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate motions + PCA distribution analysis')
    parser.add_argument('--mode', choices=['generate', 'plot', 'all'],
                        default='all')
    parser.add_argument('--model_path', type=str, required=True,
                        help='path to model checkpoint (.pt)')
    parser.add_argument('--model_tag', type=str, required=True,
                        help='label for output dirs and plot titles')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='per-model output directory')
    parser.add_argument('--num_samples_per_track', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip_domain_gap', action='store_true',
                        help='skip expensive KL/MMD computation')
    parser.add_argument('--force', action='store_true',
                        help='regenerate even if samples already exist')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode in ('generate', 'all'):
        run_generate(args)

    if args.mode in ('plot', 'all'):
        run_plot(args)


if __name__ == '__main__':
    main()

# [end]
# [end]
