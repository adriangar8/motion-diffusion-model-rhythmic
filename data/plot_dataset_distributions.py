# [start]
# [start]

import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist

# -- reproducibility --
RNG = np.random.default_rng(42)
N_SUBSAMPLE = 50_000

# -- paths --
HUMANML_DIR = "dataset/HumanML3D/new_joint_vecs"
AIST_DIR = "dataset/aist_raw/processed/motions_263"
FINEDANCE_DIR = "dataset/finedance_raw/processed/motions_263"
MEAN_PATH = "dataset/HumanML3D/Mean.npy"
STD_PATH = "dataset/HumanML3D/Std.npy"
OUT_DIR = "figures"
OUT_PATH = os.path.join(OUT_DIR, "dataset_pca_distribution.png")

# -- 263-dim hml-vec feature groups (from motion_process.py lines 332-365) --
FEATURE_GROUPS = {
    "Root (4d)":             slice(0, 4),
    "Joint pos (63d)":       slice(4, 67),
    "Joint rot (126d)":      slice(67, 193),
    "Joint vel (66d)":       slice(193, 259),
    "Foot contact (4d)":     slice(259, 263),
}

DATASET_COLORS = {
    "HumanML3D": "#2196F3",
    "AIST++":    "#F44336",
    "FineDance": "#4CAF50",
}


def load_frames(motion_dir: str) -> np.ndarray:
    # -- stack all (T, 263) arrays from directory --
    files = sorted(glob(os.path.join(motion_dir, "*.npy")))
    if not files:
        raise FileNotFoundError(f"no .npy files found in {motion_dir}")
    arrays = [np.load(f).astype(np.float32) for f in files]
    # -- handle shape (263,) single-frame edge case --
    arrays = [a if a.ndim == 2 else a[None] for a in arrays]
    return np.concatenate(arrays, axis=0)


def subsample(frames: np.ndarray, n: int) -> np.ndarray:
    if len(frames) <= n:
        return frames
    idx = RNG.choice(len(frames), size=n, replace=False)
    return frames[idx]


def add_kde_contours(ax, xy, color, levels=5):
    # -- lightweight 2d histogram contour --
    x, y = xy[:, 0], xy[:, 1]
    h, xedges, yedges = np.histogram2d(x, y, bins=100)
    h = h.T
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    # -- smooth slightly --
    h_smooth = gaussian_filter(h, sigma=2)
    ax.contour(xc, yc, h_smooth, levels=levels, colors=[
               color], linewidths=0.8, alpha=0.7)


def plot_feature_group_pca(sub: dict):
    # -- per-feature-group pca scatter, 1 subplot per group --
    n_groups = len(FEATURE_GROUPS)
    fig, axes = plt.subplots(1, n_groups, figsize=(4 * n_groups, 4))

    for ax, (gname, sl) in zip(axes, FEATURE_GROUPS.items()):
        # -- slice feature group from each dataset --
        sliced = {name: frames[:, sl] for name, frames in sub.items()}
        combined = np.concatenate(list(sliced.values()), axis=0)
        pca = PCA(n_components=2).fit(combined)
        var = pca.explained_variance_ratio_ * 100

        for name, frames in sliced.items():
            xy = pca.transform(frames)
            ax.scatter(xy[:, 0], xy[:, 1], c=DATASET_COLORS[name], s=0.5,
                       alpha=0.1, rasterized=True)
            add_kde_contours(ax, xy, color=DATASET_COLORS[name], levels=4)

        ax.set_title(gname, fontsize=10)
        ax.set_xlabel(f"PC1 ({var[0]:.1f}%)", fontsize=8)
        ax.set_ylabel(f"PC2 ({var[1]:.1f}%)", fontsize=8)
        ax.tick_params(labelsize=7)

    # -- shared legend --
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=DATASET_COLORS[n], markersize=6, label=n)
        for n in sub
    ]
    axes[-1].legend(handles=handles, fontsize=7, loc="upper right")

    fig.suptitle("PCA by Feature Group", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "dataset_pca_by_feature_group.png")
    fig.savefig(path, dpi=150)
    print(f"saved → {path}")


def symmetric_kl(mu1, cov1, mu2, cov2):
    # -- symmetric kl divergence between two gaussians --
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
    # -- mmd^2 with rbf kernel --
    def rbf(A, B):
        # -- (n, d) x (m, d) -> (n, m) kernel matrix --
        dists_sq = np.sum(A**2, axis=1, keepdims=True) + np.sum(B**2, axis=1) - 2 * A @ B.T
        return np.exp(-dists_sq / (2 * sigma**2))

    xx = rbf(X, X).mean()
    yy = rbf(Y, Y).mean()
    xy = rbf(X, Y).mean()
    return xx + yy - 2 * xy


def compute_domain_gap(sub: dict):
    # -- quantitative domain gap: kl divergence + mmd --
    names = list(sub.keys())
    n = len(names)

    # -- reduce to 50 pca components for stable covariance --
    combined = np.concatenate(list(sub.values()), axis=0)
    n_comp = min(50, combined.shape[1])
    pca = PCA(n_components=n_comp).fit(combined)
    reduced = {name: pca.transform(frames) for name, frames in sub.items()}
    print(f"\ndomain gap (PCA-{n_comp} space, {pca.explained_variance_ratio_.sum()*100:.1f}% var retained):")

    # -- fit gaussians --
    stats = {}
    for name, X in reduced.items():
        stats[name] = (X.mean(axis=0), np.cov(X, rowvar=False))

    # -- rbf bandwidth via median heuristic on a subsample --
    sample = combined[RNG.choice(len(combined), size=min(5000, len(combined)), replace=False)]
    sample_r = pca.transform(sample)
    sigma = np.median(pdist(sample_r))

    # -- compute pairwise metrics --
    kl_mat = np.zeros((n, n))
    mmd_mat = np.zeros((n, n))
    # -- subsample for mmd (expensive) --
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
    print(f"  {'':>12s}  {'  '.join(f'{n:>12s}' for n in names)}")
    for i, ni in enumerate(names):
        row = "  ".join(f"{kl_mat[i, j]:12.1f}" for j in range(n))
        print(f"  {ni:>12s}  {row}")

    print("\n  MMD² (RBF, σ={:.1f}):".format(sigma))
    print(f"  {'':>12s}  {'  '.join(f'{n:>12s}' for n in names)}")
    for i, ni in enumerate(names):
        row = "  ".join(f"{mmd_mat[i, j]:12.6f}" for j in range(n))
        print(f"  {ni:>12s}  {row}")

    # -- heatmap figure --
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for ax, mat, title, fmt in [
        (ax1, kl_mat, "Symmetric KL Divergence", ".1f"),
        (ax2, mmd_mat, "MMD² (RBF)", ".4f"),
    ]:
        im = ax.imshow(mat, cmap="YlOrRd")
        ax.set_xticks(range(n))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_yticks(range(n))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_title(title, fontsize=11)
        # -- annotate cells --
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{mat[i, j]:{fmt}}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "dataset_domain_gap.png")
    fig.savefig(path, dpi=150)
    print(f"\nsaved → {path}")


def main():
    # -- load normalization stats --
    mean = np.load(MEAN_PATH).astype(np.float32)   # (263,)
    std = np.load(STD_PATH).astype(np.float32)    # (263,)
    # -- avoid division by zero --
    std = np.where(std < 1e-6, 1.0, std)

    print("loading frames...")
    raw = {
        "HumanML3D": load_frames(HUMANML_DIR),
        "AIST++":    load_frames(AIST_DIR),
        "FineDance": load_frames(FINEDANCE_DIR),
    }

    for name, frames in raw.items():
        print(f"  {name}: {len(frames):,} frames, shape {frames.shape}")

    # -- normalize --
    normed = {name: (f - mean) / std for name, f in raw.items()}

    # -- subsample --
    sub = {name: subsample(f, N_SUBSAMPLE) for name, f in normed.items()}
    for name, s in sub.items():
        print(f"  {name}: {len(s):,} subsampled frames")

    # -- fit pca on combined pool --
    combined = np.concatenate(list(sub.values()), axis=0)
    print(f"\nfitting PCA on {len(combined):,} combined frames...")
    pca = PCA(n_components=2)
    pca.fit(combined)
    var = pca.explained_variance_ratio_ * 100
    print(f"  PC1 explains {var[0]:.1f}%,  PC2 explains {var[1]:.1f}%")

    # -- project each dataset --
    proj = {name: pca.transform(s) for name, s in sub.items()}

    # -- plot --
    fig, ax = plt.subplots(figsize=(9, 7))

    for name, xy in proj.items():
        c = DATASET_COLORS[name]
        ax.scatter(xy[:, 0], xy[:, 1], c=c, s=1,
                   alpha=0.15, label=name, rasterized=True)
        add_kde_contours(ax, xy, color=c)

    ax.set_xlabel(f"PC1 ({var[0]:.1f}% var)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% var)", fontsize=12)
    ax.set_title(
        "Motion Frame Distribution (PCA, 263-dim HML-Vec)", fontsize=13)

    # -- legend with larger markers --
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=DATASET_COLORS[n], markersize=8, label=n)
        for n in proj
    ]
    ax.legend(handles=handles, fontsize=11)

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"\nsaved → {OUT_PATH}")

    # -- feature-group pca --
    plot_feature_group_pca(sub)

    # -- quantitative domain gap --
    compute_domain_gap(sub)


if __name__ == "__main__":
    main()

# [end]
# [end]
