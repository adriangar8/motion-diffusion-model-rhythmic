"""
Compute FID between GT AIST++ features and generated motion features.

Usage (from project root):
    # Compare EDGE vs GT:
    python -m eval.aist_fid.compute_fid \
        --gt_path   eval/aist_fid/cache/gt_features.npz \
        --gen_path  eval/aist_fid/cache/edge_features.npz \
        --label     EDGE

    # Compare Stage-2 vs GT:
    python -m eval.aist_fid.compute_fid \
        --gt_path   eval/aist_fid/cache/gt_features.npz \
        --gen_path  eval/aist_fid/cache/stage2_wav2clip_beataware_features.npz \
        --label     "Stage2 (wav2clip-beataware)"

    # Compare multiple models at once:
    python -m eval.aist_fid.compute_fid \
        --gt_path   eval/aist_fid/cache/gt_features.npz \
        --gen_paths eval/aist_fid/cache/edge_features.npz \
                    eval/aist_fid/cache/stage2_wav2clip_beataware_features.npz \
        --labels    EDGE Stage2-BeatAware

Reports FID_kinetic, FID_manual, FID_combined for each model.
"""

import argparse
import os
import sys

import numpy as np
from scipy import linalg

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Fréchet distance
# ---------------------------------------------------------------------------

def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray,
                     mu2: np.ndarray, sigma2: np.ndarray,
                     eps: float = 1e-6) -> float:
    """Compute the Fréchet distance between two multivariate Gaussians.

    FD = ||mu1-mu2||² + Tr(Σ1 + Σ2 - 2√(Σ1·Σ2))

    Numerically stable implementation based on action2motion/Bailando.
    Handles rank-deficient covariances (e.g. when N < D) via progressive
    diagonal regularisation.
    """
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape,       "Mean vector shape mismatch"
    assert sigma1.shape == sigma2.shape, "Covariance shape mismatch"

    diff = mu1 - mu2

    # Try progressively larger regularisation until sqrtm is real-valued.
    for reg in [0.0, eps, eps * 10, eps * 100, eps * 1000]:
        if reg > 0:
            offset = np.eye(sigma1.shape[0]) * reg
            s1, s2 = sigma1 + offset, sigma2 + offset
        else:
            s1, s2 = sigma1, sigma2

        covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)

        if not np.isfinite(covmean).all():
            continue

        if np.iscomplexobj(covmean):
            imag_max = np.max(np.abs(covmean.imag))
            if imag_max > 1e-2:
                # Too much imaginary component; try more regularisation
                continue
            if imag_max > 1e-4:
                print(f"  [FID warning] small imaginary component ({imag_max:.2e}), taking real part.")
            covmean = covmean.real

        return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))

    # Fallback: just take real part and warn
    covmean = covmean.real
    print("  [FID warning] Could not fully regularise covariance; result may be inaccurate.")
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def compute_stats(features: np.ndarray):
    """Return (mu, sigma) for a set of feature vectors."""
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def normalize_features(gt: np.ndarray, gen: np.ndarray):
    """Z-score normalize both arrays using GT mean/std (matching Bailando's metrics_new.py).

    Returns (gt_norm, gen_norm).
    """
    mean = gt.mean(axis=0)
    std  = gt.std(axis=0)
    return (gt - mean) / (std + 1e-10), (gen - mean) / (std + 1e-10)


def fid_from_arrays(gt: np.ndarray, gen: np.ndarray) -> float:
    """Compute FID given two (N, D) feature arrays.

    Features are z-score normalized using GT statistics before computing FID,
    matching the Bailando evaluation protocol (metrics_new.py).
    """
    gt_n, gen_n = normalize_features(gt, gen)
    mu_gt,  sigma_gt  = compute_stats(gt_n)
    mu_gen, sigma_gen = compute_stats(gen_n)
    return frechet_distance(mu_gt, sigma_gt, mu_gen, sigma_gen)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_one(gt_path: str, gen_path: str, label: str) -> dict:
    gt  = np.load(gt_path)
    gen = np.load(gen_path)

    results = {"label": label}
    for key in ("kinetic", "manual", "combined"):
        if key not in gt or key not in gen:
            print(f"  Warning: '{key}' not found in feature files, skipping.")
            continue
        gt_feat  = gt[key].astype(np.float64)
        gen_feat = gen[key].astype(np.float64)

        n_gt, n_gen = gt_feat.shape[0], gen_feat.shape[0]
        d = gt_feat.shape[1]
        if n_gt < d or n_gen < d:
            print(f"  Warning: N ({n_gt} GT, {n_gen} gen) < D ({d}) for '{key}'. "
                  f"Covariance will be rank-deficient — FID may be unreliable.")

        fid = fid_from_arrays(gt_feat, gen_feat)
        results[f"FID_{key}"] = fid

    return results


def print_table(all_results: list):
    """Pretty-print a table of FID scores."""
    keys = ["FID_kinetic", "FID_manual", "FID_combined"]
    header = f"{'Model':<40} {'FID_kinetic':>14} {'FID_manual':>12} {'FID_combined':>14}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in all_results:
        row = f"{r['label']:<40}"
        for k in keys:
            val = r.get(k, float("nan"))
            row += f" {val:>14.4f}"
        print(row)
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(description="Compute AIST++ FID for one or more models")
    parser.add_argument("--gt_path",   type=str,   required=True,
                        help="Path to GT features .npz (from gt_features.py)")
    # Allow either --gen_path (single) or --gen_paths (multiple)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gen_path",   type=str,
                       help="Path to one generated features .npz")
    group.add_argument("--gen_paths",  type=str, nargs="+",
                       help="Paths to multiple generated features .npz files")
    parser.add_argument("--label",     type=str,   default="model",
                        help="Label for a single model (used with --gen_path)")
    parser.add_argument("--labels",    type=str,   nargs="+",
                        help="Labels for multiple models (used with --gen_paths)")
    args = parser.parse_args()

    if args.gen_path:
        gen_paths = [args.gen_path]
        labels    = [args.label]
    else:
        gen_paths = args.gen_paths
        labels    = args.labels if args.labels else [os.path.basename(p) for p in gen_paths]

    all_results = []
    for gen_path, label in zip(gen_paths, labels):
        print(f"\nEvaluating: {label}")
        print(f"  GT:        {args.gt_path}  ({np.load(args.gt_path)['kinetic'].shape[0]} seqs)")
        print(f"  Generated: {gen_path}  ({np.load(gen_path)['kinetic'].shape[0]} seqs)")
        results = evaluate_one(args.gt_path, gen_path, label)
        all_results.append(results)
        for k, v in results.items():
            if k != "label":
                print(f"  {k}: {v:.4f}")

    if len(all_results) > 1:
        print_table(all_results)


if __name__ == "__main__":
    main()
