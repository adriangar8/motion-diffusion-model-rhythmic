"""
Compute FID using per-frame root centering (PFRC) instead of first-frame root centering.

Per-frame root centering removes root trajectory entirely, isolating body pose quality.
This separates pose quality from root motion drift artifacts in recover_from_ric().
"""
import numpy as np
import sys
import glob
import os
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval.aist_fid.joint_utils import load_stage2_npy_joints22, load_edge_pkl_joints22, resample_positions
from eval.aist_fid.extract_features import extract_kinetic, extract_manual
from eval.aist_fid.compute_fid import normalize_features, frechet_distance, compute_stats


def per_frame_root_center(positions: np.ndarray) -> np.ndarray:
    """Subtract root joint (joint 0) at every frame independently."""
    return positions - positions[:, 0:1, :]


def extract_features(clips, src_fps: float, desc: str = ""):
    kins, mans = [], []
    for clip in tqdm(clips, desc=desc):
        r = per_frame_root_center(resample_positions(clip, src_fps, 60.0))
        kins.append(extract_kinetic(r, fps=60.0))
        mans.append(extract_manual(r))
    return np.stack(kins), np.stack(mans)


def fid(gt_feat, gen_feat):
    gt_n, gen_n = normalize_features(gt_feat.astype(np.float64), gen_feat.astype(np.float64))
    return frechet_distance(*compute_stats(gt_n), *compute_stats(gen_n))


def main():
    GT_DIR   = "/Data/yash.bhardwaj/datasets/aist/processed/joints_22"
    EDGE_DIR = "save/aist_fid_eval/edge_motions"
    S2_DIR   = "save/aist_fid_eval/stage2_wav2clip_beataware_motions"
    CACHE    = "eval/aist_fid/cache"
    os.makedirs(CACHE, exist_ok=True)

    gt_files   = sorted(glob.glob(os.path.join(GT_DIR, "*.npy")))
    edge_files = sorted(glob.glob(os.path.join(EDGE_DIR, "*.pkl")))
    s2_files   = sorted(glob.glob(os.path.join(S2_DIR, "*.npy")))

    print(f"GT: {len(gt_files)} clips | EDGE: {len(edge_files)} clips | Stage2: {len(s2_files)} clips")

    gt_k, gt_m = extract_features([np.load(f) for f in gt_files], 20.0, "GT")
    np.savez(os.path.join(CACHE, "pfrc_gt_features.npz"), kinetic=gt_k, manual=gt_m)
    print(f"GT features saved: {gt_k.shape}")

    edge_k, edge_m = extract_features([load_edge_pkl_joints22(f) for f in edge_files], 30.0, "EDGE")
    np.savez(os.path.join(CACHE, "pfrc_edge_features.npz"), kinetic=edge_k, manual=edge_m)
    print(f"EDGE features saved: {edge_k.shape}")

    s2_k, s2_m = extract_features([load_stage2_npy_joints22(f) for f in s2_files], 20.0, "Stage2")
    np.savez(os.path.join(CACHE, "pfrc_s2_features.npz"), kinetic=s2_k, manual=s2_m)
    print(f"Stage2 features saved: {s2_k.shape}")

    print("\n=== FID with per-frame root centering ===")
    print(f"{'Model':<35} {'FID_kinetic':>14} {'FID_manual':>14} {'FID_combined':>14}")
    print("-" * 80)

    for name, gk, gm in [
        (f"EDGE ({len(edge_files)} clips)", edge_k, edge_m),
        (f"Stage2-beataware ({len(s2_files)} clips)", s2_k, s2_m),
    ]:
        fk = fid(gt_k, gk)
        fm = fid(gt_m, gm)
        fc = fid(np.concatenate([gt_k, gt_m], axis=1), np.concatenate([gk, gm], axis=1))
        print(f"{name:<35} {fk:>14.4f} {fm:>14.4f} {fc:>14.4f}")


if __name__ == "__main__":
    main()
