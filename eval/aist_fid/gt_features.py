"""
Extract GT features from AIST++ processed joints_22/ directory.

Usage (from project root):
    python -m eval.aist_fid.gt_features \
        --joints_dir /Data/yash.bhardwaj/datasets/aist/processed/joints_22 \
        --out_path    eval/aist_fid/cache/gt_features.npz \
        [--test_only]

With --test_only the script filters to the 20 sequences in crossmodal_test.txt.
Without it (default) ALL available sequences are used — recommended for FID
because it gives a well-conditioned GT covariance (1363 sequences >> 98 dims).

Output .npz contains:
    kinetic  (N, 66)  float32
    manual   (N, 32)  float32
    combined (N, 98)  float32
    names    (N,)     str     — sequence names
"""

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from eval.aist_fid.extract_features import extract_all
from eval.aist_fid.joint_utils import resample_positions

# joints_22/ files are produced by data/preprocess_aist.py which downsamples
# AIST++ from 60 fps → 20 fps before saving. TARGET_FPS is the canonical rate
# used for all FID computations; GT will be upsampled 20→60.
GT_FPS = 20.0
TARGET_FPS = 60.0  # canonical FPS for all FID computations


def load_test_names(test_split_path: str) -> set:
    with open(test_split_path) as f:
        return {line.strip() for line in f if line.strip()}


def main():
    parser = argparse.ArgumentParser(description="Extract GT kinetic+manual features from AIST++ joints_22/")
    parser.add_argument("--joints_dir", type=str,
                        default="/Data/yash.bhardwaj/datasets/aist/processed/joints_22",
                        help="Directory containing per-sequence (T, 22, 3) .npy files")
    parser.add_argument("--test_split", type=str,
                        default=None,
                        help="Path to crossmodal_test.txt (used only with --test_only)")
    parser.add_argument("--test_only", action="store_true",
                        help="Only use the 20 crossmodal-test sequences (not recommended for GT stats)")
    parser.add_argument("--out_path", type=str,
                        default=os.path.join(_REPO_ROOT, "eval/aist_fid/cache/gt_features.npz"),
                        help="Output .npz path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    # Optionally find test split file automatically
    if args.test_only and args.test_split is None:
        candidates = [
            os.path.join(_REPO_ROOT, "EDGE/data/splits/crossmodal_test.txt"),
            "/Data/yash.bhardwaj/datasets/aist/splits/crossmodal_test.txt",
        ]
        for c in candidates:
            if os.path.exists(c):
                args.test_split = c
                break
        if args.test_split is None:
            raise FileNotFoundError(
                "Could not auto-locate crossmodal_test.txt. Pass --test_split explicitly."
            )

    all_files = sorted([f for f in os.listdir(args.joints_dir) if f.endswith(".npy")])
    print(f"Found {len(all_files)} .npy files in {args.joints_dir}")

    if args.test_only:
        test_names = load_test_names(args.test_split)
        all_files = [f for f in all_files if os.path.splitext(f)[0] in test_names]
        print(f"Filtered to {len(all_files)} test sequences")

    kinetics, manuals, names = [], [], []

    for fname in tqdm(all_files, desc="Extracting GT features"):
        seq_name = os.path.splitext(fname)[0]
        path = os.path.join(args.joints_dir, fname)
        positions = np.load(path).astype(np.float32)  # (T, 22, 3)

        if positions.shape[1] != 22 or positions.shape[2] != 3:
            print(f"  Skipping {fname}: unexpected shape {positions.shape}")
            continue

        # Resample to canonical FPS if needed
        if GT_FPS != TARGET_FPS:
            positions = resample_positions(positions, src_fps=GT_FPS, tgt_fps=TARGET_FPS)

        try:
            feats = extract_all(positions, fps=TARGET_FPS)
        except Exception as e:
            print(f"  Skipping {fname}: {e}")
            continue

        kinetics.append(feats["kinetic"])
        manuals.append(feats["manual"])
        names.append(seq_name)

    kinetics = np.stack(kinetics, axis=0)  # (N, 66)
    manuals  = np.stack(manuals,  axis=0)  # (N, 32)
    combined = np.concatenate([kinetics, manuals], axis=-1)  # (N, 98)

    np.savez(
        args.out_path,
        kinetic=kinetics,
        manual=manuals,
        combined=combined,
        names=np.array(names),
    )
    print(f"\nSaved {len(names)} GT feature vectors → {args.out_path}")
    print(f"  kinetic  shape: {kinetics.shape}")
    print(f"  manual   shape: {manuals.shape}")
    print(f"  combined shape: {combined.shape}")


if __name__ == "__main__":
    main()
