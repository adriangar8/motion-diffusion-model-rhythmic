"""
Extract features from EDGE-generated motion .pkl files.

Usage (from project root):
    python -m eval.aist_fid.edge_features \
        --motions_dir /path/to/edge/output_motions \
        --out_path    eval/aist_fid/cache/edge_features.npz

EDGE saves one .pkl per generated sequence. Each .pkl contains:
    full_pose  (T, 24, 3)  world-space joint positions at 30 fps
    smpl_poses (T, 72)     axis-angle SMPL params (not used here)
    smpl_trans (T, 3)      root translation (not used here)

We slice to the first 22 joints (drop lhand/rhand), resample to 60 fps, and
extract kinetic + manual features.

Output .npz:
    kinetic  (N, 66)  float32
    manual   (N, 32)  float32
    combined (N, 98)  float32
    names    (N,)     str
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
from eval.aist_fid.joint_utils import load_edge_pkl_joints22, resample_positions

EDGE_FPS   = 30.0
TARGET_FPS = 60.0


def main():
    parser = argparse.ArgumentParser(description="Extract features from EDGE output .pkl files")
    parser.add_argument("--motions_dir", type=str, required=True,
                        help="Directory containing EDGE output .pkl files")
    parser.add_argument("--out_path", type=str,
                        default=os.path.join(_REPO_ROOT, "eval/aist_fid/cache/edge_features.npz"),
                        help="Output .npz path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    pkl_files = sorted([f for f in os.listdir(args.motions_dir) if f.endswith(".pkl")])
    if not pkl_files:
        raise RuntimeError(f"No .pkl files found in {args.motions_dir}")
    print(f"Found {len(pkl_files)} .pkl files in {args.motions_dir}")

    kinetics, manuals, names = [], [], []

    for fname in tqdm(pkl_files, desc="Extracting EDGE features"):
        path = os.path.join(args.motions_dir, fname)
        try:
            positions = load_edge_pkl_joints22(path)  # (T, 22, 3) at EDGE_FPS
        except Exception as e:
            print(f"  Skipping {fname}: failed to load — {e}")
            continue

        if positions.shape[0] < 10:
            print(f"  Skipping {fname}: too short ({positions.shape[0]} frames)")
            continue

        # Resample from 30 fps → 60 fps
        positions = resample_positions(positions, src_fps=EDGE_FPS, tgt_fps=TARGET_FPS)

        try:
            feats = extract_all(positions, fps=TARGET_FPS)
        except Exception as e:
            print(f"  Skipping {fname}: feature extraction failed — {e}")
            continue

        kinetics.append(feats["kinetic"])
        manuals.append(feats["manual"])
        names.append(os.path.splitext(fname)[0])

    if not kinetics:
        raise RuntimeError("No sequences successfully processed.")

    kinetics = np.stack(kinetics, axis=0)
    manuals  = np.stack(manuals,  axis=0)
    combined = np.concatenate([kinetics, manuals], axis=-1)

    np.savez(
        args.out_path,
        kinetic=kinetics,
        manual=manuals,
        combined=combined,
        names=np.array(names),
    )
    print(f"\nSaved {len(names)} EDGE feature vectors → {args.out_path}")
    print(f"  kinetic  shape: {kinetics.shape}")
    print(f"  manual   shape: {manuals.shape}")
    print(f"  combined shape: {combined.shape}")


if __name__ == "__main__":
    main()
