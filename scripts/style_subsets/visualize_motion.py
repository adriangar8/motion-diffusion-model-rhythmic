#!/usr/bin/env python3
"""
Visualize a single motion by motion_id from a dataset path.

Loads motion from either new_joints/ (T, 22, 3) or new_joint_vecs/ (T, 263),
converts to joint positions if needed, and saves a stick-figure MP4.

Usage:
    python scripts/style_subsets/visualize_motion.py \\
        --motion_id 030161 \\
        --dataset_path /Data/yash.bhardwaj/datasets/Retargeted100Style/RETARGETED_100STYLE \\
        --out_file motion_030161.mp4

    python scripts/style_subsets/visualize_motion.py \\
        --motion_id 006447 \\
        --dataset_path /Data/yash.bhardwaj/datasets/HumanML3D \\
        --out_file motion_006447.mp4
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

# Repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml.utils.plot_script import plot_3d_motion


def load_joints_from_new_joints(dataset_path: str, motion_id: str) -> np.ndarray:
    """Load (T, 22, 3) from new_joints/{motion_id}.npy."""
    path = os.path.join(dataset_path, "new_joints", motion_id + ".npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    data = np.load(path)
    if data.ndim != 3 or data.shape[1] != 22 or data.shape[2] != 3:
        raise ValueError(f"Expected (T, 22, 3), got {data.shape}")
    return data.astype(np.float64)


def load_joints_from_new_joint_vecs(dataset_path: str, motion_id: str) -> np.ndarray:
    """Load (T, 263) from new_joint_vecs and convert to (T, 22, 3) via recover_from_ric."""
    path = os.path.join(dataset_path, "new_joint_vecs", motion_id + ".npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    data = np.load(path)
    if data.ndim != 2 or data.shape[1] != 263:
        raise ValueError(f"Expected (T, 263), got {data.shape}")
    # recover_from_ric expects torch, shape (..., 263)
    data_t = torch.from_numpy(data).float().unsqueeze(0)
    joints = recover_from_ric(data_t, joints_num=22)
    joints = joints.squeeze(0).numpy()
    return joints.astype(np.float64)


def load_motion(dataset_path: str, motion_id: str) -> np.ndarray:
    """Load motion as (T, 22, 3). Prefer new_joints, else new_joint_vecs."""
    new_joints_dir = os.path.join(dataset_path, "new_joints")
    new_joint_vecs_dir = os.path.join(dataset_path, "new_joint_vecs")
    if os.path.isdir(new_joints_dir) and os.path.isfile(
        os.path.join(new_joints_dir, motion_id + ".npy")
    ):
        return load_joints_from_new_joints(dataset_path, motion_id)
    if os.path.isdir(new_joint_vecs_dir) and os.path.isfile(
        os.path.join(new_joint_vecs_dir, motion_id + ".npy")
    ):
        return load_joints_from_new_joint_vecs(dataset_path, motion_id)
    raise FileNotFoundError(
        f"No motion found for {motion_id} in {dataset_path} "
        "(need new_joints/ or new_joint_vecs/)"
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize one motion by motion_id")
    parser.add_argument("--motion_id", type=str, required=True, help="e.g. 030161 or M030161")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Dataset root (HumanML3D or RETARGETED_100STYLE)",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default=None,
        help="Output MP4 path (default: <motion_id>.mp4 in cwd)",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--dataset",
        type=str,
        default="humanml",
        choices=("humanml", "kit"),
        help="Scale/convention for plot_3d_motion",
    )
    parser.add_argument("--title", type=str, default=None, help="Title on video (default: motion_id)")
    args = parser.parse_args()

    dataset_path = os.path.expanduser(args.dataset_path)
    if not os.path.isdir(dataset_path):
        print(f"ERROR: not a directory: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    motion_id = args.motion_id.strip()
    joints = load_motion(dataset_path, motion_id)
    title = args.title if args.title else motion_id
    out_file = args.out_file or (motion_id + ".mp4")
    out_file = os.path.abspath(out_file)
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)

    skeleton = paramUtil.t2m_kinematic_chain
    n_frames = joints.shape[0]
    ani = plot_3d_motion(
        out_file,
        skeleton,
        joints,
        title=title,
        dataset=args.dataset,
        fps=args.fps,
    )
    ani.write_videofile(out_file, fps=args.fps, logger=None)
    print(f"Saved: {out_file} ({n_frames} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
