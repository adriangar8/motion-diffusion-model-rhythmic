"""
Joint space utilities for AIST++ FID evaluation.

Joint space notes
-----------------
SMPL24 (Bailando convention): 24 joints, indices 0-23.
HML22  (HumanML3D convention): 22 joints = SMPL24 without lhand (22) and rhand (23).
Both representations share the same indices for joints 0-21.

EDGE outputs full_pose (T, 24, 3) in SMPL24 order.
Stage-2 outputs (T, 263) HumanML3D vectors which recover to (T, 22, 3) in HML22 order.
GT data in joints_22/ is (T, 22, 3) in HML22 order.

Since SMPL24[0:22] == HML22[0:22], all three can be compared after slicing EDGE to 22 joints.
All manual features only reference joint indices 0-21, so no remapping is required.
"""

import pickle
import sys
import os

import numpy as np

# ---------------------------------------------------------------------------
# Joint name constants
# ---------------------------------------------------------------------------

# Bailando SMPL-24 joint names — indices 0-21 match HML22 exactly.
SMPL_JOINT_NAMES = [
    "root",         # 0   pelvis
    "lhip",         # 1
    "rhip",         # 2
    "belly",        # 3   spine1
    "lknee",        # 4
    "rknee",        # 5
    "spine",        # 6   spine2
    "lankle",       # 7
    "rankle",       # 8
    "chest",        # 9   spine3
    "ltoes",        # 10  left_foot (ball of foot)
    "rtoes",        # 11  right_foot
    "neck",         # 12
    "linshoulder",  # 13  left_collar
    "rinshoulder",  # 14  right_collar
    "head",         # 15
    "lshoulder",    # 16
    "rshoulder",    # 17
    "lelbow",       # 18
    "relbow",       # 19
    "lwrist",       # 20
    "rwrist",       # 21
    "lhand",        # 22  — not in HML22, not used by any feature
    "rhand",        # 23  — not in HML22, not used by any feature
]

# HumanML3D 22-joint names for reference (same order as SMPL_JOINT_NAMES[0:22]).
HML22_JOINT_NAMES = SMPL_JOINT_NAMES[:22]


# ---------------------------------------------------------------------------
# EDGE output → (T, 22, 3)
# ---------------------------------------------------------------------------

def load_edge_pkl_joints22(path: str) -> np.ndarray:
    """Load an EDGE output .pkl and return world-space joints (T, 22, 3) in y-up space.

    EDGE stores 'full_pose' as (T, 24, 3) in SMPL24 order in **z-up** coordinates.
    The training dataset applies RotateAxisAngle(90, "X") to convert SMPL's native
    y-up space to z-up: (x, y, z) → (x, −z, y).

    We apply the inverse rotation to recover y-up: (x, y', z') → (x, z', −y').

    We also drop the last two joints (lhand, rhand, indices 22-23) which are
    absent in HML22, leaving 22 joints matching GT and Stage-2.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    fp = np.array(data["full_pose"], dtype=np.float32)  # (T, 24, 3), z-up

    # Inverse of RotateAxisAngle(90, "X"): (x, y', z') → (x, z', -y')
    fp_yup = np.stack([fp[:, :, 0],   # x unchanged
                       fp[:, :, 2],   # z' → y  (height)
                       -fp[:, :, 1]], # -y' → z
                      axis=-1)        # (T, 24, 3), y-up

    return fp_yup[:, :22, :]  # (T, 22, 3)


# ---------------------------------------------------------------------------
# Stage-2 output → (T, 22, 3)
# ---------------------------------------------------------------------------

def recover_from_ric_numpy(motion263: np.ndarray) -> np.ndarray:
    """Convert a denormalised HumanML3D 263-d sequence to (T, 22, 3) world joints.

    Uses the existing recover_from_ric() torch function from data_loaders.
    The input must already be denormalised (raw model output × std + mean).

    Args:
        motion263: (T, 263) float32 numpy array.

    Returns:
        (T, 22, 3) float32 numpy array of world-space joint positions.
    """
    import torch
    from data_loaders.humanml.scripts.motion_process import recover_from_ric

    tensor = torch.from_numpy(motion263).float().unsqueeze(0)  # (1, T, 263)
    joints = recover_from_ric(tensor, joints_num=22)           # (1, T, 22, 3)
    return joints.squeeze(0).numpy()                           # (T, 22, 3)


def load_stage2_npy_joints22(path: str) -> np.ndarray:
    """Load a Stage-2 output .npy file and return world-space joints (T, 22, 3).

    Stage-2 saves denormalised (T, 263) HumanML3D vectors.
    """
    motion = np.load(path).astype(np.float32)  # (T, 263)
    if motion.ndim == 3:
        # shape (1, T, 263) saved by some versions — squeeze
        motion = motion.squeeze(0)
    return recover_from_ric_numpy(motion)       # (T, 22, 3)


# ---------------------------------------------------------------------------
# Temporal resampling
# ---------------------------------------------------------------------------

def resample_positions(positions: np.ndarray, src_fps: float, tgt_fps: float = 60.0) -> np.ndarray:
    """Resample joint positions along the time axis to tgt_fps.

    Uses scipy Fourier-based resampling, which preserves motion dynamics
    better than nearest-neighbour for non-integer ratios.

    Args:
        positions: (T, J, 3) float32.
        src_fps:   source frame rate.
        tgt_fps:   target frame rate (default 60 = AIST++ native).

    Returns:
        (T_new, J, 3) float32.
    """
    from scipy.signal import resample

    T, J, C = positions.shape
    T_new = int(round(T * tgt_fps / src_fps))
    if T_new == T:
        return positions
    flat = positions.reshape(T, J * C)
    flat_res = resample(flat, T_new, axis=0)
    return flat_res.reshape(T_new, J, C).astype(np.float32)
