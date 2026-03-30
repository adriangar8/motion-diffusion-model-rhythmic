"""
Feature extraction for AIST++ FID evaluation.

Given (T, 22, 3) world-space joint positions (resampled to 60 fps), computes:
  - kinetic features  : 66-dim  (22 joints × 3: h-KE, v-KE, energy-expenditure)
  - manual features   : 32-dim  (geometric pose predicates, averaged over frames)
  - combined features : 98-dim  (kinetic ‖ manual)

All feature extractors are drawn from Bailando/utils/features/ with a minor
adaptation: we pass 22-joint data with the 24-name SMPL_JOINT_NAMES list.
This is safe because every geometric predicate in manual_new.py references only
joint indices 0-21, which are identical in both SMPL24 and HML22.
"""

import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# Make Bailando feature code importable regardless of working directory
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BAILANDO_UTILS = os.path.join(_REPO_ROOT, "Bailando")
if _BAILANDO_UTILS not in sys.path:
    sys.path.insert(0, _BAILANDO_UTILS)

from utils.features.kinetic import extract_kinetic_features, KineticFeatures  # noqa: E402
from utils.features.manual_new import extract_manual_features, ManualFeatures, SMPL_JOINT_NAMES  # noqa: E402


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def root_center(positions: np.ndarray) -> np.ndarray:
    """Subtract first-frame root joint (joint 0) from all joints at all frames.

    Matches Bailando's preprocessing (metrics_new.py):
        roott = joint3d[:1, :3]
        joint3d = joint3d - np.tile(roott, (1, 24))

    Args:
        positions: (T, J, 3) world-space joint positions.

    Returns:
        (T, J, 3) root-relative positions (initial hip at origin).
    """
    root0 = positions[0:1, 0:1, :]  # (1, 1, 3) – first-frame pelvis
    return positions - root0


def extract_kinetic(positions: np.ndarray, fps: float = 60.0) -> np.ndarray:
    """Extract 66-dim kinetic features from (T, 22, 3) positions at `fps`.

    Returns a float32 array of shape (66,).
    """
    assert positions.ndim == 3 and positions.shape[1] == 22 and positions.shape[2] == 3, \
        f"Expected (T, 22, 3), got {positions.shape}"
    frame_time = 1.0 / fps
    feat = KineticFeatures(positions, frame_time=frame_time)
    vec = []
    for i in range(22):
        vec.extend([
            feat.average_kinetic_energy_horizontal(i),
            feat.average_kinetic_energy_vertical(i),
            feat.average_energy_expenditure(i),
        ])
    return np.array(vec, dtype=np.float32)  # (66,)


def extract_manual(positions: np.ndarray) -> np.ndarray:
    """Extract 32-dim manual features from (T, 22, 3) positions.

    Returns a float32 array of shape (32,).
    The SMPL_JOINT_NAMES list (24 entries) is used for index lookups;
    only the first 22 are ever accessed so this is safe with 22-joint input.
    """
    assert positions.ndim == 3 and positions.shape[1] == 22 and positions.shape[2] == 3, \
        f"Expected (T, 22, 3), got {positions.shape}"
    return extract_manual_features(positions)  # (32,)


def extract_all(positions: np.ndarray, fps: float = 60.0) -> dict:
    """Extract all features for one motion clip.

    Args:
        positions: (T, 22, 3) world-space joint positions at `fps`.
        fps:       frame rate (used only for kinetic feature scaling).

    Returns:
        dict with keys:
          'kinetic'  : (66,) float32
          'manual'   : (32,) float32
          'combined' : (98,) float32
    """
    positions = root_center(positions)
    k = extract_kinetic(positions, fps=fps)
    m = extract_manual(positions)
    return {
        "kinetic":  k,
        "manual":   m,
        "combined": np.concatenate([k, m]),
    }


def extract_batch(clips: list, fps: float = 60.0, verbose: bool = False) -> dict:
    """Extract features for a list of (T_i, 22, 3) clips.

    Returns:
        dict with keys 'kinetic', 'manual', 'combined',
        each containing a (N, D) float32 array.
    """
    kinetics, manuals = [], []
    for i, clip in enumerate(clips):
        if verbose:
            print(f"  [{i+1}/{len(clips)}] clip shape={clip.shape}")
        feats = extract_all(clip, fps=fps)
        kinetics.append(feats["kinetic"])
        manuals.append(feats["manual"])
    kinetics = np.stack(kinetics, axis=0)   # (N, 66)
    manuals  = np.stack(manuals,  axis=0)   # (N, 32)
    return {
        "kinetic":  kinetics,
        "manual":   manuals,
        "combined": np.concatenate([kinetics, manuals], axis=-1),  # (N, 98)
    }
