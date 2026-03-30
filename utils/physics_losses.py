"""
Differentiable physics auxiliary losses for motion (Blueprint).
L_contact: penalize foot velocity when contact=1.
L_penetrate: ReLU(-y_foot) for joints below ground.
L_skating: penalize horizontal foot displacement when contact=1.
Uses HumanML3D 263-d layout; recovers 22 joints via recover_from_ric.
"""

import torch
import sys
import os

# HumanML3D motion_process for recover_from_ric
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loaders.humanml.scripts.motion_process import recover_from_ric

JOINTS_NUM = 22
# Foot joint indices (HumanML3D 22-joint): left 7,10 ; right 8,11
FID_L = [7, 10]
FID_R = [8, 11]
FOOT_JOINT_IDS = FID_L + FID_R  # [7, 10, 8, 11]
# Contact channels 259:263 correspond to these 4 foot contacts (same order)
CONTACT_DIM_START = 259


def motion_263_to_joints(motion_263, mean=None, std=None):
    """
    motion_263: (B, 263, 1, T) or (B, T, 263)
    Returns: (B, T, 22, 3) world positions.
    """
    if motion_263.dim() == 4:
        motion_263 = motion_263.squeeze(2).permute(0, 2, 1)  # (B, T, 263)
    if mean is not None and std is not None:
        motion_263 = motion_263 * std.to(motion_263.device) + mean.to(motion_263.device)
    joints = recover_from_ric(motion_263, JOINTS_NUM)  # (B, T, 22, 3)
    return joints


def physics_losses(pred_xstart, mean=None, std=None, lambda_contact=0.5, lambda_penetrate=1.0, lambda_skating=0.5,
                   floor_height=0.0, contact_threshold=0.5):
    """
    pred_xstart: (B, 263, 1, T) normalized predicted clean motion.
    mean, std: (263,) for denormalization (e.g. HumanML3D).
    Returns: dict with 'L_contact', 'L_penetrate', 'L_skating', 'physics_total' (weighted sum).
    """
    B, _, _, T = pred_xstart.shape
    device = pred_xstart.device

    joints = motion_263_to_joints(pred_xstart, mean=mean, std=std)  # (B, T, 22, 3)
    contact = pred_xstart[:, CONTACT_DIM_START:, 0, :]  # (B, 4, T)
    contact_mask = (contact > contact_threshold).float()  # (B, 4, T)

    # Foot positions (B, T, 4, 3) for joints 7,10,8,11
    foot_pos = joints[:, :, FOOT_JOINT_IDS, :]  # (B, T, 4, 3)
    foot_y = foot_pos[:, :, :, 1]  # (B, T, 4)
    foot_xz = foot_pos[:, :, :, [0, 2]]  # (B, T, 4, 2)

    # L_penetrate: ReLU(-(y - floor_height))
    below = (floor_height - foot_y).clamp(min=0.0)  # (B, T, 4)
    L_penetrate = below.mean()

    # L_contact: velocity of foot when contact=1. contact_mask (B, 4, T), we need vel (B, T, 4)
    vel = foot_pos[:, 1:, :, :] - foot_pos[:, :-1, :, :]  # (B, T-1, 4, 3)
    vel_norm_sq = (vel ** 2).sum(dim=-1)  # (B, T-1, 4)
    contact_at_t = contact_mask[:, :, 1:].permute(0, 2, 1)  # (B, T-1, 4)
    L_contact = (vel_norm_sq * contact_at_t).sum() / (contact_at_t.sum().clamp(min=1e-6))

    # L_skating: horizontal displacement when contact=1
    foot_xz_prev = foot_xz[:, :-1, :, :]  # (B, T-1, 4, 2)
    foot_xz_curr = foot_xz[:, 1:, :, :]
    horiz_disp_sq = ((foot_xz_curr - foot_xz_prev) ** 2).sum(dim=-1)  # (B, T-1, 4)
    L_skating = (horiz_disp_sq * contact_at_t).sum() / (contact_at_t.sum().clamp(min=1e-6))

    physics_total = lambda_contact * L_contact + lambda_penetrate * L_penetrate + lambda_skating * L_skating
    return {
        'L_contact': L_contact,
        'L_penetrate': L_penetrate,
        'L_skating': L_skating,
        'physics_total': physics_total,
    }
