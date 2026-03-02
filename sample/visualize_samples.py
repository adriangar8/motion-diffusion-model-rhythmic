####################################################################################[start]
####################################################################################[start]

"""

Visualize generated motion samples.

Converts 263-dim HumanML3D features back to (T, 22, 3) joint positions
using recover_from_ric, then renders skeleton animations as GIF/MP4.

Usage:
    python -m sample.visualize_samples \
        --sample_dir ./save/audio_stage2/samples_audio \
        --humanml_dir ./dataset/HumanML3D \
        --output_dir ./save/audio_stage2/vis \
        --fps 20

    # Compare audio vs text
    python -m sample.visualize_samples \
        --sample_dir ./save/audio_stage2/samples_audio \
        --compare_dir ./save/audio_stage2/samples_text \
        --humanml_dir ./dataset/HumanML3D \
        --output_dir ./save/audio_stage2/vis_compare

"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from glob import glob

# -- HumanML3D kinematic chains for skeleton drawing --

KINEMATIC_CHAINS = [
    [0, 2, 5, 8, 11],     # right leg
    [0, 1, 4, 7, 10],     # left leg
    [0, 3, 6, 9, 12, 15], # spine → head
    [9, 14, 17, 19, 21],  # right arm
    [9, 13, 16, 18, 20],  # left arm
]

CHAIN_COLORS = ['red', 'blue', 'black', 'orange', 'green']

def recover_joints(motion_263, mean, std, joints_num=22):

    """
    
    Convert (T, 263) normalized features → (T, 22, 3) joint positions.
    Uses MDM's recover_from_ric.
    
    """

    sys.path.insert(0, '.')
    from data_loaders.humanml.scripts.motion_process import recover_from_ric

    # -- denormalize --

    motion = motion_263 * std + mean

    # -- recover_from_ric expects (B, T, 263) tensor --

    motion_t = torch.from_numpy(motion).float().unsqueeze(0) # (1, T, 263)
    joints = recover_from_ric(motion_t, joints_num) # (1, T, 22, 3)
    joints = joints.squeeze(0).numpy() # (T, 22, 3)

    return joints

def plot_skeleton_frame(ax, joints, title='', elev=110, azim=-90):

    """Plot a single skeleton frame on a 3D axis."""

    ax.cla()

    for chain, color in zip(KINEMATIC_CHAINS, CHAIN_COLORS):
        
        ax.plot3D(
            joints[chain, 0],
            joints[chain, 2],
            joints[chain, 1],
            color=color, linewidth=2, marker='o', markersize=3
        )

    # -- axis limits --

    mid_x = joints[:, 0].mean()
    mid_z = joints[:, 2].mean()

    ax.set_xlim(mid_x - 1.0, mid_x + 1.0)
    ax.set_ylim(mid_z - 1.0, mid_z + 1.0)
    ax.set_zlim(0, 2.0)

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=elev, azim=azim)


def create_animation(joints, output_path, fps=20, title=''):

    """Create a GIF animation of a skeleton sequence."""

    T = joints.shape[0]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        t_str = f'{title}  t={frame} ({frame/fps:.1f}s)' if title else f't={frame} ({frame/fps:.1f}s)'
        plot_skeleton_frame(ax, joints[frame], title=t_str)

    anim = FuncAnimation(fig, update, frames=T, interval=1000//fps)

    if output_path.endswith('.gif'):
        anim.save(output_path, writer=PillowWriter(fps=fps))
    else:
        anim.save(output_path, fps=fps)

    plt.close(fig)
    print(f"Saved animation: {output_path} ({T} frames, {T/fps:.1f}s)")

def create_keyframe_grid(joints, output_path, fps=20, title='', n_frames=8):

    """Save a grid of keyframes from the motion."""

    T = joints.shape[0]
    frame_indices = np.linspace(0, T - 1, n_frames).astype(int)

    fig, axes = plt.subplots(1, n_frames, figsize=(3 * n_frames, 4),
                              subplot_kw={'projection': '3d'})

    if n_frames == 1:
        axes = [axes]

    for i, t in enumerate(frame_indices):
        plot_skeleton_frame(axes[i], joints[t], title=f'{t/fps:.1f}s')

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved keyframes: {output_path}")

def create_comparison(joints_a, joints_b, output_path, fps=20,
                       title_a='Audio+Text', title_b='Text only'):

    """Side-by-side animation comparing two motions."""

    T = min(joints_a.shape[0], joints_b.shape[0])

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    def update(frame):
        plot_skeleton_frame(ax1, joints_a[frame],
                            title=f'{title_a}  ({frame/fps:.1f}s)')
        plot_skeleton_frame(ax2, joints_b[frame],
                            title=f'{title_b}  ({frame/fps:.1f}s)')

    anim = FuncAnimation(fig, update, frames=T, interval=1000//fps)

    if output_path.endswith('.gif'):
        anim.save(output_path, writer=PillowWriter(fps=fps))
    else:
        anim.save(output_path, fps=fps)

    plt.close(fig)
    print(f"Saved comparison: {output_path} ({T} frames)")

def create_comparison_keyframes(joints_a, joints_b, output_path, fps=20,
                                 title_a='Audio+Text', title_b='Text only',
                                 n_frames=6):

    """Side-by-side keyframe comparison."""

    T = min(joints_a.shape[0], joints_b.shape[0])
    frame_indices = np.linspace(0, T - 1, n_frames).astype(int)

    fig, axes = plt.subplots(2, n_frames, figsize=(3 * n_frames, 8),
                              subplot_kw={'projection': '3d'})

    for i, t in enumerate(frame_indices):
        plot_skeleton_frame(axes[0, i], joints_a[t], title=f'{t/fps:.1f}s')
        plot_skeleton_frame(axes[1, i], joints_b[t], title=f'{t/fps:.1f}s')

    axes[0, 0].set_ylabel(title_a, fontsize=12)
    axes[1, 0].set_ylabel(title_b, fontsize=12)

    plt.suptitle(f'{title_a} vs {title_b}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved comparison keyframes: {output_path}")

def compute_motion_stats(joints):

    """Compute motion statistics for analysis."""

    T = joints.shape[0]

    # -- velocity: frame-to-frame displacement --
    
    vel = np.diff(joints, axis=0) # (T-1, 22, 3)
    speed = np.linalg.norm(vel, axis=-1) # (T-1, 22)

    # -- root trajectory --
    
    root = joints[:, 0] # (T, 3)
    root_disp = np.linalg.norm(root[-1] - root[0])

    stats = {
        'n_frames': T,
        'duration_s': T / 20.0,
        'avg_joint_speed': speed.mean(),
        'max_joint_speed': speed.max(),
        'root_height_mean': root[:, 1].mean(),
        'root_height_std': root[:, 1].std(),
        'root_displacement': root_disp,
        'joint_pos_std': joints.std(),
    }

    return stats

def main():

    parser = argparse.ArgumentParser(description='Visualize generated motion samples')

    parser.add_argument('--sample_dir', type=str, required=True,
                        help='Directory with .npy sample files')
    parser.add_argument('--compare_dir', type=str, default='',
                        help='Optional: second directory for comparison')
    parser.add_argument('--humanml_dir', type=str, default='./dataset/HumanML3D')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--max_samples', type=int, default=3,
                        help='Max number of samples to visualize')
    parser.add_argument('--format', type=str, default='gif',
                        choices=['gif', 'mp4', 'png'],
                        help='Output format (gif=animation, png=keyframes only)')

    args = parser.parse_args()

    if not args.output_dir:
        args.output_dir = os.path.join(args.sample_dir, 'vis')

    os.makedirs(args.output_dir, exist_ok=True)

    # -- load normalization stats --

    mean = np.load(os.path.join(args.humanml_dir, 'Mean.npy'))
    std = np.load(os.path.join(args.humanml_dir, 'Std.npy'))

    # -- find sample files --

    sample_files = sorted(glob(os.path.join(args.sample_dir, '*.npy')))[:args.max_samples]
    print(f"Found {len(sample_files)} samples in {args.sample_dir}")

    # -- process each sample --

    all_joints = []

    for f in sample_files:

        name = os.path.splitext(os.path.basename(f))[0]
        motion = np.load(f) # (T, 263)

        print(f"\nProcessing {name}: {motion.shape}")

        joints = recover_joints(motion, mean, std)
        all_joints.append(joints)

        print(f"  Recovered joints: {joints.shape}")

        # -- stats --

        stats = compute_motion_stats(joints)

        for k, v in stats.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        # -- keyframe grid (always) --

        create_keyframe_grid(
            joints,
            os.path.join(args.output_dir, f'{name}_keyframes.png'),
            fps=args.fps,
            title=name,
        )

        # -- animation (if requested) --

        if args.format in ['gif', 'mp4']:

            ext = args.format
            create_animation(
                joints,
                os.path.join(args.output_dir, f'{name}.{ext}'),
                fps=args.fps,
                title=name,
            )

    # -- comparison mode --

    if args.compare_dir:

        compare_files = sorted(glob(os.path.join(args.compare_dir, '*.npy')))[:args.max_samples]
        print(f"\n=== Comparison with {args.compare_dir} ===")
        print(f"Found {len(compare_files)} comparison samples")

        compare_joints = []

        for f in compare_files:
            motion = np.load(f)
            joints = recover_joints(motion, mean, std)
            compare_joints.append(joints)

            stats = compute_motion_stats(joints)
            name = os.path.splitext(os.path.basename(f))[0]
            print(f"  {name}: avg_speed={stats['avg_joint_speed']:.4f}, "
                  f"root_h_std={stats['root_height_std']:.4f}")

        # -- side-by-side comparisons --

        n_pairs = min(len(all_joints), len(compare_joints))

        for i in range(n_pairs):

            # -- keyframe comparison --

            create_comparison_keyframes(
                all_joints[i], compare_joints[i],
                os.path.join(args.output_dir, f'compare_keyframes_{i:02d}.png'),
                fps=args.fps,
                title_a='Audio+Text',
                title_b='Text only',
            )

            # -- animation comparison --

            if args.format in ['gif', 'mp4']:

                create_comparison(
                    all_joints[i], compare_joints[i],
                    os.path.join(args.output_dir, f'compare_{i:02d}.{args.format}'),
                    fps=args.fps,
                    title_a='Audio+Text',
                    title_b='Text only',
                )

    print(f"\nAll outputs saved to {args.output_dir}")

if __name__ == '__main__':
    main()

####################################################################################[end]
####################################################################################[end]