"""
1×3 spatial grid — fixed text "a person walks forward", three AIST audio tempos.

Format per cell (from visualize_bas_alignment.py):
  Top-left:  Front-Left 3D view  (azim=-70, elev=15)
  Top-right: Front     3D view   (azim=0,   elev=5)
  Bottom:    Wavestrip with music beats / motion beats / velocity / playhead

Camera: follows the character as they walk through 3D space.
No audio in the output (three different tracks can't be merged meaningfully).
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
import librosa
import soundfile as sf
import subprocess
import imageio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval.beat_align_score import (
    load_humanml_motion, get_music_beats, get_motion_beats,
    ba_score, BA_SIGMA_TIME_SEC, SMOOTH_TIME_SEC
)

parser = argparse.ArgumentParser()
parser.add_argument("--motion_dir", default="/Data/yash.bhardwaj/eval/e6_motions")
parser.add_argument("--audio_dir",  default="/Data/yash.bhardwaj/datasets/aist/audio_test_10")
parser.add_argument("--out_dir",    default="/Data/yash.bhardwaj/eval/e6_viz")
parser.add_argument("--fps",        type=float, default=20.0)
parser.add_argument("--clip_sec",   type=float, default=5.0)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
FPS = args.fps
CLIP_FRAMES = int(args.clip_sec * FPS)

CELLS = [
    ("aist_slow_mWA0", "mWA0", "81 BPM  —  Waacking (mWA0)"),
    ("aist_med_mJS3",  "mJS3", "110 BPM  —  Jazz Swing (mJS3)"),
    ("aist_fast_mBR0", "mBR0", "161 BPM  —  Breakdance (mBR0)"),
]

# ── skeleton ──────────────────────────────────────────────────────────────────
PARENTS = [0,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]
BONES   = [(i, PARENTS[i]) for i in range(1, 22)]
LIMB_COLORS = {
    "spine":"#F5A623","l_leg":"#4A90E2",
    "r_leg":"#7B68EE","l_arm":"#50C878","r_arm":"#FF6B6B",
}
def bone_color(i, j):
    if i in [3,6,9,12]  or j in [3,6,9,12,15]: return LIMB_COLORS["spine"]
    if i in [0,1,4,7]   or j in [1,4,7,10]:    return LIMB_COLORS["l_leg"]
    if i in [0,2,5,8]   or j in [2,5,8,11]:    return LIMB_COLORS["r_leg"]
    if i in [9,13,16,18] or j in [13,16,18,20]: return LIMB_COLORS["l_arm"]
    return LIMB_COLORS["r_arm"]

BG = "#0D0D0D"; BG2 = "#0C0C1A"
AMBER = "#FFB300"; CORAL = "#F06292"; MINT = "#80CBC4"

# ── load ──────────────────────────────────────────────────────────────────────
print("Loading motions...")
cell_data = []
for mot_stem, aud_stem, col_lbl in CELLS:
    joints_raw = load_humanml_motion(
        os.path.join(args.motion_dir, f"{mot_stem}.npy"))   # (T, 22, 3)
    T = min(joints_raw.shape[0], CLIP_FRAMES)
    joints_raw = joints_raw[:T]
    duration   = T / FPS

    # Shift so the first frame root sits at origin (keep trajectory from there)
    joints = joints_raw - joints_raw[0:1, 0:1, :]           # (T,22,3) — root starts at (0,0,0)
    joints[:, :, 1] += joints_raw[0, 0, 1]                  # restore floor height

    aud_path = os.path.join(args.audio_dir, f"{aud_stem}.wav")
    y_full, sr = librosa.load(aud_path, sr=None)
    y = y_full[:int(duration * sr)]

    music_beats  = get_music_beats(aud_path, FPS, max_frames=T)
    motion_beats = get_motion_beats(joints, fps=FPS)
    bas = ba_score(music_beats, motion_beats, sigma=BA_SIGMA_TIME_SEC * FPS)

    # Per-frame: local body extent around root (for camera half-width)
    # Compute body spread relative to root at each frame
    body_spread = joints - joints[:, 0:1, :]          # relative to root
    max_spread = np.abs(body_spread).max() + 0.35     # generous padding

    # Velocity for wavestrip overlay
    vel = np.mean(np.sqrt(np.sum((joints[1:]-joints[:-1])**2, axis=2)), axis=1)
    vel_s = gaussian_filter1d(vel, SMOOTH_TIME_SEC * FPS)
    vel_norm = vel_s / (vel_s.max() + 1e-8) * 0.4

    # Floor grid extents: full trajectory bounding box (for drawing floor)
    all_root = joints[:, 0, :]
    x_pad, z_pad = max_spread * 1.5, max_spread * 1.5

    cell_data.append(dict(
        joints=joints, T=T, duration=duration,
        y=y, sr=sr, aud_path=aud_path,
        music_beats=music_beats, motion_beats=motion_beats,
        bas=bas, vel_norm=vel_norm,
        max_spread=max_spread,
        all_root=all_root,
        col_lbl=col_lbl,
    ))
    root_travel = np.linalg.norm(joints[-1,0,:] - joints[0,0,:])
    print(f"  {mot_stem}: BAS={bas:.4f}  root_travel={root_travel:.2f}m  "
          f"music={len(music_beats)}  motion={len(motion_beats)}")

T_render = min(cd["T"] for cd in cell_data)

# ── figure layout ─────────────────────────────────────────────────────────────
# Each cell: two 3D views (like bas_alignment) + wavestrip below
CELL_W = 5.6
CELL_H = 4.4
FIG_W  = 3 * CELL_W           # 16.8
FIG_H  = CELL_H + 0.55        # title + content
DPI    = 100

def draw_skeleton_following(ax, joints_all, fi, max_spread, azim, elev, title):
    """Draw skeleton at frame fi with camera following the root."""
    jf   = joints_all[fi]                    # (22, 3)
    root = jf[0]                             # current root position

    sp = max_spread
    x_lim = (root[0] - sp, root[0] + sp)
    y_lim = (-0.05,          root[1] + 1.5)  # floor to slightly above head
    z_lim = (root[2] - sp, root[2] + sp)

    ax.set_facecolor(BG)
    for bi, bj in BONES:
        ax.plot([jf[bi,0],jf[bj,0]], [jf[bi,2],jf[bj,2]], [jf[bi,1],jf[bj,1]],
                color=bone_color(bi,bj), linewidth=2.2, solid_capstyle="round")
    ax.scatter(jf[:,0], jf[:,2], jf[:,1],
               c="#FFFFFF", s=12, zorder=5, depthshade=False)

    # Floor grid following the root
    gx = np.linspace(x_lim[0], x_lim[1], 6)
    gz = np.linspace(z_lim[0], z_lim[1], 6)
    for gxi in gx:
        ax.plot([gxi, gxi], [gz[0], gz[-1]], [0, 0],
                color="#2A2A2A", linewidth=0.5, alpha=0.6)
    for gzi in gz:
        ax.plot([gx[0], gx[-1]], [gzi, gzi], [0, 0],
                color="#2A2A2A", linewidth=0.5, alpha=0.6)

    # Trajectory trail — last N frames root path
    trail_start = max(0, fi - 15)
    trail_pts   = joints_all[trail_start:fi+1, 0, :]   # (N, 3)
    if len(trail_pts) > 1:
        alphas = np.linspace(0.1, 0.55, len(trail_pts))
        for k in range(len(trail_pts) - 1):
            ax.plot(
                [trail_pts[k,0], trail_pts[k+1,0]],
                [trail_pts[k,2], trail_pts[k+1,2]],
                [0.02, 0.02],
                color="#5599FF", linewidth=1.5, alpha=float(alphas[k+1]), zorder=3
            )

    ax.set_xlim(x_lim); ax.set_ylim(z_lim); ax.set_zlim(y_lim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False; pane.set_edgecolor("#222222")
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    ax.set_title(title, color="#666677", fontsize=7, pad=2, fontfamily="monospace")


def draw_wavestrip(ax, cd, t_now):
    times   = np.linspace(0, cd["duration"], len(cd["y"]))
    wav_max = np.abs(cd["y"]).max() * 1.1
    vel_t   = np.arange(len(cd["vel_norm"])) / FPS
    mb_t    = cd["music_beats"]  / FPS
    mot_t   = cd["motion_beats"] / FPS

    ax.fill_between(times,  cd["y"], alpha=0.15, color="#7986CB")
    ax.fill_between(times, -cd["y"], alpha=0.15, color="#7986CB")
    ax.plot(times,  cd["y"], color="#9FA8DA", lw=0.4, alpha=0.4)
    ax.axhline(0, color="#333355", lw=0.6, alpha=0.5, zorder=1)

    for bt in mb_t:
        ax.axvline(bt, color=AMBER, lw=1.5, alpha=0.92, zorder=5)
        ax.axvline(bt, color=AMBER, lw=5.5, alpha=0.10, zorder=4)
    for bt in mot_t:
        ax.axvline(bt, color=CORAL, lw=1.5, alpha=0.92, zorder=6)
        ax.axvline(bt, color=CORAL, lw=5.5, alpha=0.12, zorder=5)

    ax.plot(vel_t, cd["vel_norm"], color=MINT, lw=1.3, alpha=0.88, zorder=4)
    ax.axvline(t_now, color="#FFFFFF", lw=1.8, alpha=1.0,  zorder=8)
    ax.axvline(t_now, color="#FFE082", lw=6.5, alpha=0.16, zorder=7)

    ax.set_facecolor(BG2)
    ax.set_xlim(0, cd["duration"])
    ax.set_ylim(-wav_max, wav_max)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor("#1E1E33")


# ── render ─────────────────────────────────────────────────────────────────────
print(f"\nRendering {T_render} frames ({T_render/FPS:.1f}s)...")
frames_rgb = []

for fi in range(T_render):
    t_now = fi / FPS

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=BG)
    outer = gridspec.GridSpec(
        1, 3, figure=fig,
        hspace=0.0, wspace=0.06,
        left=0.02, right=0.99, top=0.91, bottom=0.03
    )

    for col, cd in enumerate(cell_data):
        # Each cell: 2 3D views (left/right) on top, wavestrip on bottom
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[0, col],
            height_ratios=[3, 1], width_ratios=[1.1, 1],
            hspace=0.04, wspace=0.05
        )
        ax_fl  = fig.add_subplot(inner[0, 0], projection="3d", facecolor=BG)
        ax_fr  = fig.add_subplot(inner[0, 1], projection="3d", facecolor=BG)
        ax_wav = fig.add_subplot(inner[1, :], facecolor=BG2)

        draw_skeleton_following(ax_fl, cd["joints"], fi, cd["max_spread"],
                                azim=-70, elev=15, title="Front-Left")
        draw_skeleton_following(ax_fr, cd["joints"], fi, cd["max_spread"],
                                azim=0,   elev=5,  title="Front")
        draw_wavestrip(ax_wav, cd, t_now)

        # BAS badge on the front-left view
        ax_fl.text2D(0.97, 0.97, f"BAS {cd['bas']:.3f}",
                     transform=ax_fl.transAxes,
                     color="#FFD700", fontsize=7, ha="right", va="top",
                     fontfamily="monospace",
                     bbox=dict(boxstyle="round,pad=0.22", fc=BG, ec="#665500", alpha=0.88))

        # Column header with BPM label
        # Place title above the two 3D views using figure text
        # Use the front-left axis title for the full column label
        mid_x = (ax_fl.get_position().x0 + ax_fr.get_position().x1) / 2
        top_y  = ax_fl.get_position().y1 + 0.005
        fig.text(mid_x, top_y + 0.028, cd["col_lbl"],
                 color="#DDDDDD", fontsize=9, ha="center", va="bottom",
                 fontfamily="monospace", fontweight="bold")

    # Shared legend bottom-right
    legend_elems = [
        Line2D([0],[0], color=AMBER,    lw=1.8, label="Music beats"),
        Line2D([0],[0], color=CORAL,    lw=1.8, label="Motion beats"),
        Line2D([0],[0], color=MINT,     lw=1.4, label="Motion velocity"),
        Line2D([0],[0], color="#5599FF",lw=1.5, label="Root trail"),
        Line2D([0],[0], color="#FFFFFF",lw=1.6, label="Playhead"),
    ]
    fig.legend(handles=legend_elems, loc="lower right", fontsize=7.5,
               facecolor=BG2, edgecolor="#2A2A45", labelcolor="#C5CAE9",
               ncol=5, framealpha=0.92, bbox_to_anchor=(0.99, 0.0))

    fig.suptitle(
        f'Prompt: "a person walks forward"  |  wav2clip_beataware  ag=1.5 tg=2.5'
        f'  |  t={t_now:.2f}s / {T_render/FPS:.1f}s',
        color="#DDDDDD", fontsize=9, y=0.98, fontfamily="monospace"
    )

    fig.canvas.draw()
    w_px, h_px = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(h_px, w_px, 4)[:, :, :3]
    buf = buf[:buf.shape[0]//2*2, :buf.shape[1]//2*2, :]
    frames_rgb.append(buf.copy())
    plt.close(fig)

    if fi % 20 == 0:
        print(f"  {fi+1}/{T_render} frames...")

# ── write video (no audio — 3 different tracks) ────────────────────────────────
print("Writing video (no audio)...")
out_video = os.path.join(args.out_dir, "e6_1x3_aist_spatial.mp4")
writer = imageio.get_writer(out_video, fps=FPS, codec="libx264",
                             quality=8, macro_block_size=1)
for frame in frames_rgb:
    writer.append_data(frame)
writer.close()
print(f"\nDone!  {out_video}")
