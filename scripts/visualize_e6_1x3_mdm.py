"""
1×3 spatial grid — same view style as MDM's plot_3d_motion.

Single front view per cell (elev=120, azim=-90).
Character stays root-centred each frame; the floor grid moves underneath,
creating the "walking through space" effect (same as the MOSPA sample videos).

No audio in output (three different audio tracks can't be mixed).
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter1d
import librosa
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
parser.add_argument("--seed_tag",   type=str, default="",
                    help="If set, motion stems become {track}_seed{tag}")
parser.add_argument("--out_name",   type=str, default="e6_1x3_mdm_style.mp4")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
FPS        = args.fps
CLIP_FRAMES= int(args.clip_sec * FPS)
RADIUS     = 3        # axis half-width, same as MDM default
SCALE      = 1.3      # humanml scale from plot_script.py

DEFAULT_CELLS = [
    ("aist_slow_mWA0", "mWA0", "81 BPM  —  Waacking (mWA0)"),
    ("aist_med_mJS3",  "mJS3", "110 BPM  —  Jazz Swing (mJS3)"),
    ("aist_fast_mBR0", "mBR0", "161 BPM  —  Breakdance (mBR0)"),
]

# ── kinematic chains (HumanML3D 22-joint, same as MDM) ────────────────────────
KINEMATIC_CHAINS = [
    [0, 2, 5, 8, 11],          # right leg
    [0, 1, 4, 7, 10],          # left  leg
    [0, 3, 6, 9, 12, 15],      # spine + head
    [9, 14, 17, 19, 21],       # right arm
    [9, 13, 16, 18, 20],       # left  arm
]
CHAIN_COLORS = ["#4A90E2", "#7B68EE", "#F5A623", "#FF6B6B", "#50C878"]
LINEWIDTHS   = [3.0, 3.0, 3.5, 2.5, 2.5]

BG = "#0D0D0D"; BG2 = "#0C0C1A"
AMBER = "#FFB300"; CORAL = "#F06292"; MINT = "#80CBC4"

# ── build cell list ───────────────────────────────────────────────────────────
if args.seed_tag:
    CELLS = [(f"{aud}_seed{args.seed_tag}", aud, lbl)
             for _, aud, lbl in DEFAULT_CELLS]
else:
    CELLS = DEFAULT_CELLS

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading motions...")
cell_data = []
for mot_stem, aud_stem, col_lbl in CELLS:
    joints_raw = load_humanml_motion(
        os.path.join(args.motion_dir, f"{mot_stem}.npy"))   # (T, 22, 3)
    T = min(joints_raw.shape[0], CLIP_FRAMES)
    joints_raw = joints_raw[:T].copy() * SCALE

    # Floor at y=0 (subtract global minimum y across whole sequence)
    height_offset = joints_raw[:, :, 1].min()
    joints_raw[:, :, 1] -= height_offset

    # Store full trajectory (original root x, z before per-frame centering)
    trajec = joints_raw[:, 0, [0, 2]].copy()   # (T, 2): root x,z per frame

    # Root-centre every frame in XZ (body stays at 0,z=0)
    joints_disp = joints_raw.copy()
    joints_disp[:, :, 0] -= joints_disp[:, 0:1, 0]
    joints_disp[:, :, 2] -= joints_disp[:, 0:1, 2]

    # Global XYZ bounds (for floor size)
    MINS = joints_raw.min(axis=0).min(axis=0)
    MAXS = joints_raw.max(axis=0).max(axis=0)

    duration  = T / FPS
    aud_path  = os.path.join(args.audio_dir, f"{aud_stem}.wav")
    y_full, sr = librosa.load(aud_path, sr=None)
    y_audio    = y_full[:int(duration * sr)]

    music_beats  = get_music_beats(aud_path, FPS, max_frames=T)
    motion_beats = get_motion_beats(joints_raw / SCALE, fps=FPS)  # unscaled for BAS
    bas = ba_score(music_beats, motion_beats, sigma=BA_SIGMA_TIME_SEC * FPS)

    vel = np.mean(np.sqrt(np.sum(
        (joints_raw[1:] - joints_raw[:-1]) ** 2, axis=2)), axis=1)
    vel_s    = gaussian_filter1d(vel, SMOOTH_TIME_SEC * FPS)
    vel_norm = vel_s / (vel_s.max() + 1e-8) * 0.4

    root_travel = np.linalg.norm(
        joints_raw[-1, 0, [0, 2]] - joints_raw[0, 0, [0, 2]])
    print(f"  {mot_stem}: BAS={bas:.4f}  travel={root_travel:.2f}m  "
          f"music={len(music_beats)}  motion={len(motion_beats)}")

    cell_data.append(dict(
        joints_disp=joints_disp, trajec=trajec, T=T, duration=duration,
        MINS=MINS, MAXS=MAXS,
        y=y_audio, sr=sr, aud_path=aud_path,
        music_beats=music_beats, motion_beats=motion_beats,
        bas=bas, vel_norm=vel_norm, col_lbl=col_lbl,
    ))

T_render = min(cd["T"] for cd in cell_data)

# ── figure ─────────────────────────────────────────────────────────────────────
CELL_W = 5.2
CELL_H = 4.8
FIG_W  = 3 * CELL_W   # 15.6
FIG_H  = CELL_H + 0.5
DPI    = 100

def draw_cell(fig, col_gs, cd, fi, t_now):
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=col_gs,
        height_ratios=[3.2, 1], hspace=0.04
    )
    ax3d  = fig.add_subplot(inner[0], projection="3d")
    ax_wav= fig.add_subplot(inner[1])

    # ── 3D skeleton ──────────────────────────────────────────────────────────
    ax3d.set_facecolor(BG)
    jf = cd["joints_disp"][fi]    # (22, 3): root-centred body

    # Moving floor — shifts so character appears to travel
    cx = cd["trajec"][fi, 0]
    cz = cd["trajec"][fi, 1]
    MINS, MAXS = cd["MINS"], cd["MAXS"]

    # Negate Z so that the -Z "forward" direction faces the camera correctly
    # (equivalent to azim flip; makes the person walk forward not backward)
    def nz(z): return -z

    # Floor as grid of lines (cleaner on dark BG than filled poly)
    gx = np.linspace(MINS[0] - cx,        MAXS[0] - cx,        7)
    gz = np.linspace(nz(MAXS[2] - cz),    nz(MINS[2] - cz),    7)
    for gxi in gx:
        ax3d.plot([gxi, gxi], [0, 0], [gz[0], gz[-1]],
                  color="#2A2A2A", lw=0.6, alpha=0.7, zorder=0)
    for gzi in gz:
        ax3d.plot([gx[0], gx[-1]], [0, 0], [gzi, gzi],
                  color="#2A2A2A", lw=0.6, alpha=0.7, zorder=0)

    # Trajectory trail on the floor (past N frames)
    trail_start = max(0, fi - 20)
    trail_pts = cd["trajec"][trail_start:fi+1]           # (N, 2): x, z
    if len(trail_pts) > 1:
        tx = trail_pts[:, 0] - cx
        tz = nz(trail_pts[:, 1] - cz)
        alphas = np.linspace(0.1, 0.7, len(trail_pts))
        for k in range(len(trail_pts) - 1):
            ax3d.plot([tx[k], tx[k+1]], [0.02, 0.02], [tz[k], tz[k+1]],
                      color="#5599FF", lw=2.0, alpha=float(alphas[k+1]), zorder=2)

    # Skeleton (negate Z to match floor orientation)
    for chain, color, lw in zip(KINEMATIC_CHAINS, CHAIN_COLORS, LINEWIDTHS):
        xs = jf[chain, 0]
        ys = jf[chain, 1]
        zs = nz(jf[chain, 2])
        ax3d.plot3D(xs, ys, zs, color=color, linewidth=lw,
                    marker='o', markersize=2.5, markerfacecolor='white')

    # Axis setup — same as MDM plot_script.py
    r = RADIUS
    ax3d.set_xlim3d([-r/2,  r/2])
    ax3d.set_ylim3d([0,     r  ])
    ax3d.set_zlim3d([-r/3,  r*2/3])
    ax3d.view_init(elev=120, azim=-90)
    ax3d.dist = 7.5

    ax3d.set_axis_off()
    ax3d.grid(False)
    ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])
    ax3d.set_facecolor(BG)
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(BG)

    # BAS badge
    ax3d.text2D(0.97, 0.97, f"BAS {cd['bas']:.3f}",
                transform=ax3d.transAxes,
                color="#FFD700", fontsize=8, ha="right", va="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec="#665500", alpha=0.88))

    # ── wavestrip ─────────────────────────────────────────────────────────────
    ax_wav.set_facecolor(BG2)
    times   = np.linspace(0, cd["duration"], len(cd["y"]))
    wav_max = np.abs(cd["y"]).max() * 1.1
    vel_t   = np.arange(len(cd["vel_norm"])) / FPS
    mb_t    = cd["music_beats"]  / FPS
    mot_t   = cd["motion_beats"] / FPS

    ax_wav.fill_between(times,  cd["y"], alpha=0.15, color="#7986CB")
    ax_wav.fill_between(times, -cd["y"], alpha=0.15, color="#7986CB")
    ax_wav.plot(times,  cd["y"], color="#9FA8DA", lw=0.4, alpha=0.4)
    ax_wav.axhline(0, color="#333355", lw=0.6, alpha=0.5, zorder=1)

    for bt in mb_t:
        ax_wav.axvline(bt, color=AMBER, lw=1.5, alpha=0.92, zorder=5)
        ax_wav.axvline(bt, color=AMBER, lw=5.5, alpha=0.10, zorder=4)
    for bt in mot_t:
        ax_wav.axvline(bt, color=CORAL, lw=1.5, alpha=0.92, zorder=6)
        ax_wav.axvline(bt, color=CORAL, lw=5.5, alpha=0.12, zorder=5)

    ax_wav.plot(vel_t, cd["vel_norm"], color=MINT, lw=1.3, alpha=0.88, zorder=4)
    ax_wav.axvline(t_now, color="#FFFFFF", lw=1.8, alpha=1.0, zorder=8)
    ax_wav.axvline(t_now, color="#FFE082", lw=6.5, alpha=0.16, zorder=7)

    ax_wav.set_xlim(0, cd["duration"])
    ax_wav.set_ylim(-wav_max, wav_max)
    ax_wav.set_xticks([]); ax_wav.set_yticks([])
    for sp in ax_wav.spines.values(): sp.set_edgecolor("#1E1E33")

    return ax3d  # return so caller can place col header above it

# ── render ─────────────────────────────────────────────────────────────────────
print(f"\nRendering {T_render} frames ({T_render/FPS:.1f}s)...")
frames_rgb = []

for fi in range(T_render):
    t_now = fi / FPS

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=BG)
    outer = gridspec.GridSpec(
        1, 3, figure=fig,
        hspace=0.0, wspace=0.04,
        left=0.01, right=0.99, top=0.91, bottom=0.02
    )

    for col, cd in enumerate(cell_data):
        ax3d = draw_cell(fig, outer[0, col], cd, fi, t_now)

        # Column header above the 3D view
        pos = ax3d.get_position()
        fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.01,
                 cd["col_lbl"], color="#DDDDDD", fontsize=9,
                 ha="center", va="bottom", fontfamily="monospace",
                 fontweight="bold")

    # Legend
    legend_elems = [
        Line2D([0],[0], color=AMBER,     lw=1.8, label="Music beats"),
        Line2D([0],[0], color=CORAL,     lw=1.8, label="Motion beats"),
        Line2D([0],[0], color=MINT,      lw=1.4, label="Motion velocity"),
        Line2D([0],[0], color="#5599FF", lw=1.5, label="Root trail"),
        Line2D([0],[0], color="#FFFFFF", lw=1.6, label="Playhead"),
    ]
    fig.legend(handles=legend_elems, loc="lower right", fontsize=7.5,
               facecolor=BG2, edgecolor="#2A2A45", labelcolor="#C5CAE9",
               ncol=5, framealpha=0.92, bbox_to_anchor=(0.99, 0.0))

    fig.suptitle(
        f'Prompt: "a person walks forward"  |  wav2clip_beataware  ag=1.5 tg=2.5'
        f'  |  t={t_now:.2f}s',
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

# ── write (no audio) ──────────────────────────────────────────────────────────
print("Writing video...")
out_video = os.path.join(args.out_dir, args.out_name)
writer = imageio.get_writer(out_video, fps=FPS, codec="libx264",
                             quality=8, macro_block_size=1)
for frame in frames_rgb:
    writer.append_data(frame)
writer.close()
print(f"\nDone!  {out_video}")
