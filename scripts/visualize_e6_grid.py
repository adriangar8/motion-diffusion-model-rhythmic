"""
E6 Grid Visualization — qualitative proof that text and audio are independent controls.

Renders a 3×2 video grid:
  Row 0 (Sub-A): Fixed text "a person walks forward", audio varies (70/120/160 BPM)
  Row 1 (Sub-B): Fixed audio (120 BPM pop), text varies (walk/jump/kick)

Each cell shows a 3D skeleton + beat timeline strip.
Also writes a static filmstrip PNG with 4 key frames per cell + beat annotations.

Usage:
  python scripts/visualize_e6_grid.py [--out_dir DIR] [--fps 20]
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

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--motion_dir", default="/Data/yash.bhardwaj/eval/e6_motions")
parser.add_argument("--audio_dir",  default="/Data/yash.bhardwaj/eval/e6_audio")
parser.add_argument("--out_dir",    default="/Data/yash.bhardwaj/eval/e6_viz")
parser.add_argument("--fps",        type=float, default=20.0)
parser.add_argument("--clip_sec",   type=float, default=5.0,
                    help="Seconds to render per cell (keeps video manageable)")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
FPS = args.fps
CLIP_FRAMES = int(args.clip_sec * FPS)

# ── experiment layout ─────────────────────────────────────────────────────────
# Each entry: (motion_stem, audio_stem, col_label, row_label)
CELLS = [
    # Row 0 — Sub-A: fixed text, variable audio
    ("subA_slow70",  "slow_ballad_70bpm",  "70 BPM\n(Slow Ballad)",   "Sub-A\nFixed text:\n'a person\nwalks forward'"),
    ("subA_med120",  "medium_pop_120bpm",  "120 BPM\n(Medium Pop)",   None),
    ("subA_fast160", "fast_dnb_160bpm",    "160 BPM\n(Fast D&B)",     None),
    # Row 1 — Sub-B: fixed audio, variable text
    ("subB_walk",    "medium_pop_120bpm",  "'a person\nwalks forward'", "Sub-B\nFixed audio:\n120 BPM Pop"),
    ("subB_jump",    "medium_pop_120bpm",  "'a person\njumps'",         None),
    ("subB_kick",    "medium_pop_120bpm",  "'a person kicks\nright leg'", None),
]
NROWS, NCOLS = 2, 3

# ── skeleton setup ────────────────────────────────────────────────────────────
PARENTS = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
BONES   = [(i, PARENTS[i]) for i in range(1, 22)]

LIMB_COLORS = {
    "spine": "#F5A623", "l_leg": "#4A90E2",
    "r_leg": "#7B68EE", "l_arm": "#50C878", "r_arm": "#FF6B6B",
}

def bone_color(i, j):
    if i in [3,6,9,12] or j in [3,6,9,12,15]: return LIMB_COLORS["spine"]
    if i in [0,1,4,7]  or j in [1,4,7,10]:    return LIMB_COLORS["l_leg"]
    if i in [0,2,5,8]  or j in [2,5,8,11]:    return LIMB_COLORS["r_leg"]
    if i in [9,13,16,18] or j in [13,16,18,20]: return LIMB_COLORS["l_arm"]
    return LIMB_COLORS["r_arm"]

# ── load all cell data ────────────────────────────────────────────────────────
print("Loading all motions and audio...")
cell_data = []
for mot_stem, aud_stem, col_lbl, row_lbl in CELLS:
    mot_path = os.path.join(args.motion_dir, f"{mot_stem}.npy")
    aud_path = os.path.join(args.audio_dir,  f"{aud_stem}.wav")

    joints = load_humanml_motion(mot_path)          # (T, 22, 3)
    T = min(joints.shape[0], CLIP_FRAMES)
    joints = joints[:T]
    duration = T / FPS

    y_full, sr = librosa.load(aud_path, sr=None)
    y = y_full[:int(duration * sr)]

    music_beats  = get_music_beats(aud_path, FPS, max_frames=T)
    motion_beats = get_motion_beats(joints, fps=FPS)
    sigma_ba     = BA_SIGMA_TIME_SEC * FPS
    bas          = ba_score(music_beats, motion_beats, sigma=sigma_ba)

    # Skeleton centred around root
    jc = joints - joints[:, 0:1, :]
    jc += np.array([0, joints[:, 0, 1].mean(), 0])
    all_pts  = jc.reshape(-1, 3)
    pad = 0.25
    x_lim = (all_pts[:,0].min()-pad, all_pts[:,0].max()+pad)
    y_lim = (all_pts[:,1].min()-pad, all_pts[:,1].max()+pad)
    z_lim = (all_pts[:,2].min()-pad, all_pts[:,2].max()+pad)

    # Velocity envelope
    vel = np.mean(np.sqrt(np.sum((joints[1:]-joints[:-1])**2, axis=2)), axis=1)
    vel_s = gaussian_filter1d(vel, SMOOTH_TIME_SEC * FPS)
    vel_norm = vel_s / (vel_s.max() + 1e-8) * 0.4

    cell_data.append(dict(
        joints=jc, T=T, duration=duration,
        y=y, sr=sr,
        music_beats=music_beats, motion_beats=motion_beats,
        bas=bas, vel_norm=vel_norm,
        x_lim=x_lim, y_lim=y_lim, z_lim=z_lim,
        col_lbl=col_lbl, row_lbl=row_lbl,
        mot_stem=mot_stem, aud_stem=aud_stem,
    ))
    print(f"  {mot_stem}: T={T}, BAS={bas:.4f}, "
          f"music={len(music_beats)}, motion={len(motion_beats)}")

# ── helpers ───────────────────────────────────────────────────────────────────
BG    = "#0D0D0D"
BG2   = "#0C0C1A"
AMBER = "#FFB300"
CORAL = "#F06292"
MINT  = "#80CBC4"

def draw_skeleton(ax, jf, x_lim, y_lim, z_lim, azim=-70, elev=12):
    ax.set_facecolor(BG)
    for bi, bj in BONES:
        ax.plot([jf[bi,0], jf[bj,0]], [jf[bi,2], jf[bj,2]], [jf[bi,1], jf[bj,1]],
                color=bone_color(bi, bj), linewidth=2.0, solid_capstyle="round")
    ax.scatter(jf[:,0], jf[:,2], jf[:,1], c="#FFFFFF", s=12, zorder=5, depthshade=False)
    ax.set_xlim(x_lim); ax.set_ylim(z_lim); ax.set_zlim(y_lim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#222222")
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")

def draw_beat_strip(ax, cd, t_now, wav_max_override=None):
    times   = np.linspace(0, cd["duration"], len(cd["y"]))
    y       = cd["y"]
    wav_max = (wav_max_override or np.abs(y).max()) * 1.1
    vel_t   = np.arange(len(cd["vel_norm"])) / FPS
    mb_t    = cd["music_beats"]  / FPS
    mot_t   = cd["motion_beats"] / FPS

    ax.fill_between(times,  y, alpha=0.15, color="#7986CB")
    ax.fill_between(times, -y, alpha=0.15, color="#7986CB")
    ax.plot(times,  y, color="#9FA8DA", lw=0.4, alpha=0.4)
    ax.axhline(0, color="#333355", lw=0.6, alpha=0.5)

    for bt in mb_t:
        ax.axvline(bt, color=AMBER, lw=1.4, alpha=0.9, zorder=5)
        ax.axvline(bt, color=AMBER, lw=5.0, alpha=0.10, zorder=4)
    for bt in mot_t:
        ax.axvline(bt, color=CORAL, lw=1.4, alpha=0.9, zorder=6)
        ax.axvline(bt, color=CORAL, lw=5.0, alpha=0.12, zorder=5)

    ax.plot(vel_t, cd["vel_norm"], color=MINT, lw=1.2, alpha=0.85, zorder=4)
    ax.axvline(t_now, color="#FFFFFF", lw=1.6, alpha=1.0, zorder=8)
    ax.axvline(t_now, color="#FFE082", lw=6.0, alpha=0.15, zorder=7)

    ax.set_facecolor(BG2)
    ax.set_xlim(0, cd["duration"])
    ax.set_ylim(-wav_max, wav_max)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#1E1E33")

# ── render grid video ─────────────────────────────────────────────────────────
T_min = min(cd["T"] for cd in cell_data)

# Figure: 3 cols × 2 rows, each cell has skeleton (top) + wavestrip (bottom)
CELL_W, CELL_H = 4.0, 3.8
FIG_W = NCOLS * CELL_W
FIG_H = NROWS * CELL_H + 0.6  # +0.6 for title row

print(f"\nRendering {T_min} frames ({T_min/FPS:.1f}s)...")
frames_rgb = []

for fi in range(T_min):
    t_now = fi / FPS

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=110, facecolor=BG)
    # outer gridspec: NROWS rows
    outer = gridspec.GridSpec(NROWS, NCOLS, figure=fig,
                              hspace=0.08, wspace=0.06,
                              left=0.07, right=0.98, top=0.93, bottom=0.04)

    for idx, cd in enumerate(cell_data):
        row = idx // NCOLS
        col = idx  % NCOLS

        # Each cell: 2 sub-rows (skeleton 3:1 split with wavestrip)
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[row, col],
            height_ratios=[3, 1], hspace=0.04
        )
        ax3d  = fig.add_subplot(inner[0], projection="3d", facecolor=BG)
        ax_wav= fig.add_subplot(inner[1], facecolor=BG2)

        draw_skeleton(ax3d,  cd["joints"][fi], cd["x_lim"], cd["y_lim"], cd["z_lim"])
        draw_beat_strip(ax_wav, cd, t_now)

        # BAS badge top-right of skeleton
        ax3d.text2D(0.97, 0.97,
                    f"BAS {cd['bas']:.3f}",
                    transform=ax3d.transAxes,
                    color="#FFD700", fontsize=6.5, ha="right", va="top",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#0D0D0D", ec="#555500", alpha=0.8))

        # Column header (only row 0)
        if row == 0:
            ax3d.set_title(cd["col_lbl"], color="#CCCCCC", fontsize=7.5,
                           pad=3, fontfamily="monospace")

        # Row label (only col 0)
        if col == 0 and cd["row_lbl"]:
            ax3d.text2D(-0.18, 0.5, cd["row_lbl"],
                        transform=ax3d.transAxes,
                        color="#AAAAAA", fontsize=6.5, ha="center", va="center",
                        rotation=90, fontfamily="monospace")

    # Legend once, bottom-right
    legend_elems = [
        Line2D([0],[0], color=AMBER, lw=1.8, label="Music beats"),
        Line2D([0],[0], color=CORAL, lw=1.8, label="Motion beats"),
        Line2D([0],[0], color=MINT,  lw=1.4, label="Velocity"),
    ]
    fig.legend(handles=legend_elems, loc="lower right", fontsize=7,
               facecolor=BG2, edgecolor="#2A2A45", labelcolor="#C5CAE9",
               ncol=3, framealpha=0.9, bbox_to_anchor=(0.98, 0.005))

    fig.suptitle(
        f"E6: Text × Audio Independence  |  wav2clip_beataware  ag=1.5 tg=2.5"
        f"  |  t={t_now:.2f}s",
        color="#DDDDDD", fontsize=9, y=0.975, fontfamily="monospace"
    )

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    h_px = fig.canvas.get_width_height()[1]
    w_px = fig.canvas.get_width_height()[0]
    buf = buf.reshape(h_px, w_px, 4)[:, :, :3]
    # libx264 requires even dimensions
    buf = buf[:buf.shape[0]//2*2, :buf.shape[1]//2*2, :]
    frames_rgb.append(buf.copy())
    plt.close(fig)

    if fi % 20 == 0:
        print(f"  {fi+1}/{T_min} frames...")

# ── write video ───────────────────────────────────────────────────────────────
print("Writing video...")
silent = os.path.join(args.out_dir, "e6_grid_silent.mp4")
writer = imageio.get_writer(silent, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
for frame in frames_rgb:
    writer.append_data(frame)
writer.close()

# Use 120 BPM track as the reference audio for the video
ref_audio = os.path.join(args.audio_dir, "medium_pop_120bpm.wav")
y_ref, sr_ref = librosa.load(ref_audio, sr=None)
y_ref = y_ref[:int(T_min / FPS * sr_ref)]
audio_tmp = os.path.join(args.out_dir, "_ref_audio.wav")
sf.write(audio_tmp, y_ref, sr_ref)

out_video = os.path.join(args.out_dir, "e6_grid.mp4")
subprocess.run([
    "ffmpeg", "-y", "-i", silent, "-i", audio_tmp,
    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out_video
], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
os.remove(silent)
os.remove(audio_tmp)
print(f"Video saved: {out_video}")

# ── filmstrip PNG ─────────────────────────────────────────────────────────────
# 4 key frames per cell: evenly spaced across the clip
print("Rendering filmstrip PNG...")

KF = 4  # key frames per cell
key_frames = [int(T_min * (k + 0.5) / KF) for k in range(KF)]

STRIP_W = KF * CELL_W
STRIP_H = NROWS * (CELL_H * 0.85)   # slightly taller cells for static view
DPI_STRIP = 130

fig_strip = plt.figure(figsize=(STRIP_W, STRIP_H), dpi=DPI_STRIP, facecolor=BG)
strip_outer = gridspec.GridSpec(
    NROWS, NCOLS, figure=fig_strip,
    hspace=0.06, wspace=0.06,
    left=0.06, right=0.99, top=0.91, bottom=0.04
)

for idx, cd in enumerate(cell_data):
    row = idx // NCOLS
    col = idx  % NCOLS

    # Each cell: top half = 4-frame skeleton filmstrip, bottom = full wavestrip
    inner = gridspec.GridSpecFromSubplotSpec(
        2, KF, subplot_spec=strip_outer[row, col],
        height_ratios=[3, 1], hspace=0.03, wspace=0.02
    )

    for ki, fi in enumerate(key_frames):
        ax3d = fig_strip.add_subplot(inner[0, ki], projection="3d", facecolor=BG)
        draw_skeleton(ax3d, cd["joints"][fi], cd["x_lim"], cd["y_lim"], cd["z_lim"])

        # Frame time label
        ax3d.set_title(f"{fi/FPS:.1f}s", color="#666677", fontsize=5.5,
                       pad=1, fontfamily="monospace")

        if ki == 0:
            # BAS badge on first frame only
            ax3d.text2D(0.97, 0.97, f"BAS {cd['bas']:.3f}",
                        transform=ax3d.transAxes,
                        color="#FFD700", fontsize=5.5, ha="right", va="top",
                        fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=0.15", fc=BG, ec="#555500", alpha=0.8))

        if row == 0 and ki == KF // 2:
            ax3d.set_title(cd["col_lbl"] + f"\n{fi/FPS:.1f}s",
                           color="#CCCCCC", fontsize=6.5, pad=2, fontfamily="monospace")

    # Wavestrip spanning all 4 frames — use gridspec spanning trick
    ax_wav = fig_strip.add_subplot(inner[1, :], facecolor=BG2)
    # Draw full wavestrip with all 4 key-frame playheads
    times = np.linspace(0, cd["duration"], len(cd["y"]))
    y     = cd["y"]
    wav_max = np.abs(y).max() * 1.1
    vel_t = np.arange(len(cd["vel_norm"])) / FPS
    mb_t  = cd["music_beats"]  / FPS
    mot_t = cd["motion_beats"] / FPS

    ax_wav.fill_between(times,  y, alpha=0.15, color="#7986CB")
    ax_wav.fill_between(times, -y, alpha=0.15, color="#7986CB")
    ax_wav.plot(times,  y, color="#9FA8DA", lw=0.35, alpha=0.35)
    ax_wav.axhline(0, color="#333355", lw=0.5, alpha=0.5)

    for bt in mb_t:
        ax_wav.axvline(bt, color=AMBER, lw=1.2, alpha=0.85, zorder=5)
    for bt in mot_t:
        ax_wav.axvline(bt, color=CORAL, lw=1.2, alpha=0.85, zorder=6)

    ax_wav.plot(vel_t, cd["vel_norm"], color=MINT, lw=1.0, alpha=0.85, zorder=4)

    for fi_kf in key_frames:
        ax_wav.axvline(fi_kf / FPS, color="#FFFFFF", lw=1.0, alpha=0.65,
                       linestyle="--", zorder=8)

    ax_wav.set_facecolor(BG2)
    ax_wav.set_xlim(0, cd["duration"])
    ax_wav.set_ylim(-wav_max, wav_max)
    ax_wav.set_xticks([]); ax_wav.set_yticks([])
    for sp in ax_wav.spines.values():
        sp.set_edgecolor("#1E1E33")

    if col == 0 and cd["row_lbl"]:
        ax_wav.set_ylabel(cd["row_lbl"], color="#AAAAAA", fontsize=5.5,
                          fontfamily="monospace", rotation=0, labelpad=48, va="center")

# Shared legend
legend_elems = [
    Line2D([0],[0], color=AMBER, lw=1.6, label="Music beats"),
    Line2D([0],[0], color=CORAL, lw=1.6, label="Motion beats"),
    Line2D([0],[0], color=MINT,  lw=1.2, label="Velocity"),
    Line2D([0],[0], color="#FFFFFF", lw=1.0, linestyle="--", label="Key frame"),
]
fig_strip.legend(handles=legend_elems, loc="lower right", fontsize=6.5,
                 facecolor=BG2, edgecolor="#2A2A45", labelcolor="#C5CAE9",
                 ncol=4, framealpha=0.9, bbox_to_anchor=(0.99, 0.005))

fig_strip.suptitle(
    "E6: Text × Audio Independence  |  wav2clip_beataware  ag=1.5 tg=2.5\n"
    "Sub-A (top): fixed text 'a person walks forward', audio BPM varies  |  "
    "Sub-B (bottom): fixed audio 120 BPM, text prompt varies",
    color="#DDDDDD", fontsize=8.5, y=0.975, fontfamily="monospace"
)

filmstrip_path = os.path.join(args.out_dir, "e6_filmstrip.png")
fig_strip.savefig(filmstrip_path, dpi=DPI_STRIP, bbox_inches="tight",
                  facecolor=BG)
plt.close(fig_strip)
print(f"Filmstrip saved: {filmstrip_path}")
print("\nAll done.")
