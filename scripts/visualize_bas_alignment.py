"""
Visualize beat-aligned motion — MDM-style top-down view.

Root-centred skeleton with moving floor grid + blue trajectory trail.
Wavestrip below with music beats (amber), motion beats (coral), velocity.

Usage:
  python scripts/visualize_bas_alignment.py \
      --motion_path <path>.npy \
      --audio_path  <path>.wav \
      --out_video   <path>.mp4 \
      --song_label  "mMH3 (Middle Hip-hop)" \
      [--fps 20] [--prompt "..."] [--model_label "..."]
"""

import argparse, os, sys
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
parser.add_argument("--motion_path", required=True)
parser.add_argument("--audio_path",  required=True)
parser.add_argument("--out_video",   required=True)
parser.add_argument("--song_label",  default="")
parser.add_argument("--fps",         type=float, default=20.0)
parser.add_argument("--prompt",      default="a person dances to music")
parser.add_argument("--model_label", default="Stage-2 wav2clip_beataware")
args_cli = parser.parse_args()

MOTION_PATH = args_cli.motion_path
AUDIO_PATH  = args_cli.audio_path
OUT_VIDEO   = args_cli.out_video
SONG_LABEL  = args_cli.song_label
FPS         = args_cli.fps
PROMPT      = args_cli.prompt
MODEL_LABEL = args_cli.model_label
OUT_DIR     = os.path.dirname(OUT_VIDEO) or "."

BG  = "#0D0D0D"
BG2 = "#0C0C1A"
AMBER = "#FFB300"
CORAL = "#F06292"
MINT  = "#80CBC4"
SCALE = 1.3       # humanml display scale (same as MDM plot_script.py)
RADIUS = 3

# ── HumanML3D 22-joint kinematic chains (same as MDM) ────────────────────────
KINEMATIC_CHAINS = [
    [0, 2, 5, 8, 11],          # right leg
    [0, 1, 4, 7, 10],          # left  leg
    [0, 3, 6, 9, 12, 15],      # spine + head
    [9, 14, 17, 19, 21],       # right arm
    [9, 13, 16, 18, 20],       # left  arm
]
CHAIN_COLORS = ["#4A90E2", "#7B68EE", "#F5A623", "#FF6B6B", "#50C878"]
LINEWIDTHS   = [3.0, 3.0, 3.5, 2.5, 2.5]

os.makedirs(OUT_DIR, exist_ok=True)

# ── load & process ────────────────────────────────────────────────────────────
print("Loading motion...")
joints_raw = load_humanml_motion(MOTION_PATH) * SCALE   # (T, 22, 3)
T = joints_raw.shape[0]
duration = T / FPS
print(f"  {T} frames @ {FPS} FPS = {duration:.1f}s")

# Floor at y=0
height_offset = joints_raw[:, :, 1].min()
joints_raw[:, :, 1] -= height_offset

# Store trajectory before root-centering
trajec = joints_raw[:, 0, [0, 2]].copy()   # (T, 2): root x,z

# Root-centre each frame (body stays at origin, floor moves)
joints_disp = joints_raw.copy()
joints_disp[:, :, 0] -= joints_disp[:, 0:1, 0]
joints_disp[:, :, 2] -= joints_disp[:, 0:1, 2]

# Global bounds for floor extent
MINS = joints_raw.min(axis=0).min(axis=0)
MAXS = joints_raw.max(axis=0).max(axis=0)

print("Loading audio...")
y_full, sr = librosa.load(AUDIO_PATH, sr=None)
y_audio = y_full[:int(duration * sr)]
times = np.linspace(0, duration, len(y_audio))

print("Extracting beats...")
music_beats  = get_music_beats(AUDIO_PATH, FPS, max_frames=T)
motion_beats = get_motion_beats(joints_raw / SCALE, fps=FPS)  # unscaled for BAS
sigma_ba     = BA_SIGMA_TIME_SEC * FPS
bas          = ba_score(music_beats, motion_beats, sigma=sigma_ba)
print(f"  Music beats: {len(music_beats)}, Motion beats: {len(motion_beats)}, BAS={bas:.4f}")

mb_t  = music_beats  / FPS
mot_t = motion_beats / FPS

vel = np.mean(np.sqrt(np.sum((joints_raw[1:]-joints_raw[:-1])**2, axis=2)), axis=1)
vel_s = gaussian_filter1d(vel, SMOOTH_TIME_SEC * FPS)
vel_norm = vel_s / (vel_s.max() + 1e-8) * 0.5
vel_t = np.arange(len(vel)) / FPS

wav_max = np.abs(y_audio).max() * 1.1

# ── render ─────────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 10, 8
DPI = 110

print("Rendering frames...")
frames_rgb = []

for fi in range(T):
    t_now = fi / FPS

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=BG)
    gs = gridspec.GridSpec(2, 1, figure=fig,
                           height_ratios=[3, 1],
                           hspace=0.05,
                           left=0.03, right=0.97,
                           top=0.93, bottom=0.06)

    ax3d  = fig.add_subplot(gs[0], projection="3d")
    ax_wav = fig.add_subplot(gs[1])

    # ── 3D skeleton (MDM-style: root-centred, floor moves) ────────────────────
    ax3d.set_facecolor(BG)
    jf = joints_disp[fi]
    cx = trajec[fi, 0]
    cz = trajec[fi, 1]

    def nz(z): return -z

    # Moving floor grid
    gx = np.linspace(MINS[0] - cx, MAXS[0] - cx, 7)
    gz = np.linspace(nz(MAXS[2] - cz), nz(MINS[2] - cz), 7)
    for gxi in gx:
        ax3d.plot([gxi, gxi], [0, 0], [gz[0], gz[-1]],
                  color="#2A2A2A", lw=0.6, alpha=0.7, zorder=0)
    for gzi in gz:
        ax3d.plot([gx[0], gx[-1]], [0, 0], [gzi, gzi],
                  color="#2A2A2A", lw=0.6, alpha=0.7, zorder=0)

    # Trajectory trail
    trail_start = max(0, fi - 25)
    trail_pts = trajec[trail_start:fi+1]
    if len(trail_pts) > 1:
        tx = trail_pts[:, 0] - cx
        tz = nz(trail_pts[:, 1] - cz)
        alphas = np.linspace(0.1, 0.7, len(trail_pts))
        for k in range(len(trail_pts) - 1):
            ax3d.plot([tx[k], tx[k+1]], [0.02, 0.02], [tz[k], tz[k+1]],
                      color="#5599FF", lw=2.5, alpha=float(alphas[k+1]), zorder=2)

    # Skeleton
    for chain, color, lw in zip(KINEMATIC_CHAINS, CHAIN_COLORS, LINEWIDTHS):
        xs = jf[chain, 0]
        ys = jf[chain, 1]
        zs = nz(jf[chain, 2])
        ax3d.plot3D(xs, ys, zs, color=color, linewidth=lw,
                    marker='o', markersize=3, markerfacecolor='white')

    r = RADIUS
    ax3d.set_xlim3d([-r/2, r/2])
    ax3d.set_ylim3d([0, r])
    ax3d.set_zlim3d([-r/3, r*2/3])
    ax3d.view_init(elev=120, azim=-90)
    ax3d.dist = 7.5
    ax3d.set_axis_off()
    ax3d.grid(False)
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(BG)

    # BAS badge
    ax3d.text2D(0.97, 0.97, f"BAS {bas:.3f}",
                transform=ax3d.transAxes,
                color="#FFD700", fontsize=10, ha="right", va="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec="#665500", alpha=0.88))

    # ── wavestrip ─────────────────────────────────────────────────────────────
    ax_wav.set_facecolor(BG2)
    ax_wav.fill_between(times,  y_audio, alpha=0.18, color="#7986CB")
    ax_wav.fill_between(times, -y_audio, alpha=0.18, color="#7986CB")
    ax_wav.plot(times,  y_audio, color="#9FA8DA", lw=0.5, alpha=0.45)
    ax_wav.axhline(0, color="#333355", lw=0.8, alpha=0.6, zorder=1)

    for bt in mb_t:
        ax_wav.axvline(bt, color=AMBER, lw=1.8, alpha=0.95, zorder=5)
        ax_wav.axvline(bt, color=AMBER, lw=6.0, alpha=0.12, zorder=4)
    for bt in mot_t:
        ax_wav.axvline(bt, color=CORAL, lw=1.8, alpha=0.95, zorder=6)
        ax_wav.axvline(bt, color=CORAL, lw=6.0, alpha=0.15, zorder=5)

    ax_wav.plot(vel_t, vel_norm, color=MINT, lw=1.4, alpha=0.90, zorder=4)
    ax_wav.axvline(t_now, color="#FFFFFF", lw=2.0, alpha=1.0, zorder=8)
    ax_wav.axvline(t_now, color="#FFE082", lw=7.0, alpha=0.18, zorder=7)

    ax_wav.set_xlim(0, duration)
    ax_wav.set_ylim(-wav_max, wav_max)
    ax_wav.set_xlabel("Time (s)", color="#6670AA", fontsize=8)
    ax_wav.tick_params(colors="#555580", labelsize=7)
    for sp in ax_wav.spines.values():
        sp.set_edgecolor("#1E1E33")

    legend_elems = [
        Line2D([0],[0], color=AMBER, lw=2.0, label=f"Music beats ({len(mb_t)})"),
        Line2D([0],[0], color=CORAL, lw=2.0, label=f"Motion beats ({len(mot_t)})"),
        Line2D([0],[0], color=MINT,  lw=1.6, label="Motion velocity"),
        Line2D([0],[0], color="#5599FF", lw=1.5, label="Root trail"),
        Line2D([0],[0], color="#FFFFFF", lw=2.0, label="Playhead"),
    ]
    legend = ax_wav.legend(handles=legend_elems, loc="upper right", fontsize=7.5,
                  facecolor=BG2, edgecolor="#2A2A45", labelcolor="#C5CAE9",
                  ncol=3, framealpha=0.95, borderpad=0.7)
    legend.set_zorder(20)

    fig.suptitle(
        f"{MODEL_LABEL}  |  {SONG_LABEL}  |  BAS = {bas:.4f}\n"
        f'Prompt: "{PROMPT}"  |  Frame {fi+1}/{T}  ({t_now:.2f}s / {duration:.1f}s)',
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
        print(f"  {fi+1}/{T} frames rendered...")

# ── write video ───────────────────────────────────────────────────────────────
print("Writing video frames...")
silent_path = OUT_VIDEO.replace(".mp4", "_silent.mp4")
writer = imageio.get_writer(silent_path, fps=FPS, codec="libx264",
                             quality=8, macro_block_size=1)
for frame in frames_rgb:
    writer.append_data(frame)
writer.close()

_song_stem = os.path.splitext(os.path.basename(AUDIO_PATH))[0]
audio_trim_path = os.path.join(OUT_DIR, f"{_song_stem}_trim.wav")
sf.write(audio_trim_path, y_audio, sr)

print("Merging audio with ffmpeg...")
subprocess.run([
    "ffmpeg", "-y",
    "-i", silent_path, "-i", audio_trim_path,
    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
    OUT_VIDEO
], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
os.remove(silent_path)
os.remove(audio_trim_path)

print(f"\nDone! Video saved to: {OUT_VIDEO}")
print(f"Duration: {duration:.1f}s, {T} frames @ {FPS} FPS")
print(f"BAS={bas:.4f} | Music beats: {len(mb_t)} | Motion beats: {len(mot_t)}")
