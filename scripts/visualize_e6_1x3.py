"""
1×3 grid: fixed text "a person walks forward", three audio tempos side by side.

Layout per cell:
  Top:    3D skeleton animation
  Bottom: waveform strip with music beats (amber) + motion beats (coral)
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
parser.add_argument("--audio_dir",  default="/Data/yash.bhardwaj/eval/e6_audio")
parser.add_argument("--out_dir",    default="/Data/yash.bhardwaj/eval/e6_viz")
parser.add_argument("--fps",        type=float, default=20.0)
parser.add_argument("--clip_sec",   type=float, default=5.0)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
FPS = args.fps
CLIP_FRAMES = int(args.clip_sec * FPS)

CELLS = [
    ("aist_slow_mWA0", "mWA0", "81.5 BPM  —  Waacking  (mWA0)"),
    ("aist_med_mJS3",  "mJS3", "110.3 BPM  —  Jazz Swing  (mJS3)"),
    ("aist_fast_mBR0", "mBR0", "160.7 BPM  —  Breakdance  (mBR0)"),
]

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

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading motions...")
cell_data = []
for mot_stem, aud_stem, col_lbl in CELLS:
    joints = load_humanml_motion(
        os.path.join(args.motion_dir, f"{mot_stem}.npy"))
    T = min(joints.shape[0], CLIP_FRAMES)
    joints = joints[:T]
    duration = T / FPS

    y_full, sr = librosa.load(
        os.path.join(args.audio_dir, f"{aud_stem}.wav"), sr=None)
    y = y_full[:int(duration * sr)]

    aud_path = os.path.join(args.audio_dir, f"{aud_stem}.wav")
    music_beats  = get_music_beats(aud_path, FPS, max_frames=T)
    motion_beats = get_motion_beats(joints, fps=FPS)
    bas = ba_score(music_beats, motion_beats, sigma=BA_SIGMA_TIME_SEC * FPS)

    jc = joints - joints[:, 0:1, :]
    jc += np.array([0, joints[:, 0, 1].mean(), 0])
    all_pts = jc.reshape(-1, 3)
    pad = 0.25
    x_lim = (all_pts[:,0].min()-pad, all_pts[:,0].max()+pad)
    y_lim = (all_pts[:,1].min()-pad, all_pts[:,1].max()+pad)
    z_lim = (all_pts[:,2].min()-pad, all_pts[:,2].max()+pad)

    vel = np.mean(np.sqrt(np.sum((joints[1:]-joints[:-1])**2, axis=2)), axis=1)
    vel_s = gaussian_filter1d(vel, SMOOTH_TIME_SEC * FPS)
    vel_norm = vel_s / (vel_s.max() + 1e-8) * 0.4

    cell_data.append(dict(
        joints=jc, T=T, duration=duration,
        y=y, sr=sr, aud_path=aud_path,
        music_beats=music_beats, motion_beats=motion_beats,
        bas=bas, vel_norm=vel_norm,
        x_lim=x_lim, y_lim=y_lim, z_lim=z_lim,
        col_lbl=col_lbl, mot_stem=mot_stem,
    ))
    print(f"  {mot_stem}: BAS={bas:.4f}  "
          f"music={len(music_beats)}  motion={len(motion_beats)}")

T_render = min(cd["T"] for cd in cell_data)

# ── render ─────────────────────────────────────────────────────────────────────
CELL_W, CELL_H = 4.4, 4.2
FIG_W = 3 * CELL_W          # 13.2
FIG_H = CELL_H + 0.55        # header space
DPI   = 110

print(f"\nRendering {T_render} frames ({T_render/FPS:.1f}s)...")
frames_rgb = []

for fi in range(T_render):
    t_now = fi / FPS

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=BG)
    outer = gridspec.GridSpec(
        1, 3, figure=fig,
        hspace=0.0, wspace=0.06,
        left=0.03, right=0.99, top=0.90, bottom=0.03
    )

    for col, cd in enumerate(cell_data):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[0, col],
            height_ratios=[3, 1], hspace=0.04
        )
        ax3d  = fig.add_subplot(inner[0], projection="3d", facecolor=BG)
        ax_wav= fig.add_subplot(inner[1], facecolor=BG2)

        # ── skeleton ──────────────────────────────────────────────────────────
        jf = cd["joints"][fi]
        ax3d.set_facecolor(BG)
        for bi, bj in BONES:
            ax3d.plot([jf[bi,0],jf[bj,0]], [jf[bi,2],jf[bj,2]], [jf[bi,1],jf[bj,1]],
                      color=bone_color(bi,bj), linewidth=2.2, solid_capstyle="round")
        ax3d.scatter(jf[:,0], jf[:,2], jf[:,1],
                     c="#FFFFFF", s=14, zorder=5, depthshade=False)
        ax3d.set_xlim(cd["x_lim"]); ax3d.set_ylim(cd["z_lim"]); ax3d.set_zlim(cd["y_lim"])
        ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])
        ax3d.view_init(elev=12, azim=-70)
        ax3d.grid(False)
        for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
            pane.fill = False; pane.set_edgecolor("#222222")
        ax3d.set_xlabel(""); ax3d.set_ylabel(""); ax3d.set_zlabel("")

        # BAS badge
        ax3d.text2D(0.97, 0.96, f"BAS {cd['bas']:.3f}",
                    transform=ax3d.transAxes,
                    color="#FFD700", fontsize=7.5, ha="right", va="top",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec="#665500", alpha=0.85))

        # Column header
        ax3d.set_title(cd["col_lbl"], color="#DDDDDD", fontsize=9,
                       pad=4, fontfamily="monospace", fontweight="bold")

        # ── wavestrip ─────────────────────────────────────────────────────────
        times   = np.linspace(0, cd["duration"], len(cd["y"]))
        wav_max = np.abs(cd["y"]).max() * 1.1
        vel_t   = np.arange(len(cd["vel_norm"])) / FPS
        mb_t    = cd["music_beats"]  / FPS
        mot_t   = cd["motion_beats"] / FPS

        ax_wav.fill_between(times,  cd["y"], alpha=0.15, color="#7986CB")
        ax_wav.fill_between(times, -cd["y"], alpha=0.15, color="#7986CB")
        ax_wav.plot(times,  cd["y"], color="#9FA8DA", lw=0.4, alpha=0.4)
        ax_wav.axhline(0, color="#333355", lw=0.6, alpha=0.5)

        for bt in mb_t:
            ax_wav.axvline(bt, color=AMBER, lw=1.5, alpha=0.9, zorder=5)
            ax_wav.axvline(bt, color=AMBER, lw=5.5, alpha=0.10, zorder=4)
        for bt in mot_t:
            ax_wav.axvline(bt, color=CORAL, lw=1.5, alpha=0.9, zorder=6)
            ax_wav.axvline(bt, color=CORAL, lw=5.5, alpha=0.12, zorder=5)

        ax_wav.plot(vel_t, cd["vel_norm"], color=MINT, lw=1.3, alpha=0.88, zorder=4)

        # playhead
        ax_wav.axvline(t_now, color="#FFFFFF", lw=1.8, alpha=1.0, zorder=8)
        ax_wav.axvline(t_now, color="#FFE082", lw=6.5, alpha=0.16, zorder=7)

        ax_wav.set_facecolor(BG2)
        ax_wav.set_xlim(0, cd["duration"])
        ax_wav.set_ylim(-wav_max, wav_max)
        ax_wav.set_xticks([]); ax_wav.set_yticks([])
        for sp in ax_wav.spines.values(): sp.set_edgecolor("#1E1E33")

        if col == 0:
            ax_wav.set_ylabel("beats", color="#556677", fontsize=6.5,
                              fontfamily="monospace")

    # legend + title
    legend_elems = [
        Line2D([0],[0], color=AMBER, lw=1.8, label="Music beats"),
        Line2D([0],[0], color=CORAL, lw=1.8, label="Motion beats"),
        Line2D([0],[0], color=MINT,  lw=1.4, label="Motion velocity"),
        Line2D([0],[0], color="#FFFFFF", lw=1.6, label="Playhead"),
    ]
    fig.legend(handles=legend_elems, loc="lower right", fontsize=7.5,
               facecolor=BG2, edgecolor="#2A2A45", labelcolor="#C5CAE9",
               ncol=4, framealpha=0.92, bbox_to_anchor=(0.99, 0.005))

    fig.suptitle(
        f'Prompt: "a person walks forward"  |  wav2clip_beataware  ag=1.5 tg=2.5'
        f'  |  t = {t_now:.2f}s',
        color="#DDDDDD", fontsize=9, y=0.975, fontfamily="monospace"
    )

    fig.canvas.draw()
    w_px, h_px = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(h_px, w_px, 4)[:, :, :3]
    buf = buf[:buf.shape[0]//2*2, :buf.shape[1]//2*2, :]  # ensure even dims
    frames_rgb.append(buf.copy())
    plt.close(fig)

    if fi % 20 == 0:
        print(f"  {fi+1}/{T_render} frames...")

# ── write video ────────────────────────────────────────────────────────────────
print("Writing video...")
out_video  = os.path.join(args.out_dir, "e6_1x3_aist_walk_vs_bpm.mp4")
silent     = out_video.replace(".mp4", "_silent.mp4")
audio_tmp  = out_video.replace(".mp4", "_audio.wav")

writer = imageio.get_writer(silent, fps=FPS, codec="libx264",
                             quality=8, macro_block_size=1)
for frame in frames_rgb:
    writer.append_data(frame)
writer.close()

# Use the medium-tempo track (mJS3) as the video's background audio
y_ref, sr_ref = librosa.load(
    os.path.join(args.audio_dir, "mJS3.wav"), sr=None)
y_ref = y_ref[:int(T_render / FPS * sr_ref)]
sf.write(audio_tmp, y_ref, sr_ref)

subprocess.run([
    "ffmpeg", "-y", "-i", silent, "-i", audio_tmp,
    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out_video
], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
os.remove(silent)
os.remove(audio_tmp)
print(f"\nDone!  {out_video}")
