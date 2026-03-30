"""
Side-by-side comparison: Ours (wav2clip_beataware) vs EDGE on same AIST++ track.

Layout per song:
  Left column:  Our model skeleton + wavestrip
  Right column: EDGE skeleton + wavestrip
  Shared audio, shared music beats; each has its own motion beats.

Both rendered with MDM-style top-down view, root-centred, moving floor, trail.

Usage:
  python scripts/visualize_sidebyside.py \
      --song mMH3 \
      --ours_path  <path>.npy \
      --edge_path  <path>.pkl \
      --audio_path <path>.wav \
      --out_video  <path>.mp4
"""

import argparse, os, sys, pickle
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
parser.add_argument("--song",       required=True, help="e.g. mMH3")
parser.add_argument("--ours_path",  required=True, help="HumanML3D .npy")
parser.add_argument("--edge_path",  required=True, help="EDGE .pkl")
parser.add_argument("--audio_path", required=True)
parser.add_argument("--out_video",  required=True)
parser.add_argument("--ours_fps",   type=float, default=20.0)
parser.add_argument("--edge_fps",   type=float, default=30.0)
args = parser.parse_args()

os.makedirs(os.path.dirname(args.out_video) or ".", exist_ok=True)

BG = "#0D0D0D"; BG2 = "#0C0C1A"
AMBER = "#FFB300"; CORAL = "#F06292"; MINT = "#80CBC4"
RADIUS = 3

# ── HumanML3D 22-joint chains ────────────────────────────────────────────────
HML_CHAINS = [
    [0,2,5,8,11], [0,1,4,7,10], [0,3,6,9,12,15],
    [9,14,17,19,21], [9,13,16,18,20],
]
HML_COLORS = ["#4A90E2","#7B68EE","#F5A623","#FF6B6B","#50C878"]
HML_LW     = [3.0, 3.0, 3.5, 2.5, 2.5]

# ── SMPL 24-joint chains ─────────────────────────────────────────────────────
SMPL_CHAINS = [
    [0,1,4,7,10], [0,2,5,8,11], [0,3,6,9,12,15],
    [9,14,17,19,21,23], [9,13,16,18,20,22],
]
SMPL_COLORS = ["#4A90E2","#7B68EE","#F5A623","#FF6B6B","#50C878"]
SMPL_LW     = [3.0, 3.0, 3.5, 2.5, 2.5]

# ── load ours (HumanML3D, Y=height) ──────────────────────────────────────────
print("Loading ours...")
SCALE_HML = 1.3
j_ours_raw = load_humanml_motion(args.ours_path) * SCALE_HML
T_ours = j_ours_raw.shape[0]

h_off = j_ours_raw[:,:,1].min()
j_ours_raw[:,:,1] -= h_off
trajec_ours = j_ours_raw[:, 0, [0, 2]].copy()
j_ours_disp = j_ours_raw.copy()
j_ours_disp[:,:,0] -= j_ours_disp[:,0:1,0]
j_ours_disp[:,:,2] -= j_ours_disp[:,0:1,2]
MINS_ours = j_ours_raw.min(axis=0).min(axis=0)
MAXS_ours = j_ours_raw.max(axis=0).max(axis=0)

# BAS for ours
dur_ours = T_ours / args.ours_fps
music_beats_ours  = get_music_beats(args.audio_path, args.ours_fps, max_frames=T_ours)
motion_beats_ours = get_motion_beats(j_ours_raw / SCALE_HML, fps=args.ours_fps)
bas_ours = ba_score(music_beats_ours, motion_beats_ours, sigma=BA_SIGMA_TIME_SEC*args.ours_fps)
print(f"  Ours: T={T_ours}, BAS={bas_ours:.4f}, music={len(music_beats_ours)}, motion={len(motion_beats_ours)}")

vel_ours = np.mean(np.sqrt(np.sum((j_ours_raw[1:]-j_ours_raw[:-1])**2, axis=2)), axis=1)
vel_ours_s = gaussian_filter1d(vel_ours, SMOOTH_TIME_SEC * args.ours_fps)
vel_ours_n = vel_ours_s / (vel_ours_s.max()+1e-8) * 0.5

# ── load EDGE (SMPL 24-joint, Z=height) ──────────────────────────────────────
print("Loading EDGE...")
with open(args.edge_path, 'rb') as f:
    edge_data = pickle.load(f)
j_edge_raw_smpl = edge_data['full_pose']   # (T, 24, 3): x,y,z where z=height
T_edge = j_edge_raw_smpl.shape[0]

# Convert EDGE axes: SMPL (x,y,z with z=height) → our convention (x,y,z with y=height)
j_edge_raw = np.zeros_like(j_edge_raw_smpl)
j_edge_raw[:,:,0] = j_edge_raw_smpl[:,:,0]   # x stays
j_edge_raw[:,:,1] = j_edge_raw_smpl[:,:,2]   # z→y (height)
j_edge_raw[:,:,2] = j_edge_raw_smpl[:,:,1]   # y→z (depth)

SCALE_EDGE = 1.3
j_edge_raw *= SCALE_EDGE

h_off_e = j_edge_raw[:,:,1].min()
j_edge_raw[:,:,1] -= h_off_e
trajec_edge = j_edge_raw[:, 0, [0, 2]].copy()
j_edge_disp = j_edge_raw.copy()
j_edge_disp[:,:,0] -= j_edge_disp[:,0:1,0]
j_edge_disp[:,:,2] -= j_edge_disp[:,0:1,2]
MINS_edge = j_edge_raw.min(axis=0).min(axis=0)
MAXS_edge = j_edge_raw.max(axis=0).max(axis=0)

# BAS for EDGE
dur_edge = T_edge / args.edge_fps
music_beats_edge  = get_music_beats(args.audio_path, args.edge_fps, max_frames=T_edge)
motion_beats_edge = get_motion_beats(j_edge_raw / SCALE_EDGE, fps=args.edge_fps)
bas_edge = ba_score(music_beats_edge, motion_beats_edge, sigma=BA_SIGMA_TIME_SEC*args.edge_fps)
print(f"  EDGE: T={T_edge}, BAS={bas_edge:.4f}, music={len(music_beats_edge)}, motion={len(motion_beats_edge)}")

vel_edge = np.mean(np.sqrt(np.sum((j_edge_raw[1:]-j_edge_raw[:-1])**2, axis=2)), axis=1)
vel_edge_s = gaussian_filter1d(vel_edge, SMOOTH_TIME_SEC * args.edge_fps)
vel_edge_n = vel_edge_s / (vel_edge_s.max()+1e-8) * 0.5

# ── shared audio ──────────────────────────────────────────────────────────────
render_fps  = args.ours_fps   # render at ours fps
render_dur  = min(dur_ours, dur_edge)
T_render    = int(render_dur * render_fps)

# Map render frame index → EDGE frame index by matching wall-clock time
# At render frame fi (t = fi / render_fps), EDGE frame = t * edge_fps
edge_frame_map = np.minimum(
    (np.arange(T_render) * args.edge_fps / render_fps).astype(int),
    T_edge - 1
)

y_full, sr = librosa.load(args.audio_path, sr=None)
y_audio = y_full[:int(render_dur * sr)]
times   = np.linspace(0, render_dur, len(y_audio))
wav_max = np.abs(y_audio).max() * 1.1

# ── draw helpers ──────────────────────────────────────────────────────────────
def draw_skeleton_mdm(ax, jf, trajec_all, fi, MINS, MAXS, chains, colors, lws, label, bas_val):
    ax.set_facecolor(BG)
    cx = trajec_all[fi, 0]
    cz = trajec_all[fi, 1]

    def nz(z): return -z

    gx = np.linspace(MINS[0]-cx, MAXS[0]-cx, 7)
    gz = np.linspace(nz(MAXS[2]-cz), nz(MINS[2]-cz), 7)
    for gxi in gx:
        ax.plot([gxi,gxi],[0,0],[gz[0],gz[-1]], color="#2A2A2A",lw=0.6,alpha=0.7,zorder=0)
    for gzi in gz:
        ax.plot([gx[0],gx[-1]],[0,0],[gzi,gzi], color="#2A2A2A",lw=0.6,alpha=0.7,zorder=0)

    # Trail
    trail_start = max(0, fi-25)
    trail_pts = trajec_all[trail_start:fi+1]
    if len(trail_pts) > 1:
        tx = trail_pts[:,0]-cx
        tz = nz(trail_pts[:,1]-cz)
        alphas = np.linspace(0.1,0.7,len(trail_pts))
        for k in range(len(trail_pts)-1):
            ax.plot([tx[k],tx[k+1]],[0.02,0.02],[tz[k],tz[k+1]],
                    color="#5599FF",lw=2.5,alpha=float(alphas[k+1]),zorder=2)

    for chain, color, lw in zip(chains, colors, lws):
        xs = jf[chain,0]
        ys = jf[chain,1]
        zs = nz(jf[chain,2])
        ax.plot3D(xs,ys,zs, color=color, linewidth=lw,
                  marker='o', markersize=2.5, markerfacecolor='white')

    r = RADIUS
    ax.set_xlim3d([-r/2,r/2])
    ax.set_ylim3d([0,r])
    ax.set_zlim3d([-r/3,r*2/3])
    ax.view_init(elev=120, azim=-90)
    ax.dist = 7.5
    ax.set_axis_off(); ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False; pane.set_edgecolor(BG)

    ax.text2D(0.5, 1.02, label, transform=ax.transAxes,
              color="#DDDDDD", fontsize=10, ha="center", va="bottom",
              fontfamily="monospace", fontweight="bold")
    ax.text2D(0.97,0.97, f"BAS {bas_val:.3f}",
              transform=ax.transAxes, color="#FFD700", fontsize=8,
              ha="right", va="top", fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.25",fc=BG,ec="#665500",alpha=0.88))


def draw_wavestrip(ax, t_now, mb_t, mot_t, vel_n, vel_fps, duration):
    ax.set_facecolor(BG2)
    ax.fill_between(times, y_audio, alpha=0.15, color="#7986CB")
    ax.fill_between(times,-y_audio, alpha=0.15, color="#7986CB")
    ax.plot(times, y_audio, color="#9FA8DA", lw=0.4, alpha=0.4)
    ax.axhline(0, color="#333355", lw=0.6, alpha=0.5, zorder=1)

    for bt in mb_t:
        ax.axvline(bt, color=AMBER, lw=1.5, alpha=0.92, zorder=5)
        ax.axvline(bt, color=AMBER, lw=5.5, alpha=0.10, zorder=4)
    for bt in mot_t:
        ax.axvline(bt, color=CORAL, lw=1.5, alpha=0.92, zorder=6)
        ax.axvline(bt, color=CORAL, lw=5.5, alpha=0.12, zorder=5)

    vt = np.arange(len(vel_n)) / vel_fps
    ax.plot(vt, vel_n, color=MINT, lw=1.3, alpha=0.88, zorder=4)
    ax.axvline(t_now, color="#FFFFFF", lw=1.8, alpha=1.0, zorder=8)
    ax.axvline(t_now, color="#FFE082", lw=6.5, alpha=0.16, zorder=7)

    ax.set_xlim(0, render_dur)
    ax.set_ylim(-wav_max, wav_max)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor("#1E1E33")


# ── render frames ─────────────────────────────────────────────────────────────
FIG_W, FIG_H = 12, 7
DPI = 110

mb_t_ours  = music_beats_ours[music_beats_ours < T_render] / render_fps
mot_t_ours = motion_beats_ours[motion_beats_ours < T_render] / render_fps
mb_t_edge  = music_beats_edge / args.edge_fps
mot_t_edge = motion_beats_edge / args.edge_fps
# Clip to render duration
mb_t_edge  = mb_t_edge[mb_t_edge <= render_dur]
mot_t_edge = mot_t_edge[mot_t_edge <= render_dur]

print(f"\nRendering {T_render} frames ({render_dur:.1f}s)...")
frames_rgb = []

for fi in range(T_render):
    t_now = fi / render_fps
    fi_edge = edge_frame_map[fi]

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=BG)
    outer = gridspec.GridSpec(2, 2, figure=fig,
                              height_ratios=[3, 1],
                              wspace=0.06, hspace=0.05,
                              left=0.02, right=0.98, top=0.92, bottom=0.04)

    ax_ours_3d  = fig.add_subplot(outer[0, 0], projection="3d")
    ax_edge_3d  = fig.add_subplot(outer[0, 1], projection="3d")
    ax_ours_wav = fig.add_subplot(outer[1, 0])
    ax_edge_wav = fig.add_subplot(outer[1, 1])

    draw_skeleton_mdm(ax_ours_3d, j_ours_disp[fi], trajec_ours, fi,
                       MINS_ours, MAXS_ours,
                       HML_CHAINS, HML_COLORS, HML_LW,
                       "Ours (beataware ag=1.5)", bas_ours)

    draw_skeleton_mdm(ax_edge_3d, j_edge_disp[fi_edge], trajec_edge, fi_edge,
                       MINS_edge, MAXS_edge,
                       SMPL_CHAINS, SMPL_COLORS, SMPL_LW,
                       "EDGE (Tseng et al.)", bas_edge)

    draw_wavestrip(ax_ours_wav, t_now, mb_t_ours, mot_t_ours,
                   vel_ours_n, render_fps, render_dur)
    draw_wavestrip(ax_edge_wav, t_now, mb_t_edge, mot_t_edge,
                   vel_edge_n, args.edge_fps, render_dur)

    # Labels under wavestrips
    ax_ours_wav.set_xlabel("Ours", color="#80CBC4", fontsize=8, fontfamily="monospace")
    ax_edge_wav.set_xlabel("EDGE", color="#80CBC4", fontsize=8, fontfamily="monospace")

    # Shared legend
    legend_elems = [
        Line2D([0],[0], color=AMBER, lw=1.8, label="Music beats"),
        Line2D([0],[0], color=CORAL, lw=1.8, label="Motion beats"),
        Line2D([0],[0], color=MINT,  lw=1.4, label="Velocity"),
        Line2D([0],[0], color="#5599FF",lw=1.5, label="Root trail"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", fontsize=7.5,
               facecolor=BG2, edgecolor="#2A2A45", labelcolor="#C5CAE9",
               ncol=4, framealpha=0.92)

    fig.suptitle(
        f"Ours vs EDGE  |  {args.song}  |  t={t_now:.2f}s / {render_dur:.1f}s",
        color="#DDDDDD", fontsize=10, y=0.97, fontfamily="monospace"
    )

    fig.canvas.draw()
    w_px, h_px = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(h_px, w_px, 4)[:,:,:3]
    buf = buf[:buf.shape[0]//2*2, :buf.shape[1]//2*2, :]
    frames_rgb.append(buf.copy())
    plt.close(fig)

    if fi % 20 == 0:
        print(f"  {fi+1}/{T_render} frames...")

# ── write video with audio ────────────────────────────────────────────────────
print("Writing video...")
silent = args.out_video.replace(".mp4", "_silent.mp4")
writer = imageio.get_writer(silent, fps=render_fps, codec="libx264",
                             quality=8, macro_block_size=1)
for frame in frames_rgb:
    writer.append_data(frame)
writer.close()

audio_tmp = args.out_video.replace(".mp4", "_audio.wav")
sf.write(audio_tmp, y_audio, sr)
subprocess.run([
    "ffmpeg", "-y", "-i", silent, "-i", audio_tmp,
    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
    args.out_video
], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
os.remove(silent)
os.remove(audio_tmp)

print(f"\nDone! {args.out_video}")
print(f"Ours BAS={bas_ours:.4f} | EDGE BAS={bas_edge:.4f}")
