"""
4 × (1×3) ablation grid videos.

Layout per video
────────────────────────────────────────────────────────────
 Col 0  Base MDM          – text-only, no audio  (same for all 4)
 Col 1  Ours (variable)   – wav2clip_beataware  ag=1.5  tg=2.5
 Col 2  Ours mBR0         – same model, mBR0 audio  (same for all 4)
────────────────────────────────────────────────────────────

One video per track in COL1_TRACKS.
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
import imageio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval.beat_align_score import (
    load_humanml_motion, get_music_beats, get_motion_beats,
    ba_score, BA_SIGMA_TIME_SEC, SMOOTH_TIME_SEC,
)

parser = argparse.ArgumentParser()
parser.add_argument("--motion_dir", default="/Data/yash.bhardwaj/eval/ablation_grid/motions")
parser.add_argument("--audio_dir",  default="/Data/yash.bhardwaj/datasets/aist/audio_test_10")
parser.add_argument("--out_dir",    default="/Data/yash.bhardwaj/eval/ablation_grid")
parser.add_argument("--fps",         type=float, default=20.0)
parser.add_argument("--clip_sec",    type=float, default=8.0)
parser.add_argument("--seed_suffix", type=str,   default="",
                    help="Appended to motion stems, e.g. '_s1' -> base_mdm_s1.npy")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
FPS         = args.fps
CLIP_FRAMES = int(args.clip_sec * FPS)
RADIUS      = 3
SCALE       = 1.3

# Varying tracks for column 1
COL1_TRACKS = ["mLO2", "mKR2", "mLH4", "mWA0"]

# Track labels for the video title / column header
TRACK_LABELS = {
    "mLO2": "Lock (mLO2) 118 BPM",
    "mKR2": "Krump (mKR2) 122 BPM",
    "mLH4": "LA-style Hiphop (mLH4) 105 BPM",
    "mWA0": "Waacking (mWA0) 81 BPM",
    "mBR0": "Breakdance (mBR0) 161 BPM",
}

# ── kinematic chains (HumanML3D 22-joint) ─────────────────────────────────────
KINEMATIC_CHAINS = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]
CHAIN_COLORS = ["#4A90E2", "#7B68EE", "#F5A623", "#FF6B6B", "#50C878"]
LINEWIDTHS   = [3.0, 3.0, 3.5, 2.5, 2.5]

BG   = "#0D0D0D"
BG2  = "#0C0C1A"
AMBER = "#FFB300"
CORAL = "#F06292"
MINT  = "#80CBC4"
FLOOR_COLOR = "#B0B0B0"   # light grey — clearly visible on dark bg


# ── helpers ───────────────────────────────────────────────────────────────────

def load_cell(npy_path, aud_path=None):
    """Load a motion npy and optionally an audio file; return a cell dict."""
    joints_raw = load_humanml_motion(npy_path)            # (T, 22, 3)
    T = min(joints_raw.shape[0], CLIP_FRAMES)
    joints_raw = joints_raw[:T].copy() * SCALE

    # Floor at y=0
    joints_raw[:, :, 1] -= joints_raw[:, :, 1].min()

    trajec    = joints_raw[:, 0, [0, 2]].copy()
    joints_disp = joints_raw.copy()
    joints_disp[:, :, 0] -= joints_disp[:, 0:1, 0]
    joints_disp[:, :, 2] -= joints_disp[:, 0:1, 2]

    MINS = joints_raw.min(axis=0).min(axis=0)
    MAXS = joints_raw.max(axis=0).max(axis=0)

    duration = T / FPS
    vel = np.mean(np.sqrt(np.sum(
        (joints_raw[1:] - joints_raw[:-1]) ** 2, axis=2)), axis=1)
    vel_s    = gaussian_filter1d(vel, SMOOTH_TIME_SEC * FPS)
    vel_norm = vel_s / (vel_s.max() + 1e-8) * 0.4

    has_audio = aud_path is not None and os.path.isfile(aud_path)
    if has_audio:
        y_full, sr  = librosa.load(aud_path, sr=None)
        y_audio     = y_full[:int(duration * sr)]
        music_beats  = get_music_beats(aud_path, FPS, max_frames=T)
        motion_beats = get_motion_beats(joints_raw / SCALE, fps=FPS)
        bas          = ba_score(music_beats, motion_beats, sigma=BA_SIGMA_TIME_SEC * FPS)
    else:
        sr = 22050
        y_audio      = np.zeros(int(duration * sr))
        music_beats  = np.array([], dtype=float)
        motion_beats = get_motion_beats(joints_raw / SCALE, fps=FPS)
        bas          = None

    return dict(
        joints_disp=joints_disp, trajec=trajec, T=T, duration=duration,
        MINS=MINS, MAXS=MAXS,
        y=y_audio, sr=sr,
        music_beats=music_beats, motion_beats=motion_beats,
        bas=bas, vel_norm=vel_norm,
        has_audio=has_audio,
    )


def draw_cell(fig, col_gs, cd, fi, t_now, col_lbl):
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=col_gs,
        height_ratios=[3.2, 1], hspace=0.04,
    )
    ax3d  = fig.add_subplot(inner[0], projection="3d")
    ax_wav = fig.add_subplot(inner[1])

    # ── 3D skeleton ──────────────────────────────────────────────────────────
    ax3d.set_facecolor(BG)
    jf = cd["joints_disp"][fi]    # (22, 3) root-centred

    cx = cd["trajec"][fi, 0]
    cz = cd["trajec"][fi, 1]
    MINS, MAXS = cd["MINS"], cd["MAXS"]

    def nz(z): return -z

    # Floor grid — light grey so it pops on dark background
    gx = np.linspace(MINS[0] - cx, MAXS[0] - cx, 9)
    gz = np.linspace(nz(MAXS[2] - cz), nz(MINS[2] - cz), 9)
    for gxi in gx:
        ax3d.plot([gxi, gxi], [0, 0], [gz[0], gz[-1]],
                  color=FLOOR_COLOR, lw=0.7, alpha=0.60, zorder=0)
    for gzi in gz:
        ax3d.plot([gx[0], gx[-1]], [0, 0], [gzi, gzi],
                  color=FLOOR_COLOR, lw=0.7, alpha=0.60, zorder=0)

    # Trajectory trail
    trail_start = max(0, fi - 20)
    trail_pts = cd["trajec"][trail_start:fi+1]
    if len(trail_pts) > 1:
        tx = trail_pts[:, 0] - cx
        tz = nz(trail_pts[:, 1] - cz)
        alphas = np.linspace(0.1, 0.7, len(trail_pts))
        for k in range(len(trail_pts) - 1):
            ax3d.plot([tx[k], tx[k+1]], [0.02, 0.02], [tz[k], tz[k+1]],
                      color="#5599FF", lw=2.0, alpha=float(alphas[k+1]), zorder=2)

    # Skeleton
    for chain, color, lw in zip(KINEMATIC_CHAINS, CHAIN_COLORS, LINEWIDTHS):
        xs = jf[chain, 0]
        ys = jf[chain, 1]
        zs = nz(jf[chain, 2])
        ax3d.plot3D(xs, ys, zs, color=color, linewidth=lw,
                    marker='o', markersize=2.5, markerfacecolor='white')

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
    bas_txt = f"BAS {cd['bas']:.3f}" if cd["bas"] is not None else "text-only"
    ax3d.text2D(0.97, 0.97, bas_txt,
                transform=ax3d.transAxes,
                color="#FFD700", fontsize=8, ha="right", va="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec="#665500", alpha=0.88))

    # ── wavestrip ─────────────────────────────────────────────────────────────
    ax_wav.set_facecolor(BG2)
    duration = cd["duration"]
    times    = np.linspace(0, duration, len(cd["y"]))
    wav_max  = max(np.abs(cd["y"]).max() * 1.1, 0.01)
    vel_t    = np.arange(len(cd["vel_norm"])) / FPS
    mb_t     = cd["music_beats"]  / FPS
    mot_t    = cd["motion_beats"] / FPS

    ax_wav.fill_between(times,  cd["y"], alpha=0.15, color="#7986CB")
    ax_wav.fill_between(times, -cd["y"], alpha=0.15, color="#7986CB")
    ax_wav.plot(times,  cd["y"], color="#9FA8DA", lw=0.4, alpha=0.4)
    ax_wav.axhline(0, color="#333355", lw=0.6, alpha=0.5, zorder=1)

    if cd["has_audio"]:
        for bt in mb_t:
            ax_wav.axvline(bt, color=AMBER, lw=1.5, alpha=0.92, zorder=5)
            ax_wav.axvline(bt, color=AMBER, lw=5.5, alpha=0.10, zorder=4)
    for bt in mot_t:
        ax_wav.axvline(bt, color=CORAL, lw=1.5, alpha=0.92, zorder=6)
        ax_wav.axvline(bt, color=CORAL, lw=5.5, alpha=0.12, zorder=5)

    ax_wav.plot(vel_t, cd["vel_norm"], color=MINT, lw=1.3, alpha=0.88, zorder=4)
    ax_wav.axvline(t_now, color="#FFFFFF", lw=1.8, alpha=1.0, zorder=8)
    ax_wav.axvline(t_now, color="#FFE082", lw=6.5, alpha=0.16, zorder=7)

    ax_wav.set_xlim(0, duration)
    ax_wav.set_ylim(-wav_max, wav_max)
    ax_wav.set_xticks([]); ax_wav.set_yticks([])
    for sp in ax_wav.spines.values():
        sp.set_edgecolor("#1E1E33")

    # Column header text placed by caller after ax3d.get_position() is stable
    return ax3d, col_lbl


# ── pre-load fixed motions ──────────────────────────────────────────────────
MOTIONS_DIR = args.motion_dir
AUDIO_DIR   = args.audio_dir

SFX = args.seed_suffix   # e.g. "" or "_s1"

print(f"Loading base_mdm{SFX}.npy  (text-only, no audio)...")
base_mdm = load_cell(
    os.path.join(MOTIONS_DIR, f"base_mdm{SFX}.npy"),
    aud_path=None,
)
print(f"  T={base_mdm['T']}  motion_beats={len(base_mdm['motion_beats'])}")

print(f"Loading ours_mBR0{SFX}.npy ...")
ours_br0  = load_cell(
    os.path.join(MOTIONS_DIR, f"ours_mBR0{SFX}.npy"),
    aud_path=os.path.join(AUDIO_DIR, "mBR0.wav"),
)
print(f"  T={ours_br0['T']}  BAS={ours_br0['bas']:.4f}")

# ── render one video per COL1 track ─────────────────────────────────────────
CELL_W = 5.2
CELL_H = 4.8
FIG_W  = 3 * CELL_W
FIG_H  = CELL_H + 0.5
DPI    = 100

for track in COL1_TRACKS:
    print(f"\n=== Rendering video for track {track}{SFX} ===")
    ours_var = load_cell(
        os.path.join(MOTIONS_DIR, f"ours_{track}{SFX}.npy"),
        aud_path=os.path.join(AUDIO_DIR, f"{track}.wav"),
    )
    print(f"  T={ours_var['T']}  BAS={ours_var['bas']:.4f}")

    T_render = min(base_mdm["T"], ours_var["T"], ours_br0["T"])
    cells = [
        (base_mdm, "Base MDM  (text-only)"),
        (ours_var, f"Ours — {TRACK_LABELS[track]}"),
        (ours_br0, f"Ours — {TRACK_LABELS['mBR0']}"),
    ]

    frames_rgb = []
    for fi in range(T_render):
        t_now = fi / FPS

        fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=BG)
        outer = gridspec.GridSpec(
            1, 3, figure=fig,
            hspace=0.0, wspace=0.04,
            left=0.01, right=0.99, top=0.91, bottom=0.02,
        )

        ax3d_refs = []
        for col, (cd, lbl) in enumerate(cells):
            ax3d, col_lbl = draw_cell(fig, outer[0, col], cd, fi, t_now, lbl)
            ax3d_refs.append((ax3d, col_lbl))

        # Draw headers after layout is complete
        fig.canvas.draw()
        for ax3d, col_lbl in ax3d_refs:
            pos = ax3d.get_position()
            fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.005,
                     col_lbl, color="#DDDDDD", fontsize=8.5,
                     ha="center", va="bottom", fontfamily="monospace",
                     fontweight="bold")

        # Legend
        legend_elems = [
            Line2D([0], [0], color=AMBER,     lw=1.8, label="Music beats"),
            Line2D([0], [0], color=CORAL,     lw=1.8, label="Motion beats"),
            Line2D([0], [0], color=MINT,      lw=1.4, label="Motion velocity"),
            Line2D([0], [0], color="#5599FF", lw=1.5, label="Root trail"),
            Line2D([0], [0], color="#FFFFFF", lw=1.6, label="Playhead"),
        ]
        fig.legend(handles=legend_elems, loc="lower right", fontsize=7.5,
                   facecolor=BG2, edgecolor="#2A2A45", labelcolor="#C5CAE9",
                   ncol=5, framealpha=0.92, bbox_to_anchor=(0.99, 0.0))

        fig.suptitle(
            f'Prompt: "a person dances"  |  wav2clip_beataware  ag=1.5 tg=2.5'
            f'  |  t={t_now:.2f}s',
            color="#DDDDDD", fontsize=8.5, y=0.98, fontfamily="monospace",
        )

        fig.canvas.draw()
        w_px, h_px = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(h_px, w_px, 4)[:, :, :3]
        buf = buf[:buf.shape[0]//2*2, :buf.shape[1]//2*2, :]
        frames_rgb.append(buf.copy())
        plt.close(fig)

        if fi % 20 == 0:
            print(f"  frame {fi+1}/{T_render}")

    out_path = os.path.join(args.out_dir, f"ablation_{track}{SFX}.mp4")
    writer = imageio.get_writer(out_path, fps=FPS, codec="libx264",
                                quality=8, macro_block_size=1)
    for frame in frames_rgb:
        writer.append_data(frame)
    writer.close()
    print(f"  Saved → {out_path}")

print("\nAll 4 ablation grid videos done.")
