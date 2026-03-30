"""
Side-by-side skinned-mesh comparison: our model vs EDGE.

Pipeline
--------
Ours  : 263-dim HumanML3D .npy
          → recover_joints_from_humanml  (196, 22, 3)
          → joints2smpl SMPLify          (196, 25, 6)  rot6d
          → Rotation2xyz                 (1, 6890, 3, 196) vertices
EDGE  : SMPL .pkl  (smpl_poses N×72, smpl_trans N×3)
          → smplx forward                (N, 6890, 3) vertices
          → time-aligned to render FPS

Rendering: pyrender, side-by-side frames, wavestrip + beat timeline below.
"""

import os, sys, argparse, pickle, warnings
import numpy as np
import torch
import pyrender
import trimesh
import librosa
import scipy.signal
import imageio
import subprocess
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore")
os.environ["PYOPENGL_PLATFORM"] = "egl"   # headless rendering

# ── Style constants (matching visualize_bas_alignment.py) ────────────────────
BG     = "#0D0D0D"
BG2    = "#0C0C1A"
AMBER  = "#FFB300"
CORAL  = "#F06292"
LIME   = "#AEDE5A"
MINT   = "#80CBC4"
STEEL  = "#5599FF"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.beat_align_score import (
    get_music_beats, get_motion_beats, ba_score,
    BA_SIGMA_TIME_SEC, SMOOTH_TIME_SEC,
    recover_joints_from_humanml as _recover_joints,
)

# ── Lightweight SMPL fitter (no GMM prior, no SMPLify dependency) ─────────────
import smplx

def _fit_smpl_to_joints(joints_np, device="cuda:0", n_iters=100, lr=5e-3):
    """
    Lightweight gradient-descent SMPL fit to 3D joint positions.
    joints_np : (T, 22, 3)  HumanML3D joint order
    Returns   : (T, 6890, 3) SMPL vertex positions
    """
    T = joints_np.shape[0]
    dev = torch.device(device)

    model = smplx.create(
        "./body_models", model_type="smpl", gender="neutral",
        ext="pkl", batch_size=T,
    ).to(dev)

    joints_t = torch.tensor(joints_np, dtype=torch.float32, device=dev)

    # Initialise from pelvis position
    init_transl = joints_t[:, 0, :].clone().detach()  # (T, 3)
    pose   = torch.zeros(T, 72,  device=dev, requires_grad=True)
    shape  = torch.zeros(T, 10,  device=dev, requires_grad=False)
    transl = init_transl.clone().requires_grad_(True)

    opt = torch.optim.Adam([pose, transl], lr=lr)

    for it in range(n_iters):
        out = model(
            global_orient = pose[:, :3],
            body_pose     = pose[:, 3:],
            betas         = shape,
            transl        = transl,
        )
        # SMPL joint regressor gives 45 joints; take first 22 (SMPL body joints)
        j_pred = out.joints[:, :22, :]          # (T, 22, 3)
        loss   = ((j_pred - joints_t) ** 2).mean()
        loss  += 1e-3 * (pose ** 2).mean()      # L2 pose regularisation
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        out = model(
            global_orient = pose[:, :3],
            body_pose     = pose[:, 3:],
            betas         = shape,
            transl        = transl,
        )
        verts = out.vertices.cpu().numpy()      # (T, 6890, 3)
        faces = model.faces
    return verts, faces


def edge_pkl_to_vertices(pkl_path, faces_out=None, device="cuda:0"):
    """Load EDGE .pkl → smplx forward → (N, 6890, 3) vertices in Y-up coords."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    smpl_poses = data["smpl_poses"].astype(np.float32)   # (N, 72)
    smpl_trans = data["smpl_trans"].astype(np.float32)   # (N, 3)
    N = smpl_poses.shape[0]

    dev = torch.device(device)
    model = smplx.create(
        "./body_models", model_type="smpl", gender="neutral",
        ext="pkl", batch_size=N,
    ).to(dev)
    with torch.no_grad():
        out = model(
            global_orient = torch.tensor(smpl_poses[:, :3],  device=dev),
            body_pose     = torch.tensor(smpl_poses[:, 3:],   device=dev),
            transl        = torch.tensor(smpl_trans,           device=dev),
        )
    verts = out.vertices.cpu().numpy()          # (N, 6890, 3)

    # AIST++ SMPL is Z-up; swap to Y-up to match our camera (same as stick-figure script)
    # x unchanged, z→y (height), y→z (depth)
    verts_yup = np.zeros_like(verts)
    verts_yup[:, :, 0] = verts[:, :, 0]
    verts_yup[:, :, 1] = verts[:, :, 2]   # Z becomes Y (height)
    verts_yup[:, :, 2] = verts[:, :, 1]   # Y becomes Z (depth)

    if faces_out is not None:
        faces_out.append(model.faces)
    return verts_yup


def ours_npy_to_vertices(npy_path, faces_out=None, device_id=0):
    """Load our 263-dim .npy → lightweight SMPL fit → (T, 6890, 3) vertices."""
    motion = np.load(npy_path)                  # (T, 263) or (263, T)
    joints = _recover_joints(motion)            # (T, 22, 3)

    dev = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    print(f"  Fitting SMPL ({joints.shape[0]} frames, {100} iters, device={dev})...")
    verts, faces = _fit_smpl_to_joints(joints, device=dev, n_iters=100)
    if faces_out is not None:
        faces_out.append(faces)
    return verts                                # (T, 6890, 3)

# ── Pyrender helpers ──────────────────────────────────────────────────────────
PANEL_W, PANEL_H = 640, 480
STRIP_H          = 80
VIDEO_W          = PANEL_W * 2
VIDEO_H          = PANEL_H + STRIP_H

# soft blue / orange materials
MAT_OURS = pyrender.MetallicRoughnessMaterial(
    baseColorFactor=[0.3, 0.6, 1.0, 1.0], roughnessFactor=0.7, metallicFactor=0.1
)
MAT_EDGE = pyrender.MetallicRoughnessMaterial(
    baseColorFactor=[1.0, 0.5, 0.15, 1.0], roughnessFactor=0.7, metallicFactor=0.1
)

def make_renderer():
    r = pyrender.OffscreenRenderer(PANEL_W, PANEL_H)
    return r

def make_scene(verts, faces, material, cam_dist=3.5, pelvis_h=0.9):
    """Build a pyrender scene with one SMPL mesh and lights."""
    scene = pyrender.Scene(bg_color=[0.12, 0.12, 0.15, 1.0],
                           ambient_light=[0.3, 0.3, 0.3])
    mesh = pyrender.Mesh.from_trimesh(
        trimesh.Trimesh(vertices=verts, faces=faces, process=False),
        material=material, smooth=True
    )
    scene.add(mesh)

    # Key light
    dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(dl, pose=_look_at([1, 2, 2], [0, pelvis_h, 0]))

    # Fill light
    dl2 = pyrender.DirectionalLight(color=[0.6, 0.7, 1.0], intensity=1.5)
    scene.add(dl2, pose=_look_at([-1, 1, -1], [0, pelvis_h, 0]))

    # Camera: front-facing, slightly above
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4)
    cam_pos = np.array([0.0, pelvis_h + 0.3, cam_dist])
    scene.add(cam, pose=_look_at(cam_pos, [0, pelvis_h * 0.7, 0]))
    return scene

def _look_at(eye, target):
    eye    = np.array(eye,    dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up     = np.array([0, 1, 0], dtype=np.float64)
    z = eye - target; z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    M = np.eye(4)
    M[:3, 0] = x; M[:3, 1] = y; M[:3, 2] = z; M[:3, 3] = eye
    return M

def render_frame(renderer, verts, faces, material, root_offset=None):
    """Root-centre the mesh, render, return (H, W, 3) uint8."""
    if root_offset is not None:
        verts = verts - root_offset
    # Estimate pelvis height after centring
    pelvis_h = float(verts[:, 1].mean())
    scene = make_scene(verts, faces, material, pelvis_h=pelvis_h)
    color, _ = renderer.render(scene)
    return color

STRIP_H = 200   # taller for matplotlib wavestrip

def make_wavestrip_mpl(
        y_audio, sr, duration, render_fps, t_now,
        mb_t, mot_t_ours, mot_t_edge,
        vel_t_ours, vel_norm_ours,
        vel_t_edge, vel_norm_edge,
        n_ours, n_edge, bas_ours, bas_edge,
        W=VIDEO_W, H=STRIP_H):
    """Matplotlib wavestrip matching visualize_bas_alignment style."""
    fig_w = W / 100.0
    fig_h = H / 100.0
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=100)
    fig.patch.set_facecolor(BG2)
    ax.set_facecolor(BG2)

    times = np.linspace(0, duration, len(y_audio))
    wav_max = np.abs(y_audio).max() * 1.1

    # Waveform fill
    ax.fill_between(times,  y_audio, alpha=0.18, color="#7986CB")
    ax.fill_between(times, -y_audio, alpha=0.18, color="#7986CB")
    ax.plot(times,  y_audio, color="#9FA8DA", lw=0.5, alpha=0.45)
    ax.axhline(0, color="#333355", lw=0.8, alpha=0.6)

    # Music beats (amber)
    for bt in mb_t:
        ax.axvline(bt, color=AMBER, lw=1.8, alpha=0.95, zorder=5)
        ax.axvline(bt, color=AMBER, lw=6.0, alpha=0.12, zorder=4)

    # Ours motion beats (coral)
    for bt in mot_t_ours:
        ax.axvline(bt, color=CORAL, lw=1.6, alpha=0.90, zorder=6)
        ax.axvline(bt, color=CORAL, lw=5.0, alpha=0.12, zorder=5)

    # EDGE motion beats (lime)
    for bt in mot_t_edge:
        ax.axvline(bt, color=LIME, lw=1.6, alpha=0.90, zorder=6)
        ax.axvline(bt, color=LIME, lw=5.0, alpha=0.12, zorder=5)

    # Velocity curves
    ax.plot(vel_t_ours, vel_norm_ours, color=CORAL, lw=1.3, alpha=0.70, zorder=4)
    ax.plot(vel_t_edge, vel_norm_edge, color=LIME,  lw=1.3, alpha=0.70, zorder=4)

    # Playhead
    ax.axvline(t_now, color="#FFFFFF", lw=2.0, alpha=1.0, zorder=9)
    ax.axvline(t_now, color="#FFE082", lw=7.0, alpha=0.18, zorder=8)

    ax.set_xlim(0, duration)
    ax.set_ylim(-wav_max, wav_max)
    ax.set_xlabel("Time (s)", color="#6670AA", fontsize=7)
    ax.tick_params(colors="#555580", labelsize=6)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1E1E33")

    legend_elems = [
        Line2D([0],[0], color=AMBER, lw=2.0, label=f"Music beats ({len(mb_t)})"),
        Line2D([0],[0], color=CORAL, lw=2.0, label=f"Ours motion beats ({n_ours})  BAS={bas_ours:.3f}"),
        Line2D([0],[0], color=LIME,  lw=2.0, label=f"EDGE motion beats ({n_edge})  BAS={bas_edge:.3f}"),
        Line2D([0],[0], color="#FFFFFF", lw=1.5, label="Playhead"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=6.5,
              facecolor=BG2, edgecolor="#2A2A45", labelcolor="#C5CAE9",
              ncol=4, framealpha=0.95)

    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    w_px, h_px = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h_px, w_px, 4)[:,:,:3]
    # resize to exact W×H
    import cv2
    buf = cv2.resize(buf, (W, H))
    plt.close(fig)
    return buf

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--song",       required=True)
    parser.add_argument("--ours_path",  required=True)
    parser.add_argument("--edge_path",  required=True)
    parser.add_argument("--audio_path", required=True)
    parser.add_argument("--out_video",  required=True)
    parser.add_argument("--ours_fps",   type=float, default=20.0)
    parser.add_argument("--edge_fps",   type=float, default=30.0)
    parser.add_argument("--render_fps", type=float, default=20.0)
    parser.add_argument("--device",     type=int,   default=0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_video), exist_ok=True)

    import cv2

    # ── Load audio ────────────────────────────────────────────────────────────
    print("Loading audio...")
    y_full, sr = librosa.load(args.audio_path, sr=None, mono=True)

    # ── Convert our model → vertices + joints ─────────────────────────────────
    print("Running SMPL fit on our model...")
    ours_motion  = np.load(args.ours_path)
    ours_joints  = _recover_joints(ours_motion)           # (T, 22, 3)
    ours_faces_l = []
    ours_verts   = ours_npy_to_vertices(args.ours_path, faces_out=ours_faces_l,
                                         device_id=args.device)
    ours_faces = ours_faces_l[0]
    T_ours = ours_verts.shape[0]
    print(f"  Ours: {T_ours} frames, {ours_verts.shape[1]} vertices")

    # ── Convert EDGE → vertices + joints ─────────────────────────────────────
    print("Running smplx forward on EDGE...")
    edge_faces_l = []
    dev_str = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    edge_verts_all = edge_pkl_to_vertices(args.edge_path, faces_out=edge_faces_l,
                                           device=dev_str)
    edge_faces = edge_faces_l[0]
    T_edge = edge_verts_all.shape[0]
    print(f"  EDGE: {T_edge} frames, {edge_verts_all.shape[1]} vertices")

    # EDGE joints from SMPL vertices (use a joint regressor subset as proxy)
    # Use key vertex landmarks as hip/knee/shoulder proxies for beat detection
    # Simpler: load full_pose from pkl for joint positions
    with open(args.edge_path, "rb") as f:
        edge_data_raw = pickle.load(f)
    edge_joints_raw = edge_data_raw["full_pose"].astype(np.float32)  # (N, 24, 3) Z-up
    # Convert to Y-up
    edge_joints = np.zeros_like(edge_joints_raw)
    edge_joints[:, :, 0] = edge_joints_raw[:, :, 0]
    edge_joints[:, :, 1] = edge_joints_raw[:, :, 2]
    edge_joints[:, :, 2] = edge_joints_raw[:, :, 1]

    # ── Align FPS ─────────────────────────────────────────────────────────────
    render_fps = args.render_fps
    dur_ours   = T_ours / args.ours_fps
    dur_edge   = T_edge / args.edge_fps
    render_dur = min(dur_ours, dur_edge, 15.0)
    T_render   = int(render_dur * render_fps)

    ours_map = np.minimum(
        (np.arange(T_render) * args.ours_fps / render_fps).astype(int), T_ours - 1)
    edge_map = np.minimum(
        (np.arange(T_render) * args.edge_fps / render_fps).astype(int), T_edge - 1)

    # ── Beat/velocity for wavestrip ───────────────────────────────────────────
    y_audio = y_full[:int(render_dur * sr)]
    duration = render_dur

    mb_frames  = get_music_beats(args.audio_path, args.ours_fps, max_frames=T_ours)
    mb_t       = mb_frames / args.ours_fps

    mot_ours_f = get_motion_beats(ours_joints, fps=args.ours_fps)
    mot_ours_t = mot_ours_f / args.ours_fps

    mot_edge_f = get_motion_beats(edge_joints, fps=args.edge_fps)
    mot_edge_t = mot_edge_f / args.edge_fps

    sigma_ours = BA_SIGMA_TIME_SEC * args.ours_fps
    sigma_edge = BA_SIGMA_TIME_SEC * args.edge_fps
    bas_ours   = ba_score(mb_frames, mot_ours_f, sigma=sigma_ours)
    bas_edge   = ba_score(
        get_music_beats(args.audio_path, args.edge_fps, max_frames=T_edge),
        mot_edge_f, sigma=sigma_edge)
    print(f"  BAS — Ours: {bas_ours:.4f}  EDGE: {bas_edge:.4f}")

    vel_ours = np.mean(np.sqrt(np.sum(np.diff(ours_joints, axis=0)**2, axis=2)), axis=1)
    vel_ours_s = gaussian_filter1d(vel_ours, SMOOTH_TIME_SEC * args.ours_fps)
    vel_ours_n = vel_ours_s / (vel_ours_s.max() + 1e-8) * (np.abs(y_audio).max() * 0.6)
    vel_t_ours = np.arange(len(vel_ours)) / args.ours_fps

    vel_edge = np.mean(np.sqrt(np.sum(np.diff(edge_joints, axis=0)**2, axis=2)), axis=1)
    vel_edge_s = gaussian_filter1d(vel_edge, SMOOTH_TIME_SEC * args.edge_fps)
    vel_edge_n = vel_edge_s / (vel_edge_s.max() + 1e-8) * (np.abs(y_audio).max() * 0.6)
    vel_t_edge = np.arange(len(vel_edge)) / args.edge_fps

    # ── Render ────────────────────────────────────────────────────────────────
    renderer = make_renderer()
    frames = []
    print(f"Rendering {T_render} frames ({render_dur:.1f}s)...")

    for fi in range(T_render):
        if fi % 20 == 0:
            print(f"  {fi+1}/{T_render} frames...")

        oi = ours_map[fi]
        ei = edge_map[fi]
        t_now = fi / render_fps

        ov = ours_verts[oi].copy()
        ev = edge_verts_all[ei].copy()

        ov -= np.array([ov[:, 0].mean(), 0, ov[:, 2].mean()])
        ev -= np.array([ev[:, 0].mean(), 0, ev[:, 2].mean()])

        left  = render_frame(renderer, ov, ours_faces, MAT_OURS).copy()
        right = render_frame(renderer, ev, edge_faces, MAT_EDGE).copy()

        # Labels on panels
        cv2.putText(left,  f"Ours  BAS={bas_ours:.3f}", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255,255,255), 2)
        cv2.putText(right, f"EDGE  BAS={bas_edge:.3f}", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255,255,255), 2)
        cv2.putText(left,  args.song, (PANEL_W//2 - 30, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)

        # Rich matplotlib wavestrip
        strip = make_wavestrip_mpl(
            y_audio, sr, duration, render_fps, t_now,
            mb_t, mot_ours_t, mot_edge_t,
            vel_t_ours, vel_ours_n,
            vel_t_edge, vel_edge_n,
            len(mot_ours_t), len(mot_edge_t),
            bas_ours, bas_edge,
        )

        row = np.concatenate([left, right], axis=1)
        # Ensure even dimensions
        row   = row  [:row.shape[0]//2*2,   :row.shape[1]//2*2]
        strip = strip[:strip.shape[0]//2*2, :strip.shape[1]//2*2]
        if strip.shape[1] != row.shape[1]:
            strip = cv2.resize(strip, (row.shape[1], strip.shape[0]))
        frame = np.concatenate([row, strip], axis=0)
        frames.append(frame)

    renderer.delete()

    # ── Write video ───────────────────────────────────────────────────────────
    print("Writing video...")
    tmp_video = args.out_video.replace(".mp4", "_silent.mp4")
    writer = imageio.get_writer(tmp_video, fps=render_fps, codec="libx264",
                                 quality=8, macro_block_size=1)
    for f in frames:
        writer.append_data(f)
    writer.close()

    # Add audio
    audio_clip = args.out_video.replace(".mp4", "_audio_clip.wav")
    sf.write(audio_clip, y_audio, sr)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", tmp_video, "-i", audio_clip,
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        args.out_video,
    ], check=True)
    os.remove(tmp_video)
    os.remove(audio_clip)

    print(f"\nDone! {args.out_video}")
    print(f"Ours BAS={bas_ours:.4f} | EDGE BAS={bas_edge:.4f}")


if __name__ == "__main__":
    main()
