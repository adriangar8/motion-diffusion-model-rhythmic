"""
Beat Alignment Score (BAS) evaluation for dance generation models.

Supports two motion formats:
  - Stage 2 (ours): .npy files in HumanML3D 263-dim at 20 FPS
  - EDGE:           .pkl files with 'full_pose' key, shape (T, 24, 3) at 30 FPS

Music beats are extracted from raw .wav files using librosa.

Usage examples:

  # Our Stage 2 model
  python -m eval.beat_align_score \\
      --motion_dir ./save/audio_stage2_wav2clip_beataware/samples \\
      --audio_dir ./dataset/aist/audio \\
      --format humanml --fps 20

  # EDGE
  python -m eval.beat_align_score \\
      --motion_dir ./EDGE/experiments/test/pkl \\
      --audio_dir ./dataset/aist/audio \\
      --format edge --fps 30

  # Single audio file paired with all motions (e.g. qualitative eval)
  python -m eval.beat_align_score \\
      --motion_dir ./save/audio_stage2_wav2clip_beataware/samples \\
      --audio_path ./dataset/aist/audio/mJB0.wav \\
      --format humanml --fps 20

Pairing logic:
  By default each motion file is paired with the audio file whose stem is a
  prefix of the motion file stem (e.g. motion "mJB0_sample_00.npy" -> audio
  "mJB0.wav").  If --audio_path is given, that single file is used for all
  motions.
"""

import argparse
import os
import sys
import glob
import pickle
import numpy as np
import torch
from scipy.ndimage import gaussian_filter as gaussian_filter1d
from scipy.signal import argrelextrema

# ── librosa ──────────────────────────────────────────────────────────────────

try:
    import librosa
except ImportError:
    print("ERROR: librosa is required. Install with: pip install librosa")
    sys.exit(1)


# ── HumanML3D motion recovery ────────────────────────────────────────────────

def _qrot(q, v):
    """Rotate vector v by quaternion q (both tensors)."""
    assert q.shape[-1] == 4 and v.shape[-1] == 3
    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return v + 2 * (q[..., :1] * uv + uuv)


def _qinv(q):
    """Invert quaternion (negate imaginary part)."""
    inv = q.clone()
    inv[..., 1:] *= -1
    return inv


def recover_joints_from_humanml(data_263):
    """
    Convert HumanML3D 263-dim motion to (T, 22, 3) world joint positions.

    Args:
        data_263: np.ndarray of shape (T, 263) — raw (unnormalized) motion.

    Returns:
        np.ndarray of shape (T, 22, 3)
    """
    data = torch.from_numpy(data_263).float()

    rot_vel = data[..., 0]           # (T,)
    r_rot_ang = torch.zeros_like(rot_vel)
    r_rot_ang[1:] = rot_vel[:-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[0], 4)
    r_rot_quat[:, 0] = torch.cos(r_rot_ang)
    r_rot_quat[:, 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[0], 3)
    r_pos[1:, [0, 2]] = data[:-1, 1:3]
    r_pos = _qrot(_qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=0)
    r_pos[:, 1] = data[:, 3]

    joints_num = 22
    local_pos = data[..., 4:(joints_num - 1) * 3 + 4]         # (T, 63)
    local_pos = local_pos.view(data.shape[0], joints_num - 1, 3)

    # rotate local joints by inverse root rotation
    local_pos = _qrot(
        _qinv(r_rot_quat[:, None, :]).expand(-1, joints_num - 1, -1),
        local_pos,
    )

    local_pos[..., 0] += r_pos[:, 0:1]
    local_pos[..., 2] += r_pos[:, 2:3]

    positions = torch.cat([r_pos.unsqueeze(1), local_pos], dim=1)  # (T,22,3)
    return positions.numpy()


# ── beat extraction ───────────────────────────────────────────────────────────

def get_music_beats(wav_path, fps, max_frames=None):
    """
    Extract beat frame indices from a .wav file using librosa.

    Args:
        wav_path:    path to .wav file.
        fps:         motion frame rate (used to convert beat times to frame indices).
        max_frames:  if given, discard beats beyond this frame index.  This is
                     essential when the audio is longer than the generated motion.

    Returns:
        np.ndarray of integer frame indices at the given fps.
    """
    y, sr = librosa.load(wav_path, sr=None)
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)
    beat_indices = (beat_times * fps).astype(int)
    if max_frames is not None:
        beat_indices = beat_indices[beat_indices < max_frames]
    return beat_indices


# Bailando uses sigma=5 at 60 FPS (= 0.083s smoothing window).
# To be FPS-fair we scale: sigma_frames = SMOOTH_TIME * fps.
SMOOTH_TIME_SEC = 5.0 / 60.0  # 0.0833s, matching Bailando's 5-frame sigma at 60 FPS


def get_motion_beats(joints, fps=None, sigma=None):
    """
    Extract motion beat frame indices as local minima of mean kinetic velocity.

    Args:
        joints: np.ndarray (T, J, 3)  — 3-D joint positions.
        fps:    motion frame rate.  When provided, the smoothing sigma is scaled
                to match Bailando's 0.083 s window (sigma=5 at 60 FPS).
        sigma:  explicit Gaussian smoothing sigma in frames.  Overrides the
                FPS-based calculation when given.

    Returns:
        np.ndarray of integer frame indices.
    """
    if sigma is None:
        if fps is not None:
            sigma = SMOOTH_TIME_SEC * fps
        else:
            sigma = 5  # legacy fallback
    vel = np.mean(
        np.sqrt(np.sum((joints[1:] - joints[:-1]) ** 2, axis=2)),
        axis=1,
    )  # (T-1,)
    vel_smooth = gaussian_filter1d(vel, sigma)
    (motion_beats,) = argrelextrema(vel_smooth, np.less)
    return motion_beats


# ── BA score ──────────────────────────────────────────────────────────────────

BA_SIGMA_TIME_SEC = 3.0 / 60.0  # 0.05s, matching Bailando's σ=3 at 60 FPS


def ba_score(music_beats, motion_beats, sigma=3):
    """
    Beat Alignment Score as defined in Bailando / AIST++.

    For each music beat, computes exp(-d²/(2σ²)) where d is the closest
    motion beat distance in frames.

    Returns scalar in [0, 1].
    """
    if len(music_beats) == 0 or len(motion_beats) == 0:
        return 0.0
    score = 0.0
    for mb in music_beats:
        score += np.exp(-np.min((motion_beats - mb) ** 2) / (2 * sigma ** 2))
    return score / len(music_beats)


# ── motion loaders ────────────────────────────────────────────────────────────

def load_humanml_motion(npy_path, mean_path=None, std_path=None):
    """
    Load a HumanML3D .npy motion file and return (T, 22, 3) joint positions.

    If mean/std paths are given the data is de-normalized first.
    If the array already looks like raw joint positions (shape (T,22,3) or
    (T,66)) it is returned directly.
    """
    data = np.load(npy_path)

    # Some generate scripts save denormalized samples; detect by ndim/shape.
    if data.ndim == 3 and data.shape[-2] == 22 and data.shape[-1] == 3:
        return data  # already (T, 22, 3)

    if data.ndim == 2 and data.shape[-1] == 66:
        return data.reshape(-1, 22, 3)

    # 263-dim representation — denormalize only if data appears z-normalized.
    # dim 3 encodes root height y: ~0.9 m when raw, ~0 when z-normalized.
    if data.ndim == 2 and data.shape[-1] == 263:
        is_normalized = abs(data[:, 3].mean()) < 0.3  # raw root_y > 0.5 m
        if is_normalized and mean_path and std_path \
                and os.path.exists(mean_path) and os.path.exists(std_path):
            mean = np.load(mean_path)
            std  = np.load(std_path)
            data = data * std + mean
        return recover_joints_from_humanml(data)

    # Fallback: try treating as (T, J, 3) or (T, J*3)
    if data.ndim == 3:
        return data
    raise ValueError(f"Unrecognised motion shape {data.shape} in {npy_path}")


def load_edge_motion(pkl_path):
    """Load an EDGE .pkl motion file and return (T, 24, 3) joint positions."""
    with open(pkl_path, "rb") as f:
        info = pickle.load(f)
    joints = info["full_pose"]   # (T, 24, 3)
    return np.array(joints)


# ── audio pairing ─────────────────────────────────────────────────────────────

def find_audio_for_motion(motion_stem, audio_dir):
    """
    Return the path of the audio file whose stem matches motion_stem.

    Matching rules (tried in order):
      1. motion_stem starts with audio_stem  (e.g. "mBR0_sample_00" -> "mBR0")
      2. audio_stem is contained in motion_stem  (e.g. "test_mBR0" -> "mBR0")
    Searches for .wav / .mp3 / .flac in audio_dir.
    """
    for ext in (".wav", ".mp3", ".flac"):
        for fname in os.listdir(audio_dir):
            if fname.endswith(ext):
                audio_stem = os.path.splitext(fname)[0]
                if motion_stem.startswith(audio_stem) or audio_stem in motion_stem:
                    return os.path.join(audio_dir, fname)
    return None


# ── main evaluation loop ──────────────────────────────────────────────────────

def evaluate_bas(
    motion_dir,
    fps,
    fmt,
    audio_dir=None,
    audio_path=None,
    mean_path=None,
    std_path=None,
    sigma=None,
    verbose=True,
):
    """
    Compute mean BAS over all motions in motion_dir.

    All FPS-dependent parameters (velocity smoothing sigma, BA-score tolerance
    sigma) are automatically scaled to match Bailando's original 60 FPS settings,
    keeping the evaluation fair across different frame rates.

    Music beats are truncated to the duration of each motion clip.

    Args:
        motion_dir:  directory containing .npy (humanml) or .pkl (edge) files
        fps:         motion FPS (20 for Stage2, 30 for EDGE)
        fmt:         'humanml' or 'edge'
        audio_dir:   directory of .wav files (one per motion, matched by name)
        audio_path:  single .wav path (used for all motions)
        mean_path:   path to Mean.npy for HumanML3D denormalization (optional)
        std_path:    path to Std.npy for HumanML3D denormalization (optional)
        sigma:       BA-score Gaussian width in frames.  If None (recommended),
                     automatically computed as BA_SIGMA_TIME_SEC * fps.

    Returns:
        mean_bas (float), per_file list of (filename, bas)
    """
    if sigma is None:
        sigma = BA_SIGMA_TIME_SEC * fps
    smooth_sigma = SMOOTH_TIME_SEC * fps

    ext = ".npy" if fmt == "humanml" else ".pkl"
    motion_files = sorted(glob.glob(os.path.join(motion_dir, f"*{ext}")))

    if len(motion_files) == 0:
        print(f"No {ext} files found in {motion_dir}")
        return 0.0, []

    if verbose:
        print(f"  FPS-scaled params: smooth_σ={smooth_sigma:.2f} frames "
              f"({SMOOTH_TIME_SEC*1000:.1f} ms), BA_σ={sigma:.2f} frames "
              f"({BA_SIGMA_TIME_SEC*1000:.1f} ms)")
        print()

    scores = []
    per_file = []

    for mpath in motion_files:
        stem = os.path.splitext(os.path.basename(mpath))[0]

        # ── resolve audio ──
        if audio_path:
            wav = audio_path
        elif audio_dir:
            wav = find_audio_for_motion(stem, audio_dir)
            if wav is None:
                if verbose:
                    print(f"  SKIP {stem}: no matching audio found in {audio_dir}")
                continue
        else:
            print("ERROR: provide --audio_path or --audio_dir")
            sys.exit(1)

        # ── load motion ──
        try:
            if fmt == "humanml":
                joints = load_humanml_motion(mpath, mean_path, std_path)
            else:
                joints = load_edge_motion(mpath)
        except Exception as e:
            if verbose:
                print(f"  SKIP {stem}: failed to load motion — {e}")
            continue

        # ── extract beats (truncate music beats to motion duration) ──
        n_motion_frames = joints.shape[0]
        music_beats = get_music_beats(wav, fps, max_frames=n_motion_frames)
        motion_beats = get_motion_beats(joints, fps=fps)

        if len(motion_beats) == 0:
            if verbose:
                print(f"  WARN {stem}: no motion beats detected")
            score = 0.0
        else:
            score = ba_score(music_beats, motion_beats, sigma=sigma)

        scores.append(score)
        per_file.append((stem, score))

        if verbose:
            print(f"  {stem}: BAS={score:.4f}  "
                  f"(music_beats={len(music_beats)}, motion_beats={len(motion_beats)})")

    mean_bas = float(np.mean(scores)) if scores else 0.0
    return mean_bas, per_file


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Beat Alignment Score for Stage 2 (HumanML3D .npy) and EDGE (.pkl)"
    )
    p.add_argument("--motion_dir",  required=True,
                   help="Directory containing generated motion files")
    p.add_argument("--format",      required=True, choices=["humanml", "edge"],
                   help="Motion file format: 'humanml' (.npy 263-dim) or 'edge' (.pkl)")
    p.add_argument("--fps",         type=float, default=20.0,
                   help="Motion FPS: 20 for Stage 2, 30 for EDGE (default: 20)")
    p.add_argument("--audio_dir",   default=None,
                   help="Directory of .wav files, matched to motions by filename prefix")
    p.add_argument("--audio_path",  default=None,
                   help="Single .wav file to use for all motions")
    p.add_argument("--mean_path",   default="./dataset/HumanML3D/Mean.npy",
                   help="Path to HumanML3D Mean.npy (for denormalization)")
    p.add_argument("--std_path",    default="./dataset/HumanML3D/Std.npy",
                   help="Path to HumanML3D Std.npy (for denormalization)")
    p.add_argument("--sigma",       type=float, default=None,
                   help="BA-score Gaussian sigma in frames (default: auto from FPS, "
                        "matching Bailando's σ=3 at 60 FPS)")
    p.add_argument("--quiet",       action="store_true",
                   help="Suppress per-file output")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.audio_path is None and args.audio_dir is None:
        print("ERROR: provide --audio_path or --audio_dir")
        sys.exit(1)

    print(f"\n=== Beat Alignment Score ===")
    print(f"  motion_dir : {args.motion_dir}")
    print(f"  format     : {args.format}")
    print(f"  fps        : {args.fps}")
    ba_sigma = args.sigma if args.sigma is not None else BA_SIGMA_TIME_SEC * args.fps
    print(f"  BA sigma   : {ba_sigma:.2f} frames (auto from FPS)" if args.sigma is None
          else f"  BA sigma   : {args.sigma} frames (manual)")
    if args.audio_path:
        print(f"  audio      : {args.audio_path} (single, used for all)")
    else:
        print(f"  audio_dir  : {args.audio_dir}")
    print()

    mean_bas, per_file = evaluate_bas(
        motion_dir=args.motion_dir,
        fps=args.fps,
        fmt=args.format,
        audio_dir=args.audio_dir,
        audio_path=args.audio_path,
        mean_path=args.mean_path,
        std_path=args.std_path,
        sigma=args.sigma,
        verbose=not args.quiet,
    )

    print(f"\n{'─'*40}")
    print(f"  Evaluated : {len(per_file)} motions")
    print(f"  Mean BAS  : {mean_bas:.4f}")
    print(f"{'─'*40}\n")
