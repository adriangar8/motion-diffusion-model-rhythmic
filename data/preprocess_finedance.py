# [start]
# [start]

"""

Preprocess FineDance → HumanML3D 263-dim features + librosa audio features.

Follows the same pipeline as preprocess_aist.py, adapted for FineDance format.

=== USAGE ===

    cd /Data/$USER/motion-diffusion-model-rhythmic

    python data/preprocess_finedance.py \
        --finedance_dir ./dataset/finedance_raw \
        --humanml_dir ./dataset/HumanML3D \
        --source_fps 30

=== INPUT FORMAT ===

    dataset/finedance_raw/
    ├── label_json/     # 001.json, ... {'name':..., 'style1':..., 'frames':...}
    ├── motion/         # 001.npy, ... — SMPLH motion (T, 315)
    │                   #   cols 0:3     = smpl_trans (3D translation)
    │                   #   cols 3:9     = global_orient (6D rotation)
    │                   #   cols 9:135   = body_pose (21 joints × 6D = 126D)
    │                   #   cols 135:225 = left_hand_pose (15 joints × 6D, ignored)
    │                   #   cols 225:315 = right_hand_pose (15 joints × 6D, ignored)
    │                   # Rotations use the 6D representation (Zhou et al. 2019)
    ├── music_wav/      # 001.wav, ... (named by sequence ID)
    └── music_npy/      # pre-extracted music features (FineDance format, unused)

=== OUTPUT ===

    dataset/finedance_raw/processed/
    ├── motions_263/   ← (T, 263) .npy per sequence
    ├── audio_feats/   ← (T, 145) .npy per sequence
    └── joints_22/     ← (T, 22, 3) .npy per sequence

"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from glob import glob
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Step 1: Import process_file (reuse from preprocess_aist.py)
# ---------------------------------------------------------------------------

def get_process_file(conversion_repo, example_joints_path=None):
    """Import process_file and initialize its module-level globals."""

    momask_path = os.path.join(conversion_repo, 'utils', 'motion_process.py')
    humanml3d_path = os.path.join(conversion_repo, 'motion_process.py')
    scripts_path = os.path.join(
        conversion_repo, 'scripts', 'motion_process.py')

    if os.path.exists(momask_path):
        sys.path.insert(0, conversion_repo)
        import utils.motion_process as motion_process_module
        print(f"✓ Found motion_process at: {momask_path}")

    elif os.path.exists(humanml3d_path):
        sys.path.insert(0, conversion_repo)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "motion_process", humanml3d_path)
        motion_process_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(motion_process_module)
        print(f"✓ Found motion_process at: {humanml3d_path}")

    elif os.path.exists(scripts_path):
        sys.path.insert(0, conversion_repo)
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..'))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "motion_process", scripts_path)
        motion_process_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(motion_process_module)
        print(f"✓ Found motion_process at: {scripts_path}")

    else:
        print("✗ Could not find motion_process.py")
        print(f"  Looked in: {momask_path}")
        print(f"  Looked in: {humanml3d_path}")
        print(f"  Looked in: {scripts_path}")
        sys.exit(1)

    try:
        from common.skeleton import Skeleton
        from common.quaternion import qbetween_np, qrot_np, qmul_np, qinv_np, qfix
        print("✓ Imported common.skeleton and common.quaternion")
    except ImportError:
        sys.path.insert(0, os.path.join(conversion_repo))
        from common.skeleton import Skeleton
        from common.quaternion import qbetween_np, qrot_np, qmul_np, qinv_np, qfix
        print("✓ Imported common.skeleton and common.quaternion (from conversion_repo)")

    try:
        from utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain
        print("✓ Imported paramUtil from utils/")
    except ImportError:
        try:
            from paramUtil import t2m_raw_offsets, t2m_kinematic_chain
            print("✓ Imported paramUtil from root")
        except ImportError:
            mdm_path = os.path.join('.', 'data_loaders', 'humanml', 'utils')
            sys.path.insert(0, mdm_path)
            from paramUtil import t2m_raw_offsets, t2m_kinematic_chain
            print("✓ Imported paramUtil from MDM bundled copy")

    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    kinematic_chain = t2m_kinematic_chain
    face_joint_indx = [2, 1, 17, 16]
    l_idx1, l_idx2 = 5, 8
    fid_r, fid_l = [8, 11], [7, 10]

    if example_joints_path and os.path.exists(example_joints_path):
        example_data = np.load(example_joints_path)
        if example_data.ndim == 3 and example_data.shape[1] == 22:
            pass  # (T, 22, 3) — good
        else:
            example_data = None
    else:
        example_data = None

    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')

    if example_data is not None:
        tgt_offsets = tgt_skel.get_offsets_joints(
            torch.from_numpy(example_data[0]))
    else:
        tgt_offsets = tgt_skel.get_offsets_joints(torch.zeros(22, 3))

    print(f"✓ Computed tgt_offsets: {tgt_offsets.shape}")

    motion_process_module.tgt_offsets = tgt_offsets
    motion_process_module.n_raw_offsets = n_raw_offsets
    motion_process_module.kinematic_chain = kinematic_chain
    motion_process_module.face_joint_indx = face_joint_indx
    motion_process_module.l_idx1 = l_idx1
    motion_process_module.l_idx2 = l_idx2
    motion_process_module.fid_r = fid_r
    motion_process_module.fid_l = fid_l

    if hasattr(motion_process_module, '__dict__'):
        motion_process_module.__dict__['tgt_offsets'] = tgt_offsets
        motion_process_module.__dict__['n_raw_offsets'] = n_raw_offsets
        motion_process_module.__dict__['kinematic_chain'] = kinematic_chain
        motion_process_module.__dict__['face_joint_indx'] = face_joint_indx
        motion_process_module.__dict__['l_idx1'] = l_idx1
        motion_process_module.__dict__['l_idx2'] = l_idx2
        motion_process_module.__dict__['fid_r'] = fid_r
        motion_process_module.__dict__['fid_l'] = fid_l

    motion_process_module.torch = torch
    motion_process_module.np = np

    print("✓ Initialized all module globals for process_file")
    return motion_process_module.process_file


# ---------------------------------------------------------------------------
# Step 2: FineDance motion loading  (6D rotation format)
# ---------------------------------------------------------------------------

def rot6d_to_rotmat(rot6d):
    """
    Convert 6D rotation representation (Zhou et al. 2019) to 3×3 rotation matrices.

    Args:
        rot6d: (..., 6) tensor — first two columns of a rotation matrix

    Returns:
        rotmat: (..., 3, 3) rotation matrices
    """
    # Split into two 3-vectors
    a1 = rot6d[..., :3]   # (..., 3)
    a2 = rot6d[..., 3:6]  # (..., 3)

    # Gram-Schmidt orthonormalization
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = torch.nn.functional.normalize(a2 - dot * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    # Stack into (..., 3, 3)
    return torch.stack([b1, b2, b3], dim=-1)


def rotmat_to_axisangle(rotmat):
    """
    Convert 3×3 rotation matrices to axis-angle vectors.

    Uses the Rodrigues formula via trace.

    Args:
        rotmat: (..., 3, 3)

    Returns:
        axisangle: (..., 3)
    """
    batch_shape = rotmat.shape[:-2]

    # Flatten for processing
    rot = rotmat.reshape(-1, 3, 3)
    B = rot.shape[0]

    # Angle from trace: cos(theta) = (tr(R) - 1) / 2
    trace = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)  # (B,)

    # Axis from skew-symmetric part
    rx = rot[:, 2, 1] - rot[:, 1, 2]
    ry = rot[:, 0, 2] - rot[:, 2, 0]
    rz = rot[:, 1, 0] - rot[:, 0, 1]
    axis_unnorm = torch.stack([rx, ry, rz], dim=-1)  # (B, 3)

    sin_theta = torch.sin(theta).unsqueeze(-1).clamp(min=1e-7)
    axis = axis_unnorm / (2.0 * sin_theta)

    # axis-angle = axis * theta
    axisangle = axis * theta.unsqueeze(-1)

    # For near-zero rotations, output zero vector
    near_zero = (theta < 1e-6).unsqueeze(-1).expand_as(axisangle)
    axisangle = torch.where(near_zero, torch.zeros_like(axisangle), axisangle)

    return axisangle.reshape(*batch_shape, 3)


def load_finedance_motion(npy_path):
    """
    Load FineDance motion from .npy file.

    Layout of the (T, 315) array:
        cols 0:3     → smpl_trans     (global translation, metres)
        cols 3:9     → global_orient  (6D rotation, Zhou et al. 2019)
        cols 9:135   → body_pose      (21 body joints × 6D = 126D)
        cols 135:315 → hand_pose      (30 joints × 6D, ignored)

    Converts 6D rotations → axis-angle before returning.

    Returns:
        global_orient_aa : (T, 3)   axis-angle
        body_pose_aa     : (T, 63)  axis-angle (21 joints × 3)
        smpl_trans       : (T, 3)
    """
    data = np.load(npy_path).astype(np.float32)  # (T, 315)

    if data.ndim != 2 or data.shape[1] != 315:
        raise ValueError(
            f"Unexpected shape {data.shape} — expected (T, 315)"
        )

    smpl_trans = data[:, :3]              # (T, 3)
    go_6d = torch.from_numpy(data[:, 3:9])       # (T, 6)
    bp_6d = torch.from_numpy(data[:, 9:135])     # (T, 126) = (T, 21*6)

    T = data.shape[0]

    # global_orient: (T, 6) → (T, 3, 3) → (T, 3)
    go_mat = rot6d_to_rotmat(go_6d)              # (T, 3, 3)
    go_aa = rotmat_to_axisangle(go_mat)          # (T, 3)

    # body_pose: (T, 21, 6) → (T, 21, 3, 3) → (T, 21, 3) → (T, 63)
    bp_6d_reshaped = bp_6d.reshape(T, 21, 6)     # (T, 21, 6)
    bp_mat = rot6d_to_rotmat(bp_6d_reshaped)     # (T, 21, 3, 3)
    bp_aa = rotmat_to_axisangle(bp_mat)          # (T, 21, 3)
    bp_aa_flat = bp_aa.reshape(T, 63)            # (T, 63)

    return (
        go_aa.numpy().astype(np.float32),
        bp_aa_flat.numpy().astype(np.float32),
        smpl_trans,
    )


def smpl_to_joints_22(global_orient, body_pose, smpl_trans, smpl_model):
    """
    Run SMPL forward kinematics to get (T, 22, 3) joint positions.

    Uses the neutral SMPL model (same as preprocess_aist.py).
    Translation is added back in world space after FK.
    """
    T = global_orient.shape[0]
    all_joints = []
    batch_size = 512

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)

        go_batch = torch.tensor(global_orient[start:end], dtype=torch.float32)
        bp_batch = torch.tensor(body_pose[start:end], dtype=torch.float32)
        tr_batch = torch.tensor(smpl_trans[start:end], dtype=torch.float32)

        # FineDance uses SMPLH (21 body joints = 63 dims). SMPL expects 23
        # body joints (69 dims). Pad the two missing hand root joints with zeros.
        if bp_batch.shape[-1] == 63:
            pad = torch.zeros(bp_batch.shape[0], 6, dtype=bp_batch.dtype)
            bp_batch = torch.cat([bp_batch, pad], dim=-1)  # (B, 69)

        output = smpl_model(
            body_pose=bp_batch,
            global_orient=go_batch,
            transl=torch.zeros_like(tr_batch),  # add translation manually below
        )

        joints = output.joints[:, :22]           # (B, 22, 3) in metres
        joints = joints + tr_batch.unsqueeze(1)  # add world translation

        all_joints.append(joints.detach().numpy())

    return np.concatenate(all_joints, axis=0)    # (T, 22, 3)


# ---------------------------------------------------------------------------
# Step 3: Audio file resolution via label_json
# ---------------------------------------------------------------------------

def load_label_json(label_json_dir, seq_id):
    """
    Load the label JSON for a sequence and return the music wav filename stem.

    FineDance label JSON structure (best-guess from dataset docs):
        {
          "sequence_id": "001",
          "music_name": "some_song_name",
          "coarse_genre": "Ballet Jazz",
          "fine_genre": "Ballet Jazz"
        }

    The 'music_name' field is used to find the matching wav file.

    Returns:
        music_name (str) or None if not found / JSON missing.
    """
    json_path = os.path.join(label_json_dir, f'{seq_id}.json')
    if not os.path.exists(json_path):
        return None

    with open(json_path, 'r') as f:
        label = json.load(f)

    # Try common field names for the music filename
    for key in ('music_name', 'music', 'song_name', 'audio_name', 'file_name'):
        if key in label:
            return str(label[key])

    return None


def find_wav(music_wav_dir, music_name, seq_id):
    """
    Resolve the wav path for a sequence.

    In FineDance, wav files are named by sequence ID (001.wav), so the
    direct match is tried first. The music_name fallback handles edge cases.

    Search order:
    1. <music_wav_dir>/<seq_id>.wav       (primary: FineDance naming convention)
    2. <music_wav_dir>/<music_name>.wav   (from label JSON)
    3. Glob for any wav containing music_name (fuzzy fallback)
    """
    direct = os.path.join(music_wav_dir, f'{seq_id}.wav')
    if os.path.exists(direct):
        return direct

    if music_name:
        for candidate in [
            os.path.join(music_wav_dir, f'{music_name}.wav'),
            os.path.join(music_wav_dir, music_name),
        ]:
            if os.path.exists(candidate):
                return candidate

        matches = glob(os.path.join(music_wav_dir, f'*{music_name}*'))
        if matches:
            return matches[0]

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess FineDance for audio-conditioned MDM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--finedance_dir', type=str, required=True,
                        help='Root of FineDance dataset (contains motion/, music_wav/, label_json/)')
    parser.add_argument('--humanml_dir', type=str, default='./dataset/HumanML3D',
                        help='Path to HumanML3D dataset (for normalization stats and example joints)')
    parser.add_argument('--conversion_repo', type=str, default='./data_loaders/humanml',
                        help='Path to motion_process.py repo (default: bundled copy)')
    parser.add_argument('--source_fps', type=int, default=30,
                        help='Source motion FPS (FineDance is 30fps)')
    parser.add_argument('--target_fps', type=int, default=20,
                        help='Target FPS to match HumanML3D (default: 20)')
    parser.add_argument('--feet_thre', type=float, default=0.002,
                        help='Foot contact threshold for process_file')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    # -- import process_file --

    example_path = None
    new_joints_dir = os.path.join(args.humanml_dir, 'new_joints')
    if os.path.isdir(new_joints_dir):
        candidates = sorted(glob(os.path.join(new_joints_dir, '*.npy')))
        if candidates:
            example_path = candidates[0]

    process_file = get_process_file(args.conversion_repo, example_path)

    # -- load SMPL model --

    import smplx
    smpl_model = smplx.create(
        model_path='body_models/',
        model_type='smpl',
        gender='neutral',
    )
    smpl_model.eval()

    # -- import audio feature extraction --

    sys.path.insert(0, '.')
    from model.audio_features import extract_audio_features

    # -- output directories --

    out_base = os.path.join(args.finedance_dir, 'processed')
    motion_out = os.path.join(out_base, 'motions_263')
    audio_out = os.path.join(out_base, 'audio_feats')
    joints_out = os.path.join(out_base, 'joints_22')

    os.makedirs(motion_out, exist_ok=True)
    os.makedirs(audio_out, exist_ok=True)
    os.makedirs(joints_out, exist_ok=True)

    # -- copy normalization stats from HumanML3D --

    mean_path = os.path.join(args.humanml_dir, 'Mean.npy')
    std_path = os.path.join(args.humanml_dir, 'Std.npy')

    if os.path.exists(mean_path) and os.path.exists(std_path):
        mean = np.load(mean_path)
        std = np.load(std_path)
        np.savez(os.path.join(out_base, 'processed_stats.npz'),
                 mean=mean, std=std)
        print(f"Normalization stats saved (shape: {mean.shape})")
    else:
        print(
            f"⚠ HumanML3D Mean/Std not found at {args.humanml_dir} — skipping stats")

    # -- find motion files --

    motion_dir = os.path.join(args.finedance_dir, 'motion')
    music_wav_dir = os.path.join(args.finedance_dir, 'music_wav')
    label_json_dir = os.path.join(args.finedance_dir, 'label_json')

    npy_files = sorted(glob(os.path.join(motion_dir, '*.npy')))
    print(f"Found {len(npy_files)} motion files in {motion_dir}")

    if not npy_files:
        print(f"ERROR: No .npy files found in {motion_dir}")
        sys.exit(1)

    # -- inspect first file to report source FPS assumption --

    first_data = np.load(npy_files[0])
    print(
        f"Sample motion shape: {first_data.shape}  (assuming {args.source_fps} fps source)")
    if first_data.shape[1] != 315:
        print(
            f"⚠ Expected 315 cols (3 trans + 52×6D), got {first_data.shape[1]} — adjust loading if needed")

    # -- process --

    ok, fail, skip = 0, 0, 0

    for npy_path in tqdm(npy_files, desc='FineDance'):

        seq_id = os.path.splitext(os.path.basename(npy_path))[0]  # e.g. "001"

        m_path = os.path.join(motion_out, f'{seq_id}.npy')
        a_path = os.path.join(audio_out, f'{seq_id}.npy')
        j_path = os.path.join(joints_out, f'{seq_id}.npy')

        if os.path.exists(m_path) and os.path.exists(a_path):
            ok += 1
            continue

        try:
            # -- 1. load motion → SMPL params --

            global_orient, body_pose, smpl_trans = load_finedance_motion(
                npy_path)

            # -- 2. SMPL FK → (T, 22, 3) joint positions --

            joints_src = smpl_to_joints_22(
                global_orient, body_pose, smpl_trans, smpl_model)

            # -- 3. downsample source_fps → target_fps --

            if args.source_fps != args.target_fps:
                ratio = args.source_fps / args.target_fps
                idx = np.arange(0, len(joints_src), ratio).astype(int)
                idx = idx[idx < len(joints_src)]
                joints_20 = joints_src[idx]
            else:
                joints_20 = joints_src

            if len(joints_20) < 40:  # < 2 seconds at 20fps
                tqdm.write(
                    f"  SKIP {seq_id}: too short ({len(joints_20)} frames after downsample)")
                skip += 1
                continue

            # -- 4. convert to 263-dim HumanML3D features --

            result = process_file(joints_20, args.feet_thre)

            if isinstance(result, tuple):
                features = result[0]  # (T-1, 263)
            else:
                features = result

            # -- 5. resolve audio file --

            music_name = load_label_json(label_json_dir, seq_id)
            wav = find_wav(music_wav_dir, music_name, seq_id)

            if wav is None:
                raise FileNotFoundError(
                    f"No wav found for seq {seq_id} (music_name={music_name!r})"
                )

            # -- 6. extract audio features --

            duration = features.shape[0] / args.target_fps + 1.0
            audio = extract_audio_features(
                wav, target_fps=args.target_fps, duration=duration)

            # -- 7. align lengths --

            L = min(features.shape[0], audio.shape[0])
            features = features[:L]
            audio = audio[:L]

            # -- 8. save --

            np.save(m_path, features.astype(np.float32))
            np.save(a_path, audio.astype(np.float32))
            np.save(j_path, joints_20.astype(np.float32))

            ok += 1

        except Exception as e:
            tqdm.write(f"  FAIL {seq_id}: {e}")
            fail += 1

    print(f"\nDone: {ok} ok, {fail} failed, {skip} skipped")
    print(f"Output: {out_base}")

    # -- sanity check on a random output --

    out_files = sorted(glob(os.path.join(motion_out, '*.npy')))
    if out_files:
        sample = np.load(out_files[0])
        print(
            f"\nSpot-check {os.path.basename(out_files[0])}: motion shape = {sample.shape}")
        audio_sample_path = os.path.join(
            audio_out, os.path.basename(out_files[0]))
        if os.path.exists(audio_sample_path):
            asample = np.load(audio_sample_path)
            print(f"  audio shape = {asample.shape}")
            if sample.shape[0] == asample.shape[0]:
                print("  ✓ motion and audio lengths match")
            else:
                print(
                    f"  ⚠ length mismatch: motion T={sample.shape[0]}, audio T={asample.shape[0]}")


if __name__ == '__main__':
    main()

# [end]
# [end]
