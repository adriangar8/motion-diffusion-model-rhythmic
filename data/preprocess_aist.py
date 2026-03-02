####################################################################################[start]
####################################################################################[start]

"""

Preprocess AIST++ → HumanML3D 263-dim features + librosa audio features.

This script uses the OFFICIAL HumanML3D conversion code. It requires
cloning one additional repo for the process_file function.

=== SETUP (run once, on the cluster) ===

    # Clone HumanML3D repo (contains the conversion utilities)
    cd /Data/$USER/mdm-project
    git clone https://github.com/EricGuo5513/HumanML3D.git

    # OR clone MoMask (has a standalone process_file in utils/motion_process.py)
    git clone https://github.com/EricGuo5513/momask-codes.git

=== USAGE ===

    cd /Data/$USER/mdm-project/motion-diffusion-model

    python preprocess_aist.py \
        --aist_dir /path/to/aist_plusplus \
        --music_dir /path/to/music_wav \
        --humanml_dir ./dataset/HumanML3D \
        --conversion_repo /Data/$USER/mdm-project/HumanML3D \
        --device cuda

=== OUTPUT ===

    <aist_dir>/processed/
    ├── motions_263/   ← (T, 263) .npy per sequence
    ├── audio_feats/   ← (T, 145) .npy per sequence
    └── joints_22/     ← (T, 22, 3) .npy per sequence (for visualization)
    
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch
from glob import glob
from tqdm import tqdm

# -- step 1: import process_file from HumanML3D or MoMask --

def get_process_file(conversion_repo, example_joints_path=None):
    
    """
    
    Import process_file and initialize its required module-level globals.

    process_file depends on these globals (defined outside the function):
    
        - tgt_offsets: target skeleton bone offsets for uniform_skeleton
        - n_raw_offsets: raw T-pose offsets (torch tensor)
        - kinematic_chain: list of kinematic chains
        - face_joint_indx: [r_hip, l_hip, r_shoulder, l_shoulder]
        - l_idx1, l_idx2: leg joint indices for scale computation
        - fid_r, fid_l: right/left foot joint indices for contact detection

    Set these up in the module's namespace before calling process_file.
    
    """

    # -- find and import the motion_process module --

    momask_path = os.path.join(conversion_repo, 'utils', 'motion_process.py')
    humanml3d_path = os.path.join(conversion_repo, 'motion_process.py')

    if os.path.exists(momask_path):
        
        sys.path.insert(0, conversion_repo)
        import utils.motion_process as motion_process_module
    
        print(f"✓ Found motion_process at: {momask_path}")
    
    elif os.path.exists(humanml3d_path):
    
        sys.path.insert(0, conversion_repo)
    
        import importlib.util
    
        spec = importlib.util.spec_from_file_location("motion_process", humanml3d_path)
        motion_process_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(motion_process_module)
    
        print(f"✓ Found motion_process at: {humanml3d_path}")
    
    else:
    
        print("✗ Could not find motion_process.py")
        print(f"  Looked in: {momask_path}")
        print(f"  Looked in: {humanml3d_path}")
    
        sys.exit(1)

    # -- import skeleton and param utilities --

    # -- these live in the conversion repo under common/ and paramUtil.py (HumanML3D) or under utils/paramUtil.py (MoMask) --
    
    try:
    
        from common.skeleton import Skeleton
        from common.quaternion import qbetween_np, qrot_np, qmul_np, qinv_np, qfix
    
        print("✓ Imported common.skeleton and common.quaternion")
    
    except ImportError:
    
        # -- MoMask might have them elsewhere
        
        sys.path.insert(0, os.path.join(conversion_repo))
        
        from common.skeleton import Skeleton
        from common.quaternion import qbetween_np, qrot_np, qmul_np, qinv_np, qfix
        
        print("✓ Imported common.skeleton and common.quaternion (from conversion_repo)")

    # -- import paramUtil (contains t2m_raw_offsets, t2m_kinematic_chain, etc.) --
    
    try:
    
        from utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain
        print("✓ Imported paramUtil from utils/")
    
    except ImportError:
    
        try:
    
            from paramUtil import t2m_raw_offsets, t2m_kinematic_chain
            print("✓ Imported paramUtil from root")
    
        except ImportError:
    
            # -- last resort: use MDM's bundled copy --
            
            mdm_path = os.path.join('.', 'data_loaders', 'humanml', 'utils')
            sys.path.insert(0, mdm_path)
    
            from paramUtil import t2m_raw_offsets, t2m_kinematic_chain
    
            print("✓ Imported paramUtil from MDM bundled copy")

    # -- initialize the globals that process_file needs --

    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    kinematic_chain = t2m_kinematic_chain

    # -- face_joint_indx: [right_hip, left_hip, right_shoulder, left_shoulder] --
    
    face_joint_indx = [2, 1, 17, 16]

    # -- leg indices for scale computation in uniform_skeleton --
    # -- l_idx1 = right hip joint index, l_idx2 = right knee joint index --
    
    l_idx1, l_idx2 = 5, 8 # in the offset array

    # -- foot joint indices for contact detection --
    
    fid_r, fid_l = [8, 11], [7, 10] # right/left ankle and toe

    if example_joints_path and os.path.exists(example_joints_path):
        
        # -- load a real example --
        
        example_data = np.load(example_joints_path)
        
        if example_data.ndim == 2:
    
            # -- (T, 263) format — need actual joints, not features --
            
            example_data = None
        
        elif example_data.ndim == 3 and example_data.shape[1] == 22:
            example_data = example_data # (T, 22, 3)
        
        else:
            example_data = None
    
    else:
        
        example_data = None

    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')

    if example_data is not None:
        tgt_offsets = tgt_skel.get_offsets_joints(
            torch.from_numpy(example_data[0])
        )
    else:
        
        # -- compute from a zero pose (T-pose) — this gives the canonical offsets --
        # -- this is what HumanML3D effectively does with its reference example --
        
        tgt_offsets = tgt_skel.get_offsets_joints(
            torch.zeros(22, 3)
        )

    print(f"✓ Computed tgt_offsets: {tgt_offsets.shape}")

    # -- inject globals into the motion_process module --

    motion_process_module.tgt_offsets = tgt_offsets
    motion_process_module.n_raw_offsets = n_raw_offsets
    motion_process_module.kinematic_chain = kinematic_chain
    motion_process_module.face_joint_indx = face_joint_indx
    motion_process_module.l_idx1 = l_idx1
    motion_process_module.l_idx2 = l_idx2
    motion_process_module.fid_r = fid_r
    motion_process_module.fid_l = fid_l

    # -- also inject into its global namespace via the module dict --
    
    if hasattr(motion_process_module, '__dict__'):
    
        motion_process_module.__dict__['tgt_offsets'] = tgt_offsets
        motion_process_module.__dict__['n_raw_offsets'] = n_raw_offsets
        motion_process_module.__dict__['kinematic_chain'] = kinematic_chain
        motion_process_module.__dict__['face_joint_indx'] = face_joint_indx
        motion_process_module.__dict__['l_idx1'] = l_idx1
        motion_process_module.__dict__['l_idx2'] = l_idx2
        motion_process_module.__dict__['fid_r'] = fid_r
        motion_process_module.__dict__['fid_l'] = fid_l

    # -- make sure torch and numpy are available in the module --
    
    motion_process_module.torch = torch
    motion_process_module.np = np

    print("✓ Initialized all module globals for process_file")

    return motion_process_module.process_file

# -- step 2: SMPL Forward Kinematics --

def load_aist_motion(pkl_path):
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    smpl_poses = data['smpl_poses'].astype(np.float32)
    smpl_trans = data['smpl_trans'].astype(np.float32)
    smpl_scaling = data['smpl_scaling']
    
    return smpl_poses, smpl_trans, smpl_scaling

def smpl_to_joints_22(smpl_poses, smpl_trans, smpl_scaling, smpl_model):
    
    T = smpl_poses.shape[0]
    all_joints = []
    batch_size = 512
    scaling = float(smpl_scaling[0])
    
    for start in range(0, T, batch_size):
    
        end = min(start + batch_size, T)
    
        poses_batch = torch.tensor(smpl_poses[start:end], dtype=torch.float32)
        trans_batch = torch.tensor(smpl_trans[start:end], dtype=torch.float32)
        
        output = smpl_model(
            body_pose=poses_batch[:, 3:],
            global_orient=poses_batch[:, :3],
            transl=torch.zeros_like(trans_batch), # don't pass raw trans
        )
    
        joints = output.joints[:, :22] # (B, 22, 3) in meters
        joints = joints * scaling + trans_batch.unsqueeze(1) # apply AIST++ convention
        joints = joints / 100.0 # cm → m
    
        all_joints.append(joints.detach().numpy())
    
    return np.concatenate(all_joints, axis=0)

def use_keypoints3d(aist_dir, seq_name):

    """

    Alternatively, load pre-computed 3D keypoints from AIST++ (17 COCO joints).
    These are less accurate but don't require SMPL.

    Returns None if not available.

    """

    kp3d_path = os.path.join(aist_dir, 'keypoints3d', f'{seq_name}.pkl')

    if not os.path.exists(kp3d_path):
        return None

    with open(kp3d_path, 'rb') as f:
        data = pickle.load(f)

    kp3d = data['keypoints3d'] # (T, 17, 3) COCO format

    return None # disabled — use SMPL FK instead

# -- step 3: Audio Feature Extraction --

def get_music_name(sequence_name):
    
    """Extract music ID from AIST++ sequence name."""
    
    for part in sequence_name.split('_'):
    
        if part.startswith('m') and len(part) >= 3:
    
            # -- music IDs look like: mBR0, mPO1, mLO2, etc. --
            
            if part[1:3].isalpha():
                return part
    
    return None

# -- main --

def main():
    
    parser = argparse.ArgumentParser(
        description='Preprocess AIST++ for audio-conditioned MDM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument('--aist_dir', type=str, required=True)
    parser.add_argument('--music_dir', type=str, required=True)
    parser.add_argument('--humanml_dir', type=str, default='./dataset/HumanML3D')
    parser.add_argument('--conversion_repo', type=str, required=True,
                        help='Path to cloned MoMask or HumanML3D repo')
    parser.add_argument('--source_fps', type=int, default=60)
    parser.add_argument('--target_fps', type=int, default=20)
    parser.add_argument('--feet_thre', type=float, default=0.002)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    # -- import process_file (with proper tgt_offsets from HumanML3D example) --

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

    # -- import audio feature extraction from our model/directory --
    
    sys.path.insert(0, '.')
    
    from model.audio_features import extract_audio_features

    # -- output directories --
    
    motion_out = os.path.join(args.aist_dir, 'processed', 'motions_263')
    audio_out = os.path.join(args.aist_dir, 'processed', 'audio_feats')
    joints_out = os.path.join(args.aist_dir, 'processed', 'joints_22')
    
    os.makedirs(motion_out, exist_ok=True)
    os.makedirs(audio_out, exist_ok=True)
    os.makedirs(joints_out, exist_ok=True)

    # -- copy normalization stats from HumanML3D --
    
    mean = np.load(os.path.join(args.humanml_dir, 'Mean.npy'))
    std = np.load(os.path.join(args.humanml_dir, 'Std.npy'))
    
    np.savez(os.path.join(args.aist_dir, 'processed_stats.npz'), mean=mean, std=std)
    
    print(f"Normalization stats saved (shape: {mean.shape})")

    # -- find motion .pkl files --
    
    motion_dir = os.path.join(args.aist_dir, 'motions')
    pkl_files = sorted(glob(os.path.join(motion_dir, '*.pkl')))
    
    print(f"Found {len(pkl_files)} motion files")

    # -- ignore list --
    
    ignore_path = os.path.join(args.aist_dir, 'ignore_list.txt')
    ignore = set()
    
    if os.path.exists(ignore_path):
    
        with open(ignore_path) as f:
            ignore = {l.strip() for l in f if l.strip()}
    
        print(f"Ignoring {len(ignore)} sequences")

    # -- process --
    
    ok, fail, skip = 0, 0, 0
    
    for pkl_path in tqdm(pkl_files, desc='AIST++'):
    
        name = os.path.splitext(os.path.basename(pkl_path))[0]
    
        if name in ignore:
    
            skip += 1
            continue

        m_path = os.path.join(motion_out, f'{name}.npy')
        a_path = os.path.join(audio_out, f'{name}.npy')
        j_path = os.path.join(joints_out, f'{name}.npy')

        if os.path.exists(m_path) and os.path.exists(a_path):
    
            ok += 1
            continue

        try:
    
            # -- 1. load SMPL params → joint positions (T, 22, 3) --
            
            poses, trans, scaling = load_aist_motion(pkl_path)
            joints_60 = smpl_to_joints_22(poses, trans, scaling, smpl_model)

            # -- 2. downsample 60fps → 20fps --
            
            ratio = args.source_fps / args.target_fps
            
            idx = np.arange(0, len(joints_60), ratio).astype(int)
            idx = idx[idx < len(joints_60)]
            
            joints_20 = joints_60[idx]

            if len(joints_20) < 40: # < 2s
                
                skip += 1
                continue

            # -- 3. convert to 263-dim HumanML3D features --
            # -- process_file returns: (data, global_pos, positions, l_velocity) --
            
            result = process_file(joints_20, args.feet_thre)
            
            if isinstance(result, tuple):
                features = result[0] # (T-1, 263)
            
            else:
                features = result

            # -- 4. extract audio features --
            
            music_id = get_music_name(name)
            
            if music_id is None:
                raise ValueError(f"No music ID in '{name}'")

            wav = None
            
            for candidate in [
                os.path.join(args.music_dir, f'{music_id}.wav'),
                os.path.join(args.music_dir, f'{name}.wav'),
            ]:
                if os.path.exists(candidate):
                    wav = candidate
                    break
            
            if wav is None:
                raise FileNotFoundError(f"No .wav for {music_id}")

            duration = features.shape[0] / args.target_fps + 1.0
            audio = extract_audio_features(wav, target_fps=args.target_fps,
                                           duration=duration)

            # -- 5. align lengths --
            
            L = min(features.shape[0], audio.shape[0])
            features = features[:L]
            audio = audio[:L]

            # -- 6. save --
            
            np.save(m_path, features.astype(np.float32))
            np.save(a_path, audio.astype(np.float32))
            np.save(j_path, joints_20.astype(np.float32))
            
            ok += 1

        except Exception as e:
            
            tqdm.write(f"  FAIL {name}: {e}")
            fail += 1

    print(f"\nDone: {ok} ok, {fail} failed, {skip} skipped")
    print(f"Output: {os.path.join(args.aist_dir, 'processed')}")

if __name__ == '__main__':
    main()
    
####################################################################################[end]
####################################################################################[end]
