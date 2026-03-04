####################################################################################[start]
####################################################################################[start]

"""

Visualize generated motion samples as MP4 videos with audio.

Uses MDM's plot_3d_motion for skeleton rendering, then muxes the
conditioning audio track so you can judge rhythmic alignment by eye+ear.

Usage:
    # Single sample with audio
    python -m sample.visualize_with_audio \
        --sample_dir ./save/audio_stage2/samples_audio \
        --audio_path ./dataset/aist/audio/mBR0.wav \
        --humanml_dir ./dataset/HumanML3D \
        --output_dir ./save/audio_stage2/videos

    # Side-by-side comparison: audio+text vs text-only
    python -m sample.visualize_with_audio \
        --sample_dir ./save/audio_stage2/samples_fair_audio \
        --compare_dir ./save/audio_stage2/samples_fair_noaudio \
        --audio_path ./dataset/aist/audio/mBR0.wav \
        --humanml_dir ./dataset/HumanML3D \
        --output_dir ./save/audio_stage2/videos_compare \
        --text_prompt "a person walks forward and waves"

"""

import os
import sys
import argparse
import subprocess
import numpy as np
import torch
from glob import glob

sys.path.insert(0, '.')

from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion

def recover_joints(motion_263, mean, std, joints_num=22):
    
    """Convert (T, 263) normalized features → (T, 22, 3) joint positions."""
    
    motion = motion_263 * std + mean
    motion_t = torch.from_numpy(motion).float().unsqueeze(0) # (1, T, 263)
    joints = recover_from_ric(motion_t, joints_num) # (1, T, 22, 3)
    
    return joints.squeeze(0).numpy() # (T, 22, 3)

def mux_audio(video_path, audio_path, output_path, duration=None):
    
    """Combine a video file with an audio file using ffmpeg."""
    
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'warning',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
    ]
    if duration is not None:
        cmd += ['-t', str(duration)]
    
    cmd.append(output_path)
    subprocess.run(cmd, check=True)

def hstack_videos(video_paths, output_path, labels=None):
    
    """Stack videos horizontally using ffmpeg."""
    
    n = len(video_paths)
    inputs = []
    
    for vp in video_paths:
        inputs += ['-i', vp]

    v_refs = ''.join(f'[{i}:v]' for i in range(n))
    filter_str = f'{v_refs}hstack=inputs={n}[out]'

    cmd = [
        'ffmpeg', '-y', '-loglevel', 'warning',
    ] + inputs + [
        '-filter_complex', filter_str,
        '-map', '[out]',
        '-c:v', 'libx264', '-crf', '23',
        output_path,
    ]
    subprocess.run(cmd, check=True)

def render_motion_video(joints, output_path, title='', fps=20, dataset='humanml'):
    
    """Render joints to MP4 using MDM's plot_3d_motion."""
    
    skeleton = paramUtil.t2m_kinematic_chain
    clip = plot_3d_motion(output_path, skeleton, joints,
                          dataset=dataset, title=title, fps=fps)
    clip.duration = joints.shape[0] / fps
    clip.write_videofile(output_path, fps=fps, logger=None)
    clip.close()
    
    print(f"  Rendered: {output_path}")

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_dir', type=str, required=True)
    parser.add_argument('--compare_dir', type=str, default='')
    parser.add_argument('--audio_path', type=str, default='')
    parser.add_argument('--humanml_dir', type=str, default='./dataset/HumanML3D')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--max_samples', type=int, default=3)
    parser.add_argument('--text_prompt', type=str, default='')
    parser.add_argument('--dataset', type=str, default='humanml')
    args = parser.parse_args()

    if not args.output_dir:
        args.output_dir = os.path.join(args.sample_dir, 'videos')
    os.makedirs(args.output_dir, exist_ok=True)

    # -- load normalization --
    
    mean = np.load(os.path.join(args.humanml_dir, 'Mean.npy'))
    std = np.load(os.path.join(args.humanml_dir, 'Std.npy'))

    # -- find samples --
    
    sample_files = sorted(glob(os.path.join(args.sample_dir, '*.npy')))[:args.max_samples]
    print(f"Found {len(sample_files)} samples")

    for i, f in enumerate(sample_files):
        
        name = os.path.splitext(os.path.basename(f))[0]
        motion = np.load(f)
        joints = recover_joints(motion, mean, std)
        duration = joints.shape[0] / args.fps

        title = args.text_prompt if args.text_prompt else name
        if args.audio_path:
            title += ' [audio-conditioned]'

        # -- render skeleton video --
        
        raw_video = os.path.join(args.output_dir, f'{name}_raw.mp4')
        render_motion_video(joints, raw_video, title=title, fps=args.fps,
                            dataset=args.dataset)

        # -- mux audio if provided --
        
        if args.audio_path and os.path.exists(args.audio_path):
            final_video = os.path.join(args.output_dir, f'{name}.mp4')
            mux_audio(raw_video, args.audio_path, final_video, duration=duration)
            os.remove(raw_video)
            print(f"  With audio: {final_video}")
        else:
            os.rename(raw_video, os.path.join(args.output_dir, f'{name}.mp4'))

    # -- comparison mode --
    
    if args.compare_dir:
    
        compare_files = sorted(glob(os.path.join(args.compare_dir, '*.npy')))[:args.max_samples]
        print(f"\n=== Comparison mode: {len(compare_files)} text-only samples ===")

        for i in range(min(len(sample_files), len(compare_files))):
            
            # -- audio+text sample --
            
            motion_a = np.load(sample_files[i])
            joints_a = recover_joints(motion_a, mean, std)

            # -- text-only sample --
            
            motion_b = np.load(compare_files[i])
            joints_b = recover_joints(motion_b, mean, std)

            duration = min(joints_a.shape[0], joints_b.shape[0]) / args.fps
            title_a = 'Audio+Text'
            title_b = 'Text only'

            # -- render both --
            
            vid_a = os.path.join(args.output_dir, f'compare_{i:02d}_audio.mp4')
            vid_b = os.path.join(args.output_dir, f'compare_{i:02d}_text.mp4')
            render_motion_video(joints_a, vid_a, title=title_a, fps=args.fps,
                                dataset=args.dataset)
            render_motion_video(joints_b, vid_b, title=title_b, fps=args.fps,
                                dataset=args.dataset)

            # -- stack side by side --
            
            stacked = os.path.join(args.output_dir, f'compare_{i:02d}_stacked.mp4')
            hstack_videos([vid_a, vid_b], stacked,
                          labels=['Audio+Text', 'Text only'])

            # -- mux audio onto stacked --
            
            if args.audio_path and os.path.exists(args.audio_path):
            
                final = os.path.join(args.output_dir, f'compare_{i:02d}.mp4')
                mux_audio(stacked, args.audio_path, final, duration=duration)
            
                for tmp in [vid_a, vid_b, stacked]:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                print(f"  Final comparison: {final}")
                
            else:
                os.rename(stacked, os.path.join(args.output_dir, f'compare_{i:02d}.mp4'))
                for tmp in [vid_a, vid_b]:
                    if os.path.exists(tmp):
                        os.remove(tmp)

    print(f"\nAll videos saved to {args.output_dir}")

if __name__ == '__main__':
    main()
    
####################################################################################[end]
####################################################################################[end]