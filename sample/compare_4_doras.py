"""
Compare the 4 DoRA style adapters on the same prompt and audio.

Generates motion for each style with the same text + audio, renders videos,
then builds a 2x2 grid video so you can compare styles side by side.

Usage:
  # Full pipeline: generate + render + 2x2 grid
  python -m sample.compare_4_doras --audio_path /path/to/audio.wav --text_prompt "a person performs breakdancing moves to music" --output_root ./save/audio_stage3_dora/compare_4styles --num_samples 1

  # Only build grid from existing video dirs (expects output_root/videos_old_elderly/, videos_angry_aggressive/, ...)
  python -m sample.compare_4_doras --output_root ./save/audio_stage3_dora/compare_4styles --skip_generate --skip_render
"""

import os
import sys
import argparse
import subprocess
from glob import glob

STYLES = ['old_elderly', 'angry_aggressive', 'proud_confident', 'robot_mechanical']


def run_cmd(cmd, check=True):
    print("  $ " + " ".join(cmd))
    return subprocess.run(cmd, check=check)


def grid_2x2_videos(video_paths, output_path, audio_path=None, duration=None):
    """Create 2x2 grid from 4 videos. Order: [0]=TL, [1]=TR, [2]=BL, [3]=BR."""
    assert len(video_paths) == 4 and all(os.path.isfile(p) for p in video_paths)
    filter_str = (
        "[0:v][1:v]hstack=inputs=2[top];"
        "[2:v][3:v]hstack=inputs=2[bottom];"
        "[top][bottom]vstack=inputs=2[out]"
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-i", video_paths[0], "-i", video_paths[1],
        "-i", video_paths[2], "-i", video_paths[3],
        "-filter_complex", filter_str,
        "-map", "[out]",
        "-c:v", "libx264", "-crf", "23",
        output_path,
    ]
    subprocess.run(cmd, check=True)
    if audio_path and os.path.isfile(audio_path):
        final = output_path.replace(".mp4", "_with_audio.mp4")
        if final == output_path:
            final = output_path + ".with_audio.mp4"
        mux_cmd = [
            "ffmpeg", "-y", "-loglevel", "warning",
            "-i", output_path, "-i", audio_path,
            "-c:v", "copy", "-c:a", "aac", "-shortest",
        ]
        if duration is not None:
            mux_cmd += ["-t", str(duration)]
        mux_cmd.append(final)
        subprocess.run(mux_cmd, check=True)
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(final, output_path)
        print("  Muxed audio -> " + output_path)


def main():
    p = argparse.ArgumentParser(description="Compare 4 DoRA styles: generate + render + 2x2 grid")
    p.add_argument("--stage2_dir", type=str, default="./save/audio_stage2_wav2clip_beataware",
                   help="Stage 2 checkpoint dir (default: beataware)")
    p.add_argument("--adapter_root", type=str, default="./save/audio_stage3_dora_beataware",
                   help="Stage 3 DoRA adapters root (default: beataware adapters)")
    p.add_argument("--audio_path", type=str, default="/Data/yash.bhardwaj/datasets/aist/audio/mJB0.wav",
                   help="Path to .wav (e.g. mJB0.wav ballet jazz, mBR0.wav breakdance)")
    p.add_argument("--text_prompt", type=str, default="a person performs ballet jazz dance moves to music")
    p.add_argument("--output_root", type=str, default="./save/audio_stage3_dora_beataware/compare_4styles")
    p.add_argument("--humanml_dir", type=str, default="/Data/yash.bhardwaj/datasets/HumanML3D")
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--skip_generate", action="store_true")
    p.add_argument("--skip_render", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=20)
    # GCDM (Phase 4) composite guidance
    p.add_argument("--use_gcdm", action="store_true", help="Use GCDM composite guidance for generation")
    p.add_argument("--gcdm_alpha", type=float, default=3.0)
    p.add_argument("--gcdm_beta_text", type=float, default=1.0)
    p.add_argument("--gcdm_beta_audio", type=float, default=1.5)
    p.add_argument("--gcdm_lambda_start", type=float, default=0.8)
    p.add_argument("--gcdm_lambda_end", type=float, default=0.2)
    args = p.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

    if not args.skip_generate:
        for style in STYLES:
            adapter_path = os.path.join(args.adapter_root, style, "adapter_final.pt")
            if not os.path.isfile(adapter_path):
                print("Skip " + style + ": no " + adapter_path)
                continue
            out_dir = os.path.join(args.output_root, "samples_" + style)
            print("\n=== Generate: " + style + " ===")
            gen_cmd = [
                sys.executable, "-m", "sample.generate_stage3_style",
                "--stage2_dir", os.path.abspath(args.stage2_dir),
                "--adapter_path", os.path.abspath(adapter_path),
                "--text_prompt", args.text_prompt,
                "--output_dir", os.path.abspath(out_dir),
                "--num_samples", str(args.num_samples),
                "--seed", str(args.seed),
            ]
            if args.audio_path and os.path.isfile(args.audio_path):
                gen_cmd += ["--audio_path", os.path.abspath(args.audio_path)]
            if args.use_gcdm:
                gen_cmd += [
                    "--use_gcdm",
                    "--gcdm_alpha", str(args.gcdm_alpha),
                    "--gcdm_beta_text", str(args.gcdm_beta_text),
                    "--gcdm_beta_audio", str(args.gcdm_beta_audio),
                    "--gcdm_lambda_start", str(args.gcdm_lambda_start),
                    "--gcdm_lambda_end", str(args.gcdm_lambda_end),
                ]
            run_cmd(gen_cmd)

    if not args.skip_render:
        for style in STYLES:
            sample_dir = os.path.join(args.output_root, "samples_" + style)
            video_dir = os.path.join(args.output_root, "videos_" + style)
            if not os.path.isdir(sample_dir):
                print("Skip render " + style + ": no " + sample_dir)
                continue
            print("\n=== Render: " + style + " ===")
            viz_cmd = [
                sys.executable, "-m", "sample.visualize_with_audio",
                "--sample_dir", os.path.abspath(sample_dir),
                "--humanml_dir", os.path.abspath(args.humanml_dir),
                "--output_dir", os.path.abspath(video_dir),
                "--max_samples", str(args.num_samples),
                "--samples_denormalized",
                "--fps", str(args.fps),
            ]
            if args.audio_path and os.path.isfile(args.audio_path):
                viz_cmd += ["--audio_path", os.path.abspath(args.audio_path)]
            run_cmd(viz_cmd)

    video_paths = []
    for style in STYLES:
        video_dir = os.path.join(args.output_root, "videos_" + style)
        mp4s = sorted(glob(os.path.join(video_dir, "*.mp4")))
        if not mp4s:
            print("Warning: no videos in " + video_dir)
            video_paths.append(None)
        else:
            video_paths.append(mp4s[0])

    if sum(1 for x in video_paths if x is not None) < 4:
        print("Need videos for all 4 styles to build grid.")
        return

    grid_out = os.path.join(args.output_root, "grid_2x2_sample_00.mp4")
    print("\n=== 2x2 grid -> " + grid_out + " ===")
    grid_2x2_videos(
        video_paths,
        grid_out,
        audio_path=args.audio_path if os.path.isfile(args.audio_path) else None,
    )
    print("\nDone. Grid: " + grid_out)
    print("Order: TL=old_elderly, TR=angry_aggressive, BL=proud_confident, BR=robot_mechanical.")


if __name__ == "__main__":
    main()
