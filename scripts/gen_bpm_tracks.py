"""
Generate synthetic drum-pattern audio tracks at specified BPMs.

Each track has:
  - Kick drum on beat 1 & 3
  - Snare on beat 2 & 4
  - Hi-hat on every 8th note (for texture / energy sense)
  - White noise pad scaled by energy level (low for ballad, full for D&B)

Outputs WAV files to the given directory.
"""
import argparse, os
import numpy as np
import soundfile as sf

SR = 44100
DURATION = 10.0  # seconds — enough for full motion generation

def make_drum_hit(sr, freq=200, decay=0.08, amplitude=1.0):
    """Sine-decayed drum hit."""
    t = np.linspace(0, decay, int(sr * decay), endpoint=False)
    env = np.exp(-t / (decay * 0.3))
    return amplitude * env * np.sin(2 * np.pi * freq * t)

def make_snare(sr, decay=0.06, amplitude=0.7):
    """Noise burst for snare."""
    n = int(sr * decay)
    t = np.linspace(0, decay, n, endpoint=False)
    env = np.exp(-t / (decay * 0.25))
    return amplitude * env * np.random.randn(n) * 0.5

def make_hihat(sr, decay=0.015, amplitude=0.35):
    """Short noise burst for hi-hat."""
    n = int(sr * decay)
    t = np.linspace(0, decay, n, endpoint=False)
    env = np.exp(-t / (decay * 0.3))
    return amplitude * env * np.random.randn(n)

def add_hit(audio, hit, pos_sec, sr):
    start = int(pos_sec * sr)
    end   = min(start + len(hit), len(audio))
    audio[start:end] += hit[:end - start]

def generate_track(bpm, duration, sr, energy=1.0, label=""):
    audio = np.zeros(int(duration * sr))
    beat_sec = 60.0 / bpm
    eighth_sec = beat_sec / 2.0

    kick  = make_drum_hit(sr, freq=80,  decay=0.12, amplitude=0.9 * energy)
    snare = make_snare(sr, decay=0.07,  amplitude=0.65 * energy)
    hh    = make_hihat(sr, decay=0.012, amplitude=0.25 * energy)

    t = 0.0
    beat_in_bar = 0
    while t < duration:
        eighth_in_beat = 0
        while eighth_in_beat < 2 and t < duration:
            if eighth_in_beat == 0:
                if beat_in_bar % 4 in (0, 2):   # beats 1 & 3
                    add_hit(audio, kick, t, sr)
                elif beat_in_bar % 4 in (1, 3): # beats 2 & 4
                    add_hit(audio, snare, t, sr)
            add_hit(audio, hh, t, sr)
            t += eighth_sec
            eighth_in_beat += 1
        beat_in_bar += 1

    # Soft pad noise for energy texture
    pad_amp = 0.03 * energy
    audio += pad_amp * np.random.randn(len(audio))

    # Normalize to [-0.85, 0.85]
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.85
    return audio.astype(np.float32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="/Data/yash.bhardwaj/eval/e6_audio")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tracks = [
        ("slow_ballad_70bpm",  70,  0.55),
        ("medium_pop_120bpm", 120,  0.85),
        ("fast_dnb_160bpm",   160,  1.00),
    ]
    for name, bpm, energy in tracks:
        print(f"Generating {name} ({bpm} BPM, energy={energy})...")
        np.random.seed(42)
        audio = generate_track(bpm, DURATION, SR, energy=energy, label=name)
        path = os.path.join(args.out_dir, f"{name}.wav")
        sf.write(path, audio, SR)
        print(f"  Saved {path}  ({len(audio)/SR:.1f}s)")
    print("Done.")
