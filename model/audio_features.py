####################################################################################[start]
####################################################################################[start]

"""

Audio feature extraction using librosa.

Extracts frame-level features at 20fps to match HumanML3D's motion frame rate.

Each frame gets a 145-dimensional feature vector:

  - Mel spectrogram: 128 dims
  - Onset strength: 1 dim
  - Beat indicator: 1 dim (binary)
  - RMS energy: 1 dim
  - Chroma: 12 dims
  - Spectral centroid: 1 dim
  - Tempo (global, repeated): 1 dim

"""

import numpy as np
import librosa

def extract_audio_features(audio_path, target_fps=20, duration=None):
    
    """
    
    Extract frame-level audio features from an audio file.

    Args:
        audio_path: Path to audio file (.wav, .mp3, etc.)
        target_fps: Frame rate to match motion data (default: 20 fps)
        duration: If set, only load this many seconds of audio

    Returns:
        features: np.ndarray of shape (T, 145) where T = duration * target_fps
    
    """
    
    # -- load audio at 22050 Hz (librosa default) --
    
    sr = 22050
    y, _ = librosa.load(audio_path, sr=sr, duration=duration)

    # -- hop length to get exactly target_fps frames per second --
    # -- at sr=22050 and fps=20: hop_length = 22050/20 = 1102.5 → use 1102 --
    
    hop_length = sr // target_fps
    n_frames = len(y) // hop_length

    if n_frames == 0:
        raise ValueError(f"Audio too short: {len(y)/sr:.2f}s")

    # -- extract features --

    # -- 1. mel spectrogram (128 bands) --
    
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, hop_length=hop_length, n_mels=128
    )
    
    mel_db = librosa.power_to_db(mel, ref=np.max) # (128, T')
    mel_db = mel_db[:, :n_frames].T # (T, 128)

    # -- 2. onset strength (1 dim) --
    
    onset = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop_length
    )[:n_frames] # (T,)

    # -- 3. Beat indicator (1 dim) — binary: 1 if this frame is a beat --
    
    tempo, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=hop_length
    )
    
    beat_indicator = np.zeros(n_frames)
    valid_beats = beat_frames[beat_frames < n_frames]
    beat_indicator[valid_beats] = 1.0 # (T,)

    # -- 4. RMS energy (1 dim) --
    
    rms = librosa.feature.rms(
        y=y, hop_length=hop_length
    )[0, :n_frames] # (T,)

    # -- 5. chroma (12 dims) --
    
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, hop_length=hop_length
    )[:, :n_frames].T # (T, 12)

    # -- 6. spectral centroid (1 dim) --
    
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=hop_length
    )[0, :n_frames] # (T,)

    # -- 7. tempo (global, repeated for all frames) (1 dim) --
    
    if isinstance(tempo, np.ndarray):
        tempo_val = float(tempo[0])
    else:
        tempo_val = float(tempo)
        
    tempo_arr = np.full(n_frames, tempo_val / 200.0) # normalize to ~[0, 1] range

    # -- Concatenate all features --
    
    features = np.concatenate([
        mel_db,                           # (T, 128)
        onset[:, None],                   # (T, 1)
        beat_indicator[:, None],          # (T, 1)
        rms[:, None],                     # (T, 1)
        chroma,                           # (T, 12)
        centroid[:, None] / sr * 2,       # (T, 1) normalized
        tempo_arr[:, None],               # (T, 1)
    ], axis=1) # (T, 145)

    return features.astype(np.float32)

def extract_audio_features_from_array(y, sr=22050, target_fps=20):
    
    """
    
    Same as extract_audio_features but takes a raw audio array.
    Useful when audio is already loaded or generated.
    
    """
    
    hop_length = sr // target_fps
    n_frames = len(y) // hop_length

    if n_frames == 0:
        raise ValueError(f"Audio too short: {len(y)/sr:.2f}s")

    mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)[:, :n_frames].T

    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)[:n_frames]

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_indicator = np.zeros(n_frames)
    valid_beats = beat_frames[beat_frames < n_frames]
    beat_indicator[valid_beats] = 1.0

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0, :n_frames]

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)[:, :n_frames].T

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0, :n_frames]

    if isinstance(tempo, np.ndarray):
        tempo_val = float(tempo[0])
    else:
        tempo_val = float(tempo)
    tempo_arr = np.full(n_frames, tempo_val / 200.0)

    features = np.concatenate([
        mel_db, onset[:, None], beat_indicator[:, None],
        rms[:, None], chroma, centroid[:, None] / sr * 2, tempo_arr[:, None],
    ], axis=1)

    return features.astype(np.float32)

if __name__ == "__main__":
    
    """Quick test: extract features from a sample audio file."""
    
    import sys

    if len(sys.argv) < 2:
        
        print("Usage: python audio_features.py <audio_file>")
        print("\nGenerating test with synthetic audio...")

        # -- generate a 3-second synthetic beat --
        
        sr = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # -- simple kick drum at 120 BPM (2 beats/sec) --
        
        beat_times = np.arange(0, duration, 0.5)
        y = np.zeros_like(t)
        
        for bt in beat_times:
            mask = (t >= bt) & (t < bt + 0.05)
            y[mask] = np.sin(2 * np.pi * 60 * (t[mask] - bt)) * np.exp(-30 * (t[mask] - bt))

        features = extract_audio_features_from_array(y, sr=sr)
        
        print(f"Synthetic audio: {duration}s at 20fps")
        print(f"Feature shape: {features.shape}")
        print(f"Expected: ({int(duration * 20)}, 145)")
        print(f"\nFeature ranges:")
        print(f"  Mel (0:128):  [{features[:, :128].min():.2f}, {features[:, :128].max():.2f}]")
        print(f"  Onset (128):  [{features[:, 128].min():.2f}, {features[:, 128].max():.2f}]")
        print(f"  Beat (129):   [{features[:, 129].min():.0f}, {features[:, 129].max():.0f}]")
        print(f"  RMS (130):    [{features[:, 130].min():.4f}, {features[:, 130].max():.4f}]")
        print(f"  Chroma (131:143): [{features[:, 131:143].min():.2f}, {features[:, 131:143].max():.2f}]")
        print(f"  Centroid (143): [{features[:, 143].min():.4f}, {features[:, 143].max():.4f}]")
        print(f"  Tempo (144):  [{features[:, 144].min():.4f}, {features[:, 144].max():.4f}]")
    
    else:
        
        features = extract_audio_features(sys.argv[1])
        
        print(f"Feature shape: {features.shape}")
        print(f"Duration: {features.shape[0] / 20:.2f}s at 20fps")
        
####################################################################################[end]
####################################################################################[end]
