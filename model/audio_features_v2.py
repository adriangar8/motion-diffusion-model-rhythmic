####################################################################################[start]
####################################################################################[start]

"""

Audio feature extraction v2.

Key changes from v1:
  - Mel spectrogram reduced from 128 to 32 bands (reduces dominance from 88% to 53%)
  - Added beat distance features: frames to nearest past/future beat (from Beat-It, ECCV 2024)
  - Added onset envelope (smoothed onset strength)
  - All features normalized to roughly [0, 1] range
  - Beat indicator widened: Gaussian around beats (sigma=1 frame) instead of binary spike

Feature breakdown (51 dims):
  - Mel spectrogram (32 bands, normalized):  32 dims
  - Onset strength (normalized):              1 dim
  - Onset envelope (smoothed):                1 dim
  - Beat indicator (soft, Gaussian):          1 dim
  - Beat distance past (normalized):          1 dim
  - Beat distance future (normalized):        1 dim
  - RMS energy (normalized):                  1 dim
  - Chroma:                                  12 dims
  - Spectral centroid (normalized):           1 dim
  - Tempo (normalized):                       1 dim
  Total:                                     52 dims

"""

import numpy as np
import librosa
from scipy.ndimage import gaussian_filter1d

def extract_audio_features_v2(audio_path, target_fps=20, duration=None):

    """

    Extract improved frame-level audio features from an audio file.

    Args:
        audio_path: Path to audio file (.wav, .mp3, etc.)
        target_fps: Frame rate to match motion data (default: 20 fps)
        duration: If set, only load this many seconds of audio

    Returns:
        features: np.ndarray of shape (T, 52)

    """

    sr = 22050
    y, _ = librosa.load(audio_path, sr=sr, duration=duration)

    hop_length = sr // target_fps
    n_frames = len(y) // hop_length

    if n_frames == 0:
        raise ValueError(f"Audio too short: {len(y)/sr:.2f}s")

    # -- 1. mel spectrogram (32 bands instead of 128) --
    
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, hop_length=hop_length, n_mels=32
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)[:, :n_frames].T  # (T, 32)
    
    # -- normalize from [-80, 0] to [0, 1] --
    
    mel_norm = (mel_db + 80.0) / 80.0
    mel_norm = np.clip(mel_norm, 0.0, 1.0)

    # -- 2. onset strength (normalized) --
    
    onset = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop_length
    )[:n_frames]
    onset_max = onset.max() if onset.max() > 0 else 1.0
    onset_norm = onset / onset_max  # (T,)

    # -- 3. onset envelope (smoothed version for longer-range context) --
    
    onset_env = gaussian_filter1d(onset_norm, sigma=3.0)  # ~150ms smoothing
    onset_env_max = onset_env.max() if onset_env.max() > 0 else 1.0
    onset_env = onset_env / onset_env_max  # (T,)

    # -- 4. beat detection --
    
    tempo, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=hop_length
    )
    valid_beats = beat_frames[beat_frames < n_frames]

    # -- 4a. soft beat indicator (Gaussian around each beat, sigma=1 frame) --
    
    beat_soft = np.zeros(n_frames)
    for b in valid_beats:
        beat_soft[b] = 1.0
    beat_soft = gaussian_filter1d(beat_soft, sigma=1.0)
    beat_soft_max = beat_soft.max() if beat_soft.max() > 0 else 1.0
    beat_soft = beat_soft / beat_soft_max  # (T,)

    # -- 4b. beat distance: frames to nearest past and future beat (normalized) --
    
    beat_dist_past = np.full(n_frames, n_frames, dtype=np.float32)
    beat_dist_future = np.full(n_frames, n_frames, dtype=np.float32)

    if len(valid_beats) > 0:
    
        for i in range(n_frames):
    
            # -- distance to nearest past beat --
            
            past_beats = valid_beats[valid_beats <= i]
            if len(past_beats) > 0:
                beat_dist_past[i] = i - past_beats[-1]

            # -- distance to nearest future beat --
            
            future_beats = valid_beats[valid_beats >= i]
            if len(future_beats) > 0:
                beat_dist_future[i] = future_beats[0] - i

    # -- normalize by average beat period --
    
    if len(valid_beats) > 1:
        avg_beat_period = np.mean(np.diff(valid_beats))
    else:
        avg_beat_period = n_frames
    beat_dist_past_norm = np.clip(beat_dist_past / avg_beat_period, 0.0, 1.0)
    beat_dist_future_norm = np.clip(beat_dist_future / avg_beat_period, 0.0, 1.0)

    # -- 5. RMS energy (normalized) --
    
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0, :n_frames]
    rms_max = rms.max() if rms.max() > 0 else 1.0
    rms_norm = rms / rms_max  # (T,)

    # -- 6. chroma (already in [0, 1]) --
    
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, hop_length=hop_length
    )[:, :n_frames].T # (T, 12)

    # -- 7. spectral centroid (normalized) --
    
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=hop_length
    )[0, :n_frames]
    centroid_norm = centroid / (sr / 2) # normalize by Nyquist  (T,)

    # -- 8. tempo (normalized) --
    
    if isinstance(tempo, np.ndarray):
        tempo_val = float(tempo[0])
    else:
        tempo_val = float(tempo)
    tempo_norm = np.full(n_frames, tempo_val / 200.0)  # [0, 1] for typical tempos

    # -- concatenate --
    
    features = np.concatenate([
        mel_norm,                          # (T, 32)
        onset_norm[:, None],               # (T, 1)
        onset_env[:, None],                # (T, 1)
        beat_soft[:, None],                # (T, 1)
        beat_dist_past_norm[:, None],      # (T, 1)
        beat_dist_future_norm[:, None],    # (T, 1)
        rms_norm[:, None],                 # (T, 1)
        chroma,                            # (T, 12)
        centroid_norm[:, None],            # (T, 1)
        tempo_norm[:, None],               # (T, 1)
    ], axis=1) # (T, 52)

    return features.astype(np.float32)


def extract_audio_features_v2_from_array(y, sr=22050, target_fps=20):
    
    """Same as extract_audio_features_v2 but from raw audio array."""
    
    import tempfile, soundfile as sf
    
    # -- write to temp file and use main function (simpler, avoids code duplication) --
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, y, sr)
        return extract_audio_features_v2(f.name, target_fps=target_fps)

# -- keep backward compatibility: original 145-dim extraction --

from model.audio_features import extract_audio_features, extract_audio_features_from_array

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print("Usage: python audio_features_v2.py <audio_file>")
        print("\nGenerating test with synthetic audio...")

        sr = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration))
        beat_times = np.arange(0, duration, 0.5)
        y = np.zeros_like(t)

        for bt in beat_times:

            mask = (t >= bt) & (t < bt + 0.05)
            y[mask] = np.sin(2 * np.pi * 60 * (t[mask] - bt)) * np.exp(-30 * (t[mask] - bt))

        features = extract_audio_features_v2_from_array(y, sr=sr)

        print(f"Feature shape: {features.shape}")
        print(f"Expected: ({int(duration * 20)}, 52)")
        print(f"\nFeature ranges:")
        print(f"  Mel (0:32):       [{features[:, :32].min():.3f}, {features[:, :32].max():.3f}]")
        print(f"  Onset (32):       [{features[:, 32].min():.3f}, {features[:, 32].max():.3f}]")
        print(f"  Onset env (33):   [{features[:, 33].min():.3f}, {features[:, 33].max():.3f}]")
        print(f"  Beat soft (34):   [{features[:, 34].min():.3f}, {features[:, 34].max():.3f}]")
        print(f"  Beat dist past (35):   [{features[:, 35].min():.3f}, {features[:, 35].max():.3f}]")
        print(f"  Beat dist future (36): [{features[:, 36].min():.3f}, {features[:, 36].max():.3f}]")
        print(f"  RMS (37):         [{features[:, 37].min():.3f}, {features[:, 37].max():.3f}]")
        print(f"  Chroma (38:50):   [{features[:, 38:50].min():.3f}, {features[:, 38:50].max():.3f}]")
        print(f"  Centroid (50):    [{features[:, 50].min():.3f}, {features[:, 50].max():.3f}]")
        print(f"  Tempo (51):       [{features[:, 51].min():.3f}, {features[:, 51].max():.3f}]")
    else:
        features = extract_audio_features_v2(sys.argv[1])
        print(f"Feature shape: {features.shape}")
        print(f"Duration: {features.shape[0] / 20:.2f}s at 20fps")

####################################################################################[end]
####################################################################################[end]
