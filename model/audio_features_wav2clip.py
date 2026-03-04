"""
Wav2CLIP (512-d) + 7-d librosa rhythmic features → 519-d per frame.
Blueprint: frame-level semantic (Wav2CLIP) + explicit beat/onset (librosa).

Optional dependency: pip install wav2clip (or clone lyrebird-wav2clip).
When Wav2CLIP is not available, extract_wav2clip_plus_librosa raises ImportError.
"""

import numpy as np
import librosa

# Optional: Wav2CLIP (lyrebird-wav2clip or wav2clip)
WAV2CLIP_AVAILABLE = False
_wav2clip_model = None

try:
    import torch
    # Try descriptinc/lyrebird-wav2clip API
    import wav2clip
    WAV2CLIP_AVAILABLE = True
    # Lyrebird calls librosa.util.frame(x, frame_length, hop_length) but librosa 0.11+ uses keyword-only args
    _librosa_frame_orig = librosa.util.frame
    def _frame_compat(x, *args, **kwargs):
        if len(args) >= 2:
            return _librosa_frame_orig(x, frame_length=args[0], hop_length=args[1], **kwargs)
        return _librosa_frame_orig(x, **kwargs)
    librosa.util.frame = _frame_compat
except ImportError:
    pass


def extract_librosa_rhythmic_7(audio_path, target_fps=20, duration=None, sr=22050):
    """
    Extract 7-d rhythmic features per frame (Blueprint table):
    onset strength (1), beat indicator (1), beat distance past/future (2),
    RMS (1), spectral centroid (1), tempo (1).
    Returns: (T, 7) float32.
    """
    y, _ = librosa.load(audio_path, sr=sr, duration=duration)
    hop_length = sr // target_fps
    n_frames = len(y) // hop_length
    if n_frames == 0:
        raise ValueError(f"Audio too short: {len(y)/sr:.2f}s")

    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)[:n_frames]
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_indicator = np.zeros(n_frames, dtype=np.float32)
    valid_beats = beat_frames[beat_frames < n_frames]
    beat_indicator[valid_beats] = 1.0

    # Beat distance: frames to nearest past and future beat (Beat-It style)
    beat_distance = np.zeros((n_frames, 2), dtype=np.float32)
    if len(valid_beats) > 0:
        for t in range(n_frames):
            past = valid_beats[valid_beats <= t]
            future = valid_beats[valid_beats > t]
            beat_distance[t, 0] = (t - past[-1]) / target_fps if len(past) else 0.5
            beat_distance[t, 1] = (future[0] - t) / target_fps if len(future) else 0.5

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0, :n_frames]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0, :n_frames]
    tempo_val = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
    tempo_arr = np.full(n_frames, tempo_val / 200.0, dtype=np.float32)

    return np.stack([
        onset,
        beat_indicator,
        beat_distance[:, 0],
        beat_distance[:, 1],
        rms,
        centroid / (sr * 0.5),
        tempo_arr,
    ], axis=1).astype(np.float32)


def _load_wav2clip_model(device="cpu", sr=16000, target_fps=20):
    """Load Wav2CLIP model (cached)."""
    global _wav2clip_model
    if _wav2clip_model is not None:
        return _wav2clip_model
    if not WAV2CLIP_AVAILABLE:
        raise ImportError("Wav2CLIP not available. Install: pip install wav2clip (or clone lyrebird-wav2clip)")
    import torch
    frame_length = int(sr * 0.5)
    hop_length = sr // target_fps
    _wav2clip_model = wav2clip.get_model(device=device, pretrained=True, frame_length=frame_length, hop_length=hop_length)
    return _wav2clip_model


def extract_wav2clip_512(audio_path, target_fps=20, duration=None, sr=16000, device="cpu"):
    """
    Extract Wav2CLIP 512-d per frame at target_fps.
    Lyrebird model uses frame_length=0.5s and hop_length for framing. Returns (T, 512) float32.
    """
    y, _ = librosa.load(audio_path, sr=sr, duration=duration)
    if len(y) == 0:
        raise ValueError(f"Audio empty or too short")
    frame_length = int(sr * 0.5)
    hop_length = sr // target_fps
    model = _load_wav2clip_model(device=device, sr=sr, target_fps=target_fps)
    # Pass full audio; model forwards with frame_length/hop_length and returns (1, T, 512)
    audio_batch = np.expand_dims(y.astype(np.float32), axis=0)
    import torch
    with torch.no_grad():
        out = model(torch.from_numpy(audio_batch).to(device))
    if isinstance(out, torch.Tensor):
        out = out.cpu().numpy()
    if out.ndim == 3:
        out = out[0]  # (512, T) or (T, 512)
    if out.shape[0] == 512:
        out = out.T  # (T, 512)
    return out.astype(np.float32)


def extract_wav2clip_plus_librosa(audio_path, target_fps=20, duration=None, device="cpu"):
    """
    Extract 519-d features per frame: 512 (Wav2CLIP) + 7 (librosa rhythmic).
    Returns (T, 519) float32. T is determined by the shorter of the two.
    """
    if not WAV2CLIP_AVAILABLE:
        raise ImportError("Wav2CLIP not available. Install wav2clip or use librosa-only 145-d features.")
    wav2clip_sr = 16000
    w512 = extract_wav2clip_512(audio_path, target_fps=target_fps, duration=duration, sr=wav2clip_sr, device=device)
    lib7 = extract_librosa_rhythmic_7(audio_path, target_fps=target_fps, duration=duration, sr=22050)
    T = min(w512.shape[0], lib7.shape[0])
    w512 = w512[:T]
    lib7 = lib7[:T]
    return np.concatenate([w512, lib7], axis=1).astype(np.float32)


def extract_wav2clip_plus_librosa_from_array(y, sr=22050, target_fps=20, wav2clip_device="cpu"):
    """Same as above but from raw audio array (e.g. for preprocess pipeline)."""
    if not WAV2CLIP_AVAILABLE:
        raise ImportError("Wav2CLIP not available.")
    import tempfile
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        sf.write(f.name, y, sr)
        return extract_wav2clip_plus_librosa(f.name, target_fps=target_fps, duration=len(y)/sr, device=wav2clip_device)
