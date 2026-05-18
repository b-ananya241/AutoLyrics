import numpy as np
import librosa

def preprocess_audio(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Clean singing audio before feeding to Whisper:
    1. Harmonic-percussive separation — keeps vocals, reduces drums/bass
    2. High-pass filter — removes low-frequency instrumentation
    3. Normalize volume
    """
    # Step 1 — HPSS: keep harmonic (vocal) component only
    harmonic, _ = librosa.effects.hpss(audio, margin=3.0)

    # Step 2 — high-pass filter at 150Hz to cut bass/instruments
    from scipy.signal import butter, filtfilt
    nyq = sr / 2
    cutoff = 150 / nyq
    b, a = butter(4, cutoff, btype='high')
    filtered = filtfilt(b, a, harmonic)

    # Step 3 — normalize
    max_val = np.max(np.abs(filtered))
    if max_val > 0:
        filtered = filtered / max_val * 0.95

    return filtered.astype(np.float32)