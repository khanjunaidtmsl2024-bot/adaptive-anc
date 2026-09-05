"""
Audio Provenance & Cryptographic Verification Module.
PS 26052 — Adaptive Defence ANC.

Extracts cryptographic SHA-256 hashes, acoustic physical descriptors, 
and provenance metadata (source, category, synthetic vs real).
"""

import hashlib
from pathlib import Path
from typing import Dict, Any, Union
import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None


def calculate_sha256(filepath: Union[str, Path]) -> str:
    """Computes SHA-256 checksum of an audio or data file."""
    p = Path(filepath)
    if not p.exists():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def extract_signal_descriptors(signal: np.ndarray, sample_rate: int = 16000) -> Dict[str, float]:
    """Computes acoustic physical descriptors for quality and regime analysis."""
    if len(signal) == 0:
        return {"rms": 0.0, "peak": 0.0, "crest_factor": 0.0, "kurtosis": 0.0, "spectral_centroid": 0.0}
    
    rms = float(np.sqrt(np.mean(signal ** 2) + 1e-12))
    peak = float(np.max(np.abs(signal)))
    crest_factor = float(peak / (rms + 1e-12))
    
    # Fourth standardized moment (kurtosis) for impulsiveness
    mu = float(np.mean(signal))
    std = float(np.std(signal) + 1e-12)
    kurtosis = float(np.mean(((signal - mu) / std) ** 4))
    
    # Spectral Centroid via FFT
    n = len(signal)
    fft_mag = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    sum_mag = np.sum(fft_mag) + 1e-12
    spectral_centroid = float(np.sum(freqs * fft_mag) / sum_mag)
    
    return {
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "crest_factor": round(crest_factor, 3),
        "kurtosis": round(kurtosis, 3),
        "spectral_centroid": round(spectral_centroid, 1),
    }
