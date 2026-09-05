"""
ADAPTIVE-DEFENCE ANC — Classical Baseline: Boll (1979) Spectral Subtraction
Smart India Hackathon (SIH) 2026 — Problem Statement ID: 26052
DRDO / Department of Defence Production / iDEX — Smart Vehicles

Reference:
    Boll, S. (1979). "Suppression of acoustic noise in speech using spectral subtraction".
    IEEE Transactions on Acoustics, Speech, and Signal Processing, 27(2), 113-120.
"""

from typing import Optional, Tuple
import numpy as np


class SpectralSubtraction:
    """
    Standard Boll (1979) magnitude spectral subtraction with over-subtraction
    factor (alpha) and spectral flooring (beta) to mitigate musical noise artifacts.
    """

    def __init__(
        self,
        frame_size: int = 512,
        hop_size: int = 256,
        sample_rate: int = 16000,
        alpha: float = 2.0,
        beta: float = 0.02,
    ) -> None:
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.window = np.hanning(frame_size).astype(np.float32)

    def estimate_noise_profile(
        self,
        audio: np.ndarray,
        initial_frames: int = 6,
    ) -> np.ndarray:
        """Estimates stationary noise magnitude spectrum from the leading non-speech frames."""
        stft = self._stft(audio)
        mag = np.abs(stft)
        n_lead = min(initial_frames, mag.shape[1])
        noise_mag = np.mean(mag[:, :n_lead], axis=1, keepdims=True)
        return noise_mag

    def process(
        self,
        noisy_audio: np.ndarray,
        noise_profile: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Applies magnitude spectral subtraction:
            |S_hat(w)| = max(|Y(w)| - alpha * |N(w)|, beta * |Y(w)|)
            S_hat(w) = |S_hat(w)| * exp(j * phase(Y(w)))
        """
        x = np.asarray(noisy_audio, dtype=np.float32)
        orig_len = len(x)

        stft = self._stft(x)
        mag = np.abs(stft)
        phase = np.angle(stft)

        if noise_profile is None:
            noise_profile = self.estimate_noise_profile(x)

        # Boll over-subtraction rule
        subtracted = mag - self.alpha * noise_profile
        floor = self.beta * mag
        clean_mag = np.maximum(subtracted, floor)

        # Reconstruct complex STFT with original phase
        clean_stft = clean_mag * np.exp(1j * phase)

        # Invert STFT
        enhanced = self._istft(clean_stft, orig_len)
        return enhanced.astype(np.float32)

    def _stft(self, x: np.ndarray) -> np.ndarray:
        """Computes STFT using internal framing and Hanning window."""
        n_frames = max(1, (len(x) - self.frame_size) // self.hop_size + 1)
        frames = np.zeros((self.frame_size, n_frames), dtype=np.float32)
        for i in range(n_frames):
            start = i * self.hop_size
            frames[:, i] = x[start : start + self.frame_size] * self.window
        return np.fft.rfft(frames, axis=0)

    def _istft(self, stft: np.ndarray, target_len: int) -> np.ndarray:
        """Inverts STFT via Overlap-Add (OLA)."""
        frames = np.fft.irfft(stft, axis=0)
        n_frames = frames.shape[1]
        out = np.zeros(target_len + self.frame_size, dtype=np.float32)
        norm = np.zeros(target_len + self.frame_size, dtype=np.float32)

        for i in range(n_frames):
            start = i * self.hop_size
            end = start + self.frame_size
            out[start:end] += frames[:, i] * self.window
            norm[start:end] += self.window**2

        # Safe normalization
        mask = norm > 1e-4
        out[mask] /= norm[mask]
        return out[:target_len]


if __name__ == "__main__":
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    speech = 0.5 * np.sin(2 * np.pi * 400 * t)
    noise = np.random.normal(0, 0.2, sr).astype(np.float32)
    noisy = speech + noise

    ss = SpectralSubtraction(frame_size=512, hop_size=256, sample_rate=sr)
    out = ss.process(noisy)

    in_snr = 10 * np.log10(np.mean(speech**2) / np.mean(noise**2))
    err_out = out - speech
    out_snr = 10 * np.log10(np.mean(speech**2) / (np.mean(err_out**2) + 1e-12))
    print(f"[+] Boll Spectral Subtraction: Input SNR = {in_snr:.2f} dB -> Output SNR = {out_snr:.2f} dB (Gain: +{out_snr - in_snr:.2f} dB)")
