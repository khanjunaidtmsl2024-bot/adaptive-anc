"""
ADAPTIVE-DEFENCE ANC — Classical Baseline: Wiener Filtering (Decision-Directed)
Smart India Hackathon (SIH) 2026 — Problem Statement ID: 26052
DRDO / Department of Defence Production / iDEX — Smart Vehicles

Reference:
    Ephraim, Y., & Malah, D. (1984). "Speech enhancement using a minimum mean-square error
    short-time spectral amplitude estimator". IEEE Transactions on ASSP, 32(6), 1109-1121.
"""

from typing import Optional
import numpy as np


class WienerFilter:
    """
    Parametric Wiener Filter in the STFT frequency domain using Decision-Directed
    a priori SNR tracking to minimize distortion and suppression latency.
    """

    def __init__(
        self,
        frame_size: int = 512,
        hop_size: int = 256,
        sample_rate: int = 16000,
        alpha_dd: float = 0.98,
        gain_floor: float = 0.05,
    ) -> None:
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.alpha_dd = float(alpha_dd)
        self.gain_floor = float(gain_floor)
        self.window = np.hanning(frame_size).astype(np.float32)

    def process(
        self,
        noisy_audio: np.ndarray,
        noise_profile: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Enhances noisy speech by filtering each frequency bin with Wiener gain:
            H(w, t) = xi(w, t) / (xi(w, t) + 1)
        where xi is the a priori SNR estimated via Decision-Directed method:
            xi(w, t) = alpha * (|S_hat(w, t-1)|^2 / lambda_d) + (1-alpha) * max(gamma(w,t) - 1, 0)
        """
        x = np.asarray(noisy_audio, dtype=np.float32)
        orig_len = len(x)

        # STFT
        n_frames = max(1, (len(x) - self.frame_size) // self.hop_size + 1)
        frames = np.zeros((self.frame_size, n_frames), dtype=np.float32)
        for i in range(n_frames):
            start = i * self.hop_size
            frames[:, i] = x[start : start + self.frame_size] * self.window

        stft = np.fft.rfft(frames, axis=0)
        n_bins = stft.shape[0]
        power_y = np.abs(stft) ** 2

        # Noise power estimation from initial frames if not provided
        if noise_profile is None:
            n_lead = min(6, n_frames)
            noise_power = np.mean(power_y[:, :n_lead], axis=1, keepdims=True)
        else:
            noise_power = np.asarray(noise_profile, dtype=np.float32)**2
            if noise_power.ndim == 1:
                noise_power = noise_power[:, np.newaxis]

        # Prior power buffer for Decision-Directed recursive update
        prev_s_hat_power = np.zeros(n_bins, dtype=np.float32)
        enhanced_stft = np.zeros_like(stft)

        for t_idx in range(n_frames):
            y_frame = stft[:, t_idx]
            y_pow = np.abs(y_frame) ** 2
            n_pow = np.maximum(noise_power[:, 0], 1e-12)

            # A posteriori SNR: gamma = |Y|^2 / lambda_d
            gamma = y_pow / n_pow

            # A priori SNR: xi via Decision-Directed rule
            if t_idx == 0:
                xi = np.maximum(gamma - 1.0, 0.0)
            else:
                xi = self.alpha_dd * (prev_s_hat_power / n_pow) + (1.0 - self.alpha_dd) * np.maximum(gamma - 1.0, 0.0)
            xi = np.maximum(xi, 1e-4)

            # Wiener Gain: H = xi / (xi + 1)
            hw = xi / (xi + 1.0)
            hw = np.maximum(hw, self.gain_floor)

            # Filter STFT frame
            s_hat_frame = hw * y_frame
            enhanced_stft[:, t_idx] = s_hat_frame
            prev_s_hat_power = np.abs(s_hat_frame) ** 2

        # iSTFT synthesis
        time_frames = np.fft.irfft(enhanced_stft, axis=0)
        out = np.zeros(orig_len + self.frame_size, dtype=np.float32)
        norm = np.zeros(orig_len + self.frame_size, dtype=np.float32)

        for i in range(n_frames):
            start = i * self.hop_size
            end = start + self.frame_size
            out[start:end] += time_frames[:, i] * self.window
            norm[start:end] += self.window**2

        mask = norm > 1e-4
        out[mask] /= norm[mask]
        return out[:orig_len].astype(np.float32)


if __name__ == "__main__":
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    speech = 0.5 * np.sin(2 * np.pi * 400 * t)
    noise = np.random.normal(0, 0.2, sr).astype(np.float32)
    noisy = speech + noise

    wf = WienerFilter(frame_size=512, hop_size=256, sample_rate=sr)
    out = wf.process(noisy)

    in_snr = 10 * np.log10(np.mean(speech**2) / np.mean(noise**2))
    err_out = out - speech
    out_snr = 10 * np.log10(np.mean(speech**2) / (np.mean(err_out**2) + 1e-12))
    print(f"[+] Wiener Filter (Decision-Directed): Input SNR = {in_snr:.2f} dB -> Output SNR = {out_snr:.2f} dB (Gain: +{out_snr - in_snr:.2f} dB)")
