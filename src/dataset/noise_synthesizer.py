"""
Tactical Defence Noise Synthesizer.
PS 26052 — Adaptive Defence ANC.

Generates realistic tactical acoustic disturbances:
1. Tank / Armoured Vehicle: Heavy low-frequency diesel harmonics + track squeal (50-200 Hz).
2. Helicopter Rotor: Blade passage frequency (BPF) fundamental (15-30 Hz) + high-order rotor chop.
3. Gunfire / Artillery Shock: High-energy impulsive transients with <2 ms rise and 50 ms decay.
4. Tactical Siren / Radio Whine: Frequency-modulated non-stationary tone.
"""

from typing import Optional
import numpy as np


class DefenceNoiseSynthesizer:
    """Acoustic synthesizer for standardized defence noise profiles."""

    def __init__(self, sample_rate: int = 16000):
        self.sr = int(sample_rate)

    def generate_tank_noise(self, duration: float = 5.0, seed: Optional[int] = 42) -> np.ndarray:
        """Synthesize heavy armoured vehicle diesel engine rumble and track vibration."""
        if seed is not None:
            np.random.seed(seed)

        n_samples = int(duration * self.sr)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        # Baseline broadband track rumble
        noise = np.random.normal(0, 0.3, n_samples)

        # Diesel engine fundamentals and harmonics (65 Hz, 130 Hz, 195 Hz, 260 Hz)
        engine = (
            0.6 * np.sin(2 * np.pi * 65.0 * t) +
            0.4 * np.sin(2 * np.pi * 130.0 * t + 0.5) +
            0.3 * np.sin(2 * np.pi * 195.0 * t + 1.2) +
            0.2 * np.sin(2 * np.pi * 260.0 * t + 2.0)
        )

        signal = noise + engine
        return (signal / np.max(np.abs(signal))).astype(np.float32)

    def generate_helicopter_noise(self, duration: float = 5.0, seed: Optional[int] = 42) -> np.ndarray:
        """Synthesize helicopter rotor blade-passage frequency (BPF) and tail rotor chop."""
        if seed is not None:
            np.random.seed(seed)

        n_samples = int(duration * self.sr)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        # Main rotor BPF: 22.5 Hz with sharp pulse train harmonics
        bpf = 22.5
        rotor = np.zeros(n_samples)
        for harmonic in range(1, 12):
            amp = 1.0 / (harmonic ** 0.8)
            rotor += amp * np.sin(2 * np.pi * (bpf * harmonic) * t)

        # Tail rotor: 110 Hz
        tail = 0.4 * np.sin(2 * np.pi * 110.0 * t)

        # Atmospheric wind turbulence
        wind = np.random.normal(0, 0.2, n_samples)

        signal = rotor + tail + wind
        return (signal / np.max(np.abs(signal))).astype(np.float32)

    def generate_gunfire_transient(self, duration: float = 2.0, num_shots: int = 3, seed: Optional[int] = 42) -> np.ndarray:
        """Synthesize sharp ballistic muzzle blast and shockwave transients."""
        if seed is not None:
            np.random.seed(seed)

        n_samples = int(duration * self.sr)
        signal = np.zeros(n_samples, dtype=np.float32)

        # Space shots evenly
        shot_interval = n_samples // (num_shots + 1)

        for i in range(1, num_shots + 1):
            idx = i * shot_interval
            # Friedlander blast wave waveform: P(t) = P0 * (1 - t/T) * exp(-b*t/T)
            blast_len = int(0.08 * self.sr)  # 80 ms blast + reverberation
            if idx + blast_len < n_samples:
                t_blast = np.linspace(0, 0.08, blast_len, endpoint=False)
                shock = (1.0 - t_blast / 0.08) * np.exp(-40.0 * t_blast)
                # High-frequency crack noise
                crack = np.random.normal(0, 1.0, blast_len) * np.exp(-80.0 * t_blast)
                signal[idx:idx + blast_len] += (shock * 2.0 + crack).astype(np.float32)

        max_val = np.max(np.abs(signal)) + 1e-12
        return (signal / max_val).astype(np.float32)
