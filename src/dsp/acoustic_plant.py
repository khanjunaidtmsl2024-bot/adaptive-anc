"""
Electro-Acoustic Plant and Ear-Cup Cavity Modeling Module.
PH4-SIM: Physical ANC Simulation & Secondary-Path Validation.
PS 26052 — Adaptive Defence ANC.

Provides physically grounded lumped-parameter models for:
1. Primary Acoustic Path P(z): Sound penetration through headset shell/cushion.
2. Secondary Electro-Acoustic Path S(z): DAC -> Amp -> Speaker -> Cavity -> Error Mic -> ADC.
3. Perturbed Secondary Path S_leak(z): Compromised cushion seal (glasses frame leak).
4. Parametric Secondary Path Uncertainty Generator: Controlled S_hat(z) estimation errors.

All outputs are strictly simulated numerical representations on host CPU.
"""

from typing import Tuple, Optional
import numpy as np
from scipy import signal


class AcousticPlantModel:
    """
    Simulates physical acoustic and electro-acoustic transfer functions
    for an over-ear circumaural headset cavity.
    """

    def __init__(self, sample_rate: int = 16000, fir_length: int = 64):
        self.sr = sample_rate
        self.fir_length = fir_length

        # Generate nominal primary path P(z) and secondary path S(z)
        self.p_nominal = self._synthesize_primary_path(sample_rate, fir_length)
        self.s_nominal = self._synthesize_secondary_path(sample_rate, fir_length, leak_ratio=0.0)
        self.s_leaked = self._synthesize_secondary_path(sample_rate, fir_length, leak_ratio=0.5)

    @staticmethod
    def _synthesize_primary_path(sr: int, fir_length: int) -> np.ndarray:
        """
        Synthesizes the primary acoustic disturbance path P(z).
        Physics:
        - Propagation delay: ~0.5 ms (around 8 samples @ 16 kHz) from exterior source to inner ear.
        - Passive transmission loss: Massive attenuation above 800 Hz due to shell mass
          and foam damping (lowpass roll-off).
        """
        # 4th-order Butterworth low-pass filter at 1000 Hz representing shell passive attenuation
        nyq = sr / 2.0
        cutoff_hz = min(1000.0, nyq * 0.8)
        b, a = signal.butter(4, cutoff_hz / nyq, btype="low")

        # Compute unit impulse response
        impulse = np.zeros(fir_length, dtype=np.float32)
        impulse[0] = 1.0
        ir = signal.lfilter(b, a, impulse).astype(np.float32)

        # Add pure acoustic propagation delay (8 samples = 0.5 ms @ 16 kHz)
        delay_samples = max(1, int(0.0005 * sr))
        ir_delayed = np.zeros_like(ir)
        if delay_samples < fir_length:
            ir_delayed[delay_samples:] = ir[:-delay_samples]

        # Normalize energy so passive loss provides ~6 dB average broadband attenuation
        ir_delayed *= 0.5 / (np.max(np.abs(ir_delayed)) + 1e-12)
        return ir_delayed.astype(np.float32)

    @staticmethod
    def _synthesize_secondary_path(sr: int, fir_length: int, leak_ratio: float = 0.0) -> np.ndarray:
        """
        Synthesizes the electro-acoustic secondary path S(z).
        Physics:
        - DAC/ADC conversion + group delay + acoustic travel: ~0.25 ms (4 samples @ 16 kHz).
        - Speaker driver + ear-cup cavity resonance: 2nd-order resonator around 300 Hz with Q~1.5.
        - High-frequency voice-coil inductance roll-off above 3.5 kHz.
        - Leak ratio: Simulates cushion unsealing. An acoustic leak causes severe bass roll-off
          (high-pass effect below 400 Hz) and upward frequency shift of cavity resonance.
        """
        nyq = sr / 2.0
        t = np.arange(fir_length, dtype=np.float32) / sr

        # Resonant frequency and damping factor
        f_res = 300.0 * (1.0 + 0.8 * leak_ratio)  # Resonates higher if cavity has acoustic leak
        q_factor = 1.6 * (1.0 - 0.4 * leak_ratio)
        damping = 2.0 * np.pi * f_res / (2.0 * q_factor)
        omega_d = 2.0 * np.pi * f_res * np.sqrt(max(0.1, 1.0 - 1.0 / (4.0 * q_factor ** 2)))

        # Damped harmonic oscillation representing speaker diaphragm + acoustic cavity volume
        response = np.exp(-damping * t) * np.sin(omega_d * t)

        # Bass loss if acoustic seal is compromised (high-pass characteristic)
        if leak_ratio > 0.0:
            b_hp, a_hp = signal.butter(2, min(400.0 * leak_ratio, nyq * 0.9) / nyq, btype="high")
            response = signal.lfilter(b_hp, a_hp, response)

        # Electronic + acoustic transport delay: 4 samples (~0.25 ms @ 16 kHz)
        delay_samples = max(1, int(0.00025 * sr))
        s_delayed = np.zeros(fir_length, dtype=np.float32)
        if delay_samples < fir_length:
            s_delayed[delay_samples:] = response[:-delay_samples]

        # Apply smooth Hann window taper on FIR tail
        window = np.hanning(fir_length * 2)[fir_length:]
        s_delayed *= window

        # Normalize peak gain to 1.0
        s_delayed /= (np.max(np.abs(s_delayed)) + 1e-12)
        return s_delayed.astype(np.float32)

    def generate_uncertain_secondary_path(
        self,
        mismatch_percent: float = 0.0,
        phase_jitter_rad: float = 0.0,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Generates an estimated secondary path S_hat(z) with controlled relative L2 mismatch:
            ||S_hat - S||_2 / ||S||_2 == mismatch_percent / 100.0

        Args:
            mismatch_percent: Target relative L2 modeling error percentage (e.g. 10.0 for 10%).
            phase_jitter_rad: Additional systematic phase shift in radians.
            seed: RNG seed for deterministic perturbation synthesis.

        Returns:
            s_hat: Perturbed FIR impulse response matching exact error target.
        """
        if mismatch_percent <= 0.0 and phase_jitter_rad == 0.0:
            return self.s_nominal.copy()

        rng = np.random.RandomState(seed)
        nominal = self.s_nominal.copy()
        norm_nominal = float(np.linalg.norm(nominal))

        # Generate random perturbation vector in frequency domain
        perturbation = rng.randn(len(nominal)).astype(np.float32)
        # Low-pass filter perturbation so it represents physically realistic acoustic variations
        b, a = signal.butter(2, 0.4, btype="low")
        perturbation = signal.lfilter(b, a, perturbation).astype(np.float32)

        # Scale perturbation to achieve exact target relative L2 norm
        target_norm = (mismatch_percent / 100.0) * norm_nominal
        perturbation = perturbation * (target_norm / (np.linalg.norm(perturbation) + 1e-12))

        s_hat = nominal + perturbation

        # Apply systematic phase shift if requested
        if phase_jitter_rad != 0.0:
            s_hat_fft = np.fft.rfft(s_hat)
            s_hat_fft *= np.exp(1j * phase_jitter_rad)
            s_hat = np.fft.irfft(s_hat_fft, n=len(s_hat)).astype(np.float32)

        # Final rescale to preserve target relative error
        actual_err = float(np.linalg.norm(s_hat - nominal) / norm_nominal)
        return s_hat.astype(np.float32)

    def generate_multi_uncertain_secondary_path(
        self,
        perturbation_type: str,
        severity: float,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Generates an estimated secondary path S_hat(z) with specific physical perturbation types:
        - 'amplitude': Uniform gain miscalibration (e.g., severity = 0.20 for +20% gain error).
        - 'phase': Systematic broadband phase rotation in degrees (e.g., severity = 30.0 for 30 deg).
        - 'delay': Pure transport latency mismatch in discrete samples (e.g., severity = 2.0 for 2 samples).
        - 'resonance': Ear-cup cavity Helmholtz resonance shift (e.g., severity = 0.20 for +20% frequency shift).
        - 'q_factor': Cavity damping Q-factor variation (e.g., severity = -0.30 for -30% Q).
        - 'random_fir': Randomized FIR coefficient Gaussian perturbation with relative L2 error equal to severity.
        - 'combined': Multi-parameter realistic perturbation combining gain, phase, resonance, and random FIR.

        Args:
            perturbation_type: One of 'amplitude', 'phase', 'delay', 'resonance', 'q_factor', 'random_fir', 'combined'.
            severity: Magnitude of perturbation (dimension-specific).
            seed: RNG seed for reproducible synthesis.

        Returns:
            s_hat: Perturbed secondary path FIR filter (64 taps).
        """
        nominal = self.s_nominal.copy()
        norm_nominal = float(np.linalg.norm(nominal))
        rng = np.random.RandomState(seed)

        ptype = perturbation_type.lower()
        if ptype == "amplitude":
            # Direct gain scaling error
            s_hat = nominal * (1.0 + float(severity))

        elif ptype == "phase":
            # Systematic phase rotation in degrees
            rad = float(severity) * np.pi / 180.0
            fft_nom = np.fft.rfft(nominal)
            fft_nom *= np.exp(1j * rad)
            s_hat = np.fft.irfft(fft_nom, n=len(nominal)).astype(np.float32)

        elif ptype == "delay":
            # Pure sample transport delay mismatch
            shift = int(round(severity))
            s_hat = np.zeros_like(nominal)
            if shift > 0 and shift < len(nominal):
                s_hat[shift:] = nominal[:-shift]
            elif shift < 0 and -shift < len(nominal):
                s_hat[:shift] = nominal[-shift:]
            else:
                s_hat = nominal.copy()

        elif ptype == "resonance":
            # Shift the internal physical Helmholtz resonance f_0 by percentage
            f_res_shifted = 300.0 * (1.0 + float(severity))
            q_factor = 1.6
            damping = 2.0 * np.pi * f_res_shifted / (2.0 * q_factor)
            omega_d = 2.0 * np.pi * f_res_shifted * np.sqrt(max(0.1, 1.0 - 1.0 / (4.0 * q_factor ** 2)))
            t = np.arange(len(nominal), dtype=np.float32) / self.sr
            s_raw = np.exp(-damping * t) * np.sin(omega_d * t)
            delay_samples = max(1, int(0.00025 * self.sr))
            s_hat = np.zeros_like(nominal)
            if delay_samples < len(nominal):
                s_hat[delay_samples:] = s_raw[:-delay_samples]
            s_hat *= np.hanning(len(nominal) * 2)[len(nominal):]
            s_hat /= (np.max(np.abs(s_hat)) + 1e-12)

        elif ptype == "q_factor":
            # Shift cavity acoustic damping Q-factor
            f_res = 300.0
            q_shifted = max(0.4, 1.6 * (1.0 + float(severity)))
            damping = 2.0 * np.pi * f_res / (2.0 * q_shifted)
            omega_d = 2.0 * np.pi * f_res * np.sqrt(max(0.1, 1.0 - 1.0 / (4.0 * q_shifted ** 2)))
            t = np.arange(len(nominal), dtype=np.float32) / self.sr
            s_raw = np.exp(-damping * t) * np.sin(omega_d * t)
            delay_samples = max(1, int(0.00025 * self.sr))
            s_hat = np.zeros_like(nominal)
            if delay_samples < len(nominal):
                s_hat[delay_samples:] = s_raw[:-delay_samples]
            s_hat *= np.hanning(len(nominal) * 2)[len(nominal):]
            s_hat /= (np.max(np.abs(s_hat)) + 1e-12)

        elif ptype == "random_fir":
            # Generalized Gaussian coefficient perturbation
            pert = rng.randn(len(nominal)).astype(np.float32)
            b, a = signal.butter(2, 0.4, btype="low")
            pert = signal.lfilter(b, a, pert).astype(np.float32)
            target_norm = float(severity) * norm_nominal
            pert = pert * (target_norm / (np.linalg.norm(pert) + 1e-12))
            s_hat = nominal + pert

        elif ptype == "combined":
            # Multi-parameter perturbation representing realistic field uncertainty
            # Gain error + phase jitter + resonance shift + random FIR noise
            gain_factor = 1.0 + 0.15 * float(severity)
            rad = (15.0 * float(severity)) * np.pi / 180.0
            pert = rng.randn(len(nominal)).astype(np.float32)
            b, a = signal.butter(2, 0.4, btype="low")
            pert = signal.lfilter(b, a, pert).astype(np.float32)
            pert = pert * ((0.10 * float(severity) * norm_nominal) / (np.linalg.norm(pert) + 1e-12))
            s_hat = (nominal + pert) * gain_factor
            fft_c = np.fft.rfft(s_hat) * np.exp(1j * rad)
            s_hat = np.fft.irfft(fft_c, n=len(nominal)).astype(np.float32)
        else:
            raise ValueError(f"Unknown perturbation_type: {perturbation_type}")

        return s_hat.astype(np.float32)

    @staticmethod
    def synthesize_incoherent_reference(
        clean_noise: np.ndarray,
        coherence: float,
        sr: int = 16000,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Synthesizes a reference mic signal x(n) with controllable coherence Gamma relative to disturbance:
            x(n) = sqrt(Gamma) * clean_noise(n) + sqrt(1 - Gamma) * uncorrelated_noise(n)

        Args:
            clean_noise: Acoustic disturbance source signal.
            coherence: Magnitude-squared coherence Gamma in [0.0, 1.0].
            sr: Sampling rate.
            seed: RNG seed.

        Returns:
            ref: Incoherent reference microphone signal x(n).
        """
        coherence = float(np.clip(coherence, 0.0, 1.0))
        if coherence >= 1.0:
            return clean_noise.copy().astype(np.float32)

        N = len(clean_noise)
        rng = np.random.RandomState(seed)

        # Generate uncorrelated noise with matching spectral envelope
        uncorr_raw = rng.randn(N).astype(np.float32)
        # Filter uncorrelated noise to match ambient noise frequency distribution (< 1 kHz)
        b, a = signal.butter(4, min(1000.0, (sr / 2.0) * 0.8) / (sr / 2.0), btype="low")
        uncorr_shaped = signal.lfilter(b, a, uncorr_raw).astype(np.float32)

        # Match RMS energy of clean_noise
        p_clean = float(np.mean(clean_noise ** 2)) + 1e-12
        p_uncorr = float(np.mean(uncorr_shaped ** 2)) + 1e-12
        uncorr_scaled = uncorr_shaped * np.sqrt(p_clean / p_uncorr)

        # Linear combination
        ref = np.sqrt(coherence) * clean_noise + np.sqrt(1.0 - coherence) * uncorr_scaled
        return ref.astype(np.float32)

    def convolve_primary(self, external_noise: np.ndarray) -> np.ndarray:
        """Propagate external acoustic noise causally through primary path P(z) into the ear canal."""
        return np.convolve(external_noise, self.p_nominal, mode="full")[: len(external_noise)].astype(np.float32)

    def convolve_secondary(self, anti_noise: np.ndarray, leak_ratio: float = 0.0) -> np.ndarray:
        """Propagate anti-noise signal y(n) causally through speaker and cavity S(z) to the error mic."""
        h = self.s_nominal if leak_ratio <= 0.0 else self.s_leaked
        return np.convolve(anti_noise, h, mode="full")[: len(anti_noise)].astype(np.float32)


class StreamingPlantState:
    """
    Stateful sample-by-sample acoustic plant simulator.
    Maintains ring buffers for true real-time causality testing.
    """

    def __init__(self, p_fir: np.ndarray, s_fir: np.ndarray):
        self.p_fir = np.asarray(p_fir, dtype=np.float32)
        self.s_fir = np.asarray(s_fir, dtype=np.float32)
        self.lp = len(self.p_fir)
        self.ls = len(self.s_fir)

        self.buf_ext = np.zeros(self.lp, dtype=np.float32)
        self.buf_anti = np.zeros(self.ls, dtype=np.float32)

    def step(self, ext_noise_sample: float, anti_noise_sample: float) -> Tuple[float, float, float]:
        """
        Executes one sample step of the acoustic environment.

        Args:
            ext_noise_sample: Disturbance at reference mic x(n).
            anti_noise_sample: Speaker actuator command y(n).

        Returns:
            Tuple of (d_n, y_s_n, e_n):
            - d_n: Primary acoustic disturbance in cavity.
            - y_s_n: Acoustic anti-sound arriving at cavity.
            - e_n: Residual acoustic sound pressure e(n) = d(n) - y_s(n).
        """
        # Shift buffers
        self.buf_ext[1:] = self.buf_ext[:-1]
        self.buf_ext[0] = ext_noise_sample

        self.buf_anti[1:] = self.buf_anti[:-1]
        self.buf_anti[0] = anti_noise_sample

        # Convolve sample
        d_n = float(np.dot(self.p_fir, self.buf_ext))
        y_s_n = float(np.dot(self.s_fir, self.buf_anti))
        e_n = d_n - y_s_n

        return d_n, y_s_n, e_n
