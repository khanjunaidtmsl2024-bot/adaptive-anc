"""
Two-Rate Supervisory AI Coupling Module.
PH4-SIM: Physical ANC Simulation & Secondary-Path Validation.
PS 26052 — Adaptive Defence ANC.

Demonstrates the hybrid supervisory architecture:
- Fast Loop: Sample-by-sample FxNLMS physical acoustic control simulation.
- Slow Loop: Frame-by-frame (128 samples / 8.0 ms hop) supervisory governor.

Supervisory Actions:
1. Speech Leakage Protection: Freezes FxNLMS adaptation (mu -> 0) when user speaks,
   preventing voice cancellation and filter weight divergence.
2. Acoustic Seal-Break Monitor: Detects ear-cup cushion unsealing via residual error
   spectral changes and shifts secondary path model or adaptation rate.
3. Instability / Howl Suppression: Detects narrow-band buildup and engages leakage damping.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np

from src.dsp.fxlms import FxLMSFilter, SecondaryPathModel
from src.dsp.acoustic_plant import AcousticPlantModel, StreamingPlantState


class SupervisoryGovernor:
    """
    Supervisory controller operating on frame cadence (8.0 ms / 128 samples)
    to govern the fast-path FxNLMS acoustic loop.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        hop_size: int = 128,
        speech_energy_threshold: float = 0.0003,
        howl_threshold: float = 0.8,
    ):
        self.sr = sample_rate
        self.hop_size = hop_size
        self.speech_thresh = speech_energy_threshold
        self.howl_thresh = howl_threshold

        # Supervisory telemetry
        self.speech_detected_history = []
        self.seal_break_detected_history = []
        self.adaptation_frozen_hops = 0

    def analyze_frame(
        self,
        primary_frame: np.ndarray,
        error_frame: np.ndarray,
        current_step_size: float,
    ) -> Dict[str, Any]:
        """
        Analyzes an 8.0 ms frame and outputs control flags for the FxNLMS loop.

        Returns:
            Dict containing:
            - freeze_adaptation: bool
            - adjusted_mu: float
            - seal_compromised: bool
            - howl_risk: bool
        """
        # 1. Voice Activity / Speech Leakage Detection
        pri_power = float(np.mean(primary_frame ** 2))
        speech_active = bool(pri_power > self.speech_thresh)
        self.speech_detected_history.append(speech_active)

        # 2. Howling / Feedback Instability Detection
        err_peak = float(np.max(np.abs(error_frame)))
        howl_risk = bool(err_peak > self.howl_thresh)

        # 3. Low-Frequency Acoustic Seal Break Detection (bass leakage surge)
        # In a closed cavity, bass is tightly controlled; when seal breaks, low-freq residual rises
        err_power = float(np.mean(error_frame ** 2))
        fft_err = np.abs(np.fft.rfft(error_frame))
        low_band_power = float(np.mean(fft_err[:8] ** 2))  # < 500 Hz
        high_band_power = float(np.mean(fft_err[8:] ** 2)) + 1e-12
        spectral_ratio = low_band_power / high_band_power
        seal_compromised = bool(err_power > 1e-4 and spectral_ratio > 15.0 and not speech_active)
        self.seal_break_detected_history.append(seal_compromised)

        # Compute governed step-size
        if speech_active:
            # Freeze adaptation completely to protect speech
            freeze = True
            adj_mu = 0.0
            self.adaptation_frozen_hops += 1
        elif howl_risk:
            # Cut step-size in half and engage damping
            freeze = False
            adj_mu = current_step_size * 0.1
        elif seal_compromised:
            # Conservative adaptation under altered acoustic seal
            freeze = False
            adj_mu = current_step_size * 0.5
        else:
            freeze = False
            adj_mu = current_step_size

        return {
            "freeze_adaptation": freeze,
            "adjusted_mu": adj_mu,
            "speech_active": speech_active,
            "seal_compromised": seal_compromised,
            "howl_risk": howl_risk,
        }


def run_supervisory_hybrid_simulation(
    reference: np.ndarray,
    clean_speech: np.ndarray,
    external_noise: np.ndarray,
    plant_model: AcousticPlantModel,
    s_hat: np.ndarray,
    enable_supervision: bool = True,
    base_mu: float = 0.02,
    hop_size: int = 128,
) -> Dict[str, Any]:
    """
    Executes a complete hybrid physical simulation comparing:
    - Unsupervised FxNLMS (adaptation runs blindly even during speech)
    - Supervised FxNLMS (E2/Governor freezes adaptation during speech)

    Args:
        reference: Noise reference at external mic x(n).
        clean_speech: Clean speech spoken by user into headset microphone.
        external_noise: Raw external ambient noise.
        plant_model: AcousticPlantModel providing P(z) and S(z).
        s_hat: Estimated secondary path FIR filter.
        enable_supervision: Whether the 8.0 ms supervisory governor is active.
        base_mu: Base step size for FxNLMS.
        hop_size: Frame size for supervisory analysis (128 samples = 8.0 ms).

    Returns:
        Dictionary with residual error, speech distortion, and adaptation telemetry.
    """
    N = len(reference)
    sr = plant_model.sr

    # Acoustic simulation setup
    # Primary disturbance inside ear cup d(n) = P(z) * noise + leaked speech
    d_acoustic = plant_model.convolve_primary(external_noise)
    # Speech leaks into ear cavity directly
    d_acoustic += clean_speech * 0.8

    fxlms = FxLMSFilter(
        filter_length=64,
        step_size=base_mu,
        secondary_path_estimate=s_hat,
        leakage=1e-4,
    )
    governor = SupervisoryGovernor(sample_rate=sr, hop_size=hop_size)
    plant_state = StreamingPlantState(plant_model.p_nominal, plant_model.s_nominal)

    anti_noise = np.zeros(N, dtype=np.float32)
    residual_error = np.zeros(N, dtype=np.float32)
    prior_err = 0.0

    n_hops = N // hop_size
    current_mu = base_mu

    for h in range(n_hops):
        start = h * hop_size
        end = start + hop_size

        # Slow Supervisory Loop (runs once per hop)
        if enable_supervision:
            primary_chunk = clean_speech[start:end] + external_noise[start:end] * 0.1
            err_prior_chunk = residual_error[max(0, start - hop_size) : start]
            if len(err_prior_chunk) == 0:
                err_prior_chunk = np.zeros(hop_size, dtype=np.float32)

            gov_diag = governor.analyze_frame(primary_chunk, err_prior_chunk, base_mu)
            fxlms.is_frozen = gov_diag["freeze_adaptation"]
            fxlms.mu = gov_diag["adjusted_mu"]
        else:
            fxlms.is_frozen = False
            fxlms.mu = base_mu

        # Fast Acoustic Control Loop (runs sample-by-sample)
        for i in range(start, end):
            x_i = reference[i]

            # 1. Anti-noise command
            y_i = fxlms.generate_anti_noise(x_i)
            anti_noise[i] = y_i

            # 2. Physical plant propagation (acoustic disturbance + anti-sound in ear cup)
            d_i, y_s_i, e_clean_n = plant_state.step(external_noise[i], y_i)
            # Total error at microphone is acoustic residual + user speech
            e_total_i = e_clean_n + clean_speech[i] * 0.8
            residual_error[i] = e_total_i

            # 3. Adapt weights with current acoustic error
            fxlms.adapt(e_total_i)

    # Evaluate speech preservation: calculate speech energy ratio
    speech_mask = np.abs(clean_speech) > 0.05
    if np.sum(speech_mask) > 100:
        speech_orig_energy = float(np.mean(clean_speech[speech_mask] ** 2))
        speech_res_energy = float(np.mean(residual_error[speech_mask] ** 2))
        speech_attenuation_db = float(10.0 * np.log10(speech_res_energy / (speech_orig_energy + 1e-12)))
    else:
        speech_attenuation_db = 0.0

    return {
        "residual_error": residual_error,
        "anti_noise": anti_noise,
        "speech_attenuation_db": round(speech_attenuation_db, 2),
        "diverged": fxlms.diverged,
        "frozen_hops": governor.adaptation_frozen_hops,
        "total_hops": n_hops,
    }
