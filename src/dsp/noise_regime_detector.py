"""
Noise Regime Detector — Adaptive Pipeline Selector.
PS 26052 — Adaptive Defence ANC.

Classifies incoming audio into noise regimes and dynamically selects
the optimal DSP/AI pipeline configuration:

Regimes:
  STATIONARY  — steady-state machinery, vehicle cabin, HVAC
  IMPULSIVE   — gunshots, explosions, blast overpressure
  DIFFUSE     — wind, crowd, multi-directional environmental noise
  SPEECH_ONLY — no significant noise (pass-through or light AI polish)

Decision Logic (real-time, per-frame):
  - Kurtosis > 10         → IMPULSIVE   → freeze NLMS, enable impulse gate
  - Spectral Flatness > 0.5 → DIFFUSE   → increase filter length, reduce mu
  - Spectral Flatness < 0.15 → STATIONARY → standard NLMS, moderate AI
  - Otherwise                → SPEECH_ONLY → bypass DSP, AI-only polish
"""

from typing import Dict, Any, Tuple
from enum import Enum
import numpy as np


class NoiseRegime(Enum):
    STATIONARY = "STATIONARY"
    IMPULSIVE = "IMPULSIVE"
    DIFFUSE = "DIFFUSE"
    SPEECH_ONLY = "SPEECH_ONLY"


# Pipeline parameter presets per regime
REGIME_PRESETS: Dict[NoiseRegime, Dict[str, Any]] = {
    NoiseRegime.STATIONARY: {
        "nlms_mu": 0.05,
        "filter_length": 64,
        "enable_impulse_protection": False,
        "ai_aggressiveness": 0.7,
        "description": "Standard NLMS + moderate AI enhancement",
    },
    NoiseRegime.IMPULSIVE: {
        "nlms_mu": 0.001,  # Freeze adaptation
        "filter_length": 32,
        "enable_impulse_protection": True,
        "ai_aggressiveness": 0.5,
        "description": "Frozen NLMS + impulse gate + conservative AI",
    },
    NoiseRegime.DIFFUSE: {
        "nlms_mu": 0.02,
        "filter_length": 128,
        "enable_impulse_protection": False,
        "ai_aggressiveness": 0.9,
        "description": "Long NLMS filter + aggressive AI de-reverb",
    },
    NoiseRegime.SPEECH_ONLY: {
        "nlms_mu": 0.0,  # DSP bypass
        "filter_length": 64,
        "enable_impulse_protection": False,
        "ai_aggressiveness": 0.3,
        "description": "DSP bypass + light AI polish only",
    },
}


class NoiseRegimeDetector:
    """
    Real-time noise regime classifier operating on short-time frames.

    Features computed per-frame:
      1. Kurtosis: Detects impulsive transients (gunshots, explosions).
      2. Spectral Flatness (Wiener entropy): Distinguishes tonal vs diffuse noise.
      3. Short-term energy ratio: Detects speech-dominated segments.
      4. Zero-crossing rate: Additional tonal/noise discriminator.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 512,
        smoothing_alpha: float = 0.85,
        kurtosis_threshold: float = 10.0,
        flatness_high: float = 0.50,
        flatness_low: float = 0.15,
        energy_floor_db: float = -50.0,
    ):
        self.sr = sample_rate
        self.frame_size = frame_size
        self.alpha = smoothing_alpha
        self.kurt_thresh = kurtosis_threshold
        self.flat_high = flatness_high
        self.flat_low = flatness_low
        self.energy_floor = 10 ** (energy_floor_db / 10.0)

        # State
        self.current_regime = NoiseRegime.STATIONARY
        self.smoothed_kurtosis = 0.0
        self.smoothed_flatness = 0.0
        self.smoothed_energy = 0.0
        self.noise_energy_estimate = 1e-8
        self.regime_history = []
        self.frame_count = 0

    def reset(self) -> None:
        self.current_regime = NoiseRegime.STATIONARY
        self.smoothed_kurtosis = 0.0
        self.smoothed_flatness = 0.0
        self.smoothed_energy = 0.0
        self.noise_energy_estimate = 1e-8
        self.regime_history = []
        self.frame_count = 0

    def _compute_kurtosis(self, frame: np.ndarray) -> float:
        """Excess kurtosis (0 = Gaussian, >6 = impulsive)."""
        n = len(frame)
        if n < 4:
            return 0.0
        mean = np.mean(frame)
        var = np.var(frame) + 1e-12
        m4 = np.mean((frame - mean) ** 4)
        return float(m4 / (var ** 2) - 3.0)

    def _compute_spectral_flatness(self, frame: np.ndarray) -> float:
        """Wiener entropy: geometric_mean(|X|) / arithmetic_mean(|X|)."""
        spectrum = np.abs(np.fft.rfft(frame))
        spectrum = spectrum + 1e-12
        log_mean = np.mean(np.log(spectrum))
        geom_mean = np.exp(log_mean)
        arith_mean = np.mean(spectrum)
        return float(np.clip(geom_mean / (arith_mean + 1e-12), 0.0, 1.0))

    def _compute_zcr(self, frame: np.ndarray) -> float:
        """Zero-crossing rate normalized to [0, 1]."""
        signs = np.sign(frame)
        crossings = np.sum(np.abs(np.diff(signs)) > 0)
        return float(crossings / max(len(frame) - 1, 1))

    def classify_frame(
        self,
        frame: np.ndarray,
        reference_frame: np.ndarray = None,
    ) -> Tuple[NoiseRegime, Dict[str, Any]]:
        """
        Classify a single audio frame into a noise regime.

        Args:
            frame: Primary mic frame (1D float32).
            reference_frame: Optional reference mic frame (unused for now).

        Returns:
            (regime, diagnostics) tuple.
        """
        self.frame_count += 1

        # Feature extraction
        kurt = self._compute_kurtosis(frame)
        flatness = self._compute_spectral_flatness(frame)
        energy = float(np.mean(frame ** 2))
        zcr = self._compute_zcr(frame)

        # Exponential moving average smoothing
        self.smoothed_kurtosis = self.alpha * self.smoothed_kurtosis + (1 - self.alpha) * kurt
        self.smoothed_flatness = self.alpha * self.smoothed_flatness + (1 - self.alpha) * flatness
        self.smoothed_energy = self.alpha * self.smoothed_energy + (1 - self.alpha) * energy

        # Update noise energy estimate (minimum tracker)
        if energy < self.noise_energy_estimate * 1.5 or self.frame_count < 10:
            self.noise_energy_estimate = 0.99 * self.noise_energy_estimate + 0.01 * energy

        snr_est = 10 * np.log10((self.smoothed_energy + 1e-12) / (self.noise_energy_estimate + 1e-12))

        # Decision tree
        if self.smoothed_kurtosis > self.kurt_thresh:
            regime = NoiseRegime.IMPULSIVE
        elif self.smoothed_energy < self.energy_floor:
            regime = NoiseRegime.SPEECH_ONLY
        elif self.smoothed_flatness > self.flat_high:
            regime = NoiseRegime.DIFFUSE
        elif self.smoothed_flatness < self.flat_low:
            regime = NoiseRegime.STATIONARY
        else:
            # Ambiguous region — use energy and ZCR
            if snr_est > 15.0 and zcr < 0.3:
                regime = NoiseRegime.SPEECH_ONLY
            elif zcr > 0.5:
                regime = NoiseRegime.DIFFUSE
            else:
                regime = NoiseRegime.STATIONARY

        self.current_regime = regime
        self.regime_history.append(regime)
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]

        diagnostics = {
            "regime": regime.value,
            "kurtosis": round(self.smoothed_kurtosis, 3),
            "spectral_flatness": round(self.smoothed_flatness, 4),
            "energy_db": round(10 * np.log10(self.smoothed_energy + 1e-12), 1),
            "zcr": round(zcr, 3),
            "snr_estimate_db": round(snr_est, 1),
            "frame_count": self.frame_count,
        }

        return regime, diagnostics

    def get_pipeline_params(self, regime: NoiseRegime = None) -> Dict[str, Any]:
        """Returns optimal DSP/AI parameters for the given or current regime."""
        if regime is None:
            regime = self.current_regime
        return REGIME_PRESETS[regime].copy()

    def classify_signal(self, signal: np.ndarray) -> Tuple[NoiseRegime, Dict[str, Any]]:
        """Classify an entire signal by majority-vote over frames."""
        n_frames = max(1, len(signal) // self.frame_size)
        regime_counts = {r: 0 for r in NoiseRegime}

        for i in range(n_frames):
            start = i * self.frame_size
            frame = signal[start:start + self.frame_size]
            if len(frame) < self.frame_size:
                frame = np.pad(frame, (0, self.frame_size - len(frame)))
            regime, _ = self.classify_frame(frame)
            regime_counts[regime] += 1

        dominant = max(regime_counts, key=regime_counts.get)
        total = sum(regime_counts.values())

        diag = {
            "dominant_regime": dominant.value,
            "regime_distribution": {r.value: c / total for r, c in regime_counts.items()},
            "total_frames": total,
        }

        return dominant, diag
