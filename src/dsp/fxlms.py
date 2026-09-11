"""
Filtered-x Normalized Least Mean Squares (FxNLMS) and Secondary Path Engine.
PH4-SIM: Physical ANC Simulation & Secondary-Path Validation.
PS 26052 — Adaptive Defence ANC.

Implements:
1. Real-time sample-by-sample Leaky Normalized FxNLMS Filter.
2. Direct Secondary Path Modeling S(z) and Filtered Reference Generation S_hat(z).
3. Ultra-fast Numba-accelerated batch simulation loop for multi-factor sweeps.
4. Comprehensive diagnostic tracking (cancellation depth, weight norms, stability flags).

Strictly designated for simulated physical acoustic control on host CPU.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np

try:
    import numba
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class SecondaryPathModel:
    """Models an electro-acoustic secondary path S(z) from actuator to error mic."""

    def __init__(self, impulse_response: np.ndarray):
        self.h = np.asarray(impulse_response, dtype=np.float32)
        self.L = len(self.h)
        self.buf = np.zeros(self.L, dtype=np.float32)

    def reset(self):
        self.buf.fill(0.0)

    def filter_sample(self, x_n: float) -> float:
        """Pass one sample through S(z)."""
        self.buf[1:] = self.buf[:-1]
        self.buf[0] = x_n
        return float(np.dot(self.h, self.buf))

    def filter_signal(self, signal: np.ndarray) -> np.ndarray:
        """Convolve signal with S(z) FIR filter causally."""
        return np.convolve(signal, self.h, mode="full")[: len(signal)].astype(np.float32)


class FxLMSFilter:
    """
    Filtered-x Normalized Least Mean Squares (FxNLMS) Filter.
    Causally incorporates secondary path estimate S_hat(z) and leakage:
        x'(n) = S_hat(z) * x(n)
        W(n+1) = (1 - gamma * mu) * W(n) + [mu / (||X'(n)||^2 + eps)] * e(n) * X'(n)
        y(n) = W^T(n) * X(n)
    """

    def __init__(
        self,
        filter_length: int = 64,
        step_size: float = 0.01,
        secondary_path_estimate: Optional[np.ndarray] = None,
        leakage: float = 1e-4,
        eps: float = 1e-6,
        max_anti_noise_amplitude: float = 1.0,
        saturation_mode: str = "hard",
    ):
        self.M = int(filter_length)
        self.mu = float(step_size)
        self.gamma = float(leakage)
        self.eps = float(eps)
        self.v_max = float(max_anti_noise_amplitude)
        self.saturation_mode = saturation_mode.lower()

        if secondary_path_estimate is None:
            self.s_hat = np.zeros(16, dtype=np.float32)
            self.s_hat[1] = 1.0
        else:
            self.s_hat = np.asarray(secondary_path_estimate, dtype=np.float32)

        self.sec_model = SecondaryPathModel(self.s_hat)
        self.weights = np.zeros(self.M, dtype=np.float32)
        self.ref_buf = np.zeros(self.M, dtype=np.float32)
        self.filtered_ref_buf = np.zeros(self.M, dtype=np.float32)

        self.is_frozen = False
        self.diverged = False

    def reset(self):
        """Resets internal delay buffers and filter weights."""
        self.weights.fill(0.0)
        self.ref_buf.fill(0.0)
        self.filtered_ref_buf.fill(0.0)
        self.sec_model.reset()
        self.is_frozen = False
        self.diverged = False

    def generate_anti_noise(self, ref_n: float) -> float:
        """Buffers reference x(n) and computes anti-noise y(n) with actuator saturation."""
        self.ref_buf[1:] = self.ref_buf[:-1]
        self.ref_buf[0] = ref_n

        raw_y = float(np.dot(self.weights, self.ref_buf))
        if self.saturation_mode == "soft":
            y_actuator = float(self.v_max * np.tanh(raw_y / (self.v_max + 1e-12)))
        else:
            y_actuator = float(np.clip(raw_y, -self.v_max, self.v_max))

        # Update filtered reference: x'(n) = S_hat(z) * x(n)
        filtered_ref = self.sec_model.filter_sample(ref_n)
        self.filtered_ref_buf[1:] = self.filtered_ref_buf[:-1]
        self.filtered_ref_buf[0] = filtered_ref

        return y_actuator

    def adapt(self, error_n: float):
        """Adapts weights using measured error e(n) and filtered reference."""
        if not self.is_frozen and not self.diverged:
            norm = float(np.dot(self.filtered_ref_buf, self.filtered_ref_buf)) + self.eps
            leak_factor = 1.0 - (self.gamma * self.mu)
            self.weights = leak_factor * self.weights + (self.mu / norm) * error_n * self.filtered_ref_buf

            if np.isnan(self.weights).any() or np.isinf(self.weights).any() or np.max(np.abs(self.weights)) > 50.0:
                self.diverged = True

    def step(self, ref_n: float, error_n: float) -> float:
        """Executes anti-noise generation and weight update for current sample."""
        y_n = self.generate_anti_noise(ref_n)
        self.adapt(error_n)
        return y_n


@njit(fastmath=True)
def _simulate_fxnlms_fast(
    ref: np.ndarray,
    prim_disturbance: np.ndarray,
    s_true: np.ndarray,
    s_hat: np.ndarray,
    filter_len: int,
    step_size: float,
    leakage: float,
    eps: float,
    v_max: float = 1.0,
    sat_mode: int = 0,
):
    """
    Numba-accelerated inner loop for sample-by-sample FxNLMS acoustic simulation
    with physical actuator saturation constraints.
    """
    N = len(ref)
    M = filter_len
    L_true = len(s_true)
    L_hat = len(s_hat)
    mu = step_size
    gamma = leakage

    # State buffers
    weights = np.zeros(M, dtype=np.float32)
    ref_buf = np.zeros(M, dtype=np.float32)
    filt_ref_buf = np.zeros(M, dtype=np.float32)
    s_hat_buf = np.zeros(L_hat, dtype=np.float32)
    s_true_buf = np.zeros(L_true, dtype=np.float32)

    y_out = np.zeros(N, dtype=np.float32)
    e_out = np.zeros(N, dtype=np.float32)
    w_norm = np.zeros(N, dtype=np.float32)

    diverged = False
    clip_count = 0
    tot_dist_power = 0.0

    for n in range(N):
        x_n = ref[n]
        d_n = prim_disturbance[n]

        # 1. Update reference buffer
        for k in range(M - 1, 0, -1):
            ref_buf[k] = ref_buf[k - 1]
        ref_buf[0] = x_n

        # 2. Generate anti-noise: y(n) = W^T * X(n) with physical actuator limits
        y_n = 0.0
        y_actuator = 0.0
        if not diverged:
            for k in range(M):
                y_n += weights[k] * ref_buf[k]

            if sat_mode == 1:
                # Soft driver tanh compression
                y_actuator = v_max * np.tanh(y_n / (v_max + 1e-12))
            else:
                # Hard DAC/Amp clipping
                if y_n > v_max:
                    y_actuator = v_max
                elif y_n < -v_max:
                    y_actuator = -v_max
                else:
                    y_actuator = y_n

            if abs(y_n) > v_max:
                clip_count += 1
            dist = y_n - y_actuator
            tot_dist_power += dist * dist
        else:
            y_actuator = 0.0

        y_out[n] = y_actuator

        # 3. Propagate saturated anti-noise through TRUE secondary path S(z)
        for k in range(L_true - 1, 0, -1):
            s_true_buf[k] = s_true_buf[k - 1]
        s_true_buf[0] = y_actuator

        anti_sound_n = 0.0
        for k in range(L_true):
            anti_sound_n += s_true[k] * s_true_buf[k]

        # 4. Physical cancellation in cavity: e(n) = d(n) - anti_sound(n)
        e_n = d_n - anti_sound_n
        e_out[n] = e_n

        # 5. Filter reference through S_hat(z): x'(n) = S_hat(z) * x(n)
        for k in range(L_hat - 1, 0, -1):
            s_hat_buf[k] = s_hat_buf[k - 1]
        s_hat_buf[0] = x_n

        filt_ref_n = 0.0
        for k in range(L_hat):
            filt_ref_n += s_hat[k] * s_hat_buf[k]

        for k in range(M - 1, 0, -1):
            filt_ref_buf[k] = filt_ref_buf[k - 1]
        filt_ref_buf[0] = filt_ref_n

        # 6. Adapt weights with matching error e(n) and filtered reference x'(n)
        if not diverged:
            norm = 0.0
            for k in range(M):
                norm += filt_ref_buf[k] * filt_ref_buf[k]
            norm += eps

            leak = 1.0 - (gamma * mu)
            step_mult = (mu / norm) * e_n
            for k in range(M):
                weights[k] = leak * weights[k] + step_mult * filt_ref_buf[k]

            # Check divergence
            cur_norm = 0.0
            for k in range(M):
                val = weights[k]
                if np.isnan(val) or np.isinf(val) or abs(val) > 50.0:
                    diverged = True
                    break
                cur_norm += val * val
            w_norm[n] = np.sqrt(cur_norm)
        else:
            w_norm[n] = 0.0

    return y_out, e_out, w_norm, diverged, clip_count, tot_dist_power


def run_fxnlms_simulation(
    reference: np.ndarray,
    primary_disturbance: np.ndarray,
    s_true: np.ndarray,
    s_hat: np.ndarray,
    filter_length: int = 64,
    step_size: float = 0.01,
    leakage: float = 1e-4,
    eps: float = 1e-6,
    max_anti_noise_amplitude: float = 1.0,
    saturation_mode: str = "hard",
) -> Dict[str, Any]:
    """
    High-performance simulation driver for closed-loop acoustic ANC.
    Includes actuator saturation limits and clipping distortion analysis.

    Returns dictionary with:
        - anti_noise: Actuator output signal y(n)
        - residual_error: Error microphone signal e(n)
        - weight_norm_history: L2 norm of weights over time
        - cancellation_db: Attenuation depth in decibels
        - diverged: Boolean indicating if controller became unstable
        - clip_ratio: Proportion of samples where anti-noise hit actuator saturation
        - distortion_power: Mean square power of nonlinear clipping distortion
    """
    ref = np.asarray(reference, dtype=np.float32)
    prim = np.asarray(primary_disturbance, dtype=np.float32)
    st = np.asarray(s_true, dtype=np.float32)
    sh = np.asarray(s_hat, dtype=np.float32)
    sat_int = 1 if saturation_mode.lower() == "soft" else 0

    y, e, w_norm, diverged, clip_cnt, tot_dist = _simulate_fxnlms_fast(
        ref,
        prim,
        st,
        sh,
        int(filter_length),
        float(step_size),
        float(leakage),
        float(eps),
        float(max_anti_noise_amplitude),
        sat_int,
    )

    # Compute overall attenuation in steady state (last 50% of the signal)
    half = len(e) // 2
    p_in = float(np.mean(prim[half:] ** 2)) + 1e-12
    p_err = float(np.mean(e[half:] ** 2)) + 1e-12
    cancellation_db = float(10.0 * np.log10(p_in / p_err))

    return {
        "anti_noise": y,
        "residual_error": e,
        "weight_norm": w_norm,
        "cancellation_db": round(cancellation_db, 2),
        "diverged": bool(diverged),
        "steady_state_error_power": float(p_err),
        "clip_ratio": round(float(clip_cnt / max(1, len(ref))), 4),
        "distortion_power": float(tot_dist / max(1, len(ref))),
    }
