"""
PS 26052: DRDO Adaptive ANC — Secondary-Path Repeatability Verifier (PH5-R3)
============================================================================
Performs multiple consecutive measurements of the secondary path S(z)
to evaluate:
1. Plant Repeatability: Delay-aligned waveform correlation (R_aligned >= 0.95)
2. Transport Determinism: Raw unaligned correlation (R_raw) and delay jitter (sigma_t)
3. Transducer Stability: Inter-run gain variation (Delta G <= 0.5 dB)
4. Secondary-Path Phase Stability: Inter-run phase variation (Delta phi <= 5 deg)
"""

import time
from typing import Dict, List, Union

import numpy as np

from .secondary_path_measurer import SecondaryPathMeasurer


class RepeatabilityVerifier:
    """
    Evaluates multi-run repeatability of physical secondary-path transfer function,
    separating plant repeatability from transport delay jitter.
    """

    def __init__(self, measurer: SecondaryPathMeasurer, num_runs: int = 5):
        self.measurer = measurer
        self.num_runs = int(num_runs)
        self.sr = measurer.sr

    def run_evaluation(
        self, pause_between_sec: float = 0.1
    ) -> Dict[str, Union[float, np.ndarray, List[float], bool]]:
        """
        Execute num_runs secondary path measurements and compute both raw
        and delay-aligned consistency metrics.
        """
        runs = []
        raw_firs = []
        full_irs = []
        mags = []
        phases = []

        for r in range(self.num_runs):
            res = self.measurer.measure()
            runs.append(res)
            raw_firs.append(res["s_fir"])
            full_irs.append(res["full_ir"])
            mags.append(res["mag_db"])
            phases.append(res["phase_deg"])
            if pause_between_sec > 0:
                time.sleep(pause_between_sec)

        firs_raw = np.array(raw_firs)  # (N, taps)
        mags = np.array(mags)          # (N, n_freq)
        phases = np.array(phases)      # (N, n_freq)
        freqs = runs[0]["freqs"]

        # ---------------------------------------------------------------------
        # 1. Raw Unaligned Correlation & Transport Delay Jitter
        # ---------------------------------------------------------------------
        raw_corr_matrix = np.zeros((self.num_runs, self.num_runs), dtype=np.float64)
        for i in range(self.num_runs):
            for j in range(self.num_runs):
                norm_i = np.linalg.norm(firs_raw[i]) + 1e-12
                norm_j = np.linalg.norm(firs_raw[j]) + 1e-12
                raw_corr_matrix[i, j] = np.dot(firs_raw[i], firs_raw[j]) / (norm_i * norm_j)

        off_diag_raw = []
        for i in range(self.num_runs):
            for j in range(i + 1, self.num_runs):
                off_diag_raw.append(float(raw_corr_matrix[i, j]))

        mean_raw_corr = float(np.mean(off_diag_raw)) if len(off_diag_raw) > 0 else 1.0
        min_raw_corr = float(np.min(off_diag_raw)) if len(off_diag_raw) > 0 else 1.0

        # Extract peak sample offsets across trials
        peak_indices = [int(np.argmax(np.abs(ir))) for ir in full_irs]
        peak_arr = np.array(peak_indices, dtype=np.float64)
        delay_offsets_samples = (peak_arr - peak_arr[0]).tolist()
        delay_std_samples = float(np.std(peak_arr))
        delay_std_ms = (delay_std_samples / self.sr) * 1000.0

        # ---------------------------------------------------------------------
        # 2. Delay-Aligned Plant Repeatability (Authoritative Plant Metric)
        # ---------------------------------------------------------------------
        taps = len(raw_firs[0])
        aligned_firs = []
        for p_idx, ir in zip(peak_indices, full_irs):
            start = max(0, p_idx - 10)
            fir_seg = ir[start : start + taps].copy()
            if len(fir_seg) < taps:
                fir_seg = np.pad(fir_seg, (0, taps - len(fir_seg)))
            aligned_firs.append(fir_seg)

        firs_aligned = np.array(aligned_firs)
        aligned_corr_matrix = np.zeros((self.num_runs, self.num_runs), dtype=np.float64)
        for i in range(self.num_runs):
            for j in range(self.num_runs):
                norm_i = np.linalg.norm(firs_aligned[i]) + 1e-12
                norm_j = np.linalg.norm(firs_aligned[j]) + 1e-12
                aligned_corr_matrix[i, j] = np.dot(firs_aligned[i], firs_aligned[j]) / (norm_i * norm_j)

        off_diag_aligned = []
        for i in range(self.num_runs):
            for j in range(i + 1, self.num_runs):
                off_diag_aligned.append(float(aligned_corr_matrix[i, j]))

        mean_aligned_corr = float(np.mean(off_diag_aligned)) if len(off_diag_aligned) > 0 else 1.0
        min_aligned_corr = float(np.min(off_diag_aligned)) if len(off_diag_aligned) > 0 else 1.0

        # ---------------------------------------------------------------------
        # 3. Transducer Magnitude & Phase Variations (100–1000 Hz ANC Band)
        # ---------------------------------------------------------------------
        mag_std = np.std(mags, axis=0)
        mag_mean = np.mean(mags, axis=0)
        phase_std = np.std(phases, axis=0)

        anc_mask = (freqs >= 100.0) & (freqs <= 1000.0)
        anc_mag_std_db = float(np.mean(mag_std[anc_mask])) if np.any(anc_mask) else float(np.mean(mag_std))
        anc_phase_std_deg = float(np.mean(phase_std[anc_mask])) if np.any(anc_mask) else float(np.mean(phase_std))

        # ---------------------------------------------------------------------
        # 4. Repeatability Verification Decision
        # ---------------------------------------------------------------------
        # Authoritative plant repeatability requires aligned R >= 0.95 and stable magnitude
        plant_repeatability_passed = (min_aligned_corr >= 0.95) and (anc_mag_std_db <= 0.5)
        # Transport determinism checks if timing is locked (sigma_t <= 0.05 ms / 0.8 sample at 16 kHz)
        transport_determinism_passed = delay_std_ms <= 0.05

        return {
            "num_runs": int(self.num_runs),
            # Transport determinism metrics
            "mean_raw_correlation": mean_raw_corr,
            "min_raw_correlation": min_raw_corr,
            "delay_std_samples": delay_std_samples,
            "delay_std_ms": delay_std_ms,
            "relative_delay_offsets": [float(x) for x in delay_offsets_samples],
            "transport_determinism_passed": bool(transport_determinism_passed),
            # Plant repeatability metrics
            "mean_aligned_correlation": mean_aligned_corr,
            "min_aligned_correlation": min_aligned_corr,
            "anc_band_gain_std_db": anc_mag_std_db,
            "anc_band_phase_std_deg": anc_phase_std_deg,
            "plant_repeatability_passed": bool(plant_repeatability_passed),
            # Detailed spectral arrays
            "freqs": freqs,
            "mean_mag_db": mag_mean,
            "std_mag_db": mag_std,
            "std_phase_deg": phase_std,
            "representative_fir": firs_aligned[0],
            "first_run_result": runs[0],
        }
