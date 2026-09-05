#!/usr/bin/env python3
"""
PS 26052 — Artifact 4: Adaptive Filter Laboratory
==================================================
Standalone NLMS implementation with mathematically valid signal flow.

CRITICAL DESIGN QUESTION (Section 106 of master doc):
    What is the error signal e(n) for a post-AI adaptive stage?

This artifact explores THREE candidate configurations and measures which
produces the best noise suppression without speech distortion.

Configuration A — PRE-AI Reference Canceller:
    Reference mic → NLMS → cancels correlated noise → AI enhances residual.
    Error signal: e(n) = primary(n) - w^T * reference(n)
    This is mathematically identical to classical adaptive noise cancellation.

Configuration B — POST-AI Residual Canceller:
    AI output → NLMS → suppresses residual correlated noise.
    Error signal: e(n) = AI_output(n) - w^T * reference(n)
    WARNING: This assumes reference contains only noise (no speech leakage).

Configuration C — ADAPTIVE SPECTRAL MASK:
    STFT-domain adaptive mask applied to AI output using reference coherence.
    Not classical NLMS but a common practical approach.

Each configuration is tested on synthetic data with known ground truth.

Usage:
    python 04_adaptive_filter_lab.py --mode demo
    python 04_adaptive_filter_lab.py --mode compare
    python 04_adaptive_filter_lab.py --mode interactive

Designed for: SIH 2026 Adaptive ANC — PS 26052
"""

import os
import sys
import csv
import time
import argparse
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import soundfile as sf
except ImportError:
    print("ERROR: pip install soundfile")
    sys.exit(1)

try:
    from scipy import signal as scipy_signal
except ImportError:
    print("ERROR: pip install scipy")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("adaptive_lab")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """SNR in dB."""
    noise = noisy - clean
    rms_clean = np.sqrt(np.mean(clean ** 2) + 1e-12)
    rms_noise = np.sqrt(np.mean(noise ** 2) + 1e-12)
    return 20.0 * np.log10(rms_clean / rms_noise)


def compute_rmse(signal_a: np.ndarray, signal_b: np.ndarray) -> float:
    """Root Mean Square Error."""
    return float(np.sqrt(np.mean((signal_a - signal_b) ** 2)))


# ---------------------------------------------------------------------------
# NLMS Implementation
# ---------------------------------------------------------------------------
class NLMSFilter:
    """
    Normalized Least Mean Squares adaptive filter.
    
    Math:
        w(n+1) = w(n) + mu * e(n) * x(n) / (epsilon + ||x(n)||^2)
    
    where:
        w(n)    = filter weights (length L)
        x(n)    = input/reference vector [x(n), x(n-1), ..., x(n-L+1)]
        e(n)    = desired(n) - w^T(n) * x(n)
        mu      = step size (0 < mu <= 1, typically 0.1-0.5)
        epsilon = regularization (typically 1e-6)
    """
    
    def __init__(self, filter_length: int = 64, mu: float = 0.3, epsilon: float = 1e-6):
        self.L = filter_length
        self.mu = mu
        self.epsilon = epsilon
        self.weights = np.zeros(filter_length, dtype=np.float64)
        self._input_buffer = np.zeros(filter_length, dtype=np.float64)
    
    def reset(self):
        """Reset filter weights to zero."""
        self.weights = np.zeros(self.L, dtype=np.float64)
        self._input_buffer = np.zeros(self.L, dtype=np.float64)
    
    def step(self, reference: float, desired: float) -> tuple[float, float]:
        """
        Process one sample.
        
        Args:
            reference: current reference/reference input x(n)
            desired:   desired signal d(n) — what we're trying to approximate
        
        Returns:
            (output, error) where:
                output = w^T * x (the filter's estimate)
                error  = desired - output (the adaptation signal)
        """
        # Shift input buffer
        self._input_buffer = np.roll(self._input_buffer, 1)
        self._input_buffer[0] = reference
        
        # Compute filter output
        output = np.dot(self.weights, self._input_buffer)
        
        # Compute error
        error = desired - output
        
        # NLMS weight update
        power = np.dot(self._input_buffer, self._input_buffer)
        self.weights += self.mu * error * self._input_buffer / (power + self.epsilon)
        
        return output, error


# ---------------------------------------------------------------------------
# Configuration A: Pre-AI Reference Canceller
# ---------------------------------------------------------------------------
class PreAINLCanceller:
    """
    Classical adaptive noise cancellation (pre-AI).
    
    Signal flow:
        primary mic:   d(n) = speech(n) + noise_primary(n)
        reference mic: x(n) = noise_reference(n) [correlated with noise_primary]
        
        y(n) = w^T * x(n)          — estimate of noise in primary
        e(n) = d(n) - y(n)         — cleaned signal (speech + residual noise)
        w(n+1) = NLMS update using e(n) and x(n)
        
    The error e(n) IS the enhanced signal.
    Mathematically valid because d(n) and x(n) are both observable at runtime.
    """
    
    def __init__(self, filter_length: int = 128, mu: float = 0.3):
        self.nlms = NLMSFilter(filter_length, mu)
    
    def process(self, primary: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Process a signal pair. Returns enhanced signal."""
        self.nlms.reset()
        L = self.nlms.L
        enhanced = np.zeros(len(primary), dtype=np.float64)
        
        for n in range(len(primary)):
            if n < L:
                # Not enough history yet, pass through
                enhanced[n] = primary[n]
                continue
            
            d_n = primary[n]      # desired = primary mic (speech + noise)
            x_n = reference[n]    # reference = reference mic (noise only)
            
            _, error = self.nlms.step(x_n, d_n)
            enhanced[n] = error   # output is the error
        
        return enhanced.astype(np.float32)


# ---------------------------------------------------------------------------
# Configuration B: Post-AI Residual Canceller
# ---------------------------------------------------------------------------
class PostAIResidualCanceller:
    """
    Post-AI adaptive residual noise suppression.
    
    WARNING: This configuration has a mathematical validity problem.
    
    Signal flow:
        AI output:    s_hat(n) = enhanced speech estimate
        reference mic: x(n) = noise_reference(n)
        
        y(n) = w^T * x(n)          — estimate of residual noise correlated with reference
        e(n) = s_hat(n) - y(n)     — further cleaned signal
        w(n+1) = NLMS update using e(n) and x(n)
    
    MATHEMATICAL CONCERN:
        The "desired" signal d(n) = s_hat(n) (AI output).
        The NLMS tries to minimize E[|e(n)|^2] = E[|d(n) - w^T*x(n)|^2].
        This means it tries to make w^T*x(n) approximate the AI output.
        If the reference mic contains ANY speech, this will CANCEL speech.
        
    SAFE USE CONDITIONS:
        1. Reference mic must contain NO speech (only noise)
        2. Reference mic must be correlated with residual noise in AI output
        3. Filter length must be short to avoid overfitting
        4. Step size must be very conservative (mu < 0.1)
    """
    
    def __init__(self, filter_length: int = 64, mu: float = 0.05):
        self.nlms = NLMSFilter(filter_length, mu)
    
    def process(self, ai_output: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Process AI output with reference. Returns further-cleaned signal."""
        self.nlms.reset()
        L = self.nlms.L
        enhanced = np.zeros(len(ai_output), dtype=np.float64)
        
        for n in range(len(ai_output)):
            if n < L:
                enhanced[n] = ai_output[n]
                continue
            
            d_n = ai_output[n]     # desired = AI output
            x_n = reference[n]     # reference = reference mic
            
            _, error = self.nlms.step(x_n, d_n)
            enhanced[n] = error
        
        return enhanced.astype(np.float32)


# ---------------------------------------------------------------------------
# Configuration C: Adaptive Spectral Mask (frequency-domain)
# ---------------------------------------------------------------------------
class AdaptiveSpectralMask:
    """
    STFT-domain adaptive suppression using reference coherence.
    
    Instead of time-domain NLMS, this applies an adaptive gain mask
    in the frequency domain, guided by the coherence between primary
    and reference microphones.
    
    This avoids the d(n) definition problem by working on magnitudes
    and using coherence as a suppression guide.
    
    Signal flow:
        primary STFT:   P(k)
        reference STFT: R(k)
        
        coherence: gamma(k) = |P(k)*R(k)|^2 / (|P(k)|^2 * |R(k)|^2)
        
        mask: M(k) = max(1 - alpha * gamma(k), floor)
        
        enhanced: E(k) = M(k) * P(k)
    """
    
    def __init__(self, frame_size: int = 512, alpha: float = 0.8, floor: float = 0.1):
        self.frame_size = frame_size
        self.alpha = alpha
        self.floor = floor
        self.window = np.hanning(frame_size)
    
    def process(self, primary: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Process signal pair in STFT domain."""
        hop = self.frame_size // 2
        n_frames = (len(primary) - self.frame_size) // hop + 1
        
        output = np.zeros(len(primary), dtype=np.float64)
        overlap = np.zeros(self.frame_size - hop, dtype=np.float64)
        
        # Smoothed coherence estimate (exponential moving average)
        smooth_gamma = None
        ema_alpha = 0.1  # smoothing factor
        
        for i in range(n_frames):
            start = i * hop
            primary_frame = primary[start:start + self.frame_size]
            reference_frame = reference[start:start + self.frame_size]
            
            if len(primary_frame) < self.frame_size:
                break
            
            # Window
            primary_windowed = primary_frame * self.window
            reference_windowed = reference_frame * self.window
            
            # STFT
            P = np.fft.rfft(primary_windowed)
            R = np.fft.rfft(reference_windowed)
            
            # Coherence
            P_mag2 = np.abs(P) ** 2
            R_mag2 = np.abs(R) ** 2
            PR = P * np.conj(R)
            gamma = (np.abs(PR) ** 2) / (P_mag2 * R_mag2 + 1e-12)
            
            # Smooth coherence
            if smooth_gamma is None:
                smooth_gamma = gamma
            else:
                smooth_gamma = ema_alpha * gamma + (1 - ema_alpha) * smooth_gamma
            
            # Adaptive mask
            mask = np.maximum(1 - self.alpha * smooth_gamma, self.floor)
            
            # Apply mask
            E = mask * P
            
            # iSTFT
            output_frame = np.fft.irfft(E, n=self.frame_size)
            
            # Overlap-add
            output[start:start + hop] += output_frame[:hop] * self.window[:hop]
            if start + hop < len(output):
                output[start + hop:start + self.frame_size] += (
                    output_frame[hop:] * self.window[hop:]
                )
        
        return output.astype(np.float32)


# ---------------------------------------------------------------------------
# Synthetic Test Signal Generator
# ---------------------------------------------------------------------------
def generate_test_signals(
    duration_s: float = 5.0,
    sample_rate: int = 16000,
    noise_type: str = "stationary",
    snr_db: float = 5.0,
    add_speech_leakage_to_ref: bool = False,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate synthetic test signals with known ground truth.
    
    Returns: (clean_speech, noisy_primary, reference, input_snr)
    """
    rng = np.random.default_rng(seed)
    n_samples = int(duration_s * sample_rate)
    t = np.arange(n_samples) / sample_rate
    
    # --- Clean speech: modulated harmonic signal ---
    f0 = 150  # fundamental
    speech = np.zeros(n_samples)
    for h in range(1, 8):
        speech += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t)
    
    # Amplitude modulation (syllable rhythm, ~4 Hz)
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 4.5 * t))
    speech *= envelope
    
    # Normalize
    speech = speech / (np.max(np.abs(speech)) + 1e-12) * 0.8
    
    # --- Noise ---
    if noise_type == "stationary":
        # Low-frequency rumble (engine-like)
        noise_primary = rng.standard_normal(n_samples) * 0.3
        b, a = scipy_signal.butter(4, 400 / (sample_rate / 2), btype="low")
        noise_primary = scipy_signal.lfilter(b, a, noise_primary)
        # Add harmonics (engine harmonics)
        for f in [50, 100, 150]:
            noise_primary += 0.15 * np.sin(2 * np.pi * f * t)
        
        # Reference: correlated copy with different amplitude + independent noise
        correlation = 0.8
        independent = rng.standard_normal(n_samples) * 0.2
        reference = correlation * noise_primary + np.sqrt(1 - correlation**2) * independent
        
    elif noise_type == "non_stationary":
        # Rotor-like amplitude-modulated noise
        noise_primary = rng.standard_normal(n_samples) * 0.3
        mod = 0.5 * (1 + np.sin(2 * np.pi * 8 * t))  # 8 Hz modulation
        noise_primary *= mod
        
        # Reference with delay and different modulation
        correlation = 0.6
        ref_noise = rng.standard_normal(n_samples) * 0.3
        ref_mod = 0.5 * (1 + np.sin(2 * np.pi * 8 * t + 0.3))
        ref_noise *= ref_mod
        reference = correlation * noise_primary + np.sqrt(1 - correlation**2) * ref_noise
        
    else:  # impulsive
        # Sparse impulse noise
        noise_primary = rng.standard_normal(n_samples) * 0.02
        n_impulses = rng.integers(5, 15)
        for _ in range(n_impulses):
            pos = rng.integers(int(0.1 * n_samples), int(0.9 * n_samples))
            imp_len = rng.integers(int(0.01 * sample_rate), int(0.1 * sample_rate))
            imp_len = min(imp_len, n_samples - pos)
            impulse = rng.standard_normal(imp_len) * 0.8
            decay = np.exp(-np.arange(imp_len) / (imp_len * 0.2))
            noise_primary[pos:pos + imp_len] += impulse * decay
        
        reference = rng.standard_normal(n_samples) * 0.05
        for _ in range(n_impulses // 2):
            pos = rng.integers(int(0.1 * n_samples), int(0.9 * n_samples))
            imp_len = rng.integers(int(0.005 * sample_rate), int(0.05 * sample_rate))
            imp_len = min(imp_len, n_samples - pos)
            reference[pos:pos + imp_len] += rng.standard_normal(imp_len) * 0.3
    
    # Add speech leakage to reference if configured
    if add_speech_leakage_to_ref:
        leakage_level = 0.15
        reference += leakage_level * speech
        log.info(f"  Reference contains speech leakage at {leakage_level:.0%} level")
    
    # --- Create noisy primary: speech + noise ---
    rms_speech = np.sqrt(np.mean(speech ** 2))
    rms_noise = np.sqrt(np.mean(noise_primary ** 2))
    alpha = rms_speech / (rms_noise * 10 ** (snr_db / 20.0))
    noisy_primary = speech + alpha * noise_primary
    
    input_snr = compute_snr(speech, noisy_primary)
    
    return speech.astype(np.float32), noisy_primary.astype(np.float32), reference.astype(np.float32), input_snr


# ---------------------------------------------------------------------------
# Configuration Comparison
# ---------------------------------------------------------------------------
@dataclass
class ConfigResult:
    """Result for one configuration test."""
    config_name: str
    noise_type: str
    speech_leakage: bool
    input_snr_db: float
    output_snr_db: float
    snr_improvement_db: float
    rmse_to_clean: float
    filter_length: int
    mu: float


def compare_configurations(duration_s: float = 5.0, snr_db: float = 5.0):
    """Run all three configurations and compare results."""
    
    results = []
    
    for noise_type in ["stationary", "non_stationary", "impulsive"]:
        for leakage in [False, True]:
            leakage_label = "with leakage" if leakage else "no leakage"
            log.info(f"\n  Testing: {noise_type} noise, {leakage_label}")
            
            clean, noisy, reference, input_snr = generate_test_signals(
                duration_s=duration_s,
                noise_type=noise_type,
                snr_db=snr_db,
                add_speech_leakage_to_ref=leakage,
            )
            
            # --- Config A: Pre-AI Reference Canceller ---
            nlc = PreAINLCanceller(filter_length=128, mu=0.3)
            enhanced_a = nlc.process(noisy, reference)
            out_snr_a = compute_snr(clean, enhanced_a)
            rmse_a = compute_rmse(clean, enhanced_a)
            
            results.append(ConfigResult(
                config_name="A_PreAI_NLCanceller",
                noise_type=noise_type,
                speech_leakage=leakage,
                input_snr_db=round(input_snr, 2),
                output_snr_db=round(out_snr_a, 2),
                snr_improvement_db=round(out_snr_a - input_snr, 2),
                rmse_to_clean=round(rmse_a, 6),
                filter_length=128,
                mu=0.3,
            ))
            
            # --- Config B: Post-AI Residual Canceller ---
            # Simulate AI output as "noisy but better" signal
            # (In real use, this would be the AI model output)
            ai_output = noisy.copy() * 0.7 + clean * 0.3  # crude AI simulation
            ai_output = ai_output / (np.max(np.abs(ai_output)) + 1e-12) * 0.8
            
            parc = PostAIResidualCanceller(filter_length=64, mu=0.05)
            enhanced_b = parc.process(ai_output, reference)
            out_snr_b = compute_snr(clean, enhanced_b)
            rmse_b = compute_rmse(clean, enhanced_b)
            
            results.append(ConfigResult(
                config_name="B_PostAI_Residual",
                noise_type=noise_type,
                speech_leakage=leakage,
                input_snr_db=round(input_snr, 2),
                output_snr_db=round(out_snr_b, 2),
                snr_improvement_db=round(out_snr_b - input_snr, 2),
                rmse_to_clean=round(rmse_b, 6),
                filter_length=64,
                mu=0.05,
            ))
            
            # --- Config C: Adaptive Spectral Mask ---
            asm = AdaptiveSpectralMask(frame_size=512, alpha=0.8, floor=0.1)
            enhanced_c = asm.process(noisy, reference)
            out_snr_c = compute_snr(clean, enhanced_c)
            rmse_c = compute_rmse(clean, enhanced_c)
            
            results.append(ConfigResult(
                config_name="C_AdaptiveSpectralMask",
                noise_type=noise_type,
                speech_leakage=leakage,
                input_snr_db=round(input_snr, 2),
                output_snr_db=round(out_snr_c, 2),
                snr_improvement_db=round(out_snr_c - input_snr, 2),
                rmse_to_clean=round(rmse_c, 6),
                filter_length=512,
                mu=0.0,
            ))
    
    # Print comparison table
    print(f"\n{'='*100}")
    print(f"CONFIGURATION COMPARISON — Adaptive Filter Laboratory")
    print(f"{'='*100}")
    print(f"{'Config':<30s} {'Noise':<15s} {'Leakage':<10s} "
          f"{'In SNR':>8s} {'Out SNR':>8s} {'Improvement':>12s} {'RMSE':>10s}")
    print(f"{'-'*100}")
    
    for r in results:
        leakage_str = "YES" if r.speech_leakage else "no"
        print(f"{r.config_name:<30s} {r.noise_type:<15s} {leakage_str:<10s} "
              f"{r.input_snr_db:>8.2f} {r.output_snr_db:>8.2f} "
              f"{r.snr_improvement_db:>12.2f} {r.rmse_to_clean:>10.6f}")
    
    print(f"{'='*100}")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    csv_path = "results/adaptive_filter_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        from dataclasses import asdict
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"\nResults saved to {csv_path}")
    
    # Recommendation
    print(f"\n{'='*100}")
    print(f"RECOMMENDATION")
    print(f"{'='*100}")
    
    # Find best config (highest SNR improvement, lowest RMSE)
    # For no-leakage case
    no_leakage = [r for r in results if not r.speech_leakage]
    best = max(no_leakage, key=lambda r: r.snr_improvement_db)
    print(f"\n  Best config (no speech leakage):  {best.config_name}")
    print(f"    SNR improvement: {best.snr_improvement_db:+.2f} dB")
    
    # For leakage case
    with_leakage = [r for r in results if r.speech_leakage]
    best_leak = max(with_leakage, key=lambda r: r.snr_improvement_db)
    print(f"\n  Best config (with speech leakage): {best_leak.config_name}")
    print(f"    SNR improvement: {best_leak.snr_improvement_db:+.2f} dB")
    
    print(f"\n  MATHEMATICAL VALIDITY NOTES:")
    print(f"  Config A (Pre-AI):  ALWAYS valid. Error = primary - w^T*ref. Both observable.")
    print(f"  Config B (Post-AI): VALID ONLY IF ref has no speech. Otherwise cancels speech.")
    print(f"  Config C (Spectral): Valid but not classical NLMS. Coherence-based masking.")
    print(f"\n  FOR SIH 26052: Use Config A as primary adaptive stage, place BEFORE AI model.")
    print(f"{'='*100}\n")
    
    return results


# ---------------------------------------------------------------------------
# Interactive Demo
# ---------------------------------------------------------------------------
def interactive_demo():
    """Interactive demonstration of NLMS behavior."""
    print(f"\n{'='*60}")
    print(f"ADAPTIVE FILTER INTERACTIVE DEMO")
    print(f"{'='*60}")
    print(f"This demo shows how NLMS converges and adapts.")
    print()
    
    # Generate signals
    clean, noisy, reference, input_snr = generate_test_signals(
        duration_s=3.0, noise_type="stationary", snr_db=5.0
    )
    
    print(f"  Input SNR: {input_snr:.2f} dB")
    print(f"  Signal length: {len(noisy)} samples")
    print()
    
    # Run with different filter lengths
    for L in [32, 64, 128, 256]:
        for mu in [0.05, 0.1, 0.3, 0.5]:
            nlc = PreAINLCanceller(filter_length=L, mu=mu)
            enhanced = nlc.process(noisy, reference)
            out_snr = compute_snr(clean, enhanced)
            improvement = out_snr - input_snr
            
            marker = " <<<" if improvement > 5 else ""
            print(f"  L={L:>4d}, mu={mu:.2f}:  SNR improvement = {improvement:+.2f} dB{marker}")
    
    print(f"\n  <<< = improvement > 5 dB")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PS 26052 Adaptive Filter Laboratory")
    parser.add_argument("--mode", choices=["demo", "compare", "interactive", "single"],
                        default="compare",
                        help="Operating mode")
    parser.add_argument("--noise-type", choices=["stationary", "non_stationary", "impulsive"],
                        default="stationary")
    parser.add_argument("--snr", type=float, default=5.0, help="Input SNR in dB")
    parser.add_argument("--duration", type=float, default=5.0, help="Signal duration in seconds")
    parser.add_argument("--filter-length", type=int, default=128)
    parser.add_argument("--mu", type=float, default=0.3)
    parser.add_argument("--leakage", action="store_true", help="Add speech leakage to reference")
    args = parser.parse_args()
    
    if args.mode == "compare":
        compare_configurations(duration_s=args.duration, snr_db=args.snr)
    
    elif args.mode == "interactive":
        interactive_demo()
    
    elif args.mode == "single":
        clean, noisy, reference, input_snr = generate_test_signals(
            duration_s=args.duration,
            noise_type=args.noise_type,
            snr_db=args.snr,
            add_speech_leakage_to_ref=args.leakage,
        )
        
        nlc = PreAINLCanceller(filter_length=args.filter_length, mu=args.mu)
        enhanced = nlc.process(noisy, reference)
        out_snr = compute_snr(clean, enhanced)
        
        print(f"\n  Input SNR:  {input_snr:.2f} dB")
        print(f"  Output SNR: {out_snr:.2f} dB")
        print(f"  Improvement: {out_snr - input_snr:+.2f} dB")
        
        # Save outputs
        os.makedirs("results", exist_ok=True)
        sf.write("results/clean_reference.wav", clean, 16000, subtype="PCM_16")
        sf.write("results/noisy_primary.wav", noisy, 16000, subtype="PCM_16")
        sf.write("results/reference_mic.wav", reference, 16000, subtype="PCM_16")
        sf.write("results/enhanced_output.wav", enhanced, 16000, subtype="PCM_16")
        print(f"  Saved to results/")
    
    elif args.mode == "demo":
        compare_configurations(duration_s=3.0, snr_db=5.0)
    
    print(f"\nNext step: Integrate with streaming pipeline (03_streaming_skeleton.py)")
    print(f"Then: Connect AI model output to Config A as the pre-AI adaptive stage.")


if __name__ == "__main__":
    main()
