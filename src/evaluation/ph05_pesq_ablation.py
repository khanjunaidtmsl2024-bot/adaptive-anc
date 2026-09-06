"""
PH0.5 — PESQ Ablation Study (Diagnostic Only).
PS 26052 — Adaptive Defence ANC.

Runs controlled single-variable ablations to identify the root cause
of PESQ degradation (measured 1.50 vs target 2.50).

DOES NOT MODIFY THE AI ARCHITECTURE.

Each ablation changes exactly ONE variable from the baseline hybrid pipeline.
The baseline is: VSS-NLMS → Impulse Protection → Causal STFT → TinyEnhancer → iSTFT OLA.

Ablation Matrix:
  ABL-0: Baseline (full hybrid pipeline)
  ABL-1: AI bypass (NLMS-only output, no neural mask)
  ABL-2: NLMS bypass (AI-only on raw noisy input)
  ABL-3: Phase passthrough (AI mask on magnitude, keep NLMS residual phase)
  ABL-4: Hamming window instead of Hanning
  ABL-5: 75% overlap (hop=64) instead of 50% (hop=128)

Evidence Tier: OFFLINE EXPERIMENTALLY MEASURED (SYNTHETIC)

Output: results/csv/ph05_pesq_ablation.csv
"""

import csv
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.dsp.vss_nlms import VSSNLMSFilter
from src.dsp.impulse_protection import ImpulseProtectionController
from src.ai.tiny_enhancer import TinyEnhancerWrapper

# Import quality metrics
try:
    from pesq import pesq as pesq_eval
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False

try:
    from pystoi import stoi as stoi_eval
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False


SR = 16000
FRAME_SIZE = 256
HOP_SIZE = 128
FILTER_LENGTH = 64


def _generate_test_signal(rng: np.random.RandomState, duration_s: float = 2.0,
                          snr_db: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic test signal:
      clean = multi-harmonic speech-like signal
      noise = band-limited noise
      noisy = clean + noise at specified SNR
    Returns: (clean, noisy, reference_noise)
    """
    n = int(duration_s * SR)
    t = np.arange(n) / SR

    # Synthetic speech: fundamental + harmonics with envelope
    f0 = 150.0
    envelope = np.sin(np.pi * t / duration_s) ** 2
    clean = np.zeros(n, dtype=np.float32)
    for harmonic in [1, 2, 3, 4, 5]:
        clean += (0.3 / harmonic) * np.sin(2 * np.pi * f0 * harmonic * t)
    clean = (clean * envelope).astype(np.float32)

    # Normalize clean
    clean = clean / (np.max(np.abs(clean)) + 1e-12) * 0.5

    # Noise: band-limited (200-4000 Hz)
    noise = rng.randn(n).astype(np.float32)
    from scipy.signal import butter, filtfilt
    b, a = butter(4, [200 / (SR / 2), 4000 / (SR / 2)], btype='band')
    noise = filtfilt(b, a, noise).astype(np.float32)

    # Scale noise to target SNR
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / (noise_power + 1e-12))

    noisy = (clean + noise).astype(np.float32)
    reference = (noise * 0.9 + rng.randn(n).astype(np.float32) * 0.01).astype(np.float32)

    return clean, noisy, reference


def _process_hybrid_baseline(
    noisy: np.ndarray,
    reference: np.ndarray,
    window_type: str = "hanning",
    hop_size: int = HOP_SIZE,
    frame_size: int = FRAME_SIZE,
    bypass_ai: bool = False,
    bypass_nlms: bool = False,
    phase_passthrough: bool = False,
) -> np.ndarray:
    """
    Process signal through configurable hybrid pipeline.
    This is the core ablation dispatcher.
    """
    n = min(len(noisy), len(reference))

    nlms = VSSNLMSFilter(filter_length=FILTER_LENGTH, mu_init=0.05)
    imp_ctl = ImpulseProtectionController(sample_rate=SR)
    ai_model = TinyEnhancerWrapper()

    if window_type == "hanning":
        window = np.hanning(frame_size).astype(np.float32)
    elif window_type == "hamming":
        window = np.hamming(frame_size).astype(np.float32)
    else:
        window = np.hanning(frame_size).astype(np.float32)

    buf_primary = np.zeros(frame_size, dtype=np.float32)
    buf_reference = np.zeros(frame_size, dtype=np.float32)
    overlap_buf = np.zeros(frame_size, dtype=np.float32)
    output = np.zeros(n, dtype=np.float32)

    n_hops = (n - frame_size) // hop_size + 1

    for i in range(n_hops):
        start = i * hop_size
        p_hop = noisy[start:start + hop_size]
        r_hop = reference[start:start + hop_size]

        if len(p_hop) < hop_size:
            p_hop = np.pad(p_hop, (0, hop_size - len(p_hop)))
            r_hop = np.pad(r_hop, (0, hop_size - len(r_hop)))

        # Buffer update
        buf_primary[:-hop_size] = buf_primary[hop_size:]
        buf_primary[-hop_size:] = p_hop
        buf_reference[:-hop_size] = buf_reference[hop_size:]
        buf_reference[-hop_size:] = r_hop

        if bypass_nlms:
            # ABL-2: Skip NLMS, pass raw noisy to AI
            dsp_frame = buf_primary.copy()
        else:
            # Standard NLMS + Impulse Protection
            dsp_hop, _, _ = nlms.filter_block(p_hop, r_hop)
            dsp_hop, _, _ = imp_ctl.filter_block_protection(dsp_hop)

            # Construct full frame for STFT from buffer
            dsp_frame = buf_primary.copy()
            dsp_frame[-hop_size:] = dsp_hop

        if bypass_ai:
            # ABL-1: Output is NLMS-only
            output[start:start + hop_size] = dsp_frame[-hop_size:]
        else:
            # AI processing
            windowed = dsp_frame * window
            stft_frame = np.fft.rfft(windowed)
            freq_bins = len(stft_frame)
            mag = np.abs(stft_frame).reshape(-1, 1)
            phase_orig = np.angle(stft_frame).reshape(-1, 1)

            enh_mag, enh_phase = ai_model.enhance_spectrogram(mag, phase_orig)

            enh_mag_1d = enh_mag.flatten()[:freq_bins]

            if phase_passthrough:
                # ABL-3: Use original phase, not AI phase
                enh_phase_1d = phase_orig.flatten()[:freq_bins]
            else:
                enh_phase_1d = enh_phase.flatten()[:freq_bins]

            enh_stft = enh_mag_1d * np.exp(1j * enh_phase_1d)
            recon = np.fft.irfft(enh_stft, n=frame_size) * window

            overlap_buf += recon
            out_hop = overlap_buf[:hop_size].copy()
            overlap_buf[:-hop_size] = overlap_buf[hop_size:]
            overlap_buf[-hop_size:] = 0.0

            output[start:start + hop_size] = out_hop

    # Trim to the region actually produced by overlap-add. The final frame's
    # second half is never flushed, so samples beyond (n_hops - 1) * hop_size
    # + hop_size would remain zeros and bias every ablation's metrics.
    processed = (n_hops - 1) * hop_size + hop_size if n_hops > 0 else 0
    output = output[:processed]

    # Normalize
    peak = np.max(np.abs(output)) + 1e-12 if output.size else 0.0
    if peak > 0.98:
        output = output * 0.98 / peak

    return output


def _evaluate_quality(
    clean: np.ndarray, noisy: np.ndarray, enhanced: np.ndarray
) -> Dict[str, float]:
    """Compute PESQ, STOI, SI-SDR, SNR_out, and Delta-SNR."""
    n = min(len(clean), len(noisy), len(enhanced))
    c = clean[:n].copy()
    y = noisy[:n].copy()
    e = enhanced[:n].copy()

    # Normalize to prevent PESQ clipping issues
    c = c / (np.max(np.abs(c)) + 1e-12) * 0.7
    y = y / (np.max(np.abs(y)) + 1e-12) * 0.7
    e = e / (np.max(np.abs(e)) + 1e-12) * 0.7

    result = {}

    if PESQ_AVAILABLE:
        try:
            result["pesq_wb"] = round(float(pesq_eval(SR, c, e, "wb")), 4)
        except Exception as ex:
            result["pesq_wb"] = float("nan")
            result["pesq_error"] = str(ex)
    else:
        result["pesq_wb"] = float("nan")

    if STOI_AVAILABLE:
        try:
            result["stoi"] = round(float(stoi_eval(c, e, SR, extended=False)), 4)
        except Exception:
            result["stoi"] = float("nan")
    else:
        result["stoi"] = float("nan")

    # SI-SDR
    s_target = c * np.dot(e, c) / (np.dot(c, c) + 1e-12)
    noise_residual = e - s_target
    si_sdr = 10 * np.log10(np.sum(s_target ** 2) / (np.sum(noise_residual ** 2) + 1e-12))
    result["si_sdr_db"] = round(float(si_sdr), 2)

    # Input SNR, Output SNR, Delta SNR
    noise_in = y - c
    snr_in = 10 * np.log10(np.sum(c ** 2) / (np.sum(noise_in ** 2) + 1e-12))
    noise_out = e - c
    snr_out = 10 * np.log10(np.sum(c ** 2) / (np.sum(noise_out ** 2) + 1e-12))
    result["snr_out_db"] = round(float(snr_out), 2)
    result["delta_snr_db"] = round(float(snr_out - snr_in), 2)

    return result


def run_pesq_ablation():
    """Execute the full PH0.5 PESQ ablation study."""
    print("=" * 72)
    print("PH0.5 -- PESQ ABLATION STUDY (DIAGNOSTIC ONLY)")
    print("Evidence Tier: OFFLINE EXPERIMENTALLY MEASURED (SYNTHETIC)")
    print("AI Architecture: UNCHANGED (diagnostic variable isolation only)")
    print("=" * 72)

    out_dir = Path("results/csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Test conditions: 10 signals at various SNRs
    test_conditions = [
        (-5, 42), (-5, 43), (0, 44), (0, 45), (5, 46),
        (5, 47), (10, 48), (10, 49), (-3, 50), (3, 51),
    ]

    ablation_configs = [
        {
            "id": "ABL-0",
            "name": "Baseline Hybrid",
            "params": {},
            "hypothesis": "Baseline reference point (hop=128, frame=256, Hanning)",
            "is_secondary": False,
        },
        {
            "id": "ABL-1",
            "name": "AI Bypass (NLMS-only)",
            "params": {"bypass_ai": True},
            "hypothesis": "Is AI degrading PESQ relative to classical DSP?",
            "is_secondary": False,
        },
        {
            "id": "ABL-2",
            "name": "NLMS Bypass (AI-only)",
            "params": {"bypass_nlms": True},
            "hypothesis": "Is NLMS degrading PESQ or is AI alone better?",
            "is_secondary": False,
        },
        {
            "id": "ABL-3",
            "name": "Phase Passthrough",
            "params": {"phase_passthrough": True},
            "hypothesis": "Is neural phase estimation corrupting speech harmonics?",
            "is_secondary": False,
        },
        {
            "id": "ABL-4",
            "name": "Hamming Window",
            "params": {"window_type": "hamming"},
            "hypothesis": "Is window transition/sidelobes causing spectral leakage?",
            "is_secondary": False,
        },
        {
            "id": "ABL-5",
            "name": "Cadence Change (hop=64, 75% OLA)",
            "params": {"hop_size": 64},
            "hypothesis": "Secondary experiment: does higher temporal sampling / overlap improve reconstruction?",
            "is_secondary": True,
        },
    ]

    all_results = []

    for abl in ablation_configs:
        sec_tag = " [SECONDARY EXPERIMENT]" if abl["is_secondary"] else ""
        print(f"\n  {abl['id']}: {abl['name']}{sec_tag} -- \"{abl['hypothesis']}\"")
        pesq_values = []
        stoi_values = []
        si_sdr_values = []
        snr_out_values = []
        delta_snr_values = []

        for snr_db, seed in test_conditions:
            rng = np.random.RandomState(seed)
            clean, noisy, reference = _generate_test_signal(rng, duration_s=2.0, snr_db=snr_db)
            enhanced = _process_hybrid_baseline(noisy, reference, **abl["params"])
            metrics = _evaluate_quality(clean, noisy, enhanced)

            pesq_values.append(metrics["pesq_wb"])
            stoi_values.append(metrics["stoi"])
            si_sdr_values.append(metrics["si_sdr_db"])
            snr_out_values.append(metrics["snr_out_db"])
            delta_snr_values.append(metrics["delta_snr_db"])

        # Average over non-NaN values
        pesq_clean = [v for v in pesq_values if not np.isnan(v)]
        stoi_clean = [v for v in stoi_values if not np.isnan(v)]

        avg_pesq = round(float(np.mean(pesq_clean)), 4) if pesq_clean else float("nan")
        avg_stoi = round(float(np.mean(stoi_clean)), 4) if stoi_clean else float("nan")
        avg_si_sdr = round(float(np.mean(si_sdr_values)), 2)
        avg_snr_out = round(float(np.mean(snr_out_values)), 2)
        avg_delta_snr = round(float(np.mean(delta_snr_values)), 2)

        print(f"    PESQ={avg_pesq:.3f}  STOI={avg_stoi:.4f}  SI-SDR={avg_si_sdr:.1f} dB  Delta-SNR={avg_delta_snr:+.1f} dB  SNR_out={avg_snr_out:.1f} dB")

        all_results.append({
            "ablation_id": abl["id"],
            "name": abl["name"],
            "is_secondary": abl["is_secondary"],
            "hypothesis": abl["hypothesis"],
            "avg_pesq_wb": avg_pesq,
            "avg_stoi": avg_stoi,
            "avg_si_sdr_db": avg_si_sdr,
            "avg_delta_snr_db": avg_delta_snr,
            "avg_snr_out_db": avg_snr_out,
            "n_clips": len(test_conditions),
            "evidence_tier": "OFFLINE EXPERIMENTALLY MEASURED (SYNTHETIC)",
        })

    # Save CSV
    csv_path = out_dir / "ph05_pesq_ablation.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n  Saved: {csv_path}")

    # Summary table
    print("\n" + "=" * 80)
    print("PESQ ABLATION SUMMARY")
    print("=" * 80)
    print(f"{'Ablation':<32} | {'PESQ':>7} | {'STOI':>7} | {'SI-SDR':>8} | {'Delta-SNR':>10} | {'SNR_out':>8}")
    print("-" * 80)
    baseline_pesq = all_results[0]["avg_pesq_wb"]
    for r in all_results:
        delta = r["avg_pesq_wb"] - baseline_pesq if not np.isnan(r["avg_pesq_wb"]) and not np.isnan(baseline_pesq) else float("nan")
        delta_str = f" (d={delta:+.3f})" if not np.isnan(delta) and r["ablation_id"] != "ABL-0" else ""
        print(f"{r['name']:<32} | {r['avg_pesq_wb']:>7.3f} | {r['avg_stoi']:>7.4f} | {r['avg_si_sdr_db']:>6.1f} dB | {r['avg_delta_snr_db']:>+8.1f} dB | {r['avg_snr_out_db']:>6.1f} dB{delta_str}")
    print("-" * 80)
    print(f"{'DRDO Target':<32} |  >2.500 |  >0.850 |          |   >10.0 dB   |  >15.0 dB")

    return all_results


if __name__ == "__main__":
    run_pesq_ablation()
