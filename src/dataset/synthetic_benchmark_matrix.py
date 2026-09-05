"""
Gate 4: Controlled Multi-Factor Adversarial Synthetic Benchmark Matrix.
PS 26052 — Adaptive Defence ANC.

Constructs a 560-condition factorial evaluation matrix:
- 10 Speakers: 5 Male (85-155 Hz), 5 Female (165-260 Hz) with diverse phonetic types.
- 8 Noise Conditions:
    * Stationary: white, pink, brown, tonal_hum (50+120+240 Hz)
    * Non-Stationary: engine_mod, rotor_mod, siren_sweep
    * Impulsive: gunfire_burst (+40 dB spikes)
- 7 SNR Levels: -10, -5, 0, +5, +10, +15, +20 dB

Matrix total: 10 * 8 * 7 = 560 evaluation clips.
Output metadata: data/benchmark_matrix/matrix_metadata.csv
Audio generator: Deterministic on-the-fly synthesis matching frozen contract (16 kHz).
"""

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


SPEAKER_PROFILES = [
    {"id": "SPK_M01", "gender": "male", "pitch": 100.0, "rate": 1.0, "type": "continuous"},
    {"id": "SPK_M02", "gender": "male", "pitch": 120.0, "rate": 0.8, "type": "plosive_rich"},
    {"id": "SPK_M03", "gender": "male", "pitch": 90.0,  "rate": 1.2, "type": "fast_fricatives"},
    {"id": "SPK_M04", "gender": "male", "pitch": 135.0, "rate": 1.0, "type": "quiet_speech"},
    {"id": "SPK_M05", "gender": "male", "pitch": 110.0, "rate": 1.1, "type": "loud_cadence"},
    {"id": "SPK_F01", "gender": "female", "pitch": 210.0, "rate": 1.0, "type": "continuous"},
    {"id": "SPK_F02", "gender": "female", "pitch": 185.0, "rate": 0.9, "type": "plosive_rich"},
    {"id": "SPK_F03", "gender": "female", "pitch": 240.0, "rate": 1.2, "type": "fast_fricatives"},
    {"id": "SPK_F04", "gender": "female", "pitch": 195.0, "rate": 0.8, "type": "quiet_speech"},
    {"id": "SPK_F05", "gender": "female", "pitch": 225.0, "rate": 1.1, "type": "loud_cadence"},
]

NOISE_PROFILES = [
    {"id": "stat_white", "family": "stationary", "desc": "Broadband white Gaussian noise"},
    {"id": "stat_pink", "family": "stationary", "desc": "1/f Pink noise (cockpit background)"},
    {"id": "stat_brown", "family": "stationary", "desc": "1/f^2 Brownian low-frequency drone"},
    {"id": "stat_tonal_hum", "family": "stationary", "desc": "50+120+240 Hz turbine / alternator hum"},
    {"id": "nonstat_engine_mod", "family": "non_stationary", "desc": "Armored vehicle engine RPM modulation"},
    {"id": "nonstat_rotor_mod", "family": "non_stationary", "desc": "Helicopter blade-pass amplitude modulation"},
    {"id": "nonstat_siren_sweep", "family": "non_stationary", "desc": "Emergency / tactical siren Doppler sweep"},
    {"id": "impulse_gunfire_burst", "family": "impulsive", "desc": "Artillery / small arms gunfire bursts (+40 dB)"},
]

SNR_LEVELS = [-10, -5, 0, 5, 10, 15, 20]


def synthesize_speech_profile(speaker_id: str, n_samples: int = 16000, sr: int = 16000, seed: int = 42) -> np.ndarray:
    """Synthesizes formant-modeled speech matching a specific speaker profile."""
    prof = next(p for p in SPEAKER_PROFILES if p["id"] == speaker_id)
    rng = np.random.RandomState(seed + hash(speaker_id) % 10000)
    t = np.linspace(0, n_samples / sr, n_samples, endpoint=False)

    f0 = prof["pitch"]
    rate = prof["rate"]

    # Formant frequencies based on gender
    if prof["gender"] == "male":
        f1, f2, f3 = 500.0, 1500.0, 2500.0
    else:
        f1, f2, f3 = 700.0, 1900.0, 2900.0

    # Glottal pulse train
    glottal = np.zeros(n_samples, dtype=np.float32)
    period_samples = int(sr / f0)
    for p in range(0, n_samples, max(1, period_samples)):
        pulse_len = min(15, n_samples - p)
        glottal[p : p + pulse_len] = np.hanning(pulse_len * 2)[pulse_len:]

    # Modulate with speech syllabic envelope (2-4 Hz cadence)
    cadence = 0.5 * (1.0 + np.sin(2 * np.pi * 3.0 * rate * t)) ** 2
    if prof["type"] == "quiet_speech":
        cadence *= 0.5
    elif prof["type"] == "loud_cadence":
        cadence *= 1.5

    # Filter with formants
    formants = (
        0.5 * np.sin(2 * np.pi * f1 * t) +
        0.3 * np.sin(2 * np.pi * f2 * t) +
        0.2 * np.sin(2 * np.pi * f3 * t)
    )

    # Phonetic enhancements
    if prof["type"] == "fast_fricatives":
        fricative_noise = rng.randn(n_samples) * (cadence < 0.2) * 0.3
        raw_speech = (glottal * formants * cadence + fricative_noise).astype(np.float32)
    elif prof["type"] == "plosive_rich":
        plosives = np.zeros(n_samples, dtype=np.float32)
        for burst_idx in range(sr // 2, n_samples, sr // 3):
            plosives[burst_idx : burst_idx + 80] += rng.uniform(-1.0, 1.0, 80)
        raw_speech = (glottal * formants * cadence + plosives * 0.4).astype(np.float32)
    else:
        raw_speech = (glottal * formants * cadence).astype(np.float32)

    # Normalize to standard -26 dBFS
    rms = np.sqrt(np.mean(raw_speech ** 2)) + 1e-12
    return raw_speech * (0.05 / rms)


def synthesize_noise_profile(noise_id: str, n_samples: int = 16000, sr: int = 16000, seed: int = 42) -> np.ndarray:
    """Synthesizes calibrated noise profile."""
    rng = np.random.RandomState(seed + hash(noise_id) % 10000)
    t = np.linspace(0, n_samples / sr, n_samples, endpoint=False)

    if noise_id == "stat_white":
        noise = rng.randn(n_samples).astype(np.float32)

    elif noise_id == "stat_pink":
        raw = rng.randn(n_samples + 32).astype(np.float32)
        pink_filt = np.array([0.0499, 0.0905, 0.0805, 0.0632, 0.0469, 0.0336, 0.0240])
        noise = np.convolve(raw, pink_filt, mode="same")[:n_samples]

    elif noise_id == "stat_brown":
        white = rng.randn(n_samples).astype(np.float32)
        noise = np.cumsum(white).astype(np.float32)
        noise = noise - np.mean(noise)

    elif noise_id == "stat_tonal_hum":
        noise = (
            0.5 * np.sin(2 * np.pi * 50 * t) +
            0.3 * np.sin(2 * np.pi * 120 * t) +
            0.2 * np.sin(2 * np.pi * 240 * t)
        ).astype(np.float32)

    elif noise_id == "nonstat_engine_mod":
        # Swept engine RPM 120 -> 280 Hz
        rpm_freq = 120.0 + 80.0 * (1.0 + np.sin(2 * np.pi * 0.8 * t))
        phase = 2 * np.pi * np.cumsum(rpm_freq) / sr
        noise = (0.6 * np.sin(phase) + 0.4 * rng.randn(n_samples)).astype(np.float32)

    elif noise_id == "nonstat_rotor_mod":
        # Blade pass frequency 25 Hz modulation
        carrier = rng.randn(n_samples).astype(np.float32)
        mod = 0.5 * (1.0 + np.sin(2 * np.pi * 25.0 * t))
        noise = (carrier * mod).astype(np.float32)

    elif noise_id == "nonstat_siren_sweep":
        # Frequency chirp 600 Hz -> 1500 Hz
        siren_freq = 1000.0 + 400.0 * np.sin(2 * np.pi * 1.5 * t)
        phase = 2 * np.pi * np.cumsum(siren_freq) / sr
        noise = np.sin(phase).astype(np.float32)

    elif noise_id == "impulse_gunfire_burst":
        bg = rng.randn(n_samples).astype(np.float32) * 0.1
        # 3 isolated sharp gunfire transients (+40 dB spikes)
        for burst_time in [0.25, 0.55, 0.85]:
            idx = int(burst_time * sr)
            if idx < n_samples - 100:
                bg[idx : idx + 40] += np.hanning(40) * 8.0 * (1 if rng.rand() > 0.5 else -1)
        noise = bg.astype(np.float32)

    else:
        noise = rng.randn(n_samples).astype(np.float32)

    # Calibrate to unit RMS
    rms = np.sqrt(np.mean(noise ** 2)) + 1e-12
    return (noise / rms).astype(np.float32)


def generate_benchmark_matrix(
    output_csv: str = "data/benchmark_matrix/matrix_metadata.csv",
    sr: int = 16000,
    duration_s: float = 1.0,
) -> List[Dict[str, Any]]:
    """Builds and writes the complete 560-condition factorial matrix metadata."""
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = int(sr * duration_s)
    records = []
    clip_id = 0

    print(f"[GATE 4] Generating 560-Condition Synthetic Benchmark Matrix...", flush=True)

    for spk in SPEAKER_PROFILES:
        for noise in NOISE_PROFILES:
            for snr in SNR_LEVELS:
                clip_id += 1
                row = {
                    "clip_id": f"BM_{clip_id:04d}",
                    "speaker_id": spk["id"],
                    "gender": spk["gender"],
                    "pitch_hz": spk["pitch"],
                    "speech_type": spk["type"],
                    "noise_id": noise["id"],
                    "noise_family": noise["family"],
                    "noise_description": noise["desc"],
                    "snr_db": snr,
                    "sample_rate": sr,
                    "duration_s": duration_s,
                    "n_samples": n_samples,
                    "provenance": "synthetic_procedural_mathematical",
                }
                records.append(row)

    # Write CSV metadata
    fieldnames = list(records[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"[GATE 4] Generated {len(records)} benchmark definitions in {out_path}", flush=True)
    return records


def get_benchmark_pair(clip_row: Dict[str, Any], seed: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    On-the-fly deterministic generation of (clean_speech, primary_mic, reference_mic)
    for a specific benchmark condition row.
    """
    sr = int(clip_row["sample_rate"])
    n_samples = int(clip_row["n_samples"])
    snr_db = float(clip_row["snr_db"])

    clean = synthesize_speech_profile(clip_row["speaker_id"], n_samples=n_samples, sr=sr, seed=seed)
    noise_base = synthesize_noise_profile(clip_row["noise_id"], n_samples=n_samples, sr=sr, seed=seed)

    # Scale noise to achieve requested SNR: SNR = 10 log10(P_speech / P_noise)
    p_speech = np.mean(clean ** 2) + 1e-12
    p_target_noise = p_speech / (10.0 ** (snr_db / 10.0))
    noise_scaled = noise_base * np.sqrt(p_target_noise)

    # Primary mic has acoustic transmission
    h_acoustic = np.array([0.85, 0.35, 0.15], dtype=np.float32)
    h_acoustic /= np.sum(np.abs(h_acoustic))
    primary_noise = np.convolve(noise_scaled, h_acoustic, mode="same")
    primary_mic = clean + primary_noise

    # Reference mic has noise + slight nominal acoustic transmission delay
    ref_mic = np.roll(noise_scaled, 2)

    return clean.astype(np.float32), primary_mic.astype(np.float32), ref_mic.astype(np.float32)


if __name__ == "__main__":
    generate_benchmark_matrix()
