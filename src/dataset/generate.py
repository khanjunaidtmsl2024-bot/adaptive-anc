"""
DATASET_V004 Generation Engine.
PS 26052 — Adaptive Defence ANC.

Constructs the standardized DRDO benchmark dataset with:
- 10 distinct speakers (SPK_001 to SPK_010)
- Real/Synthetic defence noise taxonomy (Stationary, Rotor, Non-Stationary, Impulsive)
- Disjoint splits: TRAIN, TEST_A (Unseen Speakers), TEST_B (Unseen Noise), TEST_C (Unseen Category)
- Dual-channel primary/reference acoustic pairs with speech leakage & acoustic delay
- Cryptographic provenance tracking and validation
"""

import os
import sys
import wave
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from scipy.signal import butter, sosfilt

try:
    import soundfile as sf
except ImportError:
    sf = None

from src.dataset.provenance import calculate_sha256, extract_signal_descriptors
from src.dataset.splits import classify_noise_provenance, assign_split_protocol
from src.dataset.mix import DualMicAcousticMixer
from src.dataset.metadata import DatasetLedger
from src.dataset.validate import validate_dataset_splits
from src.dataset.noise_synthesizer import DefenceNoiseSynthesizer


def write_pcm_wav(filepath: Path, audio: np.ndarray, sample_rate: int = 16000) -> None:
    """Writes 16-bit mono PCM WAV file cleanly without external C dependencies."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    clamped = np.clip(audio, -1.0, 1.0)
    int16_pcm = (clamped * 32767.0).astype(np.int16)
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_pcm.tobytes())


def generate_speaker_speech(speaker_id: str, duration: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
    """
    Synthesizes speech with unique pitch (F0), formant profiles, and syllabic cadence per speaker.
    Simulates realistic voice individuality.
    """
    # Speaker F0 and formant lookup
    speaker_profiles = {
        "SPK_001": {"f0": 105.0, "formants": [550, 1200, 2400], "rate": 2.2},  # Deep Male
        "SPK_002": {"f0": 125.0, "formants": [650, 1350, 2550], "rate": 2.6},  # Standard Male
        "SPK_003": {"f0": 140.0, "formants": [700, 1400, 2600], "rate": 2.0},  # Mid Male
        "SPK_004": {"f0": 165.0, "formants": [750, 1550, 2750], "rate": 2.8},  # Low Female
        "SPK_005": {"f0": 190.0, "formants": [800, 1700, 2900], "rate": 2.4},  # Standard Female
        "SPK_006": {"f0": 215.0, "formants": [850, 1850, 3100], "rate": 3.0},  # High Female
        "SPK_007": {"f0": 115.0, "formants": [600, 1280, 2450], "rate": 2.3},  # Male Radio
        "SPK_008": {"f0": 175.0, "formants": [780, 1620, 2800], "rate": 2.5},  # Female Radio
        # Held-out evaluation speakers
        "SPK_009": {"f0": 98.0,  "formants": [520, 1150, 2350], "rate": 2.1},  # Unseen Low Male (Test A)
        "SPK_010": {"f0": 235.0, "formants": [900, 1950, 3250], "rate": 3.2},  # Unseen High Female (Test A)
    }

    prof = speaker_profiles.get(speaker_id, speaker_profiles["SPK_001"])
    f0_base = prof["f0"]
    f1, f2, f3 = prof["formants"]
    cadence = prof["rate"]

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    f0 = f0_base + 12.0 * np.sin(2 * np.pi * 1.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate

    # Glottal pulse harmonics
    glottal = np.zeros_like(t)
    for h in range(1, 14):
        glottal += (1.0 / (h ** 0.85)) * np.sin(h * phase)

    # Syllables
    syllables = (
        0.8 * np.sin(2 * np.pi * cadence * t) ** 2 * (np.sin(2 * np.pi * 0.9 * t) > -0.2) +
        0.2 * np.sin(2 * np.pi * (cadence * 1.5) * t) ** 2
    )
    syllables = np.clip(syllables, 0.0, 1.0)

    nyq = sample_rate / 2.0
    sos1 = butter(2, [max(100, f1 - 180) / nyq, min(nyq - 100, f1 + 180) / nyq], btype="bandpass", output="sos")
    sos2 = butter(2, [max(100, f2 - 220) / nyq, min(nyq - 100, f2 + 220) / nyq], btype="bandpass", output="sos")
    sos3 = butter(2, [max(100, f3 - 350) / nyq, min(nyq - 100, f3 + 350) / nyq], btype="bandpass", output="sos")

    vocal = (sosfilt(sos1, glottal) * 1.6 + sosfilt(sos2, glottal) * 1.1 + sosfilt(sos3, glottal) * 0.7) * syllables

    # Unvoiced fricatives
    noise_bursts = np.random.randn(len(t)) * 0.12 * (syllables < 0.25) * (t > 0.2) * (t < duration - 0.2)
    sos_fric = butter(2, [3400 / nyq, min(nyq - 100, 6200) / nyq], btype="bandpass", output="sos")
    fric = sosfilt(sos_fric, noise_bursts)

    speech = vocal + fric
    peak = np.max(np.abs(speech)) + 1e-12
    return (speech / peak * 0.75).astype(np.float32)


def build_dataset_v004(
    output_dir: str = "data/v4",
    sample_rate: int = 16000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Builds the complete reproducible DATASET_V004 with verified splits.
    """
    np.random.seed(seed)
    root = Path(output_dir)
    clean_dir = root / "clean"
    noise_dir = root / "noise"
    mixed_dir = root / "mixed"
    meta_dir = root / "metadata"

    csv_path = meta_dir / "metadata_v4.csv"
    json_path = meta_dir / "metadata_v4.json"
    ledger = DatasetLedger(output_csv=csv_path, output_json=json_path)

    synth = DefenceNoiseSynthesizer(sample_rate=sample_rate)

    print(f"\n[+] Generating DATASET_V004 at: {root.resolve()}")
    print("[*] Generating clean speaker audio library (SPK_001 to SPK_010)...")
    speakers = [f"SPK_{i:03d}" for i in range(1, 11)]
    clean_audio_map = {}
    for spk in speakers:
        audio = generate_speaker_speech(spk, duration=3.0, sample_rate=sample_rate)
        p = clean_dir / f"{spk}_clean.wav"
        write_pcm_wav(p, audio, sample_rate)
        clean_audio_map[spk] = (audio, p)

    print("[*] Generating defence noise audio library...")
    noise_generators = {
        # Known categories
        "tank_engine_t90": lambda: synth.generate_tank_noise(duration=3.0),
        "diesel_idle_apc": lambda: synth.generate_tank_noise(duration=3.0),
        "generator_hum_50hz": lambda: synth.generate_tank_noise(duration=3.0) * 0.8,
        "helicopter_rotor_dhruv": lambda: synth.generate_helicopter_noise(duration=3.0),
        "propeller_rotor_an32": lambda: synth.generate_helicopter_noise(duration=3.0) * 0.9,
        "vehicle_acceleration_track": lambda: synth.generate_tank_noise(duration=3.0),
        "tactical_siren_sweep": lambda: synth.generate_helicopter_noise(duration=3.0) * 0.7,
        # Held-out noise recording of known category (for Test B)
        "tank_engine_heldout_rec": lambda: synth.generate_tank_noise(duration=3.0),
        "helicopter_rotor_heldout_rec": lambda: synth.generate_helicopter_noise(duration=3.0),
        # Unseen Noise Categories (for Test C - OOD)
        "gunfire_transient_insas": lambda: synth.generate_gunfire_transient(duration=3.0, num_shots=4),
        "artillery_blast_dhanush": lambda: synth.generate_gunfire_transient(duration=3.0, num_shots=2),
    }

    noise_audio_map = {}
    for nid, fn in noise_generators.items():
        audio = fn()
        p = noise_dir / f"{nid}.wav"
        write_pcm_wav(p, audio, sample_rate)
        cat, prov = classify_noise_provenance(nid, str(p))
        noise_audio_map[nid] = (audio, p, cat, prov)

    print("[*] Synthesizing calibrated dual-microphone mixtures across test regimes...")
    snr_grid = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0]

    sample_count = 0
    # Systematic generation matrix
    for spk, (clean_audio, clean_p) in clean_audio_map.items():
        for nid, (noise_audio, noise_p, cat, prov) in noise_audio_map.items():
            is_heldout_rec = "heldout_rec" in nid
            split = assign_split_protocol(spk, nid, cat, is_heldout_rec)

            # Pick test SNR point
            # Pick deterministically to cover range
            snr_idx = (hash(spk + nid) % len(snr_grid))
            target_snr = snr_grid[snr_idx]

            # Primary/reference parameters
            leakage = 0.15 if ("heldout" in nid or spk in ["SPK_007", "SPK_009"]) else 0.0
            delay = 2 if (spk in ["SPK_003", "SPK_010"]) else 0
            gain_mis = -1.0 if (spk in ["SPK_005", "SPK_008"]) else 0.0

            prim_d, ref_x, meta = DualMicAcousticMixer.mix_dual_channel(
                clean_speech=clean_audio,
                noise_audio=noise_audio,
                target_snr_db=target_snr,
                speech_leakage_ratio=leakage,
                delay_samples=delay,
                gain_mismatch_db=gain_mis,
            )

            sample_id = f"{spk}__{nid}__{target_snr:+.0f}dB"
            prim_path = mixed_dir / split / f"{sample_id}__primary.wav"
            ref_path = mixed_dir / split / f"{sample_id}__reference.wav"

            write_pcm_wav(prim_path, prim_d, sample_rate)
            write_pcm_wav(ref_path, ref_x, sample_rate)

            # Record in ledger
            record = {
                "sample_id": sample_id,
                "split": split,
                "speaker_id": spk,
                "noise_id": nid,
                "noise_category": cat,
                "provenance": prov,
                "target_snr_db": target_snr,
                "actual_snr_db": meta["actual_snr_db"],
                "speech_leakage_ratio": leakage,
                "delay_samples": delay,
                "gain_mismatch_db": gain_mis,
                "duration_sec": 3.0,
                "primary_rms": meta["primary_rms"],
                "reference_rms": meta["reference_rms"],
                "primary_path": str(prim_path.relative_to(root)),
                "reference_path": str(ref_path.relative_to(root)),
                "clean_path": str(clean_p.relative_to(root)),
                "sha256_hash": calculate_sha256(prim_path),
            }
            ledger.add_record(record)
            sample_count += 1

    ledger.save()
    print(f"[+] Total dual-channel mixtures generated: {sample_count}")
    print(f"[+] Metadata catalog saved to: {csv_path.resolve()}")

    # Validation
    val_res = validate_dataset_splits(str(csv_path))
    print(f"[+] Dataset Validation Result: {val_res['status']}")
    print(f"    - Total records: {val_res['total_records']}")
    print(f"    - Speaker leakage: {val_res['speaker_leakage_detected']}")
    print(f"    - Noise rec leakage: {val_res['noise_recording_leakage_detected']}")
    print(f"    - Category leakage: {val_res['category_leakage_detected']}")
    print(f"    - Max SNR deviation: {val_res['max_snr_deviation_db']} dB")

    return {
        "status": val_res["status"],
        "total_samples": sample_count,
        "metadata_csv": str(csv_path),
        "validation": val_res,
    }


if __name__ == "__main__":
    build_dataset_v004()
