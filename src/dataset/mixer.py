"""
Calibrated Acoustic SNR Mixer & Provenance Logger.
PS 26052 — Adaptive Defence ANC.
"""

from typing import Tuple, Dict, Any
import json
import csv
from pathlib import Path
import numpy as np


class AcousticMixer:
    """Mixes clean speech and noise at precisely calibrated target SNR (dB)."""

    @staticmethod
    def mix_at_snr(
        clean: np.ndarray,
        noise: np.ndarray,
        target_snr_db: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Mix clean speech and noise at specified target SNR.

        Returns:
            (mixed_signal, scaled_clean, scaled_noise, actual_snr_db)
        """
        length = min(len(clean), len(noise))
        c = clean[:length].astype(np.float64)
        n = noise[:length].astype(np.float64)

        # Active speech power
        clean_power = np.mean(c ** 2) + 1e-12
        noise_power = np.mean(n ** 2) + 1e-12

        # Required noise scale factor for target SNR: SNR = 10 * log10(P_clean / (alpha^2 * P_noise))
        target_ratio = 10.0 ** (target_snr_db / 10.0)
        alpha = np.sqrt(clean_power / (noise_power * target_ratio))

        scaled_noise = n * alpha
        mixed = c + scaled_noise

        # Prevent digital clipping
        peak = np.max(np.abs(mixed)) + 1e-12
        scale = 0.95 / peak if peak > 0.95 else 1.0

        mixed = (mixed * scale).astype(np.float32)
        c = (c * scale).astype(np.float32)
        scaled_noise = (scaled_noise * scale).astype(np.float32)

        actual_snr = 10.0 * np.log10(np.mean(c ** 2) / (np.mean(scaled_noise ** 2) + 1e-12))
        return mixed, c, scaled_noise, float(actual_snr)

    @staticmethod
    def save_provenance(
        manifest_path: str,
        records: list[Dict[str, Any]]
    ) -> None:
        """Save dataset records to CSV and JSON."""
        p = Path(manifest_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = p.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        # Save CSV
        if records:
            csv_path = p.with_suffix(".csv")
            keys = list(records[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(records)
