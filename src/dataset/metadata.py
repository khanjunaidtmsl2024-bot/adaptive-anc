"""
Dataset Metadata & Provenance Ledger.
PS 26052 — Adaptive Defence ANC.

Generates and maintains the standardized CSV/JSON metadata records
for every mixture in DATASET_V004.
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Any


FIELDNAMES = [
    "sample_id",
    "split",
    "speaker_id",
    "noise_id",
    "noise_category",
    "provenance",
    "target_snr_db",
    "actual_snr_db",
    "speech_leakage_ratio",
    "delay_samples",
    "gain_mismatch_db",
    "duration_sec",
    "primary_rms",
    "reference_rms",
    "primary_path",
    "reference_path",
    "clean_path",
    "sha256_hash",
]


class DatasetLedger:
    """Manages appending and exporting dataset provenance records."""

    def __init__(self, output_csv: Path, output_json: Path):
        self.output_csv = Path(output_csv)
        self.output_json = Path(output_json)
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record: Dict[str, Any]) -> None:
        """Adds a single record to the ledger."""
        self.records.append(record)

    def save(self) -> None:
        """Flushes records to CSV and JSON on disk."""
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.output_json.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            for r in self.records:
                filtered = {k: r.get(k, "") for k in FIELDNAMES}
                writer.writerow(filtered)

        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)
