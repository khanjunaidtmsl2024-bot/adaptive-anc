"""
PH1 — Canonical Manifest Freeze.
PS 26052 — Adaptive Defence ANC.

Combines the untouched V4 base manifest (110 rows) with the deterministic
TEST_B extension (192 rows) into ONE canonical manifest, then records
SHA-256 of the manifest and of EVERY referenced audio file.

Outputs:
    data/v4/metadata/metadata_v4_extended.csv      (canonical, 302 rows)
    experiments/manifests/ph1_frozen_canonical_manifest.json

Run whenever either source manifest changes; the JSON records what was frozen.
"""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "v4"
BASE = DATA_ROOT / "metadata" / "metadata_v4.csv"
EXT = DATA_ROOT / "metadata" / "metadata_v4_extension.csv"
CANON = DATA_ROOT / "metadata" / "metadata_v4_extended.csv"
FREEZE_JSON = ROOT / "experiments" / "manifests" / "ph1_frozen_canonical_manifest.json"

BASE_SHA_EXPECTED = "83a84a08797b61e958bc92521215ebd64a4246275448c4c86a347966f5d78ff3"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    base_sha = sha256_file(BASE)
    assert base_sha == BASE_SHA_EXPECTED, (
        f"Base manifest changed unexpectedly! sha={base_sha}. "
        f"TEST_A/TEST_C/TEST splits must never be regenerated silently."
    )

    with open(BASE, encoding="utf-8") as f:
        base_rows = list(csv.DictReader(f))
    with open(EXT, encoding="utf-8") as f:
        ext_rows = list(csv.DictReader(f))

    # Canonical fieldnames = base columns + any extension provenance extras;
    # base rows are padded with empty values for the extra columns.
    ext_cols = [c for c in ext_rows[0].keys() if c not in base_rows[0]]
    fieldnames = list(base_rows[0].keys()) + ext_cols
    with open(CANON, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(base_rows)
        writer.writerows(ext_rows)

    # Hash every referenced audio file (primary, reference, clean)
    audio_hashes = {}
    missing = []
    for r in base_rows + ext_rows:
        for col in ("primary_path", "reference_path", "clean_path"):
            rel = r[col].replace("\\", "/")
            p = DATA_ROOT / rel
            if p.exists():
                audio_hashes[rel] = sha256_file(p)
            else:
                missing.append((r["sample_id"], rel))

    splits = dict(Counter(r["split"] for r in base_rows + ext_rows))
    classes = dict(Counter(r["noise_id"] for r in ext_rows))

    freeze = {
        "canonical_manifest": "data/v4/metadata/metadata_v4_extended.csv",
        "canonical_manifest_sha256": sha256_file(CANON),
        "base_manifest": "data/v4/metadata/metadata_v4.csv",
        "base_manifest_sha256": base_sha,
        "base_manifest_expected_sha256": BASE_SHA_EXPECTED,
        "extension_manifest": "data/v4/metadata/metadata_v4_extension.csv",
        "extension_manifest_sha256": sha256_file(EXT),
        "extension_generator": "experiments/scripts/ph1_testb_extension.py",
        "total_rows": len(base_rows) + len(ext_rows),
        "splits": splits,
        "test_b_noise_classes": classes,
        "provenance": "PROCEDURAL / SYNTHETIC DEFENCE-NOISE — NOT real battlefield data",
        "audio_sha256": audio_hashes,
        "missing_files": missing,
    }
    FREEZE_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(FREEZE_JSON, "w", encoding="utf-8") as f:
        json.dump(freeze, f, indent=2)

    print(f"canonical: {CANON} ({freeze['total_rows']} rows)")
    print(f"splits: {splits}")
    print(f"extension classes: {len(classes)} -> {sorted(classes)}")
    print(f"audio files hashed: {len(audio_hashes)} | missing: {len(missing)}")
    print(f"frozen: {FREEZE_JSON}")
    print(f"canonical sha256: {freeze['canonical_manifest_sha256']}")


if __name__ == "__main__":
    main()
