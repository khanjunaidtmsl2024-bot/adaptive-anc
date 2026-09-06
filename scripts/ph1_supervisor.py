"""
PH1 Training Supervisor — Frozen Experiment Runner (Phase 13).
PS 26052 — Adaptive Defence ANC.

Runs the PH1 controlled matrix SEQUENTIALLY on this CPU-only host:

    Model × input_mode
      TinyEnhancer V3 | CRN-Micro (129-bin contract geometry)
      RAW_PRIMARY     | NLMS_RESIDUAL

Every job uses the same frozen contract (frame=256, hop=128, Hann,
center=False, seed=42) — asserted against configs/ph1_experiment_contract.yaml
before anything runs — and identical data splits, so the only difference
between the two input legs is the AI input distribution (the RAW-vs-RESIDUAL
experimental question).

NOT covered here (documented blockers, fail-loud):
- DTLN: no waveform-level training loop exists in the repo (only an inference
  wrapper); training it requires a DTLN trainer, which is a separate work item.
- DFN2: not present in this repository (literature only).

Artifacts per job: checkpoint (.pt) + SHA-256 sidecar (.json, from train.py).
Matrix results: results/csv/ph1_training_matrix.csv + .json summary.

Usage:
    python scripts/ph1_supervisor.py            # full 4-job matrix
    python scripts/ph1_supervisor.py --only TinyEnhancer --mode NLMS_RESIDUAL
    python scripts/ph1_supervisor.py --max-epochs 30
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ai.train import train_spectral_mask_model
from src.ai.tiny_enhancer import TinyEnhancerNet
from src.ai.crn import CRNNet

CONTRACT_YAML = Path("configs/ph1_experiment_contract.yaml")
RESULTS_CSV = Path("results/csv/ph1_training_matrix.csv")
RESULTS_JSON = Path("results/csv/ph1_training_matrix.json")

# Frozen contract values (must match configs/ph1_experiment_contract.yaml).
CONTRACT = {"frame_size": 256, "hop_size": 128, "seed": 42}

# (name, model factory, epochs, lr)
MODELS: Dict[str, Dict[str, Any]] = {
    "TinyEnhancer": {"factory": TinyEnhancerNet, "epochs": 30, "lr": 1e-3},
    "CRN_Micro": {
        "factory": lambda: CRNNet(freq_bins=129, hidden_size=128, channels=(8, 16, 32, 64, 128)),
        "epochs": 20,
        "lr": 5e-4,
    },
}

INPUT_MODES = ["RAW_PRIMARY", "NLMS_RESIDUAL"]


def assert_contract() -> None:
    """Fail loudly if the frozen YAML disagrees with the values we run with."""
    if not CONTRACT_YAML.exists():
        raise RuntimeError(f"Frozen contract not found: {CONTRACT_YAML}")
    text = CONTRACT_YAML.read_text(encoding="utf-8")
    missing = []
    for key, value in CONTRACT.items():
        if f"{key}: {value}" not in text and f"{key}: {value}," not in text:
            missing.append(f"{key}={value}")
    if missing:
        raise RuntimeError(
            f"Contract mismatch: {CONTRACT_YAML} does not contain {missing}. "
            f"Refusing to run — no experiment may silently override the frozen contract."
        )


def load_history_rows() -> List[Dict[str, Any]]:
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    return []


def run_job(model_name: str, input_mode: str, max_epochs: int) -> Dict[str, Any]:
    spec = MODELS[model_name]
    epochs = min(spec["epochs"], max_epochs)
    save_path = f"checkpoints/ph1/{model_name.lower()}_{input_mode.lower()}.pt"
    print(f"\n{'=' * 70}\n  JOB: {model_name} / {input_mode} ({epochs} epochs)\n{'=' * 70}", flush=True)
    t0 = time.perf_counter()
    result = train_spectral_mask_model(
        model=spec["factory"](),
        epochs=epochs,
        lr=spec["lr"],
        save_path=save_path,
        seed=CONTRACT["seed"],
        frame_size=CONTRACT["frame_size"],
        hop_size=CONTRACT["hop_size"],
        input_mode=input_mode,
    )
    result["model"] = model_name
    result["input_mode"] = input_mode
    result["wall_time_s"] = round(time.perf_counter() - t0, 1)
    return result


def write_rows(rows: List[Dict[str, Any]]) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "input_mode", "status", "final_train_loss", "final_val_loss",
        "total_time_s", "wall_time_s", "checkpoint_path", "checkpoint_sha256",
        "reason",
    ]
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "contract": CONTRACT,
                "contract_yaml": str(CONTRACT_YAML.resolve()),
                "rows": rows,
            },
            f, indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=sorted(MODELS), default=None)
    parser.add_argument("--mode", choices=INPUT_MODES, default=None)
    parser.add_argument("--max-epochs", type=int, default=10**9)
    args = parser.parse_args()

    assert_contract()
    print(f"[SUPERVISOR] Contract asserted: {CONTRACT}", flush=True)
    print(f"[SUPERVISOR] Sequential execution on CPU host (max 1 concurrent job).", flush=True)

    jobs = [(m, im) for m in MODELS for im in INPUT_MODES]
    if args.only:
        jobs = [j for j in jobs if j[0] == args.only]
    if args.mode:
        jobs = [j for j in jobs if j[1] == args.mode]

    rows = load_history_rows()
    for model_name, input_mode in jobs:
        # Skip jobs already completed in a previous supervised run (idempotent).
        if any(r.get("model") == model_name and r.get("input_mode") == input_mode
               and r.get("status") == "COMPLETED" for r in rows):
            print(f"[SUPERVISOR] {model_name}/{input_mode} already COMPLETED; skipping.", flush=True)
            continue
        try:
            result = run_job(model_name, input_mode, args.max_epochs)
        except Exception as ex:
            print(f"[SUPERVISOR] JOB FAILED {model_name}/{input_mode}: {ex}", flush=True)
            result = {
                "model": model_name, "input_mode": input_mode,
                "status": "FAILED", "reason": str(ex),
            }
        rows = [r for r in rows if not (r.get("model") == model_name and r.get("input_mode") == input_mode)]
        rows.append(result)
        write_rows(rows)

    print("\n" + "=" * 70)
    print("  PH1 TRAINING MATRIX SUMMARY")
    print("=" * 70)
    for r in rows:
        if r.get("status") == "COMPLETED":
            print(f"  {r['model']:12s} {r['input_mode']:<14s} | Train: {r['final_train_loss']:.4f} "
                  f"| Val: {r['final_val_loss']:.4f} | {r['wall_time_s']}s "
                  f"| {r.get('checkpoint_sha256', '')[:12]}")
        else:
            print(f"  {r['model']:12s} {r['input_mode']:<14s} | {r.get('status')}: {r.get('reason', '')}")
    print(f"\n[SUPERVISOR] Matrix saved to {RESULTS_CSV.resolve()}", flush=True)


if __name__ == "__main__":
    main()