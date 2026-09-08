"""
PH1 Training Supervisor — Frozen Experiment Runner (Authoritative PH1 Reset).
PS 26052 — Adaptive Defence ANC.

Runs the authoritative PH1 controlled matrix SEQUENTIALLY on CPU host:

    Experiment | Model           | Input Mode
    -----------+-----------------+------------------
    E1         | TinyEnhancer V3 | RAW_PRIMARY
    E2         | TinyEnhancer V3 | NLMS_RESIDUAL
    E3         | CRN_Micro       | RAW_PRIMARY
    E4         | CRN_Micro       | NLMS_RESIDUAL

Every job strictly enforces configs/ph1_experiment_contract.yaml:
- STFT geometry: frame=256, hop=128, Hann, center=False, 16 kHz
- Seed: 42
- Optimizer: AdamW
- Learning rate: 1e-4
- Canonical manifest: data/v4/metadata/metadata_v4_extended.csv (302 rows)
- Early stopping / best model metric: val_si_sdr (on causal WOLA reconstructed waveform)
- Clean checkpoint directory: checkpoints/ph1_clean/
- Results directory: results/ph1_clean/
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ai.train import train_spectral_mask_model
from src.ai.tiny_enhancer import TinyEnhancerNet
from src.ai.crn import CRNNet

CONTRACT_YAML = Path("configs/ph1_experiment_contract.yaml")
RESULTS_DIR = Path("results/ph1_clean")
CHECKPOINTS_DIR = Path("checkpoints/ph1_clean")
RESULTS_CSV = RESULTS_DIR / "training_matrix.csv"
RESULTS_JSON = RESULTS_DIR / "training_matrix.json"


def load_and_audit_contract() -> Dict[str, Any]:
    """Parse YAML contract and assert strict consistency against repository."""
    if not CONTRACT_YAML.exists():
        raise RuntimeError(f"[FATAL] Frozen contract YAML not found: {CONTRACT_YAML}")

    with open(CONTRACT_YAML, "r", encoding="utf-8") as f:
        contract = yaml.safe_load(f)

    print("\n" + "=" * 78, flush=True)
    print("  PH1 CONTRACT AUDIT — SINGLE SOURCE OF TRUTH VERIFICATION", flush=True)
    print("=" * 78, flush=True)

    checks = [
        ("contract_version", contract.get("contract_version"), 1),
        ("sample_rate", contract.get("sample_rate"), 16000),
        ("frame_size", contract.get("frame_size"), 256),
        ("hop_size", contract.get("hop_size"), 128),
        ("window", contract.get("window"), "hann"),
        ("overlap", contract.get("overlap"), 0.5),
        ("center", contract.get("center"), False),
        ("ola_normalization", contract.get("ola_normalization"), "wola"),
        ("seed", contract.get("seed"), 42),
        ("optimizer", contract.get("training", {}).get("optimizer"), "adamw"),
        ("learning_rate", float(contract.get("training", {}).get("lr", 0)), 1e-4),
        ("early_stopping", contract.get("training", {}).get("early_stopping"), "val_si_sdr"),
        ("gradient_clipping", float(contract.get("training", {}).get("gradient_clipping", 0)), 5.0),
    ]

    manifest_rel = contract.get("dataset", {}).get("canonical_manifest")
    manifest_path = Path(manifest_rel) if manifest_rel else None
    manifest_exists = manifest_path.exists() if manifest_path else False
    checks.append(("canonical_manifest", manifest_path.as_posix() if manifest_path else None, "data/v4/metadata/metadata_v4_extended.csv"))
    checks.append(("manifest_exists", manifest_exists, True))

    all_passed = True
    print(f"  {'Field':<24} | {'YAML Value':<24} | {'Expected':<18} | {'Status':<6}", flush=True)
    print("  " + "-" * 74, flush=True)
    for name, actual, expected in checks:
        status = "PASS" if actual == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  {name:<24} | {str(actual):<24} | {str(expected):<18} | {status:<6}", flush=True)

    # Validate dataset rows and speaker separation
    if manifest_exists:
        with open(manifest_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        total_rows = len(rows)
        train_speakers = set(r["speaker_id"] for r in rows if r["split"] == "TRAIN")
        test_a_speakers = set(r["speaker_id"] for r in rows if r["split"] == "TEST_A_UNSEEN_SPEAKER")
        speaker_overlap = train_speakers & test_a_speakers
        print(f"  {'manifest_rows':<24} | {total_rows:<24} | 302                | {'PASS' if total_rows >= 300 else 'WARN'}", flush=True)
        print(f"  {'speaker_separation':<24} | overlap={len(speaker_overlap):<16} | overlap=0          | {'PASS' if len(speaker_overlap) == 0 else 'FAIL'}", flush=True)
        if len(speaker_overlap) > 0:
            all_passed = False

    print("=" * 78, flush=True)

    if not all_passed:
        raise RuntimeError("[FATAL] Contract audit failed. Silently proceeding with drifted contract is forbidden.")

    return contract


# (id, name, model factory, epochs, lr)
import torch
import subprocess

def _get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"

EXPERIMENTS = [
    {
        "id": "E1", "model": "TinyEnhancer", "input_mode": "RAW_PRIMARY",
        "factory": TinyEnhancerNet, "epochs": 30, "lr": 1e-4,
        "checkpoint": "E1_tinyenhancer_raw_primary.pt",
    },
    {
        "id": "E2", "model": "TinyEnhancer", "input_mode": "NLMS_RESIDUAL",
        "factory": TinyEnhancerNet, "epochs": 30, "lr": 1e-4,
        "checkpoint": "E2_tinyenhancer_nlms_residual.pt",
    },
    {
        "id": "E3", "model": "CRN_Micro", "input_mode": "RAW_PRIMARY",
        "factory": lambda: CRNNet(freq_bins=129, hidden_size=128, channels=(8, 16, 32, 64, 128)),
        "epochs": 20, "lr": 1e-4,
        "checkpoint": "E3_crn_raw_primary.pt",
    },
    {
        "id": "E4", "model": "CRN_Micro", "input_mode": "NLMS_RESIDUAL",
        "factory": lambda: CRNNet(freq_bins=129, hidden_size=128, channels=(8, 16, 32, 64, 128)),
        "epochs": 20, "lr": 1e-4,
        "checkpoint": "E4_crn_nlms_residual.pt",
    },
]


def run_single_job(exp: Dict[str, Any], max_epochs: Optional[int] = None) -> Dict[str, Any]:
    """Run a single experiment job enforcing the frozen contract and verifying reloadability."""
    epochs = min(exp["epochs"], max_epochs) if max_epochs else exp["epochs"]
    save_path = str(CHECKPOINTS_DIR / exp["checkpoint"])

    print(f"\n{'=' * 78}\n  LAUNCHING {exp['id']}: {exp['model']} / {exp['input_mode']} ({epochs} epochs, LR={exp['lr']})\n{'=' * 78}", flush=True)
    t0 = time.perf_counter()

    result = train_spectral_mask_model(
        model=exp["factory"](),
        metadata_csv="data/v4/metadata/metadata_v4_extended.csv",
        epochs=epochs,
        lr=exp["lr"],
        frame_size=256,
        hop_size=128,
        save_path=save_path,
        seed=42,
        input_mode=exp["input_mode"],
    )

    # Verification: strictly reload checkpoint
    print(f"[VERIFY] Strictly reloading checkpoint {save_path}...", flush=True)
    reload_model = exp["factory"]()
    state_dict = torch.load(save_path, map_location="cpu", weights_only=True)
    reload_model.load_state_dict(state_dict, strict=True)
    param_count = sum(p.numel() for p in reload_model.parameters())
    print(f"[VERIFY] Checkpoint successfully reloaded with strict=True. Params: {param_count:,d}", flush=True)

    result["experiment_id"] = exp["id"]
    result["model"] = exp["model"]
    result["input_mode"] = exp["input_mode"]
    result["parameters"] = param_count
    result["manifest"] = "data/v4/metadata/metadata_v4_extended.csv"
    result["seed"] = 42
    result["optimizer"] = "AdamW"
    result["learning_rate"] = exp["lr"]
    result["max_epochs"] = exp["epochs"]
    result["actual_epochs"] = len(result.get("history", {}).get("train_loss", []))
    result["git_commit"] = _get_git_commit()
    result["wall_time_s"] = round(time.perf_counter() - t0, 1)
    return result


def write_results(rows: List[Dict[str, Any]], contract: Dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_id", "model", "input_mode", "status", "parameters",
        "manifest", "seed", "optimizer", "learning_rate", "max_epochs",
        "actual_epochs", "best_epoch", "best_val_si_sdr", "final_train_loss",
        "final_val_loss", "final_val_si_sdr", "wall_time_s", "checkpoint_path",
        "checkpoint_sha256", "git_commit", "reason",
    ]
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "contract_version": contract.get("contract_version"),
                "canonical_manifest": "data/v4/metadata/metadata_v4_extended.csv",
                "git_commit": _get_git_commit(),
                "rows": rows,
            },
            f, indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["E1", "E2", "E3", "E4"], default=None,
                        help="Run only one specific experiment ID.")
    parser.add_argument("--smoke-e4", action="store_true",
                        help="Run 2-epoch smoke test for E4 (CRN_Micro + NLMS_RESIDUAL).")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Cap max epochs for quick debugging/smoke tests.")
    args = parser.parse_args()

    contract = load_and_audit_contract()
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke_e4:
        print("\n[SMOKE] Running E4 (CRN_Micro × NLMS_RESIDUAL) 2-epoch smoke test...", flush=True)
        e4_exp = [e for e in EXPERIMENTS if e["id"] == "E4"][0].copy()
        e4_exp["checkpoint"] = "smoke_E4_crn_nlms.pt"
        result = run_single_job(e4_exp, max_epochs=2)
        print("\n[SMOKE RESULT] E4 Smoke Test:", flush=True)
        print(f"  Status: {result.get('status')}")
        print(f"  Best Epoch: {result.get('best_epoch')}")
        print(f"  Best Val SI-SDR: {result.get('best_val_si_sdr'):+.2f} dB")
        print(f"  Final Val SI-SDR: {result.get('final_val_si_sdr'):+.2f} dB")
        print(f"  Checkpoint: {result.get('checkpoint_path')}")
        print(f"  SHA-256: {result.get('checkpoint_sha256')}")
        return

    jobs = EXPERIMENTS
    if args.only:
        jobs = [e for e in EXPERIMENTS if e["id"] == args.only]

    rows: List[Dict[str, Any]] = []
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    for exp in jobs:
        if any(r.get("experiment_id") == exp["id"] and r.get("status") == "COMPLETED" for r in rows):
            print(f"[SUPERVISOR] {exp['id']} ({exp['model']}/{exp['input_mode']}) already COMPLETED; skipping.", flush=True)
            continue

        try:
            res = run_single_job(exp, max_epochs=args.max_epochs)
        except Exception as ex:
            print(f"[SUPERVISOR] JOB FAILED {exp['id']}: {ex}", flush=True)
            res = {
                "experiment_id": exp["id"], "model": exp["model"], "input_mode": exp["input_mode"],
                "status": "FAILED", "reason": str(ex),
            }

        rows = [r for r in rows if r.get("experiment_id") != exp["id"]]
        rows.append(res)
        write_results(rows, contract)

    print("\n" + "=" * 78, flush=True)
    print("  PH1 AUTHORITATIVE CLEAN TRAINING MATRIX SUMMARY", flush=True)
    print("=" * 78, flush=True)
    for r in sorted(rows, key=lambda x: x.get("experiment_id", "")):
        if r.get("status") == "COMPLETED":
            print(f"  {r.get('experiment_id'):<4} | {r.get('model'):14s} | {r.get('input_mode'):<14s} | "
                  f"Train: {float(r.get('final_train_loss', 0)):.4f} | Val SI-SDR: {float(r.get('best_val_si_sdr', 0)):+.2f} dB | "
                  f"{r.get('wall_time_s')}s | {str(r.get('checkpoint_sha256', ''))[:12]}", flush=True)
        else:
            print(f"  {r.get('experiment_id'):<4} | {r.get('model'):14s} | {r.get('input_mode'):<14s} | {r.get('status')}: {r.get('reason', '')}", flush=True)
    print(f"\n[SUPERVISOR] Matrix saved to {RESULTS_CSV.resolve()}", flush=True)


if __name__ == "__main__":
    main()