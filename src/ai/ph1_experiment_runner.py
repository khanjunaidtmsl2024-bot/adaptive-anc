"""
PH1 MASTER EXPERIMENT RUNNER
SIH26052 Adaptive AI/ML ANC — Controlled Multi-Model × Multi-Input Benchmark

This script executes the full PH1 experiment matrix:

  EXPERIMENT MATRIX (6 cells):
  ┌──────────────────┬──────────────┬──────────────────┐
  │ Model            │ RAW_PRIMARY  │ NLMS_RESIDUAL    │
  ├──────────────────┼──────────────┼──────────────────┤
  │ TinyEnhancer     │ E1           │ E2               │
  │ CRN-Micro        │ E3           │ E4               │
  │ DTLN             │ E5           │ E6               │
  └──────────────────┴──────────────┴──────────────────┘

  + BASELINES (no AI):
  ┌──────────────────┬──────────────────────────────────┐
  │ B0               │ Noisy (unprocessed primary mic)   │
  │ B1               │ NLMS-only (VSS-NLMS residual)    │
  └──────────────────┴──────────────────────────────────┘

For each trained model:
  1. Train for 30 epochs (TinyEnhancer/DTLN) or 20 epochs (CRN)
  2. Evaluate on TEST_A (unseen speakers), TEST_B (unseen noise recs),
     TEST_C (unseen noise category) using REAL PESQ + STOI + SI-SDR + SNR
  3. Save checkpoint + JSON sidecar with full provenance
  4. Output per-clip CSV + summary results

CANONICAL CONTRACT: frame=256, hop=128, 16kHz, hann, center=False, seed=42
"""

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[FATAL] PyTorch not available. Cannot run experiments.", flush=True)
    sys.exit(1)

import soundfile as sf

from src.ai.tiny_enhancer import TinyEnhancerNet
from src.ai.crn import CRNNet
from src.ai.dtln import DTLNNet
from src.ai.train import train_spectral_mask_model, DatasetLoader, _prepare_residual_inputs, _causal_istft
from src.ai.train_dtln import train_dtln_model
from src.dsp.vss_nlms import VSSNLMSFilter
from src.evaluation.metrics import (
    compute_snr, compute_si_snr, compute_stoi, compute_pesq,
    get_metric_provenance,
)

# ============================================================
# CONFIGURATION — frozen from ph1_experiment_contract.yaml
# ============================================================
SAMPLE_RATE = 16000
FRAME_SIZE = 256
HOP_SIZE = 128
SEED = 42
METADATA_CSV = "data/v4/metadata/metadata_v4.csv"

RESULTS_DIR = Path("results/ph1")
CHECKPOINT_DIR = Path("checkpoints/ph1")

# Experiment definitions
EXPERIMENTS = [
    {
        "id": "E1", "model": "TinyEnhancer", "input_mode": "RAW_PRIMARY",
        "epochs": 30, "lr": 1e-3, "type": "spectral_mask",
    },
    {
        "id": "E2", "model": "TinyEnhancer", "input_mode": "NLMS_RESIDUAL",
        "epochs": 30, "lr": 1e-3, "type": "spectral_mask",
    },
    {
        "id": "E3", "model": "CRN", "input_mode": "RAW_PRIMARY",
        "epochs": 20, "lr": 5e-4, "type": "spectral_mask",
    },
    {
        "id": "E4", "model": "CRN", "input_mode": "NLMS_RESIDUAL",
        "epochs": 20, "lr": 5e-4, "type": "spectral_mask",
    },
    {
        "id": "E5", "model": "DTLN", "input_mode": "RAW_PRIMARY",
        "epochs": 30, "lr": 1e-4, "type": "waveform",
    },
    {
        "id": "E6", "model": "DTLN", "input_mode": "NLMS_RESIDUAL",
        "epochs": 30, "lr": 1e-4, "type": "waveform",
    },
]

EVAL_SPLITS = ["TEST_A_UNSEEN_SPEAKER", "TEST_B_UNSEEN_NOISE_REC", "TEST_C_UNSEEN_NOISE_CATEGORY"]


def _create_model(model_name: str) -> torch.nn.Module:
    """Instantiate a fresh model with deterministic init."""
    torch.manual_seed(SEED)
    if model_name == "TinyEnhancer":
        return TinyEnhancerNet()
    elif model_name == "CRN":
        return CRNNet(freq_bins=FRAME_SIZE // 2 + 1, hidden_size=128, channels=(8, 16, 32, 64, 128))
    elif model_name == "DTLN":
        return DTLNNet(frame_size=FRAME_SIZE, hop_size=HOP_SIZE, hidden_size=128, encoder_size=256)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _load_checkpoint(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """Load checkpoint with strict verification."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _enhance_spectral_mask(
    model: torch.nn.Module,
    noisy_wav: np.ndarray,
) -> np.ndarray:
    """Run spectral-mask model (TinyEnhancer/CRN) on a waveform."""
    window = torch.hann_window(FRAME_SIZE)
    noisy_t = torch.from_numpy(noisy_wav).float()

    noisy_stft = torch.stft(noisy_t, FRAME_SIZE, HOP_SIZE, window=window,
                            return_complex=True, center=False)
    noisy_mag = noisy_stft.abs()
    noisy_phase = noisy_stft.angle()

    noisy_input = noisy_mag.unsqueeze(0).unsqueeze(0)  # (1, 1, F, T)
    with torch.no_grad():
        pred_mask = model(noisy_input)

    if pred_mask.dim() == 4:
        pred_mask = pred_mask.squeeze(0).squeeze(0)
    elif pred_mask.dim() == 3:
        pred_mask = pred_mask.squeeze(0)

    f_min = min(pred_mask.shape[0], noisy_mag.shape[0])
    t_min = min(pred_mask.shape[1], noisy_mag.shape[1])
    pred_mask = pred_mask[:f_min, :t_min]

    enhanced_mag = noisy_mag[:f_min, :t_min] * pred_mask
    enhanced_phase = noisy_phase[:f_min, :t_min]
    enhanced_stft = enhanced_mag * torch.exp(1j * enhanced_phase)

    full_stft = torch.zeros_like(noisy_stft)
    full_stft[:f_min, :t_min] = enhanced_stft

    enhanced_wav = _causal_istft(full_stft, FRAME_SIZE, HOP_SIZE, window, length=len(noisy_wav))
    return enhanced_wav.numpy()


def _enhance_waveform(
    model: torch.nn.Module,
    noisy_wav: np.ndarray,
) -> np.ndarray:
    """Run waveform model (DTLN) on a waveform."""
    noisy_t = torch.from_numpy(noisy_wav).float().unsqueeze(0).unsqueeze(0)  # (1, 1, T)
    with torch.no_grad():
        enhanced = model(noisy_t).squeeze(0).squeeze(0)  # (T,)
    out = enhanced.numpy()
    # Align length
    if len(out) > len(noisy_wav):
        out = out[:len(noisy_wav)]
    elif len(out) < len(noisy_wav):
        out = np.pad(out, (0, len(noisy_wav) - len(out)))
    return out


def evaluate_model(
    model: torch.nn.Module,
    model_type: str,
    exp_id: str,
    input_mode: str,
    eval_split: str,
) -> List[Dict[str, Any]]:
    """Evaluate a trained model on a held-out split. Returns per-clip metrics."""
    meta_path = Path(METADATA_CSV)
    data_root = meta_path.parent.parent

    loader = DatasetLoader(str(meta_path), str(data_root), split_filter=eval_split)
    if len(loader) == 0:
        print(f"  [WARN] No samples for split={eval_split}, skipping.", flush=True)
        return []

    # Precompute residuals if needed
    residuals = None
    if input_mode == "NLMS_RESIDUAL":
        residuals = _prepare_residual_inputs(loader)

    model.eval()
    results = []

    for i in range(len(loader)):
        clean, primary, reference, row = loader[i]

        if input_mode == "NLMS_RESIDUAL":
            noisy_wav = residuals[i]
        else:
            noisy_wav = primary

        # Enhance
        if model_type == "spectral_mask":
            enhanced = _enhance_spectral_mask(model, noisy_wav)
        else:
            enhanced = _enhance_waveform(model, noisy_wav)

        # Clip for safety
        enhanced = np.clip(enhanced, -1.0, 1.0)

        # Compute metrics
        min_len = min(len(clean), len(enhanced))
        c = clean[:min_len]
        e = enhanced[:min_len]

        try:
            snr = compute_snr(c, e)
        except Exception:
            snr = float("nan")
        try:
            si_sdr = compute_si_snr(c, e)
        except Exception:
            si_sdr = float("nan")
        try:
            stoi_val = compute_stoi(c, e, SAMPLE_RATE)
        except Exception:
            stoi_val = float("nan")
        try:
            pesq_val = compute_pesq(c, e, SAMPLE_RATE)
        except Exception:
            pesq_val = float("nan")

        results.append({
            "experiment_id": exp_id,
            "model": row.get("model", exp_id),
            "input_mode": input_mode,
            "eval_split": eval_split,
            "sample_id": row.get("sample_id", f"sample_{i}"),
            "speaker_id": row.get("speaker_id", ""),
            "noise_id": row.get("noise_id", ""),
            "noise_category": row.get("noise_category", ""),
            "target_snr_db": row.get("target_snr_db", ""),
            "snr_db": round(snr, 4),
            "si_sdr_db": round(si_sdr, 4),
            "stoi": round(stoi_val, 4),
            "pesq": round(pesq_val, 4),
        })

    return results


def evaluate_baselines(eval_split: str) -> List[Dict[str, Any]]:
    """Evaluate B0 (noisy) and B1 (NLMS-only) baselines."""
    meta_path = Path(METADATA_CSV)
    data_root = meta_path.parent.parent
    loader = DatasetLoader(str(meta_path), str(data_root), split_filter=eval_split)
    if len(loader) == 0:
        return []

    residuals = _prepare_residual_inputs(loader)
    results = []

    for i in range(len(loader)):
        clean, primary, reference, row = loader[i]
        sample_id = row.get("sample_id", f"sample_{i}")

        for baseline_id, baseline_name, wav in [
            ("B0", "Noisy", primary),
            ("B1", "NLMS_Only", residuals[i]),
        ]:
            min_len = min(len(clean), len(wav))
            c = clean[:min_len]
            e = wav[:min_len]

            try:
                snr = compute_snr(c, e)
            except Exception:
                snr = float("nan")
            try:
                si_sdr = compute_si_snr(c, e)
            except Exception:
                si_sdr = float("nan")
            try:
                stoi_val = compute_stoi(c, e, SAMPLE_RATE)
            except Exception:
                stoi_val = float("nan")
            try:
                pesq_val = compute_pesq(c, e, SAMPLE_RATE)
            except Exception:
                pesq_val = float("nan")

            results.append({
                "experiment_id": baseline_id,
                "model": baseline_name,
                "input_mode": "N/A",
                "eval_split": eval_split,
                "sample_id": sample_id,
                "speaker_id": row.get("speaker_id", ""),
                "noise_id": row.get("noise_id", ""),
                "noise_category": row.get("noise_category", ""),
                "target_snr_db": row.get("target_snr_db", ""),
                "snr_db": round(snr, 4),
                "si_sdr_db": round(si_sdr, 4),
                "stoi": round(stoi_val, 4),
                "pesq": round(pesq_val, 4),
            })

    return results


def write_csv(rows: List[Dict], path: Path):
    """Write results to CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [CSV] {path} ({len(rows)} rows)", flush=True)


def summarize_results(all_results: List[Dict]) -> str:
    """Generate a markdown summary table grouped by experiment × split."""
    from collections import defaultdict

    groups = defaultdict(list)
    for r in all_results:
        key = (r["experiment_id"], r["model"], r["input_mode"], r["eval_split"])
        groups[key].append(r)

    lines = []
    lines.append("# PH1 Experiment Results Summary")
    lines.append(f"\nGenerated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"\nMetric provenance: {json.dumps(get_metric_provenance(), indent=2)}")
    lines.append("")
    lines.append("| Exp | Model | Input | Split | N | SNR (dB) | SI-SDR (dB) | STOI | PESQ |")
    lines.append("|-----|-------|-------|-------|---|----------|-------------|------|------|")

    for key in sorted(groups.keys()):
        rows = groups[key]
        exp_id, model, input_mode, split = key
        n = len(rows)
        snr_vals = [r["snr_db"] for r in rows if not np.isnan(r["snr_db"])]
        si_vals = [r["si_sdr_db"] for r in rows if not np.isnan(r["si_sdr_db"])]
        stoi_vals = [r["stoi"] for r in rows if not np.isnan(r["stoi"])]
        pesq_vals = [r["pesq"] for r in rows if not np.isnan(r["pesq"])]

        snr_mean = f"{np.mean(snr_vals):.2f}" if snr_vals else "N/A"
        si_mean = f"{np.mean(si_vals):.2f}" if si_vals else "N/A"
        stoi_mean = f"{np.mean(stoi_vals):.4f}" if stoi_vals else "N/A"
        pesq_mean = f"{np.mean(pesq_vals):.2f}" if pesq_vals else "N/A"

        # Shorten split name
        split_short = split.replace("TEST_", "").replace("UNSEEN_", "")
        lines.append(f"| {exp_id} | {model} | {input_mode} | {split_short} | {n} | {snr_mean} | {si_mean} | {stoi_mean} | {pesq_mean} |")

    return "\n".join(lines)


def main():
    """Execute the full PH1 experiment campaign."""
    t_start = time.perf_counter()

    print("=" * 70, flush=True)
    print("  PH1 MASTER EXPERIMENT CAMPAIGN", flush=True)
    print("  SIH26052 — Adaptive AI/ML ANC", flush=True)
    print("  6 experiments + 2 baselines × 3 eval splits", flush=True)
    print("=" * 70, flush=True)
    print(f"  Contract: frame={FRAME_SIZE}, hop={HOP_SIZE}, sr={SAMPLE_RATE}, "
          f"window=hann, center=False", flush=True)
    print(f"  Seed: {SEED}", flush=True)
    print(f"  Torch: {torch.__version__}", flush=True)
    print(f"  Device: cpu", flush=True)
    print("=" * 70, flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    training_summary = {}

    # ========================================
    # PHASE 1: TRAIN ALL EXPERIMENTS
    # ========================================
    for exp in EXPERIMENTS:
        exp_id = exp["id"]
        model_name = exp["model"]
        input_mode = exp["input_mode"]
        epochs = exp["epochs"]
        lr = exp["lr"]
        exp_type = exp["type"]

        print(f"\n{'=' * 70}", flush=True)
        print(f"  [{exp_id}] Training {model_name} × {input_mode}", flush=True)
        print(f"  Epochs: {epochs}, LR: {lr}, Type: {exp_type}", flush=True)
        print(f"{'=' * 70}", flush=True)

        ckpt_path = str(CHECKPOINT_DIR / f"{exp_id}_{model_name.lower()}_{input_mode.lower()}.pt")
        model = _create_model(model_name)

        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {param_count:,d}", flush=True)

        if exp_type == "spectral_mask":
            result = train_spectral_mask_model(
                model=model,
                metadata_csv=METADATA_CSV,
                epochs=epochs,
                lr=lr,
                frame_size=FRAME_SIZE,
                hop_size=HOP_SIZE,
                device="cpu",
                save_path=ckpt_path,
                seed=SEED,
                input_mode=input_mode,
            )
        else:  # waveform (DTLN)
            result = train_dtln_model(
                model=model,
                metadata_csv=METADATA_CSV,
                epochs=epochs,
                lr=lr,
                device="cpu",
                save_path=ckpt_path,
                seed=SEED,
                input_mode=input_mode,
            )

        training_summary[exp_id] = {
            "model": model_name,
            "input_mode": input_mode,
            "status": result["status"],
            "final_train_loss": result.get("final_train_loss"),
            "final_val_loss": result.get("final_val_loss"),
            "total_time_s": result.get("total_time_s"),
            "checkpoint_sha256": result.get("checkpoint_sha256"),
            "param_count": param_count,
        }

        if result["status"] != "COMPLETED":
            print(f"  [!] {exp_id} FAILED: {result.get('reason', 'unknown')}", flush=True)
            continue

        print(f"  [{exp_id}] Training complete: train_loss={result['final_train_loss']:.4f}, "
              f"val_loss={result['final_val_loss']:.4f}", flush=True)

    # ========================================
    # PHASE 2: EVALUATE ALL ON HELD-OUT SETS
    # ========================================
    print(f"\n{'=' * 70}", flush=True)
    print("  PHASE 2: HELD-OUT EVALUATION", flush=True)
    print(f"{'=' * 70}", flush=True)

    # Baselines first
    for split in EVAL_SPLITS:
        print(f"\n  Evaluating baselines on {split}...", flush=True)
        baseline_results = evaluate_baselines(split)
        all_results.extend(baseline_results)
        print(f"    B0+B1: {len(baseline_results)} rows", flush=True)

    # Trained models
    for exp in EXPERIMENTS:
        exp_id = exp["id"]
        model_name = exp["model"]
        input_mode = exp["input_mode"]
        exp_type = exp["type"]

        ckpt_path = str(CHECKPOINT_DIR / f"{exp_id}_{model_name.lower()}_{input_mode.lower()}.pt")
        if not Path(ckpt_path).exists():
            print(f"  [{exp_id}] Checkpoint not found, skipping evaluation.", flush=True)
            continue

        print(f"\n  Loading {exp_id} checkpoint for evaluation...", flush=True)
        model = _create_model(model_name)
        model = _load_checkpoint(model, ckpt_path)

        for split in EVAL_SPLITS:
            print(f"  [{exp_id}] Evaluating on {split}...", flush=True)
            exp_results = evaluate_model(
                model=model,
                model_type=exp_type,
                exp_id=exp_id,
                input_mode=input_mode,
                eval_split=split,
            )
            # Tag results with model name
            for r in exp_results:
                r["model"] = model_name
            all_results.extend(exp_results)
            if exp_results:
                si_mean = np.mean([r["si_sdr_db"] for r in exp_results
                                   if not np.isnan(r["si_sdr_db"])])
                print(f"    {len(exp_results)} clips, mean SI-SDR: {si_mean:.2f} dB", flush=True)

    # ========================================
    # PHASE 3: SAVE RESULTS
    # ========================================
    print(f"\n{'=' * 70}", flush=True)
    print("  PHASE 3: SAVING RESULTS", flush=True)
    print(f"{'=' * 70}", flush=True)

    # Per-clip CSV
    csv_path = RESULTS_DIR / "ph1_per_clip_results.csv"
    write_csv(all_results, csv_path)

    # Summary markdown
    summary_md = summarize_results(all_results)
    summary_path = RESULTS_DIR / "PH1_RESULTS.md"
    summary_path.write_text(summary_md, encoding="utf-8")
    print(f"  [MD] {summary_path}", flush=True)

    # Training summary JSON
    training_json_path = RESULTS_DIR / "ph1_training_summary.json"
    with open(training_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "contract": {
                "frame_size": FRAME_SIZE,
                "hop_size": HOP_SIZE,
                "sample_rate": SAMPLE_RATE,
                "seed": SEED,
                "center": False,
                "window": "hann",
            },
            "metric_provenance": get_metric_provenance(),
            "experiments": training_summary,
        }, f, indent=2)
    print(f"  [JSON] {training_json_path}", flush=True)

    # ========================================
    # FINAL SUMMARY
    # ========================================
    total_time = time.perf_counter() - t_start
    print(f"\n{'=' * 70}", flush=True)
    print("  PH1 CAMPAIGN COMPLETE", flush=True)
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)", flush=True)
    print(f"{'=' * 70}", flush=True)

    # Print training summary
    print(f"\n  TRAINING SUMMARY:", flush=True)
    for eid, s in training_summary.items():
        status = s["status"]
        if status == "COMPLETED":
            print(f"    {eid:4s} {s['model']:15s} × {s['input_mode']:15s} | "
                  f"Train: {s['final_train_loss']:.4f} | Val: {s['final_val_loss']:.4f} | "
                  f"Time: {s['total_time_s']:.0f}s | Params: {s['param_count']:,d}", flush=True)
        else:
            print(f"    {eid:4s} {s['model']:15s} × {s['input_mode']:15s} | {status}", flush=True)

    # Print evaluation summary
    if all_results:
        print(f"\n  EVALUATION SUMMARY (mean across all held-out clips):", flush=True)
        from collections import defaultdict
        by_exp = defaultdict(list)
        for r in all_results:
            by_exp[r["experiment_id"]].append(r)

        print(f"  {'Exp':4s} {'Model':15s} {'Input':15s} | {'SNR':>8s} {'SI-SDR':>8s} {'STOI':>8s} {'PESQ':>6s} | N", flush=True)
        print(f"  {'-'*4} {'-'*15} {'-'*15} | {'-'*8} {'-'*8} {'-'*8} {'-'*6} | -", flush=True)
        for eid in sorted(by_exp.keys()):
            rows = by_exp[eid]
            model_name = rows[0]["model"]
            input_mode = rows[0]["input_mode"]
            snr_m = np.nanmean([r["snr_db"] for r in rows])
            si_m = np.nanmean([r["si_sdr_db"] for r in rows])
            stoi_m = np.nanmean([r["stoi"] for r in rows])
            pesq_m = np.nanmean([r["pesq"] for r in rows])
            print(f"  {eid:4s} {model_name:15s} {input_mode:15s} | "
                  f"{snr_m:8.2f} {si_m:8.2f} {stoi_m:8.4f} {pesq_m:6.2f} | {len(rows)}", flush=True)

    print(f"\n  Results: {csv_path}", flush=True)
    print(f"  Summary: {summary_path}", flush=True)
    print(f"  Training metadata: {training_json_path}", flush=True)

    return summary_md


if __name__ == "__main__":
    main()
