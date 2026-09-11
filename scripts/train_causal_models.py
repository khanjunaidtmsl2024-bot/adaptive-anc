"""
Training Script for PH2A Causal Models: E1_causal and E2_causal.
PS 26052 — Adaptive Defence ANC.

Trains TinyEnhancerNet with strictly causal 2D convolutions (zero lookahead).
Contract parameters:
- Epochs: 30
- Optimizer: AdamW, LR: 1e-4, weight_decay: 1e-2
- Frame: 256, Hop: 128, center: False
- Loss: 0.5 * L1_mag + 0.5 * (-SI_SDR)
- Train split: TRAIN (56 clips, SPK_001..SPK_008)
- Validation split: TEST_A_UNSEEN_SPEAKER (18 clips, SPK_009..SPK_010)
  (Documented: circular validation used solely for early stopping/checkpoint selection)
- Checkpoints saved:
  - checkpoints/E1_causal.pt (RAW_PRIMARY)
  - checkpoints/E2_causal.pt (NLMS_RESIDUAL)
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.ai.tiny_enhancer import TinyEnhancerNet
from src.ai.train import train_spectral_mask_model


def main():
    print("=" * 80, flush=True)
    print("  PH2A: TRAINING STRICTLY CAUSAL TINYENHANCER MODELS", flush=True)
    print("  PS 26052 — Adaptive Defence ANC", flush=True)
    print("=" * 80, flush=True)

    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    metadata_csv = "data/v4/metadata/metadata_v4_extended.csv"

    # --- 1. Train E1_causal (RAW_PRIMARY) ---
    print("\n" + "#" * 80, flush=True)
    print("  [1/2] Training E1_causal: TinyEnhancer Causal / RAW_PRIMARY (30 epochs)", flush=True)
    print("#" * 80, flush=True)
    t0 = time.perf_counter()
    e1_model = TinyEnhancerNet()
    e1_result = train_spectral_mask_model(
        model=e1_model,
        metadata_csv=metadata_csv,
        epochs=30,
        lr=1e-4,
        frame_size=256,
        hop_size=128,
        alpha=0.5,
        device="cpu",
        save_path=str(checkpoints_dir / "E1_causal.pt"),
        train_split="TRAIN",
        val_split="TEST_A_UNSEEN_SPEAKER",
        seed=42,
        input_mode="RAW_PRIMARY",
    )
    t_e1 = time.perf_counter() - t0
    print(f"\n[+] E1_causal completed in {t_e1:.1f}s | Status: {e1_result.get('status')}", flush=True)
    print(f"    Best Epoch: {e1_result.get('best_epoch')} | Best Val SI-SDR: {e1_result.get('best_val_si_sdr'):+.2f} dB", flush=True)
    print(f"    Checkpoint SHA-256: {e1_result.get('checkpoint_sha256')}", flush=True)

    # --- 2. Train E2_causal (NLMS_RESIDUAL) ---
    print("\n" + "#" * 80, flush=True)
    print("  [2/2] Training E2_causal: TinyEnhancer Causal / NLMS_RESIDUAL (30 epochs)", flush=True)
    print("#" * 80, flush=True)
    t0 = time.perf_counter()
    e2_model = TinyEnhancerNet()
    e2_result = train_spectral_mask_model(
        model=e2_model,
        metadata_csv=metadata_csv,
        epochs=30,
        lr=1e-4,
        frame_size=256,
        hop_size=128,
        alpha=0.5,
        device="cpu",
        save_path=str(checkpoints_dir / "E2_causal.pt"),
        train_split="TRAIN",
        val_split="TEST_A_UNSEEN_SPEAKER",
        seed=42,
        input_mode="NLMS_RESIDUAL",
    )
    t_e2 = time.perf_counter() - t0
    print(f"\n[+] E2_causal completed in {t_e2:.1f}s | Status: {e2_result.get('status')}", flush=True)
    print(f"    Best Epoch: {e2_result.get('best_epoch')} | Best Val SI-SDR: {e2_result.get('best_val_si_sdr'):+.2f} dB", flush=True)
    print(f"    Checkpoint SHA-256: {e2_result.get('checkpoint_sha256')}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("  PH2A TRAINING SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print(f"  E1_causal: Best Val SI-SDR = {e1_result.get('best_val_si_sdr'):+.2f} dB | SHA-256: {e1_result.get('checkpoint_sha256')}", flush=True)
    print(f"  E2_causal: Best Val SI-SDR = {e2_result.get('best_val_si_sdr'):+.2f} dB | SHA-256: {e2_result.get('checkpoint_sha256')}", flush=True)


if __name__ == "__main__":
    main()
