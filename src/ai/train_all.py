"""
Train All AI Models for Phase 4 Benchmark.
PS 26052 -- Adaptive Defence ANC.

Trains in order of parameter count (fastest first):
  1. TinyEnhancer V3 (9.6K params) -- 30 epochs
  2. CRN-Micro (986K params) -- 20 epochs
  3. DTLN (989K params) -- handled separately (waveform model)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if not TORCH_AVAILABLE:
    print("[!] PyTorch not available. Cannot train.", flush=True)
    sys.exit(1)

from src.ai.tiny_enhancer import TinyEnhancerNet
from src.ai.crn import CRNNet
from src.ai.train import train_spectral_mask_model


def main():
    print("=" * 70, flush=True)
    print("  MULTI-MODEL TRAINING PIPELINE", flush=True)
    print("  PS 26052 -- Adaptive Defence ANC", flush=True)
    print("=" * 70, flush=True)

    results = {}

    # 1. TinyEnhancer V3
    print("\n" + "-" * 70, flush=True)
    print("  [1/2] Training TinyEnhancer V3 (9.6K params, 30 epochs)", flush=True)
    print("-" * 70, flush=True)
    result = train_spectral_mask_model(
        model=TinyEnhancerNet(),
        epochs=30,
        lr=1e-3,
        save_path="checkpoints/tiny_enhancer_v3.pt",
    )
    results["TinyEnhancer_V3"] = result
    print(f"  -> Status: {result['status']}", flush=True)
    if result["status"] == "COMPLETED":
        print(f"  -> Final Train Loss: {result['final_train_loss']:.4f}", flush=True)
        print(f"  -> Final Val Loss:   {result['final_val_loss']:.4f}", flush=True)

    # 2. CRN-Micro
    print("\n" + "-" * 70, flush=True)
    print("  [2/2] Training CRN-Micro (986K params, 20 epochs)", flush=True)
    print("-" * 70, flush=True)
    result = train_spectral_mask_model(
        model=CRNNet(freq_bins=257, hidden_size=128, channels=(8, 16, 32, 64, 128)),
        epochs=20,
        lr=5e-4,
        save_path="checkpoints/crn_micro.pt",
    )
    results["CRN_Micro"] = result
    print(f"  -> Status: {result['status']}", flush=True)
    if result["status"] == "COMPLETED":
        print(f"  -> Final Train Loss: {result['final_train_loss']:.4f}", flush=True)
        print(f"  -> Final Val Loss:   {result['final_val_loss']:.4f}", flush=True)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("  TRAINING SUMMARY", flush=True)
    print("=" * 70, flush=True)
    for name, r in results.items():
        if r["status"] == "COMPLETED":
            print(f"  {name:20s} | Train: {r['final_train_loss']:.4f} | Val: {r['final_val_loss']:.4f} | Time: {r['total_time_s']:.1f}s", flush=True)
        else:
            print(f"  {name:20s} | {r['status']}: {r.get('reason', 'unknown')}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
