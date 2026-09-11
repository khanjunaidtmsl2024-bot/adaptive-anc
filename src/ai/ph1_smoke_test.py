"""
Quick 2-epoch smoke test of the PH1 experiment pipeline.
Validates that training + evaluation works end-to-end before
committing to full 20-30 epoch runs.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import numpy as np

from src.ai.tiny_enhancer import TinyEnhancerNet
from src.ai.crn import CRNNet
from src.ai.dtln import DTLNNet
from src.ai.train import train_spectral_mask_model
from src.ai.train_dtln import train_dtln_model

FRAME_SIZE = 256
HOP_SIZE = 128
SEED = 42
METADATA_CSV = "data/v4/metadata/metadata_v4.csv"

print("=" * 60)
print("  PH1 SMOKE TEST — 2 epochs each")
print("=" * 60)

# Test 1: TinyEnhancer × RAW
print("\n[1/4] TinyEnhancer × RAW_PRIMARY (2 epochs)...")
torch.manual_seed(SEED)
model = TinyEnhancerNet()
t0 = time.perf_counter()
r = train_spectral_mask_model(
    model=model, metadata_csv=METADATA_CSV,
    epochs=2, lr=1e-3, frame_size=FRAME_SIZE, hop_size=HOP_SIZE,
    save_path="checkpoints/ph1/smoke_te_raw.pt",
    seed=SEED, input_mode="RAW_PRIMARY",
)
print(f"  Result: {r['status']}, time: {time.perf_counter()-t0:.1f}s")
assert r["status"] == "COMPLETED", f"TinyEnhancer RAW failed: {r}"

# Test 2: TinyEnhancer × NLMS
print("\n[2/4] TinyEnhancer × NLMS_RESIDUAL (2 epochs)...")
torch.manual_seed(SEED)
model = TinyEnhancerNet()
t0 = time.perf_counter()
r = train_spectral_mask_model(
    model=model, metadata_csv=METADATA_CSV,
    epochs=2, lr=1e-3, frame_size=FRAME_SIZE, hop_size=HOP_SIZE,
    save_path="checkpoints/ph1/smoke_te_nlms.pt",
    seed=SEED, input_mode="NLMS_RESIDUAL",
)
print(f"  Result: {r['status']}, time: {time.perf_counter()-t0:.1f}s")
assert r["status"] == "COMPLETED", f"TinyEnhancer NLMS failed: {r}"

# Test 3: CRN × RAW
print("\n[3/4] CRN-Micro × RAW_PRIMARY (2 epochs)...")
torch.manual_seed(SEED)
model = CRNNet(freq_bins=FRAME_SIZE//2+1, hidden_size=128, channels=(8,16,32,64,128))
print(f"  CRN params: {sum(p.numel() for p in model.parameters()):,d}")
t0 = time.perf_counter()
r = train_spectral_mask_model(
    model=model, metadata_csv=METADATA_CSV,
    epochs=2, lr=5e-4, frame_size=FRAME_SIZE, hop_size=HOP_SIZE,
    save_path="checkpoints/ph1/smoke_crn_raw.pt",
    seed=SEED, input_mode="RAW_PRIMARY",
)
print(f"  Result: {r['status']}, time: {time.perf_counter()-t0:.1f}s")
assert r["status"] == "COMPLETED", f"CRN RAW failed: {r}"

# Test 4: DTLN × RAW
print("\n[4/4] DTLN × RAW_PRIMARY (2 epochs)...")
torch.manual_seed(SEED)
model = DTLNNet(frame_size=FRAME_SIZE, hop_size=HOP_SIZE, hidden_size=128, encoder_size=256)
print(f"  DTLN params: {sum(p.numel() for p in model.parameters()):,d}")
t0 = time.perf_counter()
r = train_dtln_model(
    model=model, metadata_csv=METADATA_CSV,
    epochs=2, lr=1e-4,
    save_path="checkpoints/ph1/smoke_dtln_raw.pt",
    seed=SEED, input_mode="RAW_PRIMARY",
)
print(f"  Result: {r['status']}, time: {time.perf_counter()-t0:.1f}s")
assert r["status"] == "COMPLETED", f"DTLN RAW failed: {r}"

# Quick evaluation smoke test
print("\n[5/5] Evaluation smoke test (TinyEnhancer on 2 TEST_A clips)...")
from src.ai.ph1_experiment_runner import evaluate_model, evaluate_baselines

torch.manual_seed(SEED)
model = TinyEnhancerNet()
model.load_state_dict(torch.load("checkpoints/ph1/smoke_te_raw.pt", map_location="cpu", weights_only=True))
model.eval()

eval_results = evaluate_model(
    model=model, model_type="spectral_mask",
    exp_id="SMOKE", input_mode="RAW_PRIMARY",
    eval_split="TEST_A_UNSEEN_SPEAKER",
)
print(f"  Eval results: {len(eval_results)} clips")
for er in eval_results[:3]:
    print(f"    {er['sample_id']}: SI-SDR={er['si_sdr_db']:.2f} dB, "
          f"STOI={er['stoi']:.4f}, PESQ={er['pesq']:.2f}")

# Baselines
baseline_results = evaluate_baselines("TEST_A_UNSEEN_SPEAKER")
print(f"  Baseline results: {len(baseline_results)} rows")
for br in baseline_results[:4]:
    print(f"    {br['experiment_id']} {br['sample_id']}: SI-SDR={br['si_sdr_db']:.2f} dB, "
          f"STOI={br['stoi']:.4f}, PESQ={br['pesq']:.2f}")

print("\n" + "=" * 60)
print("  SMOKE TEST PASSED — all 4 training + evaluation pipelines work")
print("=" * 60)
