"""
Phase 2: ONNX Export and Numerical Equivalence Verification for E2_causal.
PS 26052 — Defence-Grade Adaptive ANC.

Exports the exact frozen E2_causal checkpoint to FP32 ONNX and verifies
numerical equivalence against PyTorch across multiple representative inputs.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
import onnx
import onnxruntime as ort

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.ai.tiny_enhancer import TinyEnhancerNet


def export_and_verify_e2_causal(
    ckpt_path: str = "checkpoints/E2_causal.pt",
    output_onnx_path: str = "models/E2_causal.onnx",
    opset_version: int = 17,
) -> Dict[str, Any]:
    print("=" * 70)
    print("PHASE 2: ONNX EXPORT & NUMERICAL EQUIVALENCE VERIFICATION")
    print("=" * 70)

    # 1. Verify checkpoint existence and SHA-256
    ckpt_file = PROJECT_ROOT / ckpt_path
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    with open(ckpt_file, "rb") as f:
        ckpt_bytes = f.read()
    ckpt_sha256 = hashlib.sha256(ckpt_bytes).hexdigest()
    expected_sha256 = "0c8f08a1bdd41ff9543b45f122f48cc3c5208b7e8395ec656c557f4bd6482acd"
    print(f"[+] Loaded Checkpoint: {ckpt_file}")
    print(f"    SHA-256: {ckpt_sha256}")
    assert ckpt_sha256 == expected_sha256, f"SHA256 mismatch! Expected {expected_sha256}, got {ckpt_sha256}"

    # 2. Instantiate exact model and load weights
    model = TinyEnhancerNet().eval()
    state_dict = torch.load(str(ckpt_file), map_location="cpu")
    model.load_state_dict(state_dict)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[+] Verified Model Parameters: {param_count:,d} (must be exactly 9,569)")
    assert param_count == 9569, f"Parameter count mismatch: {param_count}"

    out_file = PROJECT_ROOT / output_onnx_path
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 3. Export to ONNX with dynamic time dimension
    dummy_input = torch.randn(1, 1, 129, 64, dtype=torch.float32)
    print(f"[+] Exporting to ONNX (opset {opset_version})...")

    torch.onnx.export(
        model,
        dummy_input,
        str(out_file),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_spectrogram"],
        output_names=["output_mask"],
        dynamic_axes={
            "input_spectrogram": {0: "batch_size", 3: "time_frames"},
            "output_mask": {0: "batch_size", 3: "time_frames"},
        },
    )

    onnx_size_bytes = out_file.stat().st_size
    with open(out_file, "rb") as f:
        onnx_sha256 = hashlib.sha256(f.read()).hexdigest()
    print(f"[+] Exported ONNX: {out_file} ({onnx_size_bytes:,d} bytes, {onnx_size_bytes/1024:.2f} KB)")
    print(f"    ONNX SHA-256: {onnx_sha256}")

    # 4. Validate ONNX model with onnx.checker
    onnx_model = onnx.load(str(out_file))
    onnx.checker.check_model(onnx_model)
    print("[+] ONNX model integrity check passed!")

    # 5. Initialize ONNX Runtime session
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(out_file), sess_options, providers=["CPUExecutionProvider"])

    # 6. Test numerical equivalence across multiple representative inputs
    test_cases = [
        ("T=1 (Single streaming frame)", 1, 1, 129, 1, 42),
        ("T=9 (Receptive field window)", 1, 1, 129, 9, 101),
        ("T=32 (Short phrase)", 1, 1, 129, 32, 202),
        ("T=64 (Standard block)", 1, 1, 129, 64, 303),
        ("T=128 (Long block)", 1, 1, 129, 128, 404),
        ("T=500 (Full 4-second clip)", 1, 1, 129, 500, 505),
        ("Batch=4, T=64 (Multi-stream)", 4, 1, 129, 64, 606),
    ]

    results = []
    max_abs_err_all = 0.0
    mean_abs_err_all = 0.0

    print("\n--- NUMERICAL EQUIVALENCE TESTS (PyTorch vs ONNX Runtime) ---")
    for name, B, C, F_dim, T_dim, seed in test_cases:
        rng = np.random.RandomState(seed)
        x_np = rng.randn(B, C, F_dim, T_dim).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        with torch.no_grad():
            y_torch = model(x_torch, stateful=False).numpy()

        ort_inputs = {session.get_inputs()[0].name: x_np}
        y_ort = session.run(None, ort_inputs)[0]

        abs_err = np.abs(y_torch - y_ort)
        max_err = float(np.max(abs_err))
        mean_err = float(np.mean(abs_err))

        if max_err > max_abs_err_all:
            max_abs_err_all = max_err
        mean_abs_err_all += mean_err

        status = "PASS [OK]" if max_err < 1e-5 else "FAIL [X]"
        print(f"  {name:30s}: Max Err = {max_err:.2e}, Mean Err = {mean_err:.2e} -> {status}")

        results.append({
            "case": name,
            "shape": [B, C, F_dim, T_dim],
            "max_abs_err": max_err,
            "mean_abs_err": mean_err,
            "passed": bool(max_err < 1e-5),
        })

    mean_abs_err_all /= len(test_cases)
    overall_passed = bool(max_abs_err_all < 1e-5)
    print(f"\nOverall Equivalence: {'PASS [OK]' if overall_passed else 'FAIL [X]'}")
    print(f"Peak Absolute Discrepancy: {max_abs_err_all:.2e}")
    print(f"Average Absolute Discrepancy: {mean_abs_err_all:.2e}")

    summary = {
        "status": "success" if overall_passed else "failed",
        "checkpoint_path": str(ckpt_file),
        "checkpoint_sha256": ckpt_sha256,
        "parameter_count": param_count,
        "onnx_path": str(out_file),
        "onnx_size_bytes": onnx_size_bytes,
        "onnx_sha256": onnx_sha256,
        "opset_version": opset_version,
        "peak_max_abs_error": max_abs_err_all,
        "overall_mean_abs_error": mean_abs_err_all,
        "test_cases": results,
    }

    out_json = PROJECT_ROOT / "results" / "ph3_embedded" / "onnx_export_verification.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Saved verification summary to {out_json}")
    return summary


if __name__ == "__main__":
    export_and_verify_e2_causal()
