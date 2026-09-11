"""
Phase 3 & Phase 7: ONNX FP32 vs INT8 Benchmark & Objective Quality Evaluation.
PS 26052 — Defence-Grade Adaptive ANC.

Quantizes E2_causal.onnx to INT8, benchmarks inference latency on host CPU,
and evaluates acoustic quality (SI-SDR, STOI, PESQ, Delta SNR, Attenuation)
to verify whether INT8 incurs acoustic regression.
"""

import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import soundfile as sf
import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ai.tiny_enhancer import TinyEnhancerNet
from src.dsp.vss_nlms_fast import VSSNLMSFilterFast
from src.evaluation.metrics import compute_si_snr, compute_snr, compute_stoi, compute_pesq


def quantize_e2_causal(
    fp32_onnx_path: str = "models/E2_causal.onnx",
    int8_onnx_path: str = "models/E2_causal_int8.onnx",
) -> Dict[str, Any]:
    print("\n--- PHASE 7: INT8 DYNAMIC QUANTIZATION ---", flush=True)
    fp32_file = PROJECT_ROOT / fp32_onnx_path
    int8_file = PROJECT_ROOT / int8_onnx_path

    if not fp32_file.exists():
        raise FileNotFoundError(f"FP32 ONNX model not found: {fp32_file}")

    print(f"[+] Quantizing {fp32_file} -> {int8_file}...")
    quantize_dynamic(
        model_input=str(fp32_file),
        model_output=str(int8_file),
        weight_type=QuantType.QInt8,
    )

    fp32_size = fp32_file.stat().st_size
    int8_size = int8_file.stat().st_size
    compression_ratio = fp32_size / int8_size

    with open(int8_file, "rb") as f:
        int8_sha256 = hashlib.sha256(f.read()).hexdigest()

    print(f"[+] FP32 Model Size: {fp32_size:,d} bytes ({fp32_size / 1024:.2f} KB)")
    print(f"[+] INT8 Model Size: {int8_size:,d} bytes ({int8_size / 1024:.2f} KB)")
    print(f"[+] Size Ratio:      {compression_ratio:.2f}x")
    print(f"[+] INT8 SHA-256:    {int8_sha256}")

    return {
        "fp32_path": str(fp32_file),
        "fp32_size_bytes": fp32_size,
        "int8_path": str(int8_file),
        "int8_size_bytes": int8_size,
        "int8_sha256": int8_sha256,
        "compression_ratio": compression_ratio,
    }


def benchmark_inference_latency(
    fp32_onnx_path: str = "models/E2_causal.onnx",
    int8_onnx_path: str = "models/E2_causal_int8.onnx",
    e2_ckpt_path: str = "checkpoints/E2_causal.pt",
    n_iters: int = 1000,
) -> Dict[str, Any]:
    print("\n--- INFERENCE LATENCY BENCHMARK (1,000 steady-state iterations) ---", flush=True)
    device = "cpu"
    torch.set_num_threads(4)

    # 1. PyTorch Model
    pytorch_model = TinyEnhancerNet().to(device).eval()
    pytorch_model.load_state_dict(torch.load(str(PROJECT_ROOT / e2_ckpt_path), map_location=device))

    # 2. ONNX Runtime Sessions
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    ort_fp32 = ort.InferenceSession(str(PROJECT_ROOT / fp32_onnx_path), sess_opts, providers=["CPUExecutionProvider"])
    ort_int8 = ort.InferenceSession(str(PROJECT_ROOT / int8_onnx_path), sess_opts, providers=["CPUExecutionProvider"])

    # 3. Fast DSP
    fast_nlms = VSSNLMSFilterFast(filter_length=64)

    # Pre-generate inputs
    rng = np.random.RandomState(42)
    stft_t1_np = [rng.randn(1, 1, 129, 1).astype(np.float32) for _ in range(n_iters)]
    stft_t1_torch = [torch.from_numpy(arr) for arr in stft_t1_np]
    stft_t9_np = [rng.randn(1, 1, 129, 9).astype(np.float32) for _ in range(n_iters)]
    primary_hops = [rng.randn(128).astype(np.float32) * 0.1 for _ in range(n_iters)]
    ref_hops = [rng.randn(128).astype(np.float32) * 0.1 for _ in range(n_iters)]

    # Warmup
    pytorch_model.reset_state()
    for _ in range(50):
        with torch.no_grad():
            pytorch_model(stft_t1_torch[0], stateful=True)
        ort_fp32.run(None, {"input_spectrogram": stft_t1_np[0]})
        ort_fp32.run(None, {"input_spectrogram": stft_t9_np[0]})
        ort_int8.run(None, {"input_spectrogram": stft_t1_np[0]})
        fast_nlms.filter_block(primary_hops[0], ref_hops[0])

    # A. PyTorch Stateful T=1
    pytorch_model.reset_state()
    times_torch = []
    with torch.no_grad():
        for i in range(n_iters):
            t0 = time.perf_counter_ns()
            pytorch_model(stft_t1_torch[i], stateful=True)
            t1 = time.perf_counter_ns()
            times_torch.append((t1 - t0) / 1e6)

    # B. ORT FP32 T=1
    times_ort_fp32_t1 = []
    inp_name_fp32 = ort_fp32.get_inputs()[0].name
    for i in range(n_iters):
        t0 = time.perf_counter_ns()
        ort_fp32.run(None, {inp_name_fp32: stft_t1_np[i]})
        t1 = time.perf_counter_ns()
        times_ort_fp32_t1.append((t1 - t0) / 1e6)

    # C. ORT FP32 T=9 (Receptive Field Window for pure stateless inference)
    times_ort_fp32_t9 = []
    for i in range(n_iters):
        t0 = time.perf_counter_ns()
        ort_fp32.run(None, {inp_name_fp32: stft_t9_np[i]})
        t1 = time.perf_counter_ns()
        times_ort_fp32_t9.append((t1 - t0) / 1e6)

    # D. ORT INT8 T=1
    times_ort_int8_t1 = []
    inp_name_int8 = ort_int8.get_inputs()[0].name
    for i in range(n_iters):
        t0 = time.perf_counter_ns()
        ort_int8.run(None, {inp_name_int8: stft_t1_np[i]})
        t1 = time.perf_counter_ns()
        times_ort_int8_t1.append((t1 - t0) / 1e6)

    # E. Fast DSP (Numba)
    times_dsp = []
    for i in range(n_iters):
        t0 = time.perf_counter_ns()
        fast_nlms.filter_block(primary_hops[i], ref_hops[i])
        t1 = time.perf_counter_ns()
        times_dsp.append((t1 - t0) / 1e6)

    hop_budget_ms = 8.0

    def calc_stats(times: List[float]) -> Dict[str, float]:
        arr = np.array(times)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "p99_9": float(np.percentile(arr, 99.9)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "rtf_p50": float(np.percentile(arr, 50) / hop_budget_ms),
            "rtf_p95": float(np.percentile(arr, 95) / hop_budget_ms),
        }

    stats = {
        "pytorch_fp32_t1": calc_stats(times_torch),
        "ort_fp32_t1": calc_stats(times_ort_fp32_t1),
        "ort_fp32_t9_window": calc_stats(times_ort_fp32_t9),
        "ort_int8_t1": calc_stats(times_ort_int8_t1),
        "numba_fast_dsp": calc_stats(times_dsp),
        "total_ort_fp32_plus_dsp": calc_stats([a + d for a, d in zip(times_ort_fp32_t1, times_dsp)]),
        "total_ort_int8_plus_dsp": calc_stats([a + d for a, d in zip(times_ort_int8_t1, times_dsp)]),
    }

    print(f"  PyTorch FP32 (T=1):     P50 = {stats['pytorch_fp32_t1']['p50']:.3f} ms | P95 = {stats['pytorch_fp32_t1']['p95']:.3f} ms | Max = {stats['pytorch_fp32_t1']['max']:.3f} ms")
    print(f"  ONNX Runtime FP32 (T=1):P50 = {stats['ort_fp32_t1']['p50']:.3f} ms | P95 = {stats['ort_fp32_t1']['p95']:.3f} ms | Max = {stats['ort_fp32_t1']['max']:.3f} ms")
    print(f"  ONNX Runtime FP32 (T=9):P50 = {stats['ort_fp32_t9_window']['p50']:.3f} ms | P95 = {stats['ort_fp32_t9_window']['p95']:.3f} ms | Max = {stats['ort_fp32_t9_window']['max']:.3f} ms")
    print(f"  ONNX Runtime INT8 (T=1):P50 = {stats['ort_int8_t1']['p50']:.3f} ms | P95 = {stats['ort_int8_t1']['p95']:.3f} ms | Max = {stats['ort_int8_t1']['max']:.3f} ms")
    print(f"  Numba Fast DSP:         P50 = {stats['numba_fast_dsp']['p50']:.3f} ms | P95 = {stats['numba_fast_dsp']['p95']:.3f} ms | Max = {stats['numba_fast_dsp']['max']:.3f} ms")
    print(f"  Total (ORT FP32 + DSP): P50 = {stats['total_ort_fp32_plus_dsp']['p50']:.3f} ms | P95 = {stats['total_ort_fp32_plus_dsp']['p95']:.3f} ms | Max = {stats['total_ort_fp32_plus_dsp']['max']:.3f} ms")
    print(f"  Total (ORT INT8 + DSP): P50 = {stats['total_ort_int8_plus_dsp']['p50']:.3f} ms | P95 = {stats['total_ort_int8_plus_dsp']['p95']:.3f} ms | Max = {stats['total_ort_int8_plus_dsp']['max']:.3f} ms")
    return stats


from src.ai.train import _causal_istft


def evaluate_row_worker(args):
    r, fp32_path, int8_path, data_root_str = args
    sess_o = ort.SessionOptions()
    sess_o.intra_op_num_threads = 1
    s_fp32 = ort.InferenceSession(fp32_path, sess_o, providers=["CPUExecutionProvider"])
    s_int8 = ort.InferenceSession(int8_path, sess_o, providers=["CPUExecutionProvider"])
    d_root = Path(data_root_str)
    win = torch.hann_window(256)

    clean, _ = sf.read(str(d_root / r["clean_path"]), dtype="float32")
    primary, _ = sf.read(str(d_root / r["primary_path"]), dtype="float32")
    reference, _ = sf.read(str(d_root / r["reference_path"]), dtype="float32")
    n = min(len(clean), len(primary), len(reference))
    clean, primary, reference = clean[:n], primary[:n], reference[:n]

    # 1. NLMS
    nlms = VSSNLMSFilterFast(filter_length=64)
    b1_res, _, _ = nlms.filter_block(primary, reference)
    b1_res = np.clip(b1_res, -0.98, 0.98)

    # 2. STFT
    t_res = torch.from_numpy(b1_res).float()
    stft = torch.stft(t_res, 256, 128, window=win, return_complex=True, center=False)
    mag_np = stft.abs().unsqueeze(0).unsqueeze(0).numpy()
    phase = stft.angle()

    # 3. FP32 Mask & Causal ISTFT
    m_fp32 = s_fp32.run(None, {"input_spectrogram": mag_np})[0].squeeze(0).squeeze(0)
    spec_fp32 = (stft.abs() * torch.from_numpy(m_fp32)) * torch.exp(1j * phase)
    wav_fp32 = _causal_istft(spec_fp32, 256, 128, win, length=n).numpy()

    # 4. INT8 Mask & Causal ISTFT
    m_int8 = s_int8.run(None, {"input_spectrogram": mag_np})[0].squeeze(0).squeeze(0)
    spec_int8 = (stft.abs() * torch.from_numpy(m_int8)) * torch.exp(1j * phase)
    wav_int8 = _causal_istft(spec_int8, 256, 128, win, length=n).numpy()

    in_snr = compute_snr(clean, primary)

    # Metrics
    si_sdr_fp32 = float(compute_si_snr(clean, wav_fp32))
    stoi_fp32 = float(compute_stoi(clean, wav_fp32, sample_rate=16000))
    dsnr_fp32 = float(compute_snr(clean, wav_fp32) - in_snr)
    pesq_fp32 = compute_pesq(clean, wav_fp32, sample_rate=16000)

    si_sdr_int8 = float(compute_si_snr(clean, wav_int8))
    stoi_int8 = float(compute_stoi(clean, wav_int8, sample_rate=16000))
    dsnr_int8 = float(compute_snr(clean, wav_int8) - in_snr)
    pesq_int8 = compute_pesq(clean, wav_int8, sample_rate=16000)

    return {
        "fp32": {"si_sdr": si_sdr_fp32, "stoi": stoi_fp32, "dsnr": dsnr_fp32, "pesq": float(pesq_fp32) if pesq_fp32 is not None else None},
        "int8": {"si_sdr": si_sdr_int8, "stoi": stoi_int8, "dsnr": dsnr_int8, "pesq": float(pesq_int8) if pesq_int8 is not None else None},
    }


def evaluate_acoustic_quality_fp32_vs_int8(
    fp32_onnx_path: str = "models/E2_causal.onnx",
    int8_onnx_path: str = "models/E2_causal_int8.onnx",
) -> Dict[str, Any]:
    print("\n--- ACOUSTIC QUALITY AUDIT: FP32 VS INT8 ---", flush=True)
    import csv
    manifest_path = PROJECT_ROOT / "data" / "v4" / "metadata" / "metadata_v4_extended.csv"
    data_root = PROJECT_ROOT / "data" / "v4"

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    session_fp32 = ort.InferenceSession(str(PROJECT_ROOT / fp32_onnx_path), sess_opts, providers=["CPUExecutionProvider"])
    session_int8 = ort.InferenceSession(str(PROJECT_ROOT / int8_onnx_path), sess_opts, providers=["CPUExecutionProvider"])

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    test_b_rows = [r for r in all_rows if r["split"] == "TEST_B_UNSEEN_NOISE_REC"]
    test_c_rows = [r for r in all_rows if r["split"] == "TEST_C_UNSEEN_NOISE_CATEGORY"]

    window = torch.hann_window(256)

    import multiprocessing as mp
    n_workers = min(6, mp.cpu_count() or 4)

    def process_split_parallel(rows: List[Dict[str, Any]], split_name: str) -> Dict[str, Any]:
        print(f"  Evaluating {split_name} ({len(rows)} clips) using {n_workers} parallel workers...", flush=True)
        t0 = time.perf_counter()
        worker_args = [(r, str(PROJECT_ROOT / fp32_onnx_path), str(PROJECT_ROOT / int8_onnx_path), str(data_root)) for r in rows]

        with mp.Pool(processes=n_workers) as pool:
            clip_results = pool.map(evaluate_row_worker, worker_args)

        elapsed = time.perf_counter() - t0
        print(f"  Completed {split_name} in {elapsed:.1f}s", flush=True)

        fp32_si_sdrs = [c["fp32"]["si_sdr"] for c in clip_results]
        int8_si_sdrs = [c["int8"]["si_sdr"] for c in clip_results]
        fp32_stois = [c["fp32"]["stoi"] for c in clip_results]
        int8_stois = [c["int8"]["stoi"] for c in clip_results]
        fp32_dsnrs = [c["fp32"]["dsnr"] for c in clip_results]
        int8_dsnrs = [c["int8"]["dsnr"] for c in clip_results]
        fp32_pesqs = [c["fp32"]["pesq"] for c in clip_results if c["fp32"]["pesq"] is not None]
        int8_pesqs = [c["int8"]["pesq"] for c in clip_results if c["int8"]["pesq"] is not None]

        m_s_fp32, m_s_int8 = float(np.mean(fp32_si_sdrs)), float(np.mean(int8_si_sdrs))
        m_st_fp32, m_st_int8 = float(np.mean(fp32_stois)), float(np.mean(int8_stois))
        m_ds_fp32, m_ds_int8 = float(np.mean(fp32_dsnrs)), float(np.mean(int8_dsnrs))
        m_p_fp32 = float(np.mean(fp32_pesqs)) if fp32_pesqs else None
        m_p_int8 = float(np.mean(int8_pesqs)) if int8_pesqs else None

        delta_si_sdr = m_s_int8 - m_s_fp32

        p32_str = f"{m_p_fp32:.4f}" if m_p_fp32 is not None else "N/A"
        p8_str = f"{m_p_int8:.4f}" if m_p_int8 is not None else "N/A"
        print(f"  {split_name} Summary:")
        print(f"    FP32 ONNX: SI-SDR = {m_s_fp32:+6.2f} dB | STOI = {m_st_fp32:.4f} | dSNR = {m_ds_fp32:+6.2f} dB | PESQ = {p32_str}", flush=True)
        print(f"    INT8 ONNX: SI-SDR = {m_s_int8:+6.2f} dB | STOI = {m_st_int8:.4f} | dSNR = {m_ds_int8:+6.2f} dB | PESQ = {p8_str}", flush=True)
        print(f"    Degradation (INT8 - FP32): {delta_si_sdr:+.3f} dB SI-SDR", flush=True)

        return {
            "clip_count": len(rows),
            "fp32": {"si_sdr": m_s_fp32, "stoi": m_st_fp32, "delta_snr": m_ds_fp32, "pesq": m_p_fp32},
            "int8": {"si_sdr": m_s_int8, "stoi": m_st_int8, "delta_snr": m_ds_int8, "pesq": m_p_int8},
            "delta_si_sdr": delta_si_sdr,
            "is_degraded": bool(delta_si_sdr < -0.5),
        }

    # Evaluate on TEST_B and TEST_C
    res_b = process_split_parallel(test_b_rows, "TEST_B_UNSEEN_NOISE_REC")
    res_c = process_split_parallel(test_c_rows, "TEST_C_UNSEEN_NOISE_CATEGORY")

    # Clean speech attenuation
    clean_wav, _ = sf.read(str(data_root / "clean" / "SPK_001_clean.wav"), dtype="float32")
    in_rms = float(np.sqrt(np.mean(clean_wav ** 2)))
    prim_t = torch.from_numpy(clean_wav).float()
    stft = torch.stft(prim_t, 256, 128, window=window, return_complex=True, center=False)
    mag_np = stft.abs().unsqueeze(0).unsqueeze(0).numpy()

    m_fp32 = session_fp32.run(None, {"input_spectrogram": mag_np})[0].squeeze(0).squeeze(0)
    w_fp32 = _causal_istft((stft.abs() * torch.from_numpy(m_fp32)) * torch.exp(1j * stft.angle()), 256, 128, window, length=len(clean_wav)).numpy()
    atten_fp32 = float(20.0 * math.log10(np.sqrt(np.mean(w_fp32 ** 2)) / in_rms))

    m_int8 = session_int8.run(None, {"input_spectrogram": mag_np})[0].squeeze(0).squeeze(0)
    w_int8 = _causal_istft((stft.abs() * torch.from_numpy(m_int8)) * torch.exp(1j * stft.angle()), 256, 128, window, length=len(clean_wav)).numpy()
    atten_int8 = float(20.0 * math.log10(np.sqrt(np.mean(w_int8 ** 2)) / in_rms))

    print(f"\n  Clean Speech Attenuation:")
    print(f"    FP32 ONNX: {atten_fp32:+.4f} dB")
    print(f"    INT8 ONNX: {atten_int8:+.4f} dB")

    return {
        "TEST_B": res_b,
        "TEST_C": res_c,
        "clean_speech_attenuation": {
            "fp32_db": atten_fp32,
            "int8_db": atten_int8,
            "delta_db": atten_int8 - atten_fp32,
        },
    }


def main():
    quant_info = quantize_e2_causal()
    latency_info = benchmark_inference_latency()
    quality_info = evaluate_acoustic_quality_fp32_vs_int8()

    full_report = {
        "quantization_info": quant_info,
        "latency_benchmark": latency_info,
        "acoustic_quality_comparison": quality_info,
    }

    out_json = PROJECT_ROOT / "results" / "ph3_embedded" / "onnx_fp32_vs_int8_report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"\n[+] Full benchmark report saved to {out_json}")


if __name__ == "__main__":
    main()
