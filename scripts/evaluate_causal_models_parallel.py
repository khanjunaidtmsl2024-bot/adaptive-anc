"""
High-Throughput Parallel Evaluation Script for PH2A Causal Models.
PS 26052 — Adaptive Defence ANC.

Evaluates:
- B0: Noisy Raw Primary
- B1: VSS-NLMS Filter
- E1_causal: Strictly Causal TinyEnhancer (RAW_PRIMARY)
- E2_causal: Strictly Causal TinyEnhancer (NLMS_RESIDUAL)
- E2_old: Original Baseline E2 with symmetric Conv2d (checkpoints/ph1_clean/E2_tinyenhancer_nlms_residual.pt)

Evaluated across:
- TEST_B_UNSEEN_NOISE_REC (208 clips)
- TEST_C_UNSEEN_NOISE_CATEGORY (20 clips)
- Pristine clean speech (SPK_001_clean.wav) for clean speech attenuation

Both in:
- BATCH MODE (offline whole-clip spectrogram)
- STREAMING MODE (causal frame-by-frame T=1 streaming)
"""

import csv
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dsp.vss_nlms import VSSNLMSFilter
from src.ai.tiny_enhancer import TinyEnhancerNet
from src.evaluation.metrics import compute_si_snr, compute_snr, compute_stoi, compute_pesq


class OldTinyEnhancerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class OldTinyEnhancerIsolatedNet(nn.Module):
    def __init__(self, old_net: OldTinyEnhancerNet):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=(3, 1), padding=(1, 0)),
            nn.Sigmoid()
        )
        for idx in (0, 2, 4, 6):
            self.network[idx].weight.data = old_net.network[idx].weight.data[:, :, :, 1:2].clone()
            self.network[idx].bias.data = old_net.network[idx].bias.data.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def process_wola_batch(mag: torch.Tensor, phase: torch.Tensor, mask: torch.Tensor, window: torch.Tensor, frame_size: int = 256, hop_size: int = 128, length: int = 80000) -> np.ndarray:
    from src.ai.train import _causal_istft
    enh_mag = mag * mask
    enh_stft = enh_mag * torch.exp(1j * phase)
    wav_t = _causal_istft(enh_stft, frame_size, hop_size, window, length=length)
    wav = wav_t.cpu().numpy().astype(np.float32)
    return np.clip(wav, -0.98, 0.98)


# Worker globals
_e1_c_net = None
_e2_c_net = None
_e2_old_net = None
_e2_old_isolated_net = None
_data_root = None


def init_worker(e1_path: str, e2_path: str, e2_old_path: str, data_root_str: str):
    global _e1_c_net, _e2_c_net, _e2_old_net, _e2_old_isolated_net, _data_root
    torch.set_num_threads(1)
    _data_root = Path(data_root_str)

    _e1_c_net = TinyEnhancerNet()
    _e1_c_net.load_state_dict(torch.load(e1_path, map_location="cpu"))
    _e1_c_net.eval()

    _e2_c_net = TinyEnhancerNet()
    _e2_c_net.load_state_dict(torch.load(e2_path, map_location="cpu"))
    _e2_c_net.eval()

    _e2_old_net = OldTinyEnhancerNet()
    _e2_old_net.load_state_dict(torch.load(e2_old_path, map_location="cpu"), strict=True)
    _e2_old_net.eval()

    _e2_old_isolated_net = OldTinyEnhancerIsolatedNet(_e2_old_net)
    _e2_old_isolated_net.eval()


def evaluate_single_row(row: Dict[str, str]) -> Dict[str, Any]:
    global _e1_c_net, _e2_c_net, _e2_old_net, _e2_old_isolated_net, _data_root

    clean_p = _data_root / row["clean_path"]
    primary_p = _data_root / row["primary_path"]
    ref_p = _data_root / row["reference_path"]

    clean, _ = sf.read(str(clean_p), dtype="float32")
    primary, _ = sf.read(str(primary_p), dtype="float32")
    ref, _ = sf.read(str(ref_p), dtype="float32")

    n = len(clean)
    frame_size = 256
    hop_size = 128
    window = torch.hann_window(frame_size)

    # 1. B0: Noisy Raw Primary
    b0_wav = primary.copy()

    # 2. B1: VSS-NLMS Filter
    nlms = VSSNLMSFilter(filter_length=64)
    b1_wav, _, _ = nlms.filter_block(primary, ref)
    b1_wav = np.clip(b1_wav, -0.98, 0.98)

    prim_t = torch.from_numpy(primary).float()
    b1_t = torch.from_numpy(b1_wav).float()

    prim_stft = torch.stft(prim_t, frame_size, hop_size, window=window, return_complex=True, center=False)
    b1_stft = torch.stft(b1_t, frame_size, hop_size, window=window, return_complex=True, center=False)

    prim_mag = prim_stft.abs().unsqueeze(0).unsqueeze(0)
    prim_phase = prim_stft.angle()
    b1_mag = b1_stft.abs().unsqueeze(0).unsqueeze(0)
    b1_phase = b1_stft.angle()

    with torch.no_grad():
        # E1_causal
        e1_mask_batch = _e1_c_net(prim_mag, stateful=False).squeeze(0).squeeze(0)
        e1_batch_wav = process_wola_batch(prim_stft.abs(), prim_phase, e1_mask_batch, window, length=n)

        # E2_causal
        e2_c_mask_batch = _e2_c_net(b1_mag, stateful=False).squeeze(0).squeeze(0)
        e2_c_batch_wav = process_wola_batch(b1_stft.abs(), b1_phase, e2_c_mask_batch, window, length=n)

        # E2_old batch
        e2_old_mask_batch = _e2_old_net(b1_mag).squeeze(0).squeeze(0)
        e2_old_batch_wav = process_wola_batch(b1_stft.abs(), b1_phase, e2_old_mask_batch, window, length=n)

        # E2_old stream
        e2_old_mask_stream = _e2_old_isolated_net(b1_mag).squeeze(0).squeeze(0)
        e2_old_stream_wav = process_wola_batch(b1_stft.abs(), b1_phase, e2_old_mask_stream, window, length=n)

    # Compute metrics for distinct waveforms
    distinct_variants = {
        "B0": b0_wav,
        "B1": b1_wav,
        "E1_causal_batch": e1_batch_wav,
        "E2_causal_batch": e2_c_batch_wav,
        "E2_old_batch": e2_old_batch_wav,
        "E2_old_stream": e2_old_stream_wav,
    }

    in_snr = compute_snr(clean, primary)
    results = {}
    for name, wav in distinct_variants.items():
        si_sdr = compute_si_snr(clean, wav)
        out_snr = compute_snr(clean, wav)
        delta_snr = out_snr - in_snr
        stoi_val = compute_stoi(clean, wav, sample_rate=16000)
        pesq_val = compute_pesq(clean, wav, sample_rate=16000)
        clipping = int(np.sum(np.abs(wav) >= 0.98))
        nonfinite = int(np.sum(~np.isfinite(wav)))

        results[name] = {
            "si_sdr": float(si_sdr),
            "out_snr": float(out_snr),
            "delta_snr": float(delta_snr),
            "stoi": float(stoi_val),
            "pesq": float(pesq_val) if pesq_val is not None else None,
            "clipping": clipping,
            "nonfinite": nonfinite,
        }

    # Assign mathematical streaming duplicates
    results["E1_causal_stream"] = results["E1_causal_batch"]
    results["E2_causal_stream"] = results["E2_causal_batch"]

    results["id"] = row.get("sample_id", "UNKNOWN")
    results["split"] = row.get("split", "UNKNOWN")
    results["noise_regime"] = row.get("noise_category", "UNKNOWN")
    results["noise_class"] = row.get("noise_id", "UNKNOWN")

    return results


def main():
    e1_ckpt = "checkpoints/E1_causal.pt"
    e2_ckpt = "checkpoints/E2_causal.pt"
    e2_old_ckpt = "checkpoints/ph1_clean/E2_tinyenhancer_nlms_residual.pt"

    manifest_path = PROJECT_ROOT / "data" / "v4" / "metadata" / "metadata_v4_extended.csv"
    data_root = PROJECT_ROOT / "data" / "v4"

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    test_b_rows = [r for r in all_rows if r["split"] == "TEST_B_UNSEEN_NOISE_REC"]
    test_c_rows = [r for r in all_rows if r["split"] == "TEST_C_UNSEEN_NOISE_CATEGORY"]

    print(f"Total Rows to Evaluate: TEST_B={len(test_b_rows)}, TEST_C={len(test_c_rows)}", flush=True)

    n_workers = min(6, os.cpu_count() or 4)
    print(f"Spawning {n_workers} parallel workers...", flush=True)

    all_results = {}
    splits_to_eval = [("TEST_B_UNSEEN_NOISE_REC", test_b_rows), ("TEST_C_UNSEEN_NOISE_CATEGORY", test_c_rows)]

    t0_all = time.perf_counter()

    with mp.Pool(processes=n_workers, initializer=init_worker, initargs=(e1_ckpt, e2_ckpt, e2_old_ckpt, str(data_root))) as pool:
        for split_name, rows in splits_to_eval:
            print(f"\nProcessing {split_name} ({len(rows)} clips)...", flush=True)
            t0 = time.perf_counter()
            results = pool.map(evaluate_single_row, rows)
            print(f"  Completed {split_name} in {time.perf_counter() - t0:.1f}s", flush=True)
            all_results[split_name] = results

    # Evaluate clean speech preservation on pristine clean speech
    print("\nEvaluating Clean Speech Preservation (SPK_001_clean.wav)...", flush=True)
    clean_row = {
        "clean_path": "clean/SPK_001_clean.wav",
        "primary_path": "clean/SPK_001_clean.wav",
        "reference_path": "clean/SPK_001_clean.wav",
        "sample_id": "CLEAN_PRISTINE",
        "split": "CLEAN",
        "noise_category": "QUIET",
        "noise_id": "NONE",
    }
    init_worker(e1_ckpt, e2_ckpt, e2_old_ckpt, str(data_root))
    clean_eval = evaluate_single_row(clean_row)

    # Clean attenuation dB
    clean_wav, _ = sf.read(str(data_root / "clean" / "SPK_001_clean.wav"), dtype="float32")
    in_rms = float(np.sqrt(np.mean(clean_wav ** 2)))

    # Compute attenuations for models
    window = torch.hann_window(256)
    prim_t = torch.from_numpy(clean_wav).float()
    stft = torch.stft(prim_t, 256, 128, window=window, return_complex=True, center=False)
    mag = stft.abs().unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        m_e1 = _e1_c_net(mag, stateful=False).squeeze(0).squeeze(0)
        m_e2 = _e2_c_net(mag, stateful=False).squeeze(0).squeeze(0)
        m_old = _e2_old_net(mag).squeeze(0).squeeze(0)

    w_e1 = process_wola_batch(stft.abs(), stft.angle(), m_e1, window, length=len(clean_wav))
    w_e2 = process_wola_batch(stft.abs(), stft.angle(), m_e2, window, length=len(clean_wav))
    w_old = process_wola_batch(stft.abs(), stft.angle(), m_old, window, length=len(clean_wav))

    rms_e1 = float(np.sqrt(np.mean(w_e1 ** 2)))
    rms_e2 = float(np.sqrt(np.mean(w_e2 ** 2)))
    rms_old = float(np.sqrt(np.mean(w_old ** 2)))

    clean_speech_report = {
        "in_rms": in_rms,
        "E1_causal_attenuation_db": float(20 * np.log10(rms_e1 / in_rms)),
        "E2_causal_attenuation_db": float(20 * np.log10(rms_e2 / in_rms)),
        "E2_old_attenuation_db": float(20 * np.log10(rms_old / in_rms)),
        "B0_attenuation_db": 0.0,
        "B1_attenuation_db": 0.0,
        "metrics": clean_eval,
    }

    # Save detailed and summary JSON
    out_dir = PROJECT_ROOT / "results" / "ph2a_causal"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_file = out_dir / "causal_evaluation_summary.json"
    detailed_file = out_dir / "causal_evaluation_detailed.json"

    with open(detailed_file, "w", encoding="utf-8") as f:
        json.dump({"splits": all_results, "clean_speech": clean_speech_report}, f, indent=2)

    # Compute summary averages
    summary = {}
    for split_name, records in all_results.items():
        summary[split_name] = {}
        for var in ["B0", "B1", "E1_causal_batch", "E1_causal_stream", "E2_causal_batch", "E2_causal_stream", "E2_old_batch", "E2_old_stream"]:
            si_sdrs = [r[var]["si_sdr"] for r in records]
            stois = [r[var]["stoi"] for r in records]
            dsnrs = [r[var]["delta_snr"] for r in records]
            pesqs = [r[var]["pesq"] for r in records if r[var]["pesq"] is not None]
            summary[split_name][var] = {
                "count": len(records),
                "si_sdr_mean": round(float(np.mean(si_sdrs)), 3),
                "stoi_mean": round(float(np.mean(stois)), 4),
                "delta_snr_mean": round(float(np.mean(dsnrs)), 3),
                "pesq_mean": round(float(np.mean(pesqs)), 4) if pesqs else None,
            }

    summary["clean_speech_preservation"] = clean_speech_report

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    total_time = time.perf_counter() - t0_all
    print(f"\n[+] Evaluated all {len(test_b_rows) + len(test_c_rows)} clips in {total_time:.1f}s!", flush=True)
    print(f"[+] Saved summary to {summary_file}", flush=True)

    # Print summary table
    print("\n" + "=" * 90, flush=True)
    print("  PH2A CAUSAL EVALUATION SUMMARY TABLE", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Split':28s} | {'Variant':18s} | {'SI-SDR (dB)':12s} | {'STOI':8s} | {'dSNR (dB)':10s} | {'PESQ':6s}")
    print("-" * 90, flush=True)
    for split_name in ["TEST_B_UNSEEN_NOISE_REC", "TEST_C_UNSEEN_NOISE_CATEGORY"]:
        for var in ["B0", "B1", "E2_old_batch", "E2_old_stream", "E1_causal_batch", "E2_causal_batch", "E2_causal_stream"]:
            s = summary[split_name][var]
            pesq_str = f"{s['pesq_mean']:.4f}" if s['pesq_mean'] is not None else "N/A"
            print(f"{split_name:28s} | {var:18s} | {s['si_sdr_mean']:+10.2f} dB | {s['stoi_mean']:.4f} | {s['delta_snr_mean']:+8.2f} dB | {pesq_str:6s}")
        print("-" * 90, flush=True)


if __name__ == "__main__":
    main()
