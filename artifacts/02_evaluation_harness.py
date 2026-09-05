#!/usr/bin/env python3
"""
PS 26052 — Artifact 2: Evaluation Harness
==========================================
Computes SNR, STOI, PESQ, SI-SNR for noisy/clean/enhanced audio triplets.
Produces per-file CSV and summary statistics.

PS 26052 Targets:
    SNR > 15 dB
    STOI > 0.85
    PESQ > 2.5

Usage:
    # Evaluate a single file
    python 02_evaluation_harness.py --clean clean.wav --noisy noisy.wav --enhanced enhanced.wav

    # Evaluate a directory of triplets
    python 02_evaluation_harness.py --triplets-dir data/test_triplets/

    # Evaluate dataset generator output
    python 02_evaluation_harness.py --dataset-dir data/mixed/test --noisy-suffix ""

    # Compare two models
    python 02_evaluation_harness.py --compare model_a_results.csv model_b_results.csv

Designed for: SIH 2026 Adaptive ANC — PS 26052
"""

import os
import sys
import csv
import glob
import argparse
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

try:
    import soundfile as sf
except ImportError:
    print("ERROR: pip install soundfile")
    sys.exit(1)

try:
    from scipy import signal as scipy_signal
except ImportError:
    print("ERROR: pip install scipy")
    sys.exit(1)

try:
    import pystoi
    HAS_STOI = True
except ImportError:
    HAS_STOI = False
    print("WARNING: pystoi not installed. STOI will be skipped. pip install pystoi")

try:
    import pesq as pesq_module
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False
    print("WARNING: pesq not installed. PESQ will be skipped. pip install pesq")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("eval_harness")


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------
def load_mono(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load audio as mono float32 at target sample rate."""
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        try:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except ImportError:
            num_samples = int(len(data) * target_sr / sr)
            data = scipy_signal.resample(data, num_samples)
            sr = target_sr
    return data, sr


def align_signals(*signals: np.ndarray) -> tuple[np.ndarray, ...]:
    """Truncate all signals to the length of the shortest."""
    min_len = min(len(s) for s in signals)
    return tuple(s[:min_len] for s in signals)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Input SNR in dB: 20*log10(rms_clean / rms_noise)."""
    noise = noisy - clean
    rms_clean = np.sqrt(np.mean(clean ** 2) + 1e-12)
    rms_noise = np.sqrt(np.mean(noise ** 2) + 1e-12)
    return 20.0 * np.log10(rms_clean / rms_noise)


def compute_output_snr(clean: np.ndarray, enhanced: np.ndarray) -> float:
    """Output SNR in dB."""
    residual = enhanced - clean
    rms_clean = np.sqrt(np.mean(clean ** 2) + 1e-12)
    rms_residual = np.sqrt(np.mean(residual ** 2) + 1e-12)
    return 20.0 * np.log10(rms_clean / rms_residual)


def compute_snr_improvement(clean: np.ndarray, noisy: np.ndarray, enhanced: np.ndarray) -> float:
    """Delta SNR = output_SNR - input_SNR."""
    input_snr = compute_snr(clean, noisy)
    output_snr = compute_output_snr(clean, enhanced)
    return output_snr - input_snr


def compute_si_snr(clean: np.ndarray, enhanced: np.ndarray) -> float:
    """Scale-Invariant SNR (SI-SNR) in dB.
    
    SI-SNR = 10*log10(||s_target||^2 / ||e_noise||^2)
    where s_target = proj(enhanced onto clean) and e_noise = enhanced - s_target
    """
    clean = clean - np.mean(clean)
    enhanced = enhanced - np.mean(enhanced)
    
    dot = np.sum(clean * enhanced)
    s_target = (dot / (np.sum(clean ** 2) + 1e-12)) * clean
    e_noise = enhanced - s_target
    
    si_snr = 10.0 * np.log10(
        np.sum(s_target ** 2) / (np.sum(e_noise ** 2) + 1e-12) + 1e-12
    )
    return si_snr


def compute_stoi(clean: np.ndarray, noisy: np.ndarray, sr: int) -> float:
    """Short-Time Objective Intelligibility (STOI). Returns 0-1."""
    if not HAS_STOI:
        return float("nan")
    try:
        return pystoi.stoi(clean, noisy, sr, extended=False)
    except Exception as e:
        log.warning(f"STOI computation failed: {e}")
        return float("nan")


def compute_pesq(clean: np.ndarray, enhanced: np.ndarray, sr: int) -> float:
    """PESQ score. Requires 8kHz or 16kHz.
    
    NOTE: pesq library uses 8kHz 'nb' mode or 16kHz 'wb' mode.
    PS 26052 targets PESQ > 2.5.
    ITU-T P.862 was withdrawn in 2024 but PS still references it.
    """
    if not HAS_PESQ:
        return float("nan")
    try:
        if sr == 8000:
            mode = "nb"
        elif sr == 16000:
            mode = "wb"
        else:
            # pesq only supports 8kHz or 16kHz
            # Resample to 16kHz for evaluation
            from scipy.signal import resample_poly
            clean_16k = resample_poly(clean, 16000, sr)
            enhanced_16k = resample_poly(enhanced, 16000, sr)
            mode = "wb"
            sr = 16000
            clean, enhanced = clean_16k, enhanced_16k
        
        score = pesq_module.pesq(sr, clean, enhanced, mode)
        return float(score)
    except Exception as e:
        log.warning(f"PESQ computation failed: {e}")
        return float("nan")


# ---------------------------------------------------------------------------
# Result Record
# ---------------------------------------------------------------------------
@dataclass
class EvalRecord:
    """Evaluation result for one audio triplet."""
    file_id: str
    input_snr_db: float
    output_snr_db: float
    snr_improvement_db: float
    si_snr_db: float
    stoi: float
    pesq: float
    clean_samples: int
    enhanced_samples: int
    sample_rate: int
    clean_path: str
    noisy_path: str
    enhanced_path: str
    timestamp: str
    noise_class: str = ""
    target_snr_db: float = float("nan")
    split: str = ""
    model_name: str = ""


# ---------------------------------------------------------------------------
# Evaluation Harness
# ---------------------------------------------------------------------------
class EvalHarness:
    """Main evaluation harness."""
    
    PS_TARGETS = {
        "snr_db": 15.0,
        "stoi": 0.85,
        "pesq": 2.5,
    }
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.records: list[EvalRecord] = []
    
    def evaluate_triplet(
        self,
        clean_path: str,
        noisy_path: str,
        enhanced_path: str,
        file_id: str = "",
        noise_class: str = "",
        target_snr_db: float = float("nan"),
        split: str = "",
        model_name: str = "",
    ) -> EvalRecord:
        """Evaluate one clean/noisy/enhanced triplet."""
        clean, sr_c = load_mono(clean_path, self.sample_rate)
        noisy, sr_n = load_mono(noisy_path, self.sample_rate)
        enhanced, sr_e = load_mono(enhanced_path, self.sample_rate)
        
        # Ensure same sample rate
        if sr_c != sr_n or sr_c != sr_e:
            log.warning(f"Sample rate mismatch: clean={sr_c}, noisy={sr_n}, enhanced={sr_e}")
        
        # Align
        clean, noisy, enhanced = align_signals(clean, noisy, enhanced)
        
        # Compute metrics
        input_snr = compute_snr(clean, noisy)
        output_snr = compute_output_snr(clean, enhanced)
        snr_imp = output_snr - input_snr
        si_snr = compute_si_snr(clean, enhanced)
        stoi_score = compute_stoi(clean, enhanced, sr_c)
        pesq_score = compute_pesq(clean, enhanced, sr_c)
        
        if not file_id:
            file_id = Path(clean_path).stem
        
        record = EvalRecord(
            file_id=file_id,
            input_snr_db=round(input_snr, 2),
            output_snr_db=round(output_snr, 2),
            snr_improvement_db=round(snr_imp, 2),
            si_snr_db=round(si_snr, 2),
            stoi=round(stoi_score, 4) if not np.isnan(stoi_score) else float("nan"),
            pesq=round(pesq_score, 2) if not np.isnan(pesq_score) else float("nan"),
            clean_samples=len(clean),
            enhanced_samples=len(enhanced),
            sample_rate=sr_c,
            clean_path=clean_path,
            noisy_path=noisy_path,
            enhanced_path=enhanced_path,
            timestamp=datetime.now().isoformat(),
            noise_class=noise_class,
            target_snr_db=target_snr_db,
            split=split,
            model_name=model_name,
        )
        self.records.append(record)
        return record
    
    def evaluate_directory(
        self,
        clean_dir: str,
        enhanced_dir: str,
        noisy_dir: Optional[str] = None,
        model_name: str = "",
    ):
        """Evaluate all matching triplets in directories."""
        clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))
        
        for clean_path in clean_files:
            fname = Path(clean_path).stem
            
            enhanced_path = os.path.join(enhanced_dir, f"{fname}.wav")
            if not os.path.exists(enhanced_path):
                log.warning(f"No enhanced file for {fname}, skipping")
                continue
            
            if noisy_dir:
                noisy_path = os.path.join(noisy_dir, f"{fname}.wav")
                if not os.path.exists(noisy_path):
                    log.warning(f"No noisy file for {fname}, skipping")
                    continue
            else:
                # If no separate noisy dir, use clean as reference and skip input SNR
                noisy_path = clean_path  # will show 0 dB input SNR
            
            self.evaluate_triplet(
                clean_path=clean_path,
                noisy_path=noisy_path,
                enhanced_path=enhanced_path,
                file_id=fname,
                model_name=model_name,
            )
    
    def print_summary(self):
        """Print summary statistics and PS target compliance."""
        if not self.records:
            log.warning("No records to summarize")
            return
        
        print(f"\n{'='*70}")
        print(f"EVALUATION SUMMARY — {len(self.records)} files evaluated")
        print(f"{'='*70}")
        
        # Collect metrics
        snr_in = [r.input_snr_db for r in self.records if not np.isnan(r.input_snr_db)]
        snr_out = [r.output_snr_db for r in self.records if not np.isnan(r.output_snr_db)]
        snr_imp = [r.snr_improvement_db for r in self.records if not np.isnan(r.snr_improvement_db)]
        si_snr = [r.si_snr_db for r in self.records if not np.isnan(r.si_snr_db)]
        stoi_vals = [r.stoi for r in self.records if not np.isnan(r.stoi)]
        pesq_vals = [r.pesq for r in self.records if not np.isnan(r.pesq)]
        
        def stats(vals, name):
            if not vals:
                print(f"  {name:25s}  N/A (no data)")
                return
            arr = np.array(vals)
            return {
                "name": name,
                "mean": np.mean(arr),
                "std": np.std(arr),
                "min": np.min(arr),
                "max": np.max(arr),
                "median": np.median(arr),
            }
        
        metrics = [
            stats(snr_in, "Input SNR (dB)"),
            stats(snr_out, "Output SNR (dB)"),
            stats(snr_imp, "SNR Improvement (dB)"),
            stats(si_snr, "SI-SNR (dB)"),
            stats(stoi_vals, "STOI"),
            stats(pesq_vals, "PESQ"),
        ]
        
        for m in metrics:
            if m is None:
                continue
            print(f"  {m['name']:25s}  mean={m['mean']:.3f}  std={m['std']:.3f}  "
                  f"min={m['min']:.3f}  max={m['max']:.3f}")
        
        # PS 26052 target compliance
        print(f"\n  PS 26052 Target Compliance:")
        print(f"  {'-'*50}")
        
        if snr_out:
            pct_snr = sum(1 for s in snr_out if s >= self.PS_TARGETS["snr_db"]) / len(snr_out) * 100
            print(f"  SNR > {self.PS_TARGETS['snr_db']:.0f} dB:  "
                  f"{pct_snr:.1f}% of files ({sum(1 for s in snr_out if s >= self.PS_TARGETS['snr_db'])}/{len(snr_out)})")
        
        if stoi_vals:
            pct_stoi = sum(1 for s in stoi_vals if s >= self.PS_TARGETS["stoi"]) / len(stoi_vals) * 100
            print(f"  STOI > {self.PS_TARGETS['stoi']:.2f}:  "
                  f"{pct_stoi:.1f}% of files ({sum(1 for s in stoi_vals if s >= self.PS_TARGETS['stoi'])}/{len(stoi_vals)})")
        
        if pesq_vals:
            pct_pesq = sum(1 for s in pesq_vals if s >= self.PS_TARGETS["pesq"]) / len(pesq_vals) * 100
            print(f"  PESQ > {self.PS_TARGETS['pesq']:.1f}:  "
                  f"{pct_pesq:.1f}% of files ({sum(1 for s in pesq_vals if s >= self.PS_TARGETS['pesq'])}/{len(pesq_vals)})")
        
        print(f"{'='*70}\n")
    
    def save_results(self, output_path: str = "results/evaluation_results.csv"):
        """Save results to CSV."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if not self.records:
            log.warning("No records to save")
            return
        
        fieldnames = list(asdict(self.records[0]).keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.records:
                writer.writerow(asdict(record))
        
        log.info(f"Results saved to {output_path}")
    
    def save_plots(self, output_dir: str = "results/plots"):
        """Generate evaluation plots."""
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            log.warning("matplotlib not installed, skipping plots")
            return
        
        snr_in = [r.input_snr_db for r in self.records if not np.isnan(r.input_snr_db)]
        snr_out = [r.output_snr_db for r in self.records if not np.isnan(r.output_snr_db)]
        stoi_vals = [r.stoi for r in self.records if not np.isnan(r.stoi)]
        pesq_vals = [r.pesq for r in self.records if not np.isnan(r.pesq)]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Input vs Output SNR
        if snr_in and snr_out:
            axes[0, 0].scatter(snr_in, snr_out, alpha=0.5, s=10)
            axes[0, 0].plot([-15, 25], [-15, 25], "r--", label="No improvement")
            axes[0, 0].set_xlabel("Input SNR (dB)")
            axes[0, 0].set_ylabel("Output SNR (dB)")
            axes[0, 0].set_title("Input vs Output SNR")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # SNR Improvement distribution
        snr_imp = [r.snr_improvement_db for r in self.records if not np.isnan(r.snr_improvement_db)]
        if snr_imp:
            axes[0, 1].hist(snr_imp, bins=30, edgecolor="black", alpha=0.7)
            axes[0, 1].axvline(0, color="r", linestyle="--", label="No improvement")
            axes[0, 1].set_xlabel("SNR Improvement (dB)")
            axes[0, 1].set_ylabel("Count")
            axes[0, 1].set_title("SNR Improvement Distribution")
            axes[0, 1].legend()
        
        # STOI distribution
        if stoi_vals:
            axes[1, 0].hist(stoi_vals, bins=20, edgecolor="black", alpha=0.7, color="green")
            axes[1, 0].axvline(self.PS_TARGETS["stoi"], color="r", linestyle="--",
                               label=f"PS Target ({self.PS_TARGETS['stoi']})")
            axes[1, 0].set_xlabel("STOI")
            axes[1, 0].set_ylabel("Count")
            axes[1, 0].set_title("STOI Distribution")
            axes[1, 0].legend()
        
        # PESQ distribution
        if pesq_vals:
            axes[1, 1].hist(pesq_vals, bins=20, edgecolor="black", alpha=0.7, color="orange")
            axes[1, 1].axvline(self.PS_TARGETS["pesq"], color="r", linestyle="--",
                               label=f"PS Target ({self.PS_TARGETS['pesq']})")
            axes[1, 1].set_xlabel("PESQ")
            axes[1, 1].set_ylabel("Count")
            axes[1, 1].set_title("PESQ Distribution")
            axes[1, 1].legend()
        
        plt.suptitle(f"PS 26052 Evaluation — {len(self.records)} files", fontsize=14)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "evaluation_summary.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        log.info(f"Plots saved to {plot_path}")


# ---------------------------------------------------------------------------
# Compare Mode
# ---------------------------------------------------------------------------
def compare_results(csv_paths: list[str]):
    """Compare evaluation results from multiple models."""
    print(f"\n{'='*70}")
    print(f"MODEL COMPARISON")
    print(f"{'='*70}")
    
    for path in csv_paths:
        records = []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
        
        name = Path(path).stem
        n = len(records)
        
        stoi_vals = [float(r["stoi"]) for r in records if r.get("stoi") and r["stoi"] != "nan"]
        pesq_vals = [float(r["pesq"]) for r in records if r.get("pesq") and r["pesq"] != "nan"]
        snr_imp = [float(r["snr_improvement_db"]) for r in records
                    if r.get("snr_improvement_db") and r["snr_improvement_db"] != "nan"]
        
        print(f"\n  Model: {name} ({n} files)")
        if snr_imp:
            print(f"    SNR Improvement:  mean={np.mean(snr_imp):.2f} dB  std={np.std(snr_imp):.2f} dB")
        if stoi_vals:
            print(f"    STOI:             mean={np.mean(stoi_vals):.4f}  std={np.std(stoi_vals):.4f}")
        if pesq_vals:
            print(f"    PESQ:             mean={np.mean(pesq_vals):.2f}  std={np.std(pesq_vals):.2f}")
    
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PS 26052 Evaluation Harness")
    parser.add_argument("--clean", type=str, help="Path to clean reference audio")
    parser.add_argument("--noisy", type=str, help="Path to noisy audio")
    parser.add_argument("--enhanced", type=str, help="Path to enhanced audio")
    parser.add_argument("--triplets-dir", type=str, help="Directory with triplets")
    parser.add_argument("--dataset-dir", type=str, help="Dataset directory to evaluate")
    parser.add_argument("--model-name", type=str, default="", help="Model name for labeling")
    parser.add_argument("--output", type=str, default="results/evaluation_results.csv",
                        help="Output CSV path")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--compare", nargs="+", help="Compare multiple result CSVs")
    args = parser.parse_args()
    
    if args.compare:
        compare_results(args.compare)
        return
    
    harness = EvalHarness()
    
    if args.clean and args.enhanced:
        # Single triplet evaluation
        harness.evaluate_triplet(
            clean_path=args.clean,
            noisy_path=args.noisy or args.clean,
            enhanced_path=args.enhanced,
            model_name=args.model_name,
        )
    elif args.dataset_dir:
        # Evaluate all .wav files in directory (assumes clean = enhanced for demo)
        harness.evaluate_directory(
            clean_dir=args.dataset_dir,
            enhanced_dir=args.dataset_dir,
            model_name=args.model_name,
        )
    elif args.triplets_dir:
        # Look for clean/, noisy/, enhanced/ subdirectories
        clean_dir = os.path.join(args.triplets_dir, "clean")
        noisy_dir = os.path.join(args.triplets_dir, "noisy")
        enhanced_dir = os.path.join(args.triplets_dir, "enhanced")
        harness.evaluate_directory(
            clean_dir=clean_dir,
            noisy_dir=noisy_dir if os.path.exists(noisy_dir) else None,
            enhanced_dir=enhanced_dir,
            model_name=args.model_name,
        )
    else:
        parser.print_help()
        print("\nExample:")
        print("  python 02_evaluation_harness.py --clean clean.wav --enhanced enhanced.wav")
        print("  python 02_evaluation_harness.py --dataset-dir data/mixed/test/")
        return
    
    # Results
    harness.print_summary()
    harness.save_results(args.output)
    if not args.no_plots:
        harness.save_plots()
    
    print(f"Results: {args.output}")
    print(f"\nNext step: Build streaming pipeline (03_streaming_skeleton.py)")


if __name__ == "__main__":
    main()
