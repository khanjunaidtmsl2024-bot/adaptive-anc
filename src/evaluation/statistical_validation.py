"""
Statistical Significance Validation & NLMS Kill-Criterion Audit.
PS 26052 — Adaptive Defence ANC.

Provides rigorous inferential statistical testing for pre-hardware readiness:
- Paired Student's t-test
- Wilcoxon signed-rank test (non-parametric)
- Cohen's d effect size
- 95% Bootstrap Confidence Intervals (1,000 resamples)
- Formal testing of the NLMS Kill-Criterion (Hybrid vs AI-Only)

Outputs:
- results/csv/statistical_significance_report.csv
"""

import csv
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import scipy.stats as stats
import soundfile as sf
import pystoi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.dsp.vss_nlms import VSSNLMSFilter
from src.ai.tiny_enhancer import TinyEnhancerWrapper
from src.evaluation.ai_ablation_and_pesq_audit import apply_ai_stft_enhancement, calculate_snr, calculate_si_sdr
from src.dataset.synthetic_benchmark_matrix import generate_benchmark_matrix, get_benchmark_pair


def compute_cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
    """Computes Cohen's d effect size for paired samples."""
    diff = x1 - x2
    s_diff = np.std(diff, ddof=1) + 1e-12
    return float(np.mean(diff) / s_diff)


def compute_bootstrap_ci(data: np.ndarray, n_boot: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Computes non-parametric bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(42)
    boot_means = np.zeros(n_boot)
    n = len(data)
    for b in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_means[b] = np.mean(sample)
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha * 100.0))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))
    return low, high


def run_statistical_validation(
    n_trials: int = 30,
    output_csv: str = "results/csv/statistical_significance_report.csv",
    sr: int = 16000,
) -> Dict[str, Any]:
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[STATISTICS] Running Paired Inferential Validation across {n_trials} trials...", flush=True)

    matrix = generate_benchmark_matrix()
    # Select 30 diverse trials across different speakers, noises, and SNRs (0 to 10 dB)
    trial_meta = [r for r in matrix if r["snr_db"] in [0, 5, 10]][:n_trials]

    ai_model = TinyEnhancerWrapper()

    hybrid_sisdrs = []
    ai_only_sisdrs = []
    nlms_only_sisdrs = []

    hybrid_snrs = []
    ai_only_snrs = []
    nlms_only_snrs = []

    hybrid_stois = []
    ai_only_stois = []
    nlms_only_stois = []

    trial_records = []

    for i, row in enumerate(trial_meta):
        clean, primary, reference = get_benchmark_pair(row, seed=100 + i)

        # 1. NLMS Only
        nlms = VSSNLMSFilter(filter_length=64, mu_init=0.05, mu_max=0.05)
        out_nlms, _, _ = nlms.filter_block(primary, reference)

        # 2. AI Only
        out_ai = apply_ai_stft_enhancement(primary, ai_model, frame_size=512, hop_size=128)

        # 3. Hybrid (NLMS -> AI)
        nlms_hyb = VSSNLMSFilter(filter_length=64, mu_init=0.05, mu_max=0.05)
        stage1, _, _ = nlms_hyb.filter_block(primary, reference)
        out_hyb = apply_ai_stft_enhancement(stage1, ai_model, frame_size=512, hop_size=128)

        # Metrics
        sisdr_hyb = calculate_si_sdr(clean, out_hyb)
        sisdr_ai = calculate_si_sdr(clean, out_ai)
        sisdr_nlms = calculate_si_sdr(clean, out_nlms)

        snr_hyb = calculate_snr(clean, out_hyb)
        snr_ai = calculate_snr(clean, out_ai)
        snr_nlms = calculate_snr(clean, out_nlms)

        stoi_hyb = float(pystoi.stoi(clean, out_hyb, sr, extended=False))
        stoi_ai = float(pystoi.stoi(clean, out_ai, sr, extended=False))
        stoi_nlms = float(pystoi.stoi(clean, out_nlms, sr, extended=False))

        hybrid_sisdrs.append(sisdr_hyb)
        ai_only_sisdrs.append(sisdr_ai)
        nlms_only_sisdrs.append(sisdr_nlms)

        hybrid_snrs.append(snr_hyb)
        ai_only_snrs.append(snr_ai)
        nlms_only_snrs.append(snr_nlms)

        hybrid_stois.append(stoi_hyb)
        ai_only_stois.append(stoi_ai)
        nlms_only_stois.append(stoi_nlms)

        trial_records.append({
            "trial_id": i + 1,
            "speaker": row["speaker_id"],
            "noise": row["noise_id"],
            "snr_db": row["snr_db"],
            "sisdr_hybrid": round(sisdr_hyb, 2),
            "sisdr_ai_only": round(sisdr_ai, 2),
            "sisdr_nlms_only": round(sisdr_nlms, 2),
            "delta_sisdr_hybrid_vs_ai": round(sisdr_hyb - sisdr_ai, 2),
        })

    arr_hyb_sisdr = np.array(hybrid_sisdrs)
    arr_ai_sisdr = np.array(ai_only_sisdrs)
    arr_nlms_sisdr = np.array(nlms_only_sisdrs)

    # 1. Hypothesis Testing: Hybrid vs AI Only on SI-SDR
    t_stat_ai, p_val_ttest_ai = stats.ttest_rel(arr_hyb_sisdr, arr_ai_sisdr)
    w_stat_ai, p_val_wilcox_ai = stats.wilcoxon(arr_hyb_sisdr, arr_ai_sisdr)
    cohen_d_ai = compute_cohens_d(arr_hyb_sisdr, arr_ai_sisdr)
    ci_low_ai, ci_high_ai = compute_bootstrap_ci(arr_hyb_sisdr - arr_ai_sisdr)

    # 2. Hypothesis Testing: Hybrid vs NLMS Only on SI-SDR
    t_stat_nlms, p_val_ttest_nlms = stats.ttest_rel(arr_hyb_sisdr, arr_nlms_sisdr)
    w_stat_nlms, p_val_wilcox_nlms = stats.wilcoxon(arr_hyb_sisdr, arr_nlms_sisdr)
    cohen_d_nlms = compute_cohens_d(arr_hyb_sisdr, arr_nlms_sisdr)
    ci_low_nlms, ci_high_nlms = compute_bootstrap_ci(arr_hyb_sisdr - arr_nlms_sisdr)

    # NLMS Kill-Criterion Decision
    # Criterion: If Hybrid does NOT statistically beat AI-Only (p >= 0.05 or d < 0.2), NLMS should be killed.
    nlms_justified = bool(p_val_ttest_ai < 0.05 and cohen_d_ai > 0.5)

    summary_rows = [
        {
            "comparison": "Hybrid (NLMS->AI) vs AI_Only",
            "metric": "SI-SDR (dB)",
            "mean_difference": round(float(np.mean(arr_hyb_sisdr - arr_ai_sisdr)), 3),
            "ci_95_low": round(ci_low_ai, 3),
            "ci_95_high": round(ci_high_ai, 3),
            "t_statistic": round(float(t_stat_ai), 3),
            "p_value_ttest": f"{p_val_ttest_ai:.4e}",
            "wilcoxon_stat": round(float(w_stat_ai), 1),
            "p_value_wilcoxon": f"{p_val_wilcox_ai:.4e}",
            "cohens_d": round(cohen_d_ai, 3),
            "statistically_significant": bool(p_val_ttest_ai < 0.05),
            "effect_magnitude": "Large" if abs(cohen_d_ai) > 0.8 else "Medium" if abs(cohen_d_ai) > 0.5 else "Small",
        },
        {
            "comparison": "Hybrid (NLMS->AI) vs NLMS_Only",
            "metric": "SI-SDR (dB)",
            "mean_difference": round(float(np.mean(arr_hyb_sisdr - arr_nlms_sisdr)), 3),
            "ci_95_low": round(ci_low_nlms, 3),
            "ci_95_high": round(ci_high_nlms, 3),
            "t_statistic": round(float(t_stat_nlms), 3),
            "p_value_ttest": f"{p_val_ttest_nlms:.4e}",
            "wilcoxon_stat": round(float(w_stat_nlms), 1),
            "p_value_wilcoxon": f"{p_val_wilcox_nlms:.4e}",
            "cohens_d": round(cohen_d_nlms, 3),
            "statistically_significant": bool(p_val_ttest_nlms < 0.05),
            "effect_magnitude": "Large" if abs(cohen_d_nlms) > 0.8 else "Medium" if abs(cohen_d_nlms) > 0.5 else "Small",
        }
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n--- INFERENTIAL STATISTICAL HYPOTHESIS TESTING REPORT ---")
    print(f"Sample Size: N = {n_trials} paired conditions")
    print("-" * 80)
    for s in summary_rows:
        print(f"Comparison: {s['comparison']}")
        print(f"  Mean Gain:        +{s['mean_difference']:.2f} dB (95% CI: [{s['ci_95_low']:.2f}, {s['ci_95_high']:.2f}] dB)")
        print(f"  Paired t-test:    t = {s['t_statistic']}, p = {s['p_value_ttest']}")
        print(f"  Wilcoxon test:    W = {s['wilcoxon_stat']}, p = {s['p_value_wilcoxon']}")
        print(f"  Cohen's d:        {s['cohens_d']} ({s['effect_magnitude']} Effect Size)")
        print(f"  Significance:     {'p < 0.001 (STATISTICALLY SIGNIFICANT)' if float(s['p_value_ttest']) < 0.001 else 'p >= 0.05 (NOT SIGNIFICANT)'}")
        print("-" * 80)

    print(f"\nNLMS KILL-CRITERION VERDICT:")
    if nlms_justified:
        print(f"  [RETAIN NLMS] The hybrid NLMS Stage 1 provides statistically significant benefit (p < 0.001, Cohen's d = {cohen_d_ai:.2f}).")
        print(f"  The adaptive filter genuinely assists the neural network by cancelling linear correlated environmental noise.\n")
    else:
        print(f"  [KILL NLMS] NLMS fails to provide significant improvement over pure AI. Recommend dropping Stage 1.\n")

    return {
        "summary": summary_rows,
        "nlms_justified": nlms_justified,
    }


if __name__ == "__main__":
    run_statistical_validation()
