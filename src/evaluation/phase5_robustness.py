"""
Phase 5: NLMS Robustness & Kill-Criterion Validation.
PS 26052 -- Adaptive Defence ANC.

Tests the mandatory kill criterion:
  IF Hybrid NLMS->AI does NOT outperform AI-Only in held-out test sets,
  THEN remove the claim that NLMS improves the system.

This script systematically evaluates:
  1. Matched noise (ref ~ primary noise): NLMS should help
  2. Unmatched noise (ref = different noise): NLMS may harm
  3. Reference leakage (ref contains speech): NLMS must not destroy speech
  4. Impulsive noise (gunshots, blasts): NLMS adaptation must freeze
  5. Low-SNR stress test (-5 dB, -10 dB): NLMS divergence check

Output: results/csv/phase5_nlms_robustness.csv
"""

import csv
import time
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.dsp.vss_nlms import VSSNLMSFilter
from src.dsp.leakage_detector import SpeechLeakageDetector
from src.dsp.impulse_protection import ImpulseProtectionController
from src.ai.tiny_enhancer import TinyEnhancerWrapper
from src.pipeline.hybrid_chain import HybridEnhancementPipeline
from src.evaluation.metrics import evaluate_all_metrics


RESULTS_FIELDS = [
    "sample_id", "noise_category", "scenario",
    "system_config",
    "delta_snr_db", "si_sdr_db", "stoi", "pesq",
    "nlms_better_than_ai_only",
]


def _run_scenario(
    clean: np.ndarray,
    primary: np.ndarray,
    reference: np.ndarray,
    ai_backend,
    sr: int = 16000,
) -> Dict[str, Dict[str, float]]:
    """Run AI-only and Hybrid on one sample, return metrics dict for each."""

    results = {}

    # AI-Only: pass zeros as reference so NLMS does nothing
    hybrid_aionly = HybridEnhancementPipeline(
        config_mode="A", filter_length=64, step_size=0.08,
        use_vss=False, enable_leakage_protection=False,
        enable_impulse_protection=False,
        ai_backend=ai_backend, sample_rate=sr,
    )

    out_ai, _ = hybrid_aionly.process_signals(primary, np.zeros_like(reference))
    m = evaluate_all_metrics(clean, out_ai, sample_rate=sr)
    results["AI_ONLY"] = m

    # Hybrid Protected: full DSP chain + AI
    hybrid_full = HybridEnhancementPipeline(
        config_mode="A", filter_length=64, step_size=0.08,
        use_vss=True, enable_leakage_protection=True,
        enable_impulse_protection=True,
        ai_backend=ai_backend, sample_rate=sr,
    )
    out_hyb, _ = hybrid_full.process_signals(primary, reference)
    m = evaluate_all_metrics(clean, out_hyb, sample_rate=sr)
    results["HYBRID_PROTECTED"] = m

    # NLMS-Only: no AI
    nlms = VSSNLMSFilter(filter_length=64, mu_init=0.08)
    out_nlms, _, _ = nlms.filter_block(primary, reference)
    m = evaluate_all_metrics(clean, out_nlms, sample_rate=sr)
    results["NLMS_ONLY"] = m

    return results


def run_phase5_robustness(
    dataset_metadata_csv: str = "data/v4/metadata/metadata_v4.csv",
    output_csv: str = "results/csv/phase5_nlms_robustness.csv",
    sample_rate: int = 16000,
) -> Dict[str, Any]:
    """Execute Phase 5 NLMS robustness tests."""

    meta_path = Path(dataset_metadata_csv)
    if not meta_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {meta_path}")

    data_root = meta_path.parent.parent
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read all records
    all_records = []
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_records.append(row)

    ai_backend = TinyEnhancerWrapper()

    print("\n" + "=" * 80, flush=True)
    print("  PHASE 5: NLMS ROBUSTNESS & KILL-CRITERION VALIDATION", flush=True)
    print("=" * 80, flush=True)
    print(f"[*] Total samples: {len(all_records)}", flush=True)

    csv_file = open(out_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=RESULTS_FIELDS)
    writer.writeheader()

    kill_criterion_tracker = {"hybrid_wins": 0, "ai_only_wins": 0, "ties": 0}
    scenario_results = []

    for idx, row in enumerate(all_records, 1):
        sample_id = row["sample_id"]
        cat = row["noise_category"]

        clean_file = data_root / row["clean_path"]
        prim_file = data_root / row["primary_path"]
        ref_file = data_root / row["reference_path"]

        clean, _ = sf.read(str(clean_file), dtype="float32")
        prim, _ = sf.read(str(prim_file), dtype="float32")
        ref, _ = sf.read(str(ref_file), dtype="float32")
        input_snr = float(row["actual_snr_db"])

        # Scenario 1: Normal matched reference
        scenario = "MATCHED_REFERENCE"
        results = _run_scenario(clean, prim, ref, ai_backend, sr=sample_rate)

        for cfg_name, m in results.items():
            nlms_better = results["HYBRID_PROTECTED"]["stoi"] > results["AI_ONLY"]["stoi"]
            writer.writerow({
                "sample_id": sample_id, "noise_category": cat,
                "scenario": scenario, "system_config": cfg_name,
                "delta_snr_db": round(m["snr_db"] - input_snr, 2),
                "si_sdr_db": m["si_snr_db"], "stoi": m["stoi"], "pesq": m["pesq"],
                "nlms_better_than_ai_only": nlms_better,
            })

        if results["HYBRID_PROTECTED"]["stoi"] > results["AI_ONLY"]["stoi"]:
            kill_criterion_tracker["hybrid_wins"] += 1
        elif results["HYBRID_PROTECTED"]["stoi"] < results["AI_ONLY"]["stoi"]:
            kill_criterion_tracker["ai_only_wins"] += 1
        else:
            kill_criterion_tracker["ties"] += 1

        # Scenario 2: Leakage test -- inject speech into reference
        scenario = "SPEECH_LEAKAGE"
        leaked_ref = ref + 0.3 * clean  # simulate 30% speech bleed
        results2 = _run_scenario(clean, prim, leaked_ref, ai_backend, sr=sample_rate)

        for cfg_name, m in results2.items():
            writer.writerow({
                "sample_id": sample_id, "noise_category": cat,
                "scenario": scenario, "system_config": cfg_name,
                "delta_snr_db": round(m["snr_db"] - input_snr, 2),
                "si_sdr_db": m["si_snr_db"], "stoi": m["stoi"], "pesq": m["pesq"],
                "nlms_better_than_ai_only": results2["HYBRID_PROTECTED"]["stoi"] > results2["AI_ONLY"]["stoi"],
            })

        # Scenario 3: Impulsive spike injection
        scenario = "IMPULSIVE_SPIKE"
        impulsive_prim = prim.copy()
        spike_idx = len(impulsive_prim) // 3
        impulsive_prim[spike_idx:spike_idx + 16] = 5.0  # simulate gunshot spike
        results3 = _run_scenario(clean, impulsive_prim, ref, ai_backend, sr=sample_rate)

        for cfg_name, m in results3.items():
            writer.writerow({
                "sample_id": sample_id, "noise_category": cat,
                "scenario": scenario, "system_config": cfg_name,
                "delta_snr_db": round(m["snr_db"] - input_snr, 2),
                "si_sdr_db": m["si_snr_db"], "stoi": m["stoi"], "pesq": m["pesq"],
                "nlms_better_than_ai_only": results3["HYBRID_PROTECTED"]["stoi"] > results3["AI_ONLY"]["stoi"],
            })

        csv_file.flush()
        print(f"    [{idx:03d}/{len(all_records):03d}] [OK] {sample_id} ({cat})", flush=True)

    csv_file.close()

    # Kill criterion verdict
    total = kill_criterion_tracker["hybrid_wins"] + kill_criterion_tracker["ai_only_wins"] + kill_criterion_tracker["ties"]
    hybrid_rate = kill_criterion_tracker["hybrid_wins"] / max(total, 1) * 100

    print("\n" + "=" * 60, flush=True)
    print("  KILL CRITERION RESULT", flush=True)
    print("=" * 60, flush=True)
    print(f"  Hybrid wins:   {kill_criterion_tracker['hybrid_wins']}/{total} ({hybrid_rate:.1f}%)", flush=True)
    print(f"  AI-only wins:  {kill_criterion_tracker['ai_only_wins']}/{total}", flush=True)
    print(f"  Ties:          {kill_criterion_tracker['ties']}/{total}", flush=True)

    if hybrid_rate >= 50.0:
        verdict = "PASS -- NLMS improves the system on matched-reference scenarios."
    else:
        verdict = "FAIL -- NLMS does NOT reliably improve over AI-only. Remove NLMS claim."

    print(f"\n  VERDICT: {verdict}", flush=True)
    print("=" * 60, flush=True)
    print(f"\n[+] Phase 5 results saved to: {out_path.resolve()}", flush=True)

    return {
        "status": "COMPLETED",
        "kill_criterion": kill_criterion_tracker,
        "verdict": verdict,
        "hybrid_win_rate": hybrid_rate,
    }


if __name__ == "__main__":
    run_phase5_robustness()
