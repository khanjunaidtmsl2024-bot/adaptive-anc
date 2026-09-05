"""
Gate 6: Causality Auditor and Anti-Cheating Verification Engine.
PS 26052 — Adaptive Defence ANC.

Mathematical Causality Criterion:
For any input signal pair A[n] and B[n] such that:
    A[n] == B[n]  for all n <= T0
    A[n] != B[n]  for all n > T0 (adversarial future mutation)

A genuinely causal system S must satisfy:
    S{A}[n] == S{B}[n]  for all n <= T0 (to within numerical precision eps < 1e-6)

If S{A}[n] != S{B}[n] for any n <= T0, the system contains non-causal lookahead,
future-frame leakage, centering padding, or future-dependent normalization.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.streaming.causal_engine import CausalStreamingEngine
from src.dsp.vss_nlms import VSSNLMSFilter


class CausalityVerifier:
    """Verifies strict mathematical causality of DSP and streaming pipelines."""

    def __init__(self, sr: int = 16000, frame_size: int = 512, hop_size: int = 128):
        self.sr = sr
        self.frame_size = frame_size
        self.hop_size = hop_size

    def verify_dsp_filter_causality(self, n_samples: int = 8000, t_cutoff: int = 4000) -> Dict[str, Any]:
        """Verifies causality of VSS-NLMS adaptive filter."""
        rng = np.random.RandomState(42)
        primary_A = rng.randn(n_samples).astype(np.float32)
        ref_A = rng.randn(n_samples).astype(np.float32)

        # Mutate future after t_cutoff
        primary_B = primary_A.copy()
        ref_B = ref_A.copy()
        primary_B[t_cutoff:] += rng.uniform(-10.0, 10.0, n_samples - t_cutoff).astype(np.float32)
        ref_B[t_cutoff:] += rng.uniform(-10.0, 10.0, n_samples - t_cutoff).astype(np.float32)

        f1 = VSSNLMSFilter(filter_length=64)
        out_A, _, _ = f1.filter_block(primary_A, ref_A)

        f2 = VSSNLMSFilter(filter_length=64)
        out_B, _, _ = f2.filter_block(primary_B, ref_B)

        # Compare outputs up to t_cutoff
        diff_past = np.max(np.abs(out_A[:t_cutoff] - out_B[:t_cutoff]))
        diff_future = np.max(np.abs(out_A[t_cutoff:] - out_B[t_cutoff:]))

        is_strictly_causal = bool(diff_past < 1e-6)
        future_is_distinct = bool(diff_future > 1.0)

        return {
            "component": "VSSNLMSFilter",
            "t_cutoff": t_cutoff,
            "max_diff_past": float(diff_past),
            "max_diff_future": float(diff_future),
            "is_strictly_causal": is_strictly_causal,
            "future_is_distinct": future_is_distinct,
        }

    def verify_streaming_engine_causality(
        self,
        n_samples: int = 8000,
        t_cutoff: int = 4096
    ) -> Dict[str, Any]:
        """
        Verifies causality of the complete CausalStreamingEngine.
        Takes into account the causal algorithmic pipeline latency (frame_size - hop_size).
        """
        rng = np.random.RandomState(123)
        t = np.linspace(0, n_samples / self.sr, n_samples, endpoint=False)
        clean = (0.5 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 1200 * t)).astype(np.float32)
        noise = rng.randn(n_samples).astype(np.float32) * 0.1

        primary_A = clean + noise
        ref_A = np.roll(noise, 2)

        primary_B = primary_A.copy()
        ref_B = ref_A.copy()
        # Add massive +50 dB chaotic transient burst after t_cutoff
        burst = rng.uniform(-20.0, 20.0, n_samples - t_cutoff).astype(np.float32)
        primary_B[t_cutoff:] += burst
        ref_B[t_cutoff:] += burst

        engine_A = CausalStreamingEngine(
            frame_size=self.frame_size,
            hop_size=self.hop_size,
            sample_rate=self.sr
        )
        out_A, _ = engine_A.process_signal(primary_A, ref_A)

        engine_B = CausalStreamingEngine(
            frame_size=self.frame_size,
            hop_size=self.hop_size,
            sample_rate=self.sr
        )
        out_B, _ = engine_B.process_signal(primary_B, ref_B)

        # In streaming hop processing, output at sample n is determined by inputs up to hop boundary.
        # Ensure that past audio up to t_cutoff is identical
        diff_past = np.max(np.abs(out_A[:t_cutoff] - out_B[:t_cutoff]))
        diff_future = np.max(np.abs(out_A[t_cutoff:] - out_B[t_cutoff:]))

        is_strictly_causal = bool(diff_past < 1e-5)

        return {
            "component": "CausalStreamingEngine",
            "t_cutoff": t_cutoff,
            "max_diff_past": float(diff_past),
            "max_diff_future": float(diff_future),
            "is_strictly_causal": is_strictly_causal,
        }


def run_full_causality_audit() -> List[Dict[str, Any]]:
    """Runs causality audit across multiple cutoff points."""
    verifier = CausalityVerifier()
    cutoffs = [1024, 2048, 4096, 6144]
    reports = []

    print("[GATE 6] Running Mathematical Causality Audit...", flush=True)

    # 1. DSP filter causality
    dsp_rep = verifier.verify_dsp_filter_causality(t_cutoff=4000)
    reports.append(dsp_rep)
    print(f"  VSS-NLMS Filter: Max Past Diff = {dsp_rep['max_diff_past']:.2e} -> Strictly Causal: {dsp_rep['is_strictly_causal']}")

    # 2. Streaming engine across cutoffs
    for t_cut in cutoffs:
        engine_rep = verifier.verify_streaming_engine_causality(t_cutoff=t_cut)
        reports.append(engine_rep)
        print(f"  Streaming Engine (t_cut={t_cut:4d}): Max Past Diff = {engine_rep['max_diff_past']:.2e} -> Strictly Causal: {engine_rep['is_strictly_causal']}")

    all_passed = all(r["is_strictly_causal"] for r in reports)
    print(f"\n[GATE 6] Causality Audit Result: {'100% CAUSAL (PASSED)' if all_passed else 'FAILED (NON-CAUSAL LEAKAGE DETECTED)'}\n", flush=True)
    return reports


if __name__ == "__main__":
    run_full_causality_audit()
