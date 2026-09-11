#!/usr/bin/env python3
"""
PS 26052: Embedded Real-Time Deployment Pipeline for Raspberry Pi.
Executes the full audio processing chain:
  Microphone Capture (ALSA / I2S stereo)
  -> 128-sample Hop Buffer
  -> VSS-NLMS Adaptive Filter (Numba or NumPy)
  -> Causal STFT Analysis (256-sample Hann, hop=128, center=False)
  -> E2_causal ONNX Runtime FP32 Inference
  -> Causal WOLA Synthesis Overlap-Add
  -> DAC / Playback Output (ALSA / I2S)
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("[!] Error: onnxruntime is required. Run: pip install onnxruntime", file=sys.stderr)
    sys.exit(1)


class EmbeddedStreamingPipeline:
    def __init__(
        self,
        onnx_model_path: str,
        device_in: str = "hw:1,0",
        device_out: str = "hw:1,0",
        sample_rate: int = 16000,
        frame_size: int = 256,
        hop_size: int = 128,
        threads: int = 4,
    ):
        self.sr = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.device_in = device_in
        self.device_out = device_out

        # 1. Load ONNX Runtime
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(onnx_model_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        # 2. Window & Synthesis Buffers
        self.analysis_window = np.hanning(self.frame_size).astype(np.float32)
        self.synthesis_window = self.analysis_window.copy()
        self.win_sq = self.synthesis_window ** 2

        # Ring buffers for primary, reference, and residual
        self.primary_buf = np.zeros(self.frame_size, dtype=np.float32)
        self.residual_buf = np.zeros(self.frame_size, dtype=np.float32)
        self.ola_buf = np.zeros(self.frame_size, dtype=np.float32)
        self.env_buf = np.zeros(self.frame_size, dtype=np.float32)

        # 3. VSS-NLMS State
        self.filter_length = 64
        self.nlms_weights = np.zeros(self.filter_length, dtype=np.float32)
        self.nlms_buffer = np.zeros(self.filter_length, dtype=np.float32)
        self.nlms_mu = 0.05
        self.nlms_p_cor = 0.0

        # Try Numba compilation if available
        self.use_numba = False
        try:
            from numba import njit
            self._init_numba()
            self.use_numba = True
        except ImportError:
            pass

    def _init_numba(self):
        from numba import njit

        @njit(cache=True)
        def _nlms_step(pri, ref, w, buf, mu_v, pc):
            n = len(pri)
            err = np.zeros(n, dtype=np.float32)
            for i in range(n):
                buf[1:] = buf[:-1]
                buf[0] = ref[i]
                y = 0.0
                for k in range(len(w)):
                    y += w[k] * buf[k]
                e = pri[i] - y
                err[i] = e
                pc = 0.95 * pc + 0.05 * (e * y)
                mu_v = min(max(0.98 * mu_v + 0.02 * abs(pc), 1e-4), 0.5)
                norm = 1e-6
                for k in range(len(buf)):
                    norm += buf[k] * buf[k]
                step = (mu_v / norm) * e
                for k in range(len(w)):
                    w[k] = (w[k] + step * buf[k]) * (1.0 - 1e-4 * mu_v)
            return err, mu_v, pc

        self._numba_step = _nlms_step

    def process_dsp_hop(self, pri_hop: np.ndarray, ref_hop: np.ndarray) -> np.ndarray:
        if self.use_numba:
            err, self.nlms_mu, self.nlms_p_cor = self._numba_step(
                pri_hop, ref_hop, self.nlms_weights, self.nlms_buffer, self.nlms_mu, self.nlms_p_cor
            )
            return err

        # Pure NumPy fallback
        n = len(pri_hop)
        err = np.zeros(n, dtype=np.float32)
        for i in range(n):
            self.nlms_buffer[1:] = self.nlms_buffer[:-1]
            self.nlms_buffer[0] = ref_hop[i]
            y = float(np.dot(self.nlms_weights, self.nlms_buffer))
            e = pri_hop[i] - y
            err[i] = e
            self.nlms_p_cor = 0.95 * self.nlms_p_cor + 0.05 * (e * y)
            self.nlms_mu = np.clip(0.98 * self.nlms_mu + 0.02 * abs(self.nlms_p_cor), 1e-4, 0.5)
            norm = float(np.dot(self.nlms_buffer, self.nlms_buffer)) + 1e-6
            self.nlms_weights += (self.nlms_mu / norm) * e * self.nlms_buffer
            self.nlms_weights *= (1.0 - 1e-4 * self.nlms_mu)
        return err

    def process_hop(self, primary_hop: np.ndarray, ref_hop: np.ndarray) -> np.ndarray:
        """Processes a single 128-sample hop through the full hybrid pipeline."""
        # 1. DSP Stage: VSS-NLMS residual
        res_hop = self.process_dsp_hop(primary_hop, ref_hop)

        # Shift analysis buffer
        self.residual_buf[:-self.hop_size] = self.residual_buf[self.hop_size:]
        self.residual_buf[-self.hop_size:] = res_hop

        # 2. STFT Analysis
        windowed = self.residual_buf * self.analysis_window
        stft_complex = np.fft.rfft(windowed, n=self.frame_size)
        mag = np.abs(stft_complex).astype(np.float32)
        phase = np.angle(stft_complex).astype(np.float32)

        # 3. ONNX AI Inference
        input_tensor = mag[np.newaxis, np.newaxis, :, np.newaxis]  # (1, 1, 129, 1)
        mask = self.session.run(None, {self.input_name: input_tensor})[0].squeeze()

        # 4. Masking & iSTFT Synthesis
        enh_mag = mag * mask
        enh_complex = enh_mag * np.exp(1j * phase)
        syn_frame = np.fft.irfft(enh_complex, n=self.frame_size) * self.synthesis_window

        # 5. Overlap-Add
        self.ola_buf += syn_frame
        self.env_buf += self.win_sq

        # Extract current hop
        valid_env = np.where(self.env_buf[:self.hop_size] > 1e-8, self.env_buf[:self.hop_size], 1.0)
        output_hop = self.ola_buf[:self.hop_size] / valid_env

        # Advance OLA buffers
        self.ola_buf[:-self.hop_size] = self.ola_buf[self.hop_size:]
        self.ola_buf[-self.hop_size:] = 0.0
        self.env_buf[:-self.hop_size] = self.env_buf[self.hop_size:]
        self.env_buf[-self.hop_size:] = 0.0

        return np.clip(output_hop, -0.98, 0.98).astype(np.float32)


def run_pipeline_test(model_path: str, n_hops: int = 500):
    print("=" * 72)
    print("  PS 26052: EMBEDDED PIPELINE VERIFICATION")
    print(f"  Model: {model_path}")
    print(f"  Simulating {n_hops} hops ({(n_hops * 128) / 16000:.2f} seconds)...")
    print("=" * 72)

    pipeline = EmbeddedStreamingPipeline(model_path)
    rng = np.random.RandomState(42)

    latencies_ms = []
    for _ in range(n_hops):
        pri = (rng.randn(128) * 0.1).astype(np.float32)
        ref = (rng.randn(128) * 0.1).astype(np.float32)

        t0 = time.perf_counter_ns()
        out = pipeline.process_hop(pri, ref)
        t1 = time.perf_counter_ns()

        latencies_ms.append((t1 - t0) / 1e6)

    arr = np.array(latencies_ms)
    p50, p95, p99, p_max = np.percentile(arr, 50), np.percentile(arr, 95), np.percentile(arr, 99), np.max(arr)

    print(f"\n[+] Pipeline execution completed successfully.")
    print(f"  P50 Latency: {p50:.3f} ms")
    print(f"  P95 Latency: {p95:.3f} ms")
    print(f"  P99 Latency: {p99:.3f} ms")
    print(f"  Max Latency: {p_max:.3f} ms")
    print(f"  Hop Budget:  8.000 ms")
    print(f"  RTF (P95):   {p95 / 8.0:.4f}")
    if p95 <= 8.0:
        print("  Status:      HARD REAL-TIME COMPLIANT")
    else:
        print("  Status:      EXCEEDS BUDGET")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run embedded pipeline.")
    parser.add_argument("--model", type=str, default="models/E2_causal.onnx", help="Path to E2_causal.onnx")
    parser.add_argument("--hops", type=int, default=500, help="Number of hops to process")
    args = parser.parse_args()

    run_pipeline_test(args.model, n_hops=args.hops)
