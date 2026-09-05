#!/usr/bin/env python3
"""
ADAPTIVE-DEFENCE ANC — Master CLI & Execution Engine
Smart India Hackathon (SIH) 2026 — Problem Statement ID: 26052
DRDO / Department of Defence Production / iDEX — Smart Vehicles

Master Command Line Interface providing single-entrypoint control over:
1. experiments    — Run the 20 Mandatory Experiments Suite (EXP-001 to EXP-020)
2. hardware-check — Execute the 14-Step Hardware Bring-Up Verification (H1 to H14)
3. demo           — Launch the Interactive Web Audio Cockpit server
4. export-edge    — Export TinyEnhancer to TorchScript JIT, INT8, and TensorRT specs
5. ichigo-bridge  — Audit and benchmark collaborator's repo (ichigo137/anc)
6. benchmark      — Edge streaming latency, jitter & Real-Time Factor (RTF) profiler
7. dsp-lab        — Compare Classical NLMS, FxLMS, Spectral Subtraction, and Wiener
8. generate-data  — Synthesize defence acoustic noise (T-90 Tank, Helicopter, Gunfire)
9. stream         — Run real-time dual-microphone streaming simulation or live I/O
10. audit         — Forensic reverse-engineering analysis of baseline checkpoints
"""

import argparse
import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path
import numpy as np

# Core Subsystems
from src.dsp.nlms import NLMSFilter
from src.dsp.spectral_subtraction import SpectralSubtraction
from src.dsp.wiener import WienerFilter
from src.pipeline.hybrid_chain import HybridEnhancementPipeline
from src.streaming.stft_engine import StreamingSTFTEngine
from src.streaming.latency_profiler import LatencyProfiler
from src.streaming.live_stream_audio import LiveAudioStreamEngine
from src.dataset.noise_synthesizer import DefenceNoiseSynthesizer
from src.dataset.mixer import AcousticMixer
from src.evaluation.metrics import evaluate_all_metrics
from src.evaluation.reporter import EvaluationReporter
from src.ai.export_onnx import export_edge_models
from src.integrations.ichigo_bridge import run_bridge_audit


def cmd_experiments(args):
    """Executes the 20 Mandatory Experiments Suite."""
    print("\n=======================================================")
    print("  PS 26052: RUNNING 20 MANDATORY EXPERIMENTS SUITE     ")
    print("=======================================================")
    script = Path("experiments/scripts/run_all_experiments.py")
    if script.exists():
        subprocess.run([sys.executable, str(script)], check=True)
    else:
        print(f"[!] Error: {script} not found.")


def cmd_hardware_check(args):
    """Executes the 14-Step Stage-1 Hardware Bring-Up Verification."""
    print("\n=======================================================")
    print("  PS 26052: 14-STEP STAGE-1 HARDWARE BRING-UP CHECK    ")
    print("=======================================================")
    script = Path("hardware/verify_hardware_bringup.py")
    if script.exists():
        subprocess.run([sys.executable, str(script)], check=True)
    else:
        print(f"[!] Error: {script} not found.")


def cmd_demo(args):
    """Launches local HTTP server and opens Interactive Web Audio Cockpit."""
    port = int(args.port)
    demo_file = Path("demo/index.html")
    if not demo_file.exists():
        print(f"[!] Demo file {demo_file} not found.")
        return

    url = f"http://localhost:{port}/demo/index.html"
    print("\n=======================================================")
    print("  PS 26052: INTERACTIVE WEB AUDIO DEMO COCKPIT         ")
    print("=======================================================")
    print(f"[*] Serving demo at: {url}")
    print("[*] Press Ctrl+C to terminate web server.\n")

    try:
        webbrowser.open(url)
        from http.server import HTTPServer, SimpleHTTPRequestHandler
        server = HTTPServer(("127.0.0.1", port), SimpleHTTPRequestHandler)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[+] Demo server stopped.")


def cmd_export_edge(args):
    """Exports model to TorchScript JIT, INT8 quantization, and TensorRT."""
    print("\n=======================================================")
    print("  PS 26052: EDGE AI EXPORT & INT8 QUANTIZATION         ")
    print("=======================================================")
    export_edge_models(output_dir=args.out_dir)


def cmd_ichigo_bridge(args):
    """Audits and benchmarks collaborator repository ichigo137/anc."""
    run_bridge_audit()


def cmd_benchmark(args):
    """Profiles edge streaming latency and Real-Time Factor (RTF)."""
    print("\n=======================================================")
    print("  PS 26052: EDGE STREAMING LATENCY & RTF PROFILER      ")
    print("=======================================================")
    sample_rate = 16000
    frame_size = 512
    hop_size = 256
    duration = float(args.duration)
    n_hops = int((duration * sample_rate) / hop_size)

    print(f"[*] Sample Rate: {sample_rate} Hz | Frame: {frame_size} ({frame_size/sample_rate*1000:.1f} ms) | Hop: {hop_size} ({hop_size/sample_rate*1000:.1f} ms)")
    print(f"[*] Profiling {n_hops} consecutive hops ({duration:.1f}s audio)...")

    engine = StreamingSTFTEngine(frame_size=frame_size, hop_size=hop_size, sample_rate=sample_rate)
    profiler = LatencyProfiler(hop_size=hop_size, sample_rate=sample_rate)
    dummy_hop = np.random.normal(0, 0.1, hop_size).astype(np.float32)

    # Warmup
    for _ in range(2):
        _ = engine.process_hop(dummy_hop)

    for _ in range(n_hops):
        t0 = time.perf_counter()
        _ = engine.process_hop(dummy_hop)
        elapsed = time.perf_counter() - t0
        profiler.record_frame_time(elapsed)

    summary = profiler.compute_summary()
    print("\n[+] LATENCY BENCHMARK RESULTS:")
    for k, v in summary.items():
        print(f"    - {k}: {v}")

    if summary.get("meets_drdo_realtime_target"):
        print("\n>>> [PASSED] MEETS DRDO PS 26052 REAL-TIME EDGE CONSTRAINTS! <<<")
    else:
        print("\n>>> [WARNING] Latency or buffer underrun limit exceeded! <<<")


def cmd_dsp_lab(args):
    """Compares adaptive and classical signal processing baselines."""
    print("\n=======================================================")
    print("  PS 26052: ADAPTIVE DSP FILTER LABORATORY             ")
    print("=======================================================")
    duration = 3.0
    sr = 16000
    synth = DefenceNoiseSynthesizer(sample_rate=sr)

    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    clean_speech = 0.5 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 1200 * t)
    noise = synth.generate_tank_noise(duration=duration)

    primary = clean_speech + noise
    reference = np.roll(noise, 4)

    # 1. Classical Boll Spectral Subtraction
    ss = SpectralSubtraction(sample_rate=sr)
    out_ss = ss.process(primary)

    # 2. Classical Wiener Filter
    wf = WienerFilter(sample_rate=sr)
    out_wf = wf.process(primary)

    # 3. Config A NLMS Filter
    nlms = NLMSFilter(filter_length=64, step_size=0.08)
    out_nlms, _ = nlms.filter_block(primary, reference)

    m_raw = evaluate_all_metrics(clean_speech, primary, sample_rate=sr)
    m_ss = evaluate_all_metrics(clean_speech, out_ss, sample_rate=sr)
    m_wf = evaluate_all_metrics(clean_speech, out_wf, sample_rate=sr)
    m_nlms = evaluate_all_metrics(clean_speech, out_nlms, sample_rate=sr)

    print(f"\n[+] Raw Primary Input:            SNR = {m_raw['snr_db']:.2f} dB | STOI = {m_raw['stoi']:.3f}")
    print(f"[+] Boll Spectral Subtraction:    SNR = {m_ss['snr_db']:.2f} dB | STOI = {m_ss['stoi']:.3f} (Delta: {m_ss['snr_db'] - m_raw['snr_db']:+.2f} dB)")
    print(f"[+] Wiener Filter (DD):           SNR = {m_wf['snr_db']:.2f} dB | STOI = {m_wf['stoi']:.3f} (Delta: {m_wf['snr_db'] - m_raw['snr_db']:+.2f} dB)")
    print(f"[+] Dual-Mic NLMS (Config A):     SNR = {m_nlms['snr_db']:.2f} dB | STOI = {m_nlms['stoi']:.3f} (Delta: {m_nlms['snr_db'] - m_raw['snr_db']:+.2f} dB)")


def cmd_generate_data(args):
    """Synthesizes defence noise or speech mixtures."""
    sr = 16000
    synth = DefenceNoiseSynthesizer(sample_rate=sr)
    preset = args.preset.lower()

    if preset == "tank":
        noise = synth.generate_tank_noise(duration=float(args.duration))
    elif preset == "helicopter":
        noise = synth.generate_helicopter_noise(duration=float(args.duration))
    elif preset == "gunfire":
        noise = synth.generate_gunfire_transient(duration=float(args.duration))
    else:
        noise = synth.generate_tank_noise(duration=float(args.duration))

    out_dir = Path("data/noise")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{preset}_synthetic.npy"
    np.save(out_file, noise)
    print(f"[+] Saved {len(noise)} samples ({len(noise)/sr:.1f}s) to: {out_file}")


def cmd_stream(args):
    """Runs real-time live audio streaming engine."""
    engine = LiveAudioStreamEngine(config_mode=args.mode)
    if args.live:
        engine.run_live(duration_sec=float(args.duration))
    else:
        res = engine.run_simulation(duration_sec=float(args.duration))
        print(f"[+] Streaming simulation complete: P50 = {res['p50_latency_ms']} ms | P95 = {res['p95_latency_ms']} ms | Target Met: {res['meets_drdo_realtime_target']}")


def cmd_audit(args):
    """Prints reverse-engineering forensic analysis of collaborator checkpoint."""
    audit_file = Path("research/dossiers/ICHIGO_ANC_FORENSIC_REVERSE_ENGINEERING.md")
    if audit_file.exists():
        with open(audit_file, "r", encoding="utf-8") as f:
            print(f.read())
    else:
        print(f"[!] Audit dossier {audit_file} not found.")


def main():
    parser = argparse.ArgumentParser(
        description="ADAPTIVE-DEFENCE ANC: Hybrid AI-DSP Edge Speech Enhancement (DRDO SIH 2026)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Operational Mode")

    # 1. experiments
    subparsers.add_parser("experiments", help="Run the complete 20 Mandatory Experiments Suite (EXP-001 to EXP-020)")

    # 2. hardware-check
    subparsers.add_parser("hardware-check", help="Execute the 14-Step Stage-1 Hardware Bring-Up verification (H1 to H14)")

    # 3. demo
    p_demo = subparsers.add_parser("demo", help="Launch the Interactive Web Audio Demo Cockpit")
    p_demo.add_argument("--port", type=int, default=8000, help="Port to bind demo server (default: 8000)")

    # 4. export-edge
    p_exp = subparsers.add_parser("export-edge", help="Export model to TorchScript JIT, INT8 quantization, and TensorRT specs")
    p_exp.add_argument("--out-dir", type=str, default="models", help="Directory for exported models")

    # 5. ichigo-bridge
    subparsers.add_parser("ichigo-bridge", help="Audit and benchmark collaborator's repo (ichigo137/anc)")

    # 6. benchmark
    p_bench = subparsers.add_parser("benchmark", help="Profile streaming latency & RTF")
    p_bench.add_argument("--duration", type=float, default=5.0, help="Benchmark duration in seconds")

    # 7. dsp-lab
    subparsers.add_parser("dsp-lab", help="Run adaptive & classical DSP laboratory")

    # 8. generate-data
    p_gen = subparsers.add_parser("generate-data", help="Synthesize defence acoustic noise")
    p_gen.add_argument("--preset", type=str, default="tank", choices=["tank", "helicopter", "gunfire"], help="Noise preset")
    p_gen.add_argument("--duration", type=float, default=5.0, help="Audio duration in seconds")

    # 9. stream
    p_stream = subparsers.add_parser("stream", help="Run real-time audio streaming")
    p_stream.add_argument("--duration", type=float, default=3.0, help="Stream duration in seconds")
    p_stream.add_argument("--mode", type=str, default="A", choices=["A", "B", "C"], help="Pipeline topology mode")
    p_stream.add_argument("--live", action="store_true", help="Capture from live hardware microphone if available")

    # 10. audit
    subparsers.add_parser("audit", help="Display forensic checkpoint reverse-engineering audit")

    args = parser.parse_args()

    if args.command == "experiments":
        cmd_experiments(args)
    elif args.command == "hardware-check":
        cmd_hardware_check(args)
    elif args.command == "demo":
        cmd_demo(args)
    elif args.command == "export-edge":
        cmd_export_edge(args)
    elif args.command == "ichigo-bridge":
        cmd_ichigo_bridge(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "dsp-lab":
        cmd_dsp_lab(args)
    elif args.command == "generate-data":
        cmd_generate_data(args)
    elif args.command == "stream":
        cmd_stream(args)
    elif args.command == "audit":
        cmd_audit(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
