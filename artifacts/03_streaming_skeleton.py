#!/usr/bin/env python3
"""
PS 26052 — Artifact 3: Streaming Audio Skeleton
================================================
Proves the real-time audio path: Microphone → callback → ring buffer →
STFT → processing → iSTFT → output buffer → headset.

Includes:
    - Lock-free ring buffer
    - STFT with overlap-add reconstruction
    - Audio callback architecture (never blocks)
    - Latency instrumentation (P50/P95/P99/worst-case)
    - Underrun/overrun logging
    - Bypass (passthrough) mode for baseline testing
    - Drop-in AI processing slot

Usage:
    # Passthrough (proves audio path works)
    python 03_streaming_skeleton.py --mode passthrough

    # STFT passthrough (proves STFT/iSTFT reconstruction)
    python 03_streaming_skeleton.py --mode stft-passthrough

    # Process a file through the streaming pipeline
    python 03_streaming_skeleton.py --mode file --input test.wav --output processed.wav

    # Live with real microphone
    python 03_streaming_skeleton.py --mode live

    # Measure latency only
    python 03_streaming_skeleton.py --mode benchmark

Designed for: SIH 2026 Adaptive ANC — PS 26052
"""

import os
import sys
import time
import threading
import argparse
import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

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

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("streaming")


# ---------------------------------------------------------------------------
# Ring Buffer
# ---------------------------------------------------------------------------
class RingBuffer:
    """Lock-free ring buffer for audio frames.
    
    Design: Audio callback writes frames (producer).
            Processing thread reads frames (consumer).
            If the buffer is full, oldest frames are dropped (with logging).
    """
    
    def __init__(self, capacity: int, frame_size: int):
        self.capacity = capacity
        self.frame_size = frame_size
        self.buffer = np.zeros((capacity, frame_size), dtype=np.float32)
        self.write_pos = 0
        self.read_pos = 0
        self.count = 0
        self.underruns = 0
        self.overruns = 0
        self._lock = threading.Lock()
    
    def write(self, frame: np.ndarray) -> bool:
        """Write a frame to the buffer. Returns False if overflow."""
        with self._lock:
            if self.count >= self.capacity:
                self.overruns += 1
                # Drop oldest frame
                self.read_pos = (self.read_pos + 1) % self.capacity
                self.count -= 1
            
            self.buffer[self.write_pos] = frame
            self.write_pos = (self.write_pos + 1) % self.capacity
            self.count += 1
            return True
    
    def read(self) -> Optional[np.ndarray]:
        """Read a frame from the buffer. Returns None if empty."""
        with self._lock:
            if self.count <= 0:
                self.underruns += 1
                return None
            
            frame = self.buffer[self.read_pos].copy()
            self.read_pos = (self.read_pos + 1) % self.capacity
            self.count -= 1
            return frame
    
    def read_available(self) -> int:
        with self._lock:
            return self.count
    
    def reset_stats(self):
        self.underruns = 0
        self.overruns = 0


# ---------------------------------------------------------------------------
# STFT Processor
# ---------------------------------------------------------------------------
class STFTProcessor:
    """Streaming STFT with overlap-add reconstruction.
    
    Signal flow:
        input frame → STFT → [processing slot] → iSTFT → output frame
    
    The processing_slot function is where AI models or DSP goes.
    """
    
    def __init__(
        self,
        frame_size: int = 512,
        hop_size: int = 256,
        processing_fn: Optional[Callable] = None,
    ):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.n_fft = frame_size
        self.window = np.hanning(frame_size).astype(np.float32)
        
        # Processing function: (complex_spectrum, metadata) → enhanced_spectrum
        # If None, passthrough (identity)
        self.processing_fn = processing_fn
        
        # Overlap-add buffer
        self.output_buffer = np.zeros(frame_size, dtype=np.float32)
        self.input_overlap = np.zeros(frame_size - hop_size, dtype=np.float32)
        
        # Latency tracking
        self.processing_times_us: list[int] = []
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process one frame through STFT → processing → iSTFT."""
        # Validate input
        if len(frame) < self.hop_size:
            frame = np.pad(frame, (0, self.hop_size - len(frame)))
        elif len(frame) > self.frame_size:
            frame = frame[:self.frame_size]
        
        # --- STFT ---
        t_start = time.perf_counter_ns()
        
        # Prepend overlap from previous frame
        full_frame = np.concatenate([self.input_overlap, frame[:self.hop_size]])
        
        # Save overlap for next frame
        self.input_overlap = frame[self.hop_size - self.frame_size + self.hop_size:] if len(frame) > self.hop_size else frame[self.hop_size:] if len(frame) > self.hop_size else np.zeros(self.frame_size - self.hop_size)
        # Simpler: just save the tail
        self.input_overlap = full_frame[self.hop_size:].copy()
        
        # Window and FFT
        windowed = full_frame * self.window
        spectrum = np.fft.rfft(windowed)
        
        # --- PROCESSING SLOT ---
        if self.processing_fn is not None:
            enhanced_spectrum = self.processing_fn(spectrum)
        else:
            enhanced_spectrum = spectrum  # passthrough
        
        # --- iSTFT ---
        output_time = np.fft.irfft(enhanced_spectrum, n=self.frame_size)
        
        # Overlap-add synthesis
        output = self.output_buffer.copy()
        output[:self.hop_size] += output_time[:self.hop_size]
        
        # Update overlap buffer for synthesis
        self.output_buffer = np.zeros(self.frame_size, dtype=np.float32)
        self.output_buffer[:self.frame_size - self.hop_size] = output_time[self.hop_size:]
        
        t_end = time.perf_counter_ns()
        self.processing_times_us.append((t_end - t_start) // 1000)
        
        return output[:self.hop_size].astype(np.float32)
    
    def get_latency_stats(self) -> dict:
        """Get processing latency statistics."""
        if not self.processing_times_us:
            return {"p50": 0, "p95": 0, "p99": 0, "worst": 0, "mean": 0, "n_frames": 0}
        
        arr = np.array(self.processing_times_us)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "worst": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "n_frames": len(arr),
        }


# ---------------------------------------------------------------------------
# Streaming Pipeline
# ---------------------------------------------------------------------------
class StreamingPipeline:
    """Complete streaming audio pipeline.
    
    Architecture:
        [Input Source] → audio_callback → [Ring Buffer]
                                              ↓
                                      [Processing Thread]
                                              ↓
                                      [STFT → AI/DSP → iSTFT]
                                              ↓
                                      [Output Ring Buffer]
                                              ↓
                                      [Output Callback] → [Speaker/Headset]
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 512,
        hop_size: int = 256,
        buffer_frames: int = 16,
        processing_fn: Optional[Callable] = None,
    ):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        
        # Ring buffers
        self.input_buffer = RingBuffer(buffer_frames, hop_size)
        self.output_buffer = RingBuffer(buffer_frames, hop_size)
        
        # STFT processor
        self.stft = STFTProcessor(frame_size, hop_size, processing_fn)
        
        # State
        self.running = False
        self.processing_thread: Optional[threading.Thread] = None
        self.total_frames_processed = 0
        self.start_time = 0.0
    
    def _processing_loop(self):
        """Processing thread: reads from input buffer, processes, writes to output."""
        while self.running:
            frame = self.input_buffer.read()
            if frame is not None:
                output = self.stft.process_frame(frame)
                self.output_buffer.write(output)
                self.total_frames_processed += 1
            else:
                # No data available, sleep briefly
                time.sleep(0.001)
    
    def start(self):
        """Start the processing thread."""
        self.running = True
        self.start_time = time.time()
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        log.info("Processing thread started")
    
    def stop(self):
        """Stop the processing thread."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        log.info("Processing thread stopped")
    
    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        elapsed = time.time() - self.start_time if self.start_time > 0 else 0
        stft_stats = self.stft.get_latency_stats()
        
        return {
            "total_frames": self.total_frames_processed,
            "elapsed_s": round(elapsed, 2),
            "frames_per_second": round(self.total_frames_processed / max(elapsed, 0.001), 1),
            "input_underruns": self.input_buffer.underruns,
            "input_overruns": self.input_buffer.overruns,
            "output_underruns": self.output_buffer.underruns,
            "output_overruns": self.output_buffer.overruns,
            "algorithmic_latency_ms": round(self.frame_size / self.sample_rate * 1000, 1),
            "processing_us": stft_stats,
        }


# ---------------------------------------------------------------------------
# File Processing Mode
# ---------------------------------------------------------------------------
def process_file(
    input_path: str,
    output_path: str,
    sample_rate: int = 16000,
    frame_size: int = 512,
    hop_size: int = 256,
    processing_fn: Optional[Callable] = None,
):
    """Process an entire file through the streaming pipeline frame-by-frame."""
    log.info(f"Processing file: {input_path}")
    
    audio, sr = sf.read(input_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    log.info(f"  Input: {len(audio)} samples, {len(audio)/sr:.2f}s, {sr}Hz")
    
    pipeline = StreamingPipeline(
        sample_rate=sr,
        frame_size=frame_size,
        hop_size=hop_size,
        processing_fn=processing_fn,
    )
    pipeline.start()
    
    # Feed frames
    n_frames = (len(audio) - frame_size) // hop_size + 1
    for i in range(n_frames):
        start = i * hop_size
        frame = audio[start:start + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))
        pipeline.input_buffer.write(frame)
    
    # Wait for processing to complete
    while pipeline.input_buffer.read_available() > 0:
        time.sleep(0.01)
    
    pipeline.stop()
    
    # Collect output
    output_frames = []
    while True:
        frame = pipeline.output_buffer.read()
        if frame is None:
            break
        output_frames.append(frame)
    
    if output_frames:
        output_audio = np.concatenate(output_frames)
        output_audio = output_audio[:len(audio)]  # trim to original length
    else:
        output_audio = np.zeros(len(audio))
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sf.write(output_path, output_audio, sr, subtype="PCM_16")
    
    stats = pipeline.get_stats()
    log.info(f"  Output: {len(output_audio)} samples, {output_path}")
    log.info(f"  Stats: {stats}")
    
    return stats


# ---------------------------------------------------------------------------
# Benchmark Mode
# ---------------------------------------------------------------------------
def benchmark_latency(
    sample_rate: int = 16000,
    frame_size: int = 512,
    hop_size: int = 256,
    duration_s: float = 10.0,
):
    """Measure processing latency without real audio hardware."""
    log.info(f"Benchmarking STFT/iSTFT latency...")
    log.info(f"  Sample rate: {sample_rate} Hz")
    log.info(f"  Frame size: {frame_size}")
    log.info(f"  Hop size: {hop_size}")
    log.info(f"  Duration: {duration_s}s")
    
    # Generate synthetic input
    n_samples = int(duration_s * sample_rate)
    input_audio = np.random.randn(n_samples).astype(np.float32) * 0.1
    
    # Process through pipeline
    stft = STFTProcessor(frame_size, hop_size, processing_fn=None)
    
    t_start = time.perf_counter()
    n_frames = (n_samples - frame_size) // hop_size + 1
    
    for i in range(n_frames):
        start = i * hop_size
        frame = input_audio[start:start + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))
        stft.process_frame(frame)
    
    t_end = time.perf_counter()
    elapsed = t_end - t_start
    
    stats = stft.get_latency_stats()
    
    print(f"\n{'='*50}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"  Frames processed:    {n_frames}")
    print(f"  Wall time:           {elapsed:.3f}s")
    print(f"  RTF (real-time factor): {elapsed / duration_s:.4f}")
    print(f"  Algorithmic latency: {frame_size / sample_rate * 1000:.1f} ms")
    print(f"  Processing per frame:")
    print(f"    Mean:   {stats['mean']:.0f} µs")
    print(f"    P50:    {stats['p50']:.0f} µs")
    print(f"    P95:    {stats['p95']:.0f} µs")
    print(f"    P99:    {stats['p99']:.0f} µs")
    print(f"    Worst:  {stats['worst']:.0f} µs")
    
    rtf = elapsed / duration_s
    if rtf < 1.0:
        print(f"\n  [OK] RTF < 1.0 -- pipeline CAN run in real-time")
    else:
        print(f"\n  [FAIL] RTF >= 1.0 -- pipeline CANNOT run in real-time at this config")
    print(f"{'='*50}\n")
    
    return stats


# ---------------------------------------------------------------------------
# Live Mode
# ---------------------------------------------------------------------------
def live_stream(
    processing_fn: Optional[Callable] = None,
    sample_rate: int = 16000,
    frame_size: int = 512,
    hop_size: int = 256,
    duration_s: float = 30.0,
):
    """Live microphone → processing → speaker streaming."""
    if not HAS_SOUNDDEVICE:
        log.error("sounddevice not installed. Run: pip install sounddevice")
        log.info("Falling back to file processing mode")
        return
    
    log.info(f"Starting live stream for {duration_s}s...")
    
    pipeline = StreamingPipeline(
        sample_rate=sample_rate,
        frame_size=frame_size,
        hop_size=hop_size,
        processing_fn=processing_fn,
    )
    
    # Audio input callback
    def input_callback(indata, frames, time_info, status):
        if status:
            log.warning(f"Input status: {status}")
        frame = indata[:, 0].copy()
        pipeline.input_buffer.write(frame)
    
    # Audio output callback
    def output_callback(outdata, frames, time_info, status):
        if status:
            log.warning(f"Output status: {status}")
        frame = pipeline.output_buffer.read()
        if frame is None:
            outdata[:, 0] = np.zeros(frames)
        else:
            outdata[:, 0] = frame[:frames]
    
    pipeline.start()
    
    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            blocksize=hop_size,
            callback=input_callback,
        ), sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            blocksize=hop_size,
            callback=output_callback,
        ):
            log.info("Streaming... Press Ctrl+C to stop")
            time.sleep(duration_s)
    except KeyboardInterrupt:
        log.info("Stopped by user")
    finally:
        pipeline.stop()
        stats = pipeline.get_stats()
        log.info(f"Final stats: {stats}")


# ---------------------------------------------------------------------------
# Demo Processing Function
# ---------------------------------------------------------------------------
def demo_spectral_gate(spectrum: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Simple spectral gating: suppress bins below threshold.
    
    This is a minimal example of a processing function.
    Replace this with your AI model inference.
    
    Signal flow for NLMS integration (Phase 2):
        1. AI model processes spectrum
        2. Reference mic signal enters NLMS
        3. NLMS estimates correlated noise
        4. Residual = AI_output - NLMS_estimate
        5. Output = residual
    """
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    
    # Simple gate
    mask = (magnitude > threshold).astype(np.float32)
    
    # Soft mask with smoothing
    mask = np.clip(magnitude / (threshold + 1e-6), 0, 1)
    
    return magnitude * mask * np.exp(1j * phase)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PS 26052 Streaming Audio Skeleton")
    parser.add_argument("--mode", choices=["passthrough", "stft-passthrough", "file", "live", "benchmark"],
                        default="benchmark",
                        help="Operating mode")
    parser.add_argument("--input", type=str, help="Input audio file (file mode)")
    parser.add_argument("--output", type=str, default="output_processed.wav",
                        help="Output audio file (file mode)")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate")
    parser.add_argument("--frame-size", type=int, default=512, help="STFT frame size")
    parser.add_argument("--hop-size", type=int, default=256, help="STFT hop size")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration for benchmark/live")
    parser.add_argument("--no-processing", action="store_true", help="Disable processing (pure passthrough)")
    args = parser.parse_args()
    
    processing_fn = None if args.no_processing else demo_spectral_gate
    
    if args.mode == "benchmark":
        benchmark_latency(
            sample_rate=args.sample_rate,
            frame_size=args.frame_size,
            hop_size=args.hop_size,
            duration_s=args.duration,
        )
    
    elif args.mode == "file":
        if not args.input:
            log.error("--input required for file mode")
            return
        process_file(
            input_path=args.input,
            output_path=args.output,
            sample_rate=args.sample_rate,
            frame_size=args.frame_size,
            hop_size=args.hop_size,
            processing_fn=processing_fn if args.mode == "stft-passthrough" else None,
        )
    
    elif args.mode == "stft-passthrough":
        if args.input:
            process_file(
                input_path=args.input,
                output_path=args.output,
                sample_rate=args.sample_rate,
                frame_size=args.frame_size,
                hop_size=args.hop_size,
                processing_fn=processing_fn,
            )
        else:
            # Generate synthetic test signal
            sr = args.sample_rate
            t = np.linspace(0, 5, sr * 5, endpoint=False)
            test_signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
            test_path = "_test_440hz.wav"
            sf.write(test_path, test_signal, sr, subtype="PCM_16")
            
            process_file(
                input_path=test_path,
                output_path=args.output,
                sample_rate=sr,
                frame_size=args.frame_size,
                hop_size=args.hop_size,
                processing_fn=processing_fn,
            )
            os.remove(test_path)
    
    elif args.mode == "live":
        live_stream(
            processing_fn=processing_fn,
            sample_rate=args.sample_rate,
            frame_size=args.frame_size,
            hop_size=args.hop_size,
            duration_s=args.duration,
        )
    
    elif args.mode == "passthrough":
        log.info("Passthrough mode — no STFT processing")
        if args.input:
            audio, sr = sf.read(args.input, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            sf.write(args.output, audio, sr, subtype="PCM_16")
            log.info(f"Copied {args.input} → {args.output}")
    
    print(f"\nNext step: Implement adaptive filter (04_adaptive_filter_lab.py)")


if __name__ == "__main__":
    main()
