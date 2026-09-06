"""
ADAPTIVE-DEFENCE ANC — Real-Time Live Audio Hardware Streaming Engine
Smart India Hackathon (SIH) 2026 — Problem Statement ID: 26052
DRDO / Department of Defence Production / iDEX — Smart Vehicles

Interfacing Dual-Microphone Hardware (INMP441 / ICS-43434) with Hybrid AI-DSP:
- Channel 0: Primary Mic (Speech + Cockpit Noise)
- Channel 1: Reference Mic (Correlated Engine / Rotor Acoustic Path)
- Channel Output: Real-Time Destructive Anti-Noise & Enhanced Speech to Headset DAC
"""

import time
import sys
from typing import Optional, Dict, Any, Callable
import numpy as np

from src.pipeline.hybrid_chain import HybridEnhancementPipeline
from src.streaming.latency_profiler import LatencyProfiler

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

# Target hardware contract (frozen at PH0/PH1 design review): any execution
# artifact must be self-describing so simulated latency can never be presented
# as hardware latency downstream.
HARDWARE_TARGET = "Raspberry Pi 4"
HARDWARE_MODEL = "Raspberry Pi 4 + WM8960 CODEC"
AUDIO_INTERFACE = "WM8960"


def _execution_provenance(simulated: bool) -> Dict[str, Any]:
    """Machine-readable execution provenance for run_live/run_simulation."""
    if simulated:
        return {
            "execution_mode": "SIMULATION",
            "is_simulated": True,
            "hardware_available": False,
            "hardware_target": HARDWARE_TARGET,
            "hardware_model": None,
            "audio_interface": None,
        }
    return {
        "execution_mode": "HARDWARE",
        "is_simulated": False,
        "hardware_available": True,
        "hardware_target": HARDWARE_TARGET,
        "hardware_model": HARDWARE_MODEL,
        "audio_interface": AUDIO_INTERFACE,
    }


class LiveAudioStreamEngine:
    """
    Low-latency streaming audio callback engine for real-time dual-mic ANC.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        hop_size: int = 256,
        block_size: int = 256,
        config_mode: str = "A",
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.block_size = block_size
        self.config_mode = config_mode
        self.input_device = input_device
        self.output_device = output_device

        self.pipeline = HybridEnhancementPipeline(config_mode=config_mode, sample_rate=sample_rate)
        self.profiler = LatencyProfiler(hop_size=hop_size, sample_rate=sample_rate)
        self.is_running = False
        self.stream = None

    def audio_callback(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """Real-time duplex audio callback."""
        t0 = time.perf_counter()

        if status:
            print(f"[!] Stream status flag: {status}", file=sys.stderr)

        # Separate Primary (Ch0) and Reference (Ch1 if dual-channel available)
        if indata.shape[1] >= 2:
            primary = indata[:, 0]
            reference = indata[:, 1]
        else:
            primary = indata[:, 0]
            reference = primary

        # Process through hybrid AI-DSP pipeline
        enhanced = self.pipeline.process_frame(primary, reference)

        # Assign to mono or stereo output
        if outdata.shape[1] == 1:
            outdata[:, 0] = enhanced
        else:
            outdata[:, 0] = enhanced
            outdata[:, 1] = enhanced

        elapsed = time.perf_counter() - t0
        self.profiler.record_frame_time(elapsed)

    def run_live(self, duration_sec: float = 5.0) -> Dict[str, Any]:
        """Runs the live full-duplex stream or simulation fallback."""
        if not SOUNDDEVICE_AVAILABLE:
            print("[*] sounddevice library not installed. Running high-fidelity streaming simulation...")
            return self.run_simulation(duration_sec=duration_sec)

        print(f"\n=======================================================")
        print(f"  STARTING LIVE AUDIO ANC STREAM ({duration_sec}s)    ")
        print(f"=======================================================")
        print(f"[*] Sample Rate: {self.sample_rate} Hz | Block Size: {self.block_size} samples ({self.block_size/self.sample_rate*1000:.1f} ms)")
        print(f"[*] Pipeline Config: Mode {self.config_mode}")

        try:
            with sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=(self.input_device, self.output_device),
                channels=(2, 2),
                dtype=np.float32,
                latency="low",
                callback=self.audio_callback,
            ):
                self.is_running = True
                time.sleep(duration_sec)
                self.is_running = False

            summary = self.profiler.compute_summary()
            summary["mode"] = "hardware"
            summary.update(_execution_provenance(simulated=False))
            print("[+] Live streaming finished. Profiling summary:")
            for k, v in summary.items():
                print(f"    - {k}: {v}")
            return summary

        except Exception as e:
            print(f"[!] Hardware audio device initialization failed ({e}). Falling back to simulation.")
            return self.run_simulation(duration_sec=duration_sec)

    def run_simulation(self, duration_sec: float = 3.0) -> Dict[str, Any]:
        """Emulates dual-channel hardware streaming with synthetic noise.

        Return dict is tagged with full execution provenance
        (execution_mode/is_simulated/hardware_*) so consumers can distinguish
        a simulated run from a real hardware run without parsing stdout
        (which may be piped/absent in batch runs).
        """
        n_blocks = int((duration_sec * self.sample_rate) / self.block_size)
        t = np.linspace(0, duration_sec, n_blocks * self.block_size, endpoint=False)

        # Clean speech + tank noise
        clean = 0.5 * np.sin(2 * np.pi * 300 * t).astype(np.float32)
        noise = np.random.normal(0, 0.2, len(t)).astype(np.float32)
        primary_full = clean + noise
        ref_full = np.roll(noise, 4)

        for i in range(n_blocks):
            start = i * self.block_size
            end = start + self.block_size
            p_block = primary_full[start:end]
            r_block = ref_full[start:end]

            t0 = time.perf_counter()
            _ = self.pipeline.process_frame(p_block, r_block)
            elapsed = time.perf_counter() - t0
            self.profiler.record_frame_time(elapsed)

        summary = self.profiler.compute_summary()
        summary["mode"] = "simulation"
        summary.update(_execution_provenance(simulated=True))
        return summary


if __name__ == "__main__":
    engine = LiveAudioStreamEngine()
    res = engine.run_simulation(duration_sec=2.0)
    print(f"[+] Simulation completed: P50 = {res['p50_latency_ms']} ms | P95 = {res['p95_latency_ms']} ms | Meets Target: {res['meets_drdo_realtime_target']}")
