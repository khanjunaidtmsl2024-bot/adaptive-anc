"""
Latency & Real-Time Factor (RTF) Edge Profiler.
PS 26052 — Adaptive Defence ANC.
"""

from typing import Dict, Any, List
import time
import numpy as np


class LatencyProfiler:
    """Profiles real-time streaming latency, jitter, and Real-Time Factor."""

    def __init__(self, hop_size: int = 256, sample_rate: int = 16000):
        self.hop_size = hop_size
        self.sr = sample_rate
        self.frame_duration_ms = (hop_size / sample_rate) * 1000.0
        self.measurements: List[float] = []

    def record_frame_time(self, elapsed_seconds: float) -> None:
        """Record processing duration for one frame in milliseconds."""
        self.measurements.append(elapsed_seconds * 1000.0)

    def compute_summary(self) -> Dict[str, Any]:
        """Compute statistical summary of latency and real-time performance."""
        if not self.measurements:
            return {"count": 0, "status": "No frames measured"}

        times = np.array(self.measurements)
        avg_time = float(np.mean(times))
        p50 = float(np.percentile(times, 50))
        p95 = float(np.percentile(times, 95))
        p99 = float(np.percentile(times, 99))
        max_time = float(np.max(times))

        # RTF = compute_time / audio_duration
        rtf = avg_time / self.frame_duration_ms
        underruns = int(np.sum(times > self.frame_duration_ms))

        return {
            "frames_processed": len(times),
            "frame_duration_ms": round(self.frame_duration_ms, 2),
            "avg_latency_ms": round(avg_time, 3),
            "p50_latency_ms": round(p50, 3),
            "p95_latency_ms": round(p95, 3),
            "p99_latency_ms": round(p99, 3),
            "max_latency_ms": round(max_time, 3),
            "real_time_factor_rtf": round(rtf, 4),
            "buffer_underruns": underruns,
            "meets_drdo_realtime_target": bool(rtf < 1.0 and underruns == 0),
        }
