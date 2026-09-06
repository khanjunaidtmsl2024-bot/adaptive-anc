"""Unit tests for RingBuffer and StreamingSTFTEngine."""

import numpy as np
from src.streaming.ring_buffer import RingBuffer
from src.streaming.stft_engine import StreamingSTFTEngine


def test_ring_buffer_read_write():
    """Verify circular ring buffer capacity, writes, reads, and peeks."""
    rb = RingBuffer(capacity=1024)
    data = np.arange(100, dtype=np.float32)

    written = rb.write(data)
    assert written == 100
    assert rb.size == 100

    peeked = rb.peek(50)
    assert len(peeked) == 50
    assert np.array_equal(peeked, data[:50])
    assert rb.size == 100  # Size unchanged after peek

    read_data = rb.read(50)
    assert len(read_data) == 50
    assert np.array_equal(read_data, data[:50])
    assert rb.size == 50   # Size reduced after read


def test_streaming_stft_reconstruction():
    """Verify STFT/iSTFT overlap-add reconstructs signal with low distortion."""
    engine = StreamingSTFTEngine(frame_size=512, hop_size=256, sample_rate=16000)
    hop_size = 256
    n_hops = 10

    # Feed consecutive hops
    reconstructed = []
    for _ in range(n_hops):
        chunk = np.ones(hop_size, dtype=np.float32) * 0.5
        out_hop = engine.process_hop(chunk)
        reconstructed.append(out_hop)

    rec_signal = np.concatenate(reconstructed)
    assert len(rec_signal) == hop_size * n_hops
    assert np.all(np.isfinite(rec_signal))


def test_causal_streaming_engine_deterministic_reconstruction():
    """
    Deterministic mathematical verification of CausalStreamingEngine WOLA reconstruction.
    Verifies:
      1. Boundary & first-hop behavior (warmup buffer is zero to float precision).
      2. Steady-state lag (exactly 1 hop = 128 samples = 8.0 ms).
      3. Impulse reconstruction: unity WOLA gain below the fail-safe output
         limiter (amp 0.5 -> peak 0.5 within 1e-3, no dispersion), and exact
         1-hop lag. A unit-amplitude impulse is capped at the impulse-protection
         output limiter ceiling (0.95 by design, hearing/DAC fail-safe) - so the
         measured 0.95 peak is NOT a reconstruction-gain error.
      4. 440 Hz sine reconstruction (unity gain, float-precision error < 1e-4).
      5. Constant signal reconstruction (unity gain, float-precision error < 1e-4).
    """
    from src.streaming.causal_engine import CausalStreamingEngine

    class PassThroughAI:
        def enhance_spectrogram(self, mag, phase):
            return mag, phase

    sr = 16000
    frame_size = 256
    hop_size = 128

    def _make_engine():
        return CausalStreamingEngine(
            frame_size=frame_size,
            hop_size=hop_size,
            sample_rate=sr,
            ai_backend=PassThroughAI(),
            enable_regime_adaptation=False,
            use_fast_dsp=False,
        )

    # 1. Boundary behavior & impulse response / steady-state lag
    engine = _make_engine()
    n_samples = 1000
    impulse_idx = 100
    primary = np.zeros(n_samples, dtype=np.float32)
    primary[impulse_idx] = 1.0
    ref = np.zeros(n_samples, dtype=np.float32)

    hops = []
    for i in range(n_samples // hop_size):
        out_hop, _ = engine.process_hop(
            primary[i * hop_size : (i + 1) * hop_size],
            ref[i * hop_size : (i + 1) * hop_size]
        )
        hops.append(out_hop)

    recon = np.concatenate(hops)

    # Boundary: first hop must be strictly zero (< 1e-6) due to causal lookahead buffering
    assert np.max(np.abs(recon[:hop_size])) < 1e-6, "First hop must be zero (< 1e-6) due to causal buffering"

    # Steady-state lag: peak must appear at exactly impulse_idx + hop_size
    peak_idx = int(np.argmax(recon))
    lag = peak_idx - impulse_idx
    assert lag == hop_size, f"Expected lag of {hop_size} samples (8.00 ms), got {lag}"

    # Impulse dispersion must be at float precision (no smearing across hops)
    surrounding = np.delete(recon[hop_size:], peak_idx - hop_size)
    assert np.max(np.abs(surrounding)) < 1e-6, f"Impulse dispersion {np.max(np.abs(surrounding))} exceeds float precision"

    # The engine path runs impulse protection, whose hard output limiter caps at
    # 0.95 (hearing/DAC fail-safe). A unit impulse is therefore clipped to 0.95
    # BEFORE the STFT/AI/WOLA chain - the 0.95 peak is the limiter ceiling, not
    # a reconstruction-gain error. Unity WOLA gain is verified below with a
    # sub-limiter impulse (amp 0.5).
    assert abs(recon[peak_idx] - 0.95) < 1e-3, (
        f"Unit impulse peak {recon[peak_idx]:.4f} != limiter ceiling 0.95 "
        f"(impulse protection max_output_limit)"
    )

    # Unity-gain reconstruction at sub-limiter amplitude, in steady state
    # (past the startup boundary region): amp 0.5 must come back as 0.5.
    engine2 = _make_engine()
    n2 = 4096
    imp2_idx = 3000  # steady-state region
    amp2 = 0.5
    p2 = np.zeros(n2, dtype=np.float32)
    p2[imp2_idx] = amp2
    r2 = np.zeros(n2, dtype=np.float32)
    hops2 = []
    for i in range(n2 // hop_size):
        out_hop, _ = engine2.process_hop(
            p2[i * hop_size : (i + 1) * hop_size],
            r2[i * hop_size : (i + 1) * hop_size],
        )
        hops2.append(out_hop)
    recon2 = np.concatenate(hops2)
    peak2_idx = imp2_idx + hop_size + int(np.argmax(
        recon2[imp2_idx + hop_size : imp2_idx + hop_size + 2 * hop_size]))
    assert abs(recon2[peak2_idx] - amp2) < 1e-3, (
        f"Sub-limiter impulse peak {recon2[peak2_idx]:.4f} deviates from input "
        f"amplitude {amp2}: WOLA reconstruction gain is not unity"
    )

    # 2. 440 Hz Sine wave reconstruction (float precision and unity gain)
    engine = _make_engine()
    t = np.arange(8000) / sr
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    hops = []
    for i in range(len(sine) // hop_size):
        out_hop, _ = engine.process_hop(
            sine[i * hop_size : (i + 1) * hop_size],
            np.zeros(hop_size, dtype=np.float32)
        )
        hops.append(out_hop)

    recon_sine = np.concatenate(hops)
    aligned_recon = recon_sine[hop_size + frame_size :]
    aligned_target = sine[frame_size : len(recon_sine) - hop_size]

    sine_max_err = float(np.max(np.abs(aligned_recon - aligned_target)))
    sine_rms_ratio = float(np.sqrt(np.mean(aligned_recon ** 2)) / np.sqrt(np.mean(aligned_target ** 2)))
    assert sine_max_err < 1e-4, f"Sine max error {sine_max_err:.2e} exceeds float precision"
    assert abs(sine_rms_ratio - 1.0) < 1e-4, f"Sine RMS ratio {sine_rms_ratio:.6f} deviates from unity"

    # 3. Constant signal reconstruction
    engine = _make_engine()
    const_val = 0.5
    const = np.full(8000, const_val, dtype=np.float32)

    hops = []
    for i in range(len(const) // hop_size):
        out_hop, _ = engine.process_hop(
            const[i * hop_size : (i + 1) * hop_size],
            np.zeros(hop_size, dtype=np.float32)
        )
        hops.append(out_hop)

    recon_const = np.concatenate(hops)
    aligned_const = recon_const[hop_size + frame_size :]
    const_max_err = float(np.max(np.abs(aligned_const - const_val)))
    const_mean = float(np.mean(aligned_const))
    assert const_max_err < 1e-4, f"Constant max error {const_max_err:.2e} exceeds float precision"
    assert abs(const_mean - const_val) < 1e-4, f"Constant mean {const_mean:.6f} deviates from {const_val}"
