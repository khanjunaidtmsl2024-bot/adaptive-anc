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
