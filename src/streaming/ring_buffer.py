"""
Circular Ring Buffer for Low-Latency Real-Time Audio Streaming.
PS 26052 — Adaptive Defence ANC.
"""

import numpy as np


class RingBuffer:
    """Thread-safe lock-free compatible circular buffer for continuous audio frames."""

    def __init__(self, capacity: int = 4096, dtype=np.float32):
        self.capacity = int(capacity)
        self.buffer = np.zeros(self.capacity, dtype=dtype)
        self.write_idx = 0
        self.read_idx = 0
        self.size = 0

    def reset(self) -> None:
        self.buffer.fill(0.0)
        self.write_idx = 0
        self.read_idx = 0
        self.size = 0

    def write(self, data: np.ndarray) -> int:
        """Write array into circular buffer. Overwrites oldest data if full."""
        n = len(data)
        if n > self.capacity:
            data = data[-self.capacity:]
            n = self.capacity

        first_chunk = min(n, self.capacity - self.write_idx)
        self.buffer[self.write_idx:self.write_idx + first_chunk] = data[:first_chunk]

        second_chunk = n - first_chunk
        if second_chunk > 0:
            self.buffer[:second_chunk] = data[first_chunk:]

        self.write_idx = (self.write_idx + n) % self.capacity
        self.size = min(self.capacity, self.size + n)
        return n

    def read(self, n_samples: int) -> np.ndarray:
        """Read and consume n_samples from circular buffer."""
        n = min(n_samples, self.size)
        out = np.zeros(n_samples, dtype=self.buffer.dtype)

        first_chunk = min(n, self.capacity - self.read_idx)
        out[:first_chunk] = self.buffer[self.read_idx:self.read_idx + first_chunk]

        second_chunk = n - first_chunk
        if second_chunk > 0:
            out[first_chunk:first_chunk + second_chunk] = self.buffer[:second_chunk]

        self.read_idx = (self.read_idx + n) % self.capacity
        self.size -= n
        return out

    def peek(self, n_samples: int) -> np.ndarray:
        """Read n_samples without advancing read index."""
        n = min(n_samples, self.size)
        out = np.zeros(n_samples, dtype=self.buffer.dtype)

        first_chunk = min(n, self.capacity - self.read_idx)
        out[:first_chunk] = self.buffer[self.read_idx:self.read_idx + first_chunk]

        second_chunk = n - first_chunk
        if second_chunk > 0:
            out[first_chunk:first_chunk + second_chunk] = self.buffer[:second_chunk]

        return out
