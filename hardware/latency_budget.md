# End-to-End Latency Budget & Real-Time Performance
**Project:** ADAPTIVE-DEFENCE ANC • **DRDO PS 26052 (SIH 2026)**  
**Constraint:** Total Algorithmic & Acoustic Round-Trip Latency $< 30.0\text{ ms}$  

---

## 1. Latency Breakdown

```
[ADC Sampling] ──> [STFT Hop Buffer] ──> [NLMS DSP] ──> [Neural Enhancer] ──> [iSTFT Synthesis] ──> [DAC Output]
   (1.5 ms)           (16.0 ms)           (0.8 ms)           (6.5 ms)             (0.5 ms)          (1.5 ms)
```

| Pipeline Stage | Buffer / Mechanism | Latency Allocation | Notes |
|:---|:---|:---:|:---|
| **ADC Ingestion** | Hardware I2S DMA Ping-Pong Buffer | $1.5\text{ ms}$ | 24 samples @ 16 kHz |
| **STFT Hop Accumulation** | Overlap buffer accumulation ($50\%$ overlap) | $16.0\text{ ms}$ | 256 samples @ 16 kHz |
| **Stage 1: NLMS Adaptive Filter** | Vectorized FIR tap update ($M=64$) | $0.8\text{ ms}$ | SIMD accelerated (NEON / AVX) |
| **Stage 2: Neural Enhancer** | ONNX / PyTorch Forward Pass | $6.5\text{ ms}$ | Benchmarked on Raspberry Pi 5 CPU |
| **iSTFT Reconstruction** | Overlap-add synthesis window | $0.5\text{ ms}$ | Real-valued inverse FFT |
| **DAC Output Buffer** | Hardware audio playback buffer | $1.5\text{ ms}$ | Direct DMA feed |
| **TOTAL ROUND-TRIP LATENCY** | **End-to-End Microphone-to-Ear** | **26.8 ms** | **PASSED (< 30.0 ms DRDO Target)** |
