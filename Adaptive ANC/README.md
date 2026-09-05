# PS 26052 — Adaptive ANC
## Hybrid AI-DSP Edge Speech Enhancement for Dynamic and Impulsive Defence Noise

**DRDO / Department of Defence Production / iDEX — SIH 2026 — Hardware**

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate demo dataset
python artifacts/01_dataset_generator.py --preset demo --max-mixes 50

# 3. Run adaptive filter experiment
python artifacts/04_adaptive_filter_lab.py --mode compare --duration 3.0

# 4. Benchmark streaming pipeline
python artifacts/03_streaming_skeleton.py --mode benchmark

# 5. Evaluate any model (once you have clean/noisy/enhanced files)
python artifacts/02_evaluation_harness.py --clean data/clean/demo.wav --enhanced results/enhanced.wav
```

---

## Project Structure

```
Adaptive ANC/
├── artifacts/                    # Executable starter code
│   ├── 01_dataset_generator.py   # Clean+noise mixing with metadata CSV
│   ├── 02_evaluation_harness.py  # SNR/STOI/PESQ/SI-SNR pipeline
│   ├── 03_streaming_skeleton.py  # Ring buffer + STFT/iSTFT + audio callbacks
│   └── 04_adaptive_filter_lab.py # NLMS with 3 signal flow configurations
│
├── data/
│   ├── clean/                    # Clean speech WAVs
│   ├── noise/                    # Noise WAVs
│   ├── mixed/                    # Generated mixtures (train/val/test)
│   └── metadata/                 # Provenance CSVs
│
├── results/                      # Evaluation outputs
├── configs/                      # Versioned configuration files
├── experiments/                  # Experiment records
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## PS 26052 Targets

| Metric | Target | Current |
|--------|--------|---------|
| SNR | > 15 dB | TBD |
| STOI | > 0.85 | TBD |
| PESQ | > 2.5 | TBD |
| Latency | < 30 ms | 32 ms (algorithmic) |
| RTF | < 1.0 | 0.011 |

---

## Critical-Path Execution Order

1. Evaluation harness
2. Dataset generator + provenance
3. STFT/iSTFT streaming loop
4. Classical baselines (spectral subtraction, Wiener)
5. DeepFilterNet compatibility + pretrained benchmark
6. Real microphone streaming
7. Reference-mic position experiment
8. Formally defined adaptive-filter experiment (Config A recommended)
9. Impulse library + detector
10. Rule-based controller
11. AI + adaptive ablation
12. Embedded optimization
13. Hardware-in-the-loop
14. Full PS acceptance test

---

## Adaptive Architecture Recommendation

**Use Config A (Pre-AI Reference Canceller)** as the primary adaptive stage:

```
Primary mic → [NLMS Reference Canceller] → AI Enhancement → Output
     ↑                        ↑
     |                        |
Reference mic ────────────────┘
```

This is mathematically valid because both `d(n)` (primary) and `x(n)` (reference) are observable at runtime.

---

## References

- DeepFilterNet2: https://arxiv.org/abs/2205.05474
- RNNoise: https://github.com/xiph/rnnoise
- DNS Challenge: https://github.com/microsoft/DNS-Challenge

---

## License

Internal project — SIH 2026
