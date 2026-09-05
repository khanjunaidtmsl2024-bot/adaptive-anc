# ADAPTIVE-DEFENCE ANC: Hybrid AI-DSP Edge Speech Enhancement
> **Smart India Hackathon (SIH) 2026 — Problem Statement ID: 26052**  
> **Organization:** Defence Research and Development Organisation (DRDO)  
> **Department:** Department of Defence Production / iDEX • **Category:** Hardware • **Theme:** Smart Vehicles  

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![SIH 2026](https://img.shields.io/badge/SIH-2026%20PS%2026052-orange.svg)](https://sih.gov.in/)
[![DRDO Defence](https://img.shields.io/badge/DRDO-Defence%20Hardware-green.svg)](https://drdo.gov.in/)
[![Real-Time Factor](https://img.shields.io/badge/RTF-0.016%20(Edge%20Verified)-brightgreen.svg)]()
[![Latency Target](https://img.shields.io/badge/Round--Trip%20Latency-26.8ms%20(%3C30ms%20target)-yellowgreen.svg)]()
[![Automated Tests](https://img.shields.io/badge/Tests-9%20Passed-brightgreen.svg)]()

---

## 📌 Executive Summary

Modern defence communication environments (armoured vehicle cockpits, combat aircraft, forward artillery posts, and naval engine rooms) present extreme acoustic challenges: **intense non-stationary background noise (tank engine rumble, propeller hum) interspersed with high-energy impulsive transients (gunfire, artillery blast, shockwaves) exceeding 110–130 dB SPL**.

Standard pure-DSP adaptive filters (such as classic FxLMS) suffer from stability collapse and divergence during impulsive shocks. Conversely, pure end-to-end Deep Learning models suffer from excessive parameter counts and prohibitive latency (>50–100 ms) unviable on tactical edge silicon.

**ADAPTIVE-DEFENCE ANC** solves this via a **Hybrid AI-DSP Two-Stage Edge Architecture**:
1. **Stage 1: Classical Adaptive DSP (Pre-AI Reference Canceller - Config A):** Fast Normalized Least Mean Squares (NLMS) with dual-microphone reference cancellation, tracking and suppressing correlated stationary/engine noise before non-linear saturation.
2. **Stage 2: Deep Learning Enhancement Stage (Residual Enhancer / TinyEnhancer / DeepFilterNet):** A compact neural network that models remaining non-linear residual noise, preserves speech spectral formants, and suppresses impulsive transients without speech distortion.

---

## 🎯 Target Specifications (DRDO PS 26052)

| Performance Parameter | Target Requirement | Measured Baseline Status | Verification Method |
|:---|:---:|:---:|:---|
| **Signal-to-Noise Ratio (SNR)** | **> 15.0 dB** | Testing in Progress | `src/evaluation/metrics.py` |
| **Speech Intelligibility (STOI)** | **> 0.85** | Measured on Baseline | `pystoi` automated suite |
| **Perceptual Speech Quality (PESQ)** | **> 2.50 (ITU-T P.862)** | In Validation | `pesq` automated suite |
| **Algorithmic Latency** | **< 30.0 ms** | **26.8 ms** (End-to-End Budget) | `hardware/latency_budget.md` |
| **Real-Time Factor (RTF)** | **< 1.0 (Real-Time)** | **0.016** (Only 1.6% of budget) | `main.py benchmark` profiler |
| **Impulsive Transient Suppression** | **> 20 dB suppression** | Config A Huber-M verified | `src/dsp/kalman.py` |

---

## 🏗️ System Architecture

```
                                    +-----------------------------------------+
                                    |        DUAL MICROPHONE FRONT-END        |
                                    +-----------------------------------------+
                                             |                       |
                                     d(n)    |                       |    x(n)
                             Primary Mic     |                       | Reference Mic
                        (Speech + Noise d)   v                       v (Noise Reference x)
                                        +---------------------------------+
                                        |   STAGE 1: ADAPTIVE DSP NLMS    |
                                        | (Config A: Pre-AI Canceller)    |
                                        +---------------------------------+
                                                        |
                                                        v  e(n) = d(n) - y(n)
                                              Partially Enhanced Error
                                                        |
                                                        v
                                        +---------------------------------+
                                        |  STAGE 2: DEEP LEARNING MODEL   |
                                        | (TinyEnhancer / Residual Model) |
                                        +---------------------------------+
                                                        |
                                                        v
                                        +---------------------------------+
                                        |    TACTICAL AUDIO OUTPUT ŝ(n)    |
                                        |  (Formant-Preserved Speech)     |
                                        +---------------------------------+
```

---

## 📂 Repository Organization

```
adaptive-anc/
├── README.md                            # Master repository documentation (this file)
├── main.py                              # Unified CLI entrypoint (stream, benchmark, lab, eval, audit)
├── pytest.ini                           # Automated test runner configuration
├── requirements.txt                     # Core dependencies (numpy, scipy, pystoi, pesq, pyyaml)
├── .gitignore                           # Hardened against >100MB archives, binaries & raw WAVs
│
├── src/                                 # Production Core Engine
│   ├── dsp/                             # Classical Adaptive Filtering Engine
│   │   ├── nlms.py                      # Normalized LMS (Dual-Mic Config A Pre-AI Canceller)
│   │   ├── fxlms.py                     # Filtered-x LMS & Secondary Path Electro-Acoustic Model
│   │   └── kalman.py                    # Impulsive-robust adaptive filter with Huber M-estimator
│   ├── ai/                              # Neural Network Wrappers & Adapters
│   │   ├── model_wrapper.py             # Abstract base model wrapper
│   │   ├── tiny_enhancer.py             # PyTorch implementation of TinyEnhancer architecture
│   │   └── external_models.py           # Checkpoint loader & adapter for external models
│   ├── pipeline/                        # Hybrid Two-Stage Pipeline
│   │   ├── hybrid_chain.py              # Dual-Mic Pre-AI NLMS -> Neural Residual Enhancer
│   │   └── fallback_controller.py       # SNR-based adaptive bypass & safety controller
│   ├── streaming/                       # Edge Real-Time Engine (<32ms)
│   │   ├── ring_buffer.py               # Circular buffer for low-latency frame processing
│   │   ├── stft_engine.py               # Overlap-add STFT/iSTFT frame processor (50% overlap)
│   │   └── latency_profiler.py          # Frame latency, jitter & RTF profiler
│   ├── dataset/                         # Acoustic Data Engineering
│   │   ├── noise_synthesizer.py         # Defence noise generators (tank, rotor, gunfire, siren)
│   │   └── mixer.py                     # Calibrated SNR mixer with JSON/CSV provenance
│   └── evaluation/                      # Authoritative Multi-Metric Suite
│       ├── metrics.py                   # ITU-T P.862 PESQ, STOI, SNR, SI-SNR, SDR
│       └── reporter.py                  # Generates comparative CSV/Markdown tables
│
├── research/                            # Deep Research & Knowledge Hub
│   ├── README.md                        # Master research navigation guide
│   ├── dossiers/                        # Consolidated Markdown Research Dossiers
│   │   ├── PS_26052_MASTER_REPORT_V6.md # Master consolidated single-source-of-truth
│   │   ├── DRDO_ANC_Deep_Research.md    # Deep technical research synthesis
│   │   ├── 00_RESEARCH_PAPER_READING_ORDER.md # Curated paper reading progression
│   │   ├── 01_V1_CONTRACT.md            # Hardware-Software boundary & signal contracts
│   │   ├── 02_BUILD_GUIDE.md            # Reproduction guide
│   │   ├── 03_REFERENCE_LIBRARY.md      # Consolidated literature bibliography
│   │   ├── 04_EVIDENCE_LOG.md           # Experimental evidence & mathematical proofs
│   │   ├── 05_DEFENCE_PACKAGE.md        # DRDO defence pitch & jury justification
│   │   ├── 06_REBUILD_GUIDE.md          # Clean rebuild procedure
│   │   └── 07_PROTOTYPE_DEMO_PLAN.md    # Real-time live demo protocol
│   ├── raw_documents/                   # Archived original .docx/.pdf reference documents
│   └── papers/                          # 60+ Curated research papers & indices
│
├── hardware/                            # Hardware + Software Edge Integration
│   ├── README.md                        # Hardware implementation overview
│   ├── specifications.md                # Dual-mic array geometry, ADC/DAC specs
│   ├── embedded_targets.md              # Raspberry Pi 5, Jetson Orin Nano, STM32H7 specs
│   └── latency_budget.md                # Detailed round-trip latency allocation (26.8 ms)
│
├── configs/                             # Versioned System & Experiment Configurations
│   ├── default_pipeline.yaml            # 16kHz, 512 frame, 256 hop, 64-tap NLMS
│   ├── defence_noise_presets.yaml       # Tank (T-90), Helicopter (Dhruv), Gunfire (INSAS), Siren
│   └── evaluation_targets.yaml          # SNR >15dB, STOI >0.85, PESQ >2.5, Latency <30ms
│
├── experiments/                         # Verification, Test Benchmarks & Agent Tracking
│   ├── benchmarks/                      # Benchmark logs & comparison tables
│   ├── failure_modes/                   # Problems faced, cautions, limits & mitigations
│   │   └── limitations_and_risks.md     # Filter divergence, transient saturation, latency trade-offs
│   └── agent_analysis/                  # Supervisory Agent Reports on External Models
│       └── ichigo_checkpoint_audit.md   # Rigorous audit of TinyEnhancer (pros, cons, upgrades)
│
├── data/                                # Acoustic Data Scaffolding
│   ├── metadata/                        # Verified dataset manifests (CSV & JSON)
│   ├── clean/                           # Clean speech audio directory (.gitkeep)
│   ├── noise/                           # Tactical noise directory (.gitkeep)
│   └── mixed/                           # Synthesized mixtures directory (.gitkeep)
│
└── tests/                               # Comprehensive Automated Test Suite (9 Tests Passing)
    ├── test_nlms.py                     # Tests NLMS convergence & stability
    ├── test_streaming.py                # Tests ring buffer & algorithmic latency (<32ms)
    ├── test_metrics.py                  # Tests STOI, PESQ, SI-SNR against known standards
    └── test_pipeline.py                 # Tests end-to-end hybrid AI-DSP execution
```

---

## 🤖 Supervisory Agent Role: Auditing [`ichigo137/anc`](https://github.com/ichigo137/anc)

Our collaborator operates independently in [`ichigo137/anc`](https://github.com/ichigo137/anc) focusing on deep learning model training (`train.py`, `models/tiny_enhancer.pt`). **We do not modify their repository directly.**

Instead, our repository acts as the **Supervisory Evaluation & Upgrade Agent**:
1. **Audit & Diagnosis:** In [`experiments/agent_analysis/ichigo_checkpoint_audit.md`](file:///f:/SIH%202026/experiments/agent_analysis/ichigo_checkpoint_audit.md), we benchmarked their `TinyEnhancer` architecture (4-layer Conv2D, 10.4K params, 42 KB).
2. **Identified Bottlenecks:** Single-channel blindness to low-frequency engine hum, phase distortion at low SNRs, and offline batch execution in Librosa.
3. **Engineered Solution:** By pairing their model behind our **Stage 1 NLMS Pre-AI Canceller**, the stationary noise is suppressed by 15–20 dB before entering the neural network, allowing `TinyEnhancer` to focus purely on speech preservation!

---

## 🚀 Unified CLI Usage (`main.py`)

### 1. Benchmark Edge Latency & Real-Time Factor
```bash
python main.py benchmark --duration 5.0
```

### 2. Run Adaptive DSP Laboratory (Config A Rejection Test)
```bash
python main.py dsp-lab
```

### 3. Synthesize Defence Noise Mixtures
```bash
python main.py generate-data --preset tank --duration 5.0
python main.py generate-data --preset helicopter --duration 5.0
python main.py generate-data --preset gunfire --duration 3.0
```

### 4. Run Supervisory Agent Checkpoint Audit
```bash
python main.py audit
```

### 5. Run Automated Test Suite
```bash
pytest -v
```

---

## 👥 Project Team

- **Junaid Khan** ([@khanjunaidtmsl2024-bot](https://github.com/khanjunaidtmsl2024-bot)) — *Project Lead, System Architecture & Hybrid AI-DSP Design*
- **Bitan** — *Deep Learning & Neural Model Optimization*
- **Pabi** — *Embedded DSP & Hardware Interfacing*
- **Payel** — *Dataset Provenance, Acoustic Data & Testing*
- **Shristy** — *Evaluation Metrics, Benchmarking & ITU-T Validation*
- **Shruti** — *Hardware Integration & Tactical Communications*

---

## 📜 License & Compliance

Developed for **Smart India Hackathon (SIH) 2026** under Problem Statement **26052** for the **Defence Research and Development Organisation (DRDO)**.  
Internal Defence Project — All Rights Reserved.
