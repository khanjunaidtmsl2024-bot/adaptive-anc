# ADAPTIVE-DEFENCE ANC: Hybrid AI-DSP Edge Speech Enhancement
> **Smart India Hackathon (SIH) 2026 — Problem Statement ID: 26052**  
> **Organization:** Defence Research and Development Organisation (DRDO)  
> **Department:** Department of Defence Production / iDEX • **Category:** Hardware • **Theme:** Smart Vehicles  

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![SIH 2026](https://img.shields.io/badge/SIH-2026%20PS%2026052-orange.svg)](https://sih.gov.in/)
[![DRDO Defence](https://img.shields.io/badge/DRDO-Defence%20Hardware-green.svg)](https://drdo.gov.in/)
[![Real-Time Factor](https://img.shields.io/badge/RTF-0.016%20(Edge%20Verified)-brightgreen.svg)]()
[![Latency Target](https://img.shields.io/badge/Round--Trip%20Latency-26.8ms%20(%3C30ms%20target)-yellowgreen.svg)]()
[![Automated Tests](https://img.shields.io/badge/Tests-15%20Passed-brightgreen.svg)]()
[![CI Status](https://img.shields.io/badge/CI-Passing-brightgreen.svg)]()

---

## 📌 Executive Summary

Modern defence communication environments (armoured vehicle cockpits, combat aircraft, forward artillery posts, and naval engine rooms) present extreme acoustic challenges: **intense non-stationary background noise (tank engine rumble, propeller hum) interspersed with high-energy impulsive transients (gunfire, artillery blast, shockwaves) exceeding 110–130 dB SPL**.

Standard pure-DSP adaptive filters (such as classic FxLMS) suffer from stability collapse and divergence during impulsive shocks. Conversely, pure end-to-end Deep Learning models suffer from excessive parameter counts and prohibitive latency (>50–100 ms) unviable on tactical edge silicon.

**ADAPTIVE-DEFENCE ANC** solves this via a **Hybrid AI-DSP Two-Stage Edge Architecture**:
1. **Stage 1: Classical Adaptive DSP (Pre-AI Reference Canceller - Config A):** Fast Normalized Least Mean Squares (NLMS) with dual-microphone reference cancellation, tracking and suppressing correlated stationary/engine noise before non-linear saturation.
2. **Stage 2: Deep Learning Enhancement Stage (Residual Enhancer / TinyEnhancer):** A compact neural network that models remaining non-linear residual noise, preserves speech spectral formants, and suppresses impulsive transients without speech distortion.
3. **Fail-Safe Supervisory Controller:** Continuous crest factor and spectral flux monitoring that freezes adaptation and applies soft limiting during high-energy ballistic shockwaves.

---

## 🎯 Target Specifications & Measured Status (DRDO PS 26052)

| Performance Parameter | Target Requirement | Measured Baseline Status | Verification Engine |
|:---|:---:|:---:|:---|
| **Signal-to-Noise Ratio (SNR)** | **> 15.0 dB** | **+21.15 dB rejection** | `src/evaluation/metrics.py` |
| **Speech Intelligibility (STOI)** | **> 0.85** | **0.865 (Measured)** | `pystoi` automated suite |
| **Perceptual Speech Quality (PESQ)** | **> 2.50 (ITU-T P.862)** | **2.68 (Measured)** | `pesq` automated suite |
| **Algorithmic Latency** | **< 30.0 ms** | **26.8 ms** (End-to-End Budget) | `hardware/latency_budget.md` |
| **Real-Time Factor (RTF)** | **< 1.0 (Real-Time)** | **0.016** (Only 1.6% of budget) | `main.py benchmark` profiler |
| **Impulsive Transient Suppression** | **> 20 dB suppression** | **23.4 dB** (Huber M-estimator) | `src/dsp/kalman.py` |
| **Stage-1 Hardware BOM** | **< ₹3,000 INR** | **₹2,750 INR** | `hardware/bill_of_materials.md` |

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
                                        |   FAIL-SAFE DYNAMIC CONTROLLER  |
                                        | (Crest Factor & Spectral Flux)  |
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
├── README.md                            # Master repository documentation
├── main.py                              # Unified master CLI execution engine
├── pytest.ini                           # Automated test runner configuration
├── requirements.txt                     # Core dependencies (numpy, scipy, pystoi, pesq, pyyaml)
├── Dockerfile                           # Containerized edge & cloud deployment
├── .gitignore                           # Hardened against >100MB archives, binaries & raw WAVs
│
├── .github/                             # Continuous Integration & Delivery
│   └── workflows/ci.yml                 # Automated test & 20-experiment validation pipeline
│
├── demo/                                # Interactive Demonstration Package
│   ├── index.html                       # Standalone Web Audio API DSP Cockpit & Spectrogram
│   └── README.md                        # Quick-start guide for demonstration
│
├── hardware/                            # Hardware Integration & Bring-Up Package
│   ├── stage1_bringup_guide.md          # 14-Step (H1-H14) Hardware Bring-Up Manual
│   ├── bill_of_materials.md             # ₹2,750 INR BOM with local Indian suppliers
│   ├── verify_hardware_bringup.py       # Automated hardware validation test script
│   ├── specifications.md                # Dual-mic array geometry, ADC/DAC specs
│   ├── embedded_targets.md              # Raspberry Pi 5, Jetson Orin Nano, STM32H7 specs
│   └── latency_budget.md                # Detailed round-trip latency allocation (26.8 ms)
│
├── experiments/                         # Verification, Test Benchmarks & Agent Tracking
│   ├── scripts/
│   │   └── run_all_experiments.py       # Automated runner for EXP-001 through EXP-020
│   ├── benchmarks/
│   │   └── experiments_journal.md       # Measured results & failure analyses (100% Pass)
│   ├── failure_modes/
│   │   └── limitations_and_risks.md     # Filter divergence, transient saturation, mitigations
│   └── agent_analysis/
│       └── ichigo_checkpoint_audit.md   # Rigorous audit of TinyEnhancer checkpoint
│
├── src/                                 # Production Core Engine
│   ├── dsp/                             # Classical & Adaptive Filtering Subsystem
│   │   ├── nlms.py                      # Normalized LMS (Dual-Mic Config A Pre-AI Canceller)
│   │   ├── fxlms.py                     # Filtered-x LMS & Secondary Path Electro-Acoustic Model
│   │   ├── kalman.py                    # Impulsive-robust filter with Huber M-estimator
│   │   ├── spectral_subtraction.py      # Boll (1979) Magnitude Spectral Subtraction baseline
│   │   └── wiener.py                    # Decision-Directed a priori SNR Wiener filter
│   ├── ai/                              # Neural Network Wrappers & Optimization
│   │   ├── model_wrapper.py             # Abstract base model wrapper
│   │   ├── tiny_enhancer.py             # PyTorch implementation of TinyEnhancer architecture
│   │   ├── external_models.py           # Checkpoint loader & baseline adapters
│   │   └── export_onnx.py               # TorchScript JIT, INT8 quantization & TensorRT exporter
│   ├── integrations/                    # External Collaborator Bridges
│   │   └── ichigo_bridge.py             # ichigo137/anc model connector & hybrid benchmark
│   ├── pipeline/                        # Hybrid Two-Stage Pipeline
│   │   ├── hybrid_chain.py              # Dual-Mic Pre-AI NLMS -> Neural Residual Enhancer
│   │   └── fallback_controller.py       # Crest-factor & spectral-flux safety controller
│   ├── streaming/                       # Edge Real-Time Engine (<32ms)
│   │   ├── ring_buffer.py               # Circular buffer for low-latency frame processing
│   │   ├── stft_engine.py               # Overlap-add STFT/iSTFT frame processor (50% overlap)
│   │   ├── latency_profiler.py          # Frame latency, jitter & RTF profiler
│   │   └── live_stream_audio.py         # Hardware callback stream for live Mic/Headphone I/O
│   ├── dataset/                         # Acoustic Data Engineering
│   │   ├── noise_synthesizer.py         # Defence noise generators (tank, rotor, gunfire, siren)
│   │   └── mixer.py                     # Calibrated SNR mixer with JSON/CSV provenance
│   └── evaluation/                      # Multi-Metric Evaluation Suite
│       ├── metrics.py                   # ITU-T P.862 PESQ, STOI, SNR, SI-SNR, SDR
│       └── reporter.py                  # Generates comparative CSV/Markdown tables
│
├── research/                            # Deep Research & Knowledge Hub
│   ├── README.md                        # Master research navigation guide
│   ├── dossiers/                        # Consolidated Markdown Research Dossiers
│   │   ├── ICHIGO_ANC_FORENSIC_REVERSE_ENGINEERING.md # Code & checkpoint disassembly
│   │   ├── SIH_2026_DEFENSE_AND_TEAM_ROADMAP.md       # RACI, 30/60/90-day plan, 28 Q&A
│   │   ├── PS_26052_MASTER_REPORT_V6.md               # Master consolidated single-source-of-truth
│   │   ├── DRDO_ANC_Deep_Research.md                  # Deep technical research synthesis
│   │   ├── 00_RESEARCH_PAPER_READING_ORDER.md         # Curated paper reading progression
│   │   ├── 01_V1_CONTRACT.md                          # Hardware-Software signal contracts
│   │   ├── 02_BUILD_GUIDE.md                          # Reproduction guide
│   │   ├── 03_REFERENCE_LIBRARY.md                    # Consolidated literature bibliography
│   │   ├── 04_EVIDENCE_LOG.md                         # Experimental evidence & proofs
│   │   ├── 05_DEFENCE_PACKAGE.md                      # DRDO defence pitch & jury justification
│   │   ├── 06_REBUILD_GUIDE.md                        # Clean rebuild procedure
│   │   └── 07_PROTOTYPE_DEMO_PLAN.md                  # Real-time live demo protocol
│   ├── raw_documents/                   # Archived original .docx/.pdf reference documents
│   └── papers/                          # 60+ Curated research papers & indices
│
├── data/                                # Packaged Demonstration & Test Samples
│   ├── samples/                         # Standardized 16kHz WAV samples (<100KB each)
│   │   ├── clean_speech_sample.wav      # Ground truth reference speech
│   │   ├── tank_t90_engine_sample.wav   # T-90 Main Battle Tank diesel engine noise
│   │   ├── helicopter_rotor_sample.wav  # ALH Dhruv blade slap acoustic signature
│   │   ├── gunfire_impulse_sample.wav   # INSAS 5.56mm rapid rifle discharge
│   │   ├── mixed_tank_0db.wav           # 0 dB SNR combat input mixture
│   │   └── hybrid_enhanced_output.wav   # Processed output (+21 dB suppression)
│   └── generate_demo_samples.py         # Zero-dependency sample generator
│
└── tests/                               # Comprehensive Automated Test Suite (15 Tests Passing)
    ├── test_baselines_and_integrations.py # Tests spectral subtraction, wiener, edge export, bridge
    ├── test_nlms.py                     # Tests NLMS convergence & stability
    ├── test_streaming.py                # Tests ring buffer & algorithmic latency (<32ms)
    ├── test_metrics.py                  # Tests STOI, PESQ, SI-SNR against known standards
    └── test_pipeline.py                 # Tests end-to-end hybrid AI-DSP execution
```

---

## 🤖 Integration & Bridge with [`ichigo137/anc`](https://github.com/ichigo137/anc)

Our collaborator operates in [`ichigo137/anc`](https://github.com/ichigo137/anc), developing pure neural network enhancements (`train.py`, `models/tiny_enhancer.pt`).

Our master repository incorporates a dedicated bridge (`src/integrations/ichigo_bridge.py`):
1. **Model Ingestion:** Ingests and inspects checkpoints (`tiny_enhancer.pt`, 10,417 params, 4-layer 2D ConvNet).
2. **Phase Restoration:** Upgrades their magnitude-only masking by applying phase-consistent STFT/iSTFT synthesis.
3. **Pre-AI Dual-Mic NLMS Coupling:** Routes incoming audio through our causal Stage 1 NLMS filter, reducing stationary engine noise by 15–20 dB before the neural net processes non-linear speech components.
4. **Benchmarking:** Run `python main.py ichigo-bridge` to see real-time comparative gains on DRDO combat noises.

---

## 🚀 Unified Master CLI Reference

All system operations can be triggered via `main.py`:

```bash
# 1. Run the 20 Mandatory Experiments Suite (EXP-001 to EXP-020)
python main.py experiments

# 2. Execute the 14-Step Stage-1 Hardware Bring-Up Verification (H1 to H14)
python main.py hardware-check

# 3. Launch Interactive Web Audio Cockpit in browser (Port 8000)
python main.py demo

# 4. Export model to TorchScript JIT, INT8 quantization & TensorRT guide
python main.py export-edge

# 5. Audit & benchmark collaborator repo (ichigo137/anc)
python main.py ichigo-bridge

# 6. Profile streaming latency & Real-Time Factor (RTF)
python main.py benchmark --duration 5.0

# 7. Run Classical & Adaptive DSP filter laboratory
python main.py dsp-lab

# 8. Synthesize defence acoustic noises (tank, helicopter, gunfire)
python main.py generate-data --preset tank --duration 5.0

# 9. Run real-time streaming audio simulation (or live hardware I/O)
python main.py stream --duration 3.0
python main.py stream --duration 5.0 --live

# 10. View forensic checkpoint reverse-engineering audit
python main.py audit

# 11. Run full automated unit test suite (15 tests)
pytest -v
```

---

## 🐳 Docker Deployment

To build and run the entire repository in an isolated container on any OS:

```bash
# Build Docker image
docker build -t adaptive-anc:latest .

# Run 20 experiments in container
docker run --rm adaptive-anc:latest

# Run Web Audio Demo on port 8000
docker run --rm -p 8000:8000 adaptive-anc:latest python main.py demo
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
