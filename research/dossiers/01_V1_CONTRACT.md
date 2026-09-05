# PS 26052 — V1 SYSTEM CONTRACT
## Adaptive ANC for Defence Speech Communication
### DRDO / iDEX / SIH 2026 — Hardware

**Version:** 1.0  
**Date:** 24 August 2026  
**Status:** FROZEN — Changes require formal scope-change proposal

---

## 1. Problem Statement

PS 26052 requires an AI/ML-enabled adaptive noise cancellation system that suppresses stationary, non-stationary, and impulsive defence noises while preserving speech intelligibility and operating in real time on embedded/edge hardware.

**Targets:** SNR > 15 dB | STOI > 0.85 | PESQ > 2.5 | Latency < 40 ms

---

## 2. What "ANC" Means in This Project

Our V1 is a **real-time AI speech-enhancement communication system** with a reference-microphone adaptive pre-cleaner. It is NOT full broadband physical anti-noise cancellation (which requires a secondary speaker, error microphone, and secondary-path modelling).

**Terminology rule:** In all reports and presentations, call this "adaptive speech enhancement" or "AI-DSP hybrid noise suppression." Only use "ANC" when quoting the problem statement directly.

---

## 3. Frozen V1 Architecture

```
PRIMARY MIC ────────────────┐
                            │
                            ▼
                      Audio Capture
                            │
REFERENCE MIC ──────────────┘
                            │
                            ▼
                   NLMS Reference Canceller
                      (Config A — Pre-AI)
                            │
                            ▼
                   AI Speech Enhancement
                   DeepFilterNet2 (primary)
                   Compact CRN (fallback)
                            │
                            ▼
                  Deterministic Controller
                            │
                            ▼
                       iSTFT / Output
                            │
                            ▼
                     Headset / Comms Output

Monitoring: latency | underruns | clipping | reference quality
Bypass: raw noisy signal available for A/B comparison
```

**Signal flow is sequential:** NLMS output feeds INTO the AI model. They are NOT parallel.

---

## 4. Frozen NLMS Signal Flow — Config A

| Signal | Definition | Observable at Runtime |
|--------|-----------|----------------------|
| d[n] | Primary microphone (speech + noise) | YES |
| x[n] | Reference microphone (correlated noise) | YES |
| y[n] | w^T * x[n] (noise estimate) | YES (computed) |
| e[n] | d[n] - y[n] (enhanced output) | YES (this IS the output) |

**Update rule:** w[n+1] = w[n] + μ * e[n] * x_vec[n] / (ε + ||x_vec[n]||²)

**Before connecting real microphones:** Verify with synthetic signals that signs, alignment, tap ordering, normalization and delay are correct.

**This is NOT FxLMS.** The reference microphone signal is used directly. There is no secondary-path estimation in V1.

---

## 5. AI Model Decision

**Primary:** DeepFilterNet2 / DeepFilterNet family  
**Fallback:** Compact CRN (if DeepFilterNet2 fails deployment gate)

**CRITICAL — Sample Rate Warning (from V6 §78):**
The official DeepFilterNet repository documents the precompiled deep-filter command as supporting **48 kHz WAV input**, not 16 kHz. The earlier generic "16 kHz first" assumption must not be carried forward.

**Required before freezing audio path:**
1. Install DeepFilterNet2 and check the exact repository version/commit
2. Run DFN-E001: check model package, input sample rate, run offline inference
3. Measure processing time, audio delay, memory, output quality
4. Check audio hardware compatibility with the model's required rate
5. Freeze the rate in experiment configuration

**Do NOT assume 16 kHz.** The model and hardware determine the rate.

**Verified anchor (V6 §127):** "The DeepFilterNet official repository currently documents the precompiled deep-filter path as accepting 48 kHz WAV files, so the project's audio-rate policy must be based on the exact installed version rather than an inherited generic 16 kHz assumption."

---

## 6. Scope Freeze

### IN V1 (Build These)
- Two-microphone capture (primary + reference)
- Streaming STFT pipeline with causal/near-causal processing
- DeepFilterNet2 as first AI benchmark; compact CRN as fallback
- NLMS reference canceller (Config A) BEFORE the AI model
- Rule-based deterministic controller (NO learned controller)
- Impulse detection and protection mode
- Real-time streaming with measured latency
- Controlled acoustic test rig
- Mandatory ablation: raw vs NLMS-only vs AI-only vs NLMS→AI vs NLMS→AI+controller
- Unseen-noise generalization test
- Fail-safe: AI failure → DSP fallback → bypass

### OUT of V1 (Phase 2 — Do Not Build)
- Neural beamforming
- Learned/controller neural network
- Physical FxLMS ANC with secondary speaker
- Online neural domain adaptation
- GAN/diffusion enhancement
- Large microphone arrays (3+ mics)
- Custom PCB or accelerator design
- Mamba/SSM architectures
- Complex-domain masking (start with magnitude-mask)
- Multi-loss composite training (start with L1/SI-SNR)

---

## 7. Acceptance Criteria

| Metric | Target | Test Condition |
|--------|--------|---------------|
| SNR improvement | > 15 dB output SNR | At 0 dB input SNR, stationary noise |
| STOI | > 0.85 | All noise categories separately |
| PESQ | > 2.5 | All noise categories separately |
| End-to-end latency | < 40 ms P95 | Measured on target hardware |
| Real-time factor | < 1.0 | Sustained 10-minute run |
| Unseen noise | Report honestly | NOISEX-92 held-out set |
| Impulse handling | T_recovery < 500 ms | Synthetic impulse library |
| Live demo | mic → processing → headset | No offline recording deception |

---

## 8. Kill Criteria

Pre-committed decision points. If a criterion is triggered, the team pivots immediately — no sunk-cost arguments.

| Day | Criterion | If Triggered |
|-----|-----------|-------------|
| 7 | NLMS doesn't converge on synthetic data | Drop NLMS from V1, go AI-only |
| 14 | DeepFilterNet2 can't export/run on target | Switch to compact CRN fallback |
| 19 | NLMS→AI doesn't beat AI-only in ablation | Remove NLMS from claimed performance |
| 22 | Streaming latency > 40 ms P95 on target | Remove controller, simplify pipeline |
| 25 | No working hardware integration | Demo on laptop only, honestly |

---

## 9. Novelty Claim (Provisional — Updated After Ablation)

> "A low-cost, experimentally validated adaptive AI-DSP pipeline for highly dynamic and impulsive defence-like noise, with explicit dataset construction, controlled unseen-noise validation, runtime state handling, and a measured deployment path — filling the gap between static commercial tactical headset profiles and the need for automatic real-time adaptation."

**Do not claim:**
- Invention of DeepFilterNet, DCCRN, or neural enhancement
- Physical broadband ANC
- Military deployment readiness
- superiority over all commercial systems

---

## 10. Decision Log

All architectural decisions are recorded in `docs/04_EVIDENCE_LOG.md`. The Contract can only be changed by:

1. Written proposal with impact assessment
2. Team vote
3. Updated Contract version number
4. Updated date

---

*This document is the authority. When in doubt, check the Contract.*
