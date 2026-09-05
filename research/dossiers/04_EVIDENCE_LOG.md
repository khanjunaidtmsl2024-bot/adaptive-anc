# PS 26052 — EVIDENCE LOG
## Decision Records, Experiment Results, and Ablation Evidence
### Grow this document during execution. Never delete a negative result.

**Document:** 04_EVIDENCE_LOG.md
**Contract:** 01_V1_CONTRACT.md
**Build Guide:** 02_BUILD_GUIDE.md

---

## How to Use This Document

1. Every architectural decision gets a Decision Record entry.
2. Every experiment gets an Experiment Record entry.
3. Every metric goes into a Results Table.
4. Negative results are preserved honestly.
5. Every claim in the final report must trace to an entry here.

---

## DECISION LOG

### D001 — Config A for NLMS Signal Flow
- **Date:** 2026-08-24
- **Decision:** Use Config A (Pre-AI Reference Canceller) as the V1 adaptive stage
- **Alternatives considered:**
  - Config B (Post-AI Residual): Requires reference mic with zero speech — unverifiable in field
  - Config C (Spectral Mask): Valid but not classical NLMS
- **Rationale:** d[n] (primary mic) and x[n] (reference mic) are both observable at runtime. Mathematically valid. Reference mic placement is an experiment, not an assumption.
- **Owner:** DSP Lead
- **Revisit if:** Ablation in Phase 9 shows NLMS before AI doesn't improve over AI-only
- **Status:** FROZEN (per Contract)

### D002 — DeepFilterNet2 as Primary AI Model
- **Date:** 2026-08-24
- **Decision:** Use DeepFilterNet2 as first AI benchmark
- **Alternatives considered:**
  - DCCRN: Strong baseline, good for comparison table
  - Compact CRN: Fallback if DeepFilterNet2 deployment fails
  - Conv-TasNet: Lower latency but weaker quality
- **Rationale:** Purpose-built for real-time embedded speech enhancement. Deep filtering handles phase well. Open source, actively maintained.
- **Owner:** ML Lead
- **Revisit if:** DeepFilterNet2 fails ONNX export or deployment gate in Phase 7/10
- **Status:** FROZEN (per Contract)

### D003 — Sample Rate Determination
- **Date:** TBD (Phase 7)
- **Decision:** Pending — verify DeepFilterNet2's actual sample rate requirement
- **Likely outcome:** 48 kHz (DeepFilterNet2 default), NOT 16 kHz
- **Impact:** Changes entire audio pipeline, microphone selection, and compute budget
- **Owner:** ML Lead + Hardware Lead
- **Revisit if:** Hardware cannot support the model's required rate

### D004 — [Placeholder for next decision]
- **Date:** TBD
- **Decision:** TBD
- **Alternatives:** TBD
- **Rationale:** TBD
- **Owner:** TBD
- **Revisit if:** TBD

---

## EXPERIMENT LOG

### EXP-001 — Environment Setup
- **Phase:** 0
- **Date:** TBD
- **Setup:** Python 3.10+, virtual environment, dependencies installed
- **Result:** TBD
- **Conclusion:** TBD
- **Gate:** PASS / FAIL

### EXP-002 — STFT/iSTFT Reconstruction
- **Phase:** 1
- **Date:** TBD
- **Setup:** Manual implementation, Hann window, various hop sizes
- **Result:** TBD
- **Reconstruction error:** TBD
- **Conclusion:** TBD
- **Gate:** PASS / FAIL

### EXP-003 — NLMS Convergence
- **Phase:** 2
- **Date:** TBD
- **Setup:** Synthetic primary/reference, mu=0.3, taps=128
- **Result:** TBD
- **Convergence time:** TBD
- **Output SNR improvement:** TBD
- **Conclusion:** TBD
- **Gate:** PASS / FAIL

### EXP-004 — [Placeholder]
- **Phase:** TBD
- **Date:** TBD
- **Setup:** TBD
- **Result:** TBD
- **Conclusion:** TBD
- **Gate:** PASS / FAIL

---

## ABBLATION MATRIX (Phase 9)

| System | Input SNR | Output SNR | Delta SNR | STOI | PESQ | Latency P95 | Artifacts | Notes |
|--------|-----------|------------|-----------|------|------|-------------|-----------|-------|
| Raw (no processing) | | | 0.0 | | | | | Baseline |
| Spectral subtraction | | | | | | | | Classical baseline |
| NLMS only | | | | | | | | Config A |
| AI only (DeepFilterNet2) | | | | | | | | No NLMS |
| NLMS then AI | | | | | | | | V1 hybrid |
| NLMS then AI + Controller | | | | | | | | Full V1 |

### By Noise Category
| Noise Type | System | SNR | STOI | PESQ |
|------------|--------|-----|------|------|
| Stationary | Raw | | | |
| Stationary | NLMS-only | | | |
| Stationary | AI-only | | | |
| Stationary | NLMS-AI | | | |
| Non-stationary | Raw | | | |
| Non-stationary | NLMS-only | | | |
| Non-stationary | AI-only | | | |
| Non-stationary | NLMS-AI | | | |
| Impulsive | Raw | | | |
| Impulsive | NLMS-only | | | |
| Impulsive | AI-only | | | |
| Impulsive | NLMS-AI | | | |
| Unseen (NOISEX-92) | Raw | | | |
| Unseen (NOISEX-92) | AI-only | | | |
| Unseen (NOISEX-92) | NLMS-AI | | | |

---

## LATENCY MEASUREMENTS

| Component | P50 (ms) | P95 (ms) | P99 (ms) | Worst (ms) |
|-----------|----------|----------|----------|------------|
| Input buffering | | | | |
| Audio driver | | | | |
| STFT framing | | | | |
| NLMS | | | | |
| AI inference | | | | |
| iSTFT | | | | |
| Output buffering | | | | |
| DAC/headphone | | | | |
| **Total end-to-end** | | | | |

---

## NEGATIVE RESULTS

Preserve every result that didn't work. A negative result is still an engineering result.

### NEG-001 — [Placeholder]
- **Date:** TBD
- **What was tried:** TBD
- **What happened:** TBD
- **Why it failed:** TBD
- **What we learned:** TBD
- **Action taken:** TBD

---

## FAILURE INJECTION RESULTS

| Fault | Detected? | Safe State? | Recovery Time | Notes |
|-------|-----------|-------------|---------------|-------|
| Reference mic disconnected | | | | |
| AI inference timeout | | | | |
| Input clipping | | | | |
| Wrong sample rate | | | | |
| Processing queue full | | | | |
| Audio device restart | | | | |

---

## UNSEEN-NOISE RESULTS

| Noise Source | Trained On? | SNR | STOI | PESQ | Generalization |
|--------------|-------------|-----|------|------|----------------|
| DNS Challenge | YES | | | | |
| MAD dataset | YES | | | | |
| NOISEX-92 babble | NO | | | | |
| NOISEX-92 f16 | NO | | | | |
| NOISEX-92 factory | NO | | | | |

---

## COST TRACKING

| Item | Source | Price | Date | Notes |
|------|--------|-------|------|-------|
| | | | | |

---

*Every claim in the Defence Package (05) must trace to an entry in this log.*
