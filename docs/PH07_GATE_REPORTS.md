# PH0.7 Gate Reports

## PH0.7 GATE 1 -- CODE INTEGRITY

**STATUS: PASS (with one item deferred to Gate 2)**

### Checked:

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 1 | train.py has center=False on every torch.stft | **DEFERRED to G2** | center= is NOT explicitly set. PyTorch default is center=True. Docstring note 6 explicitly flags this as pending Gate 2 resolution. |
| 2 | torch.istft uses causally consistent convention | **DEFERRED to G2** | L259: torch.istft inherits default center=True. Must be resolved with item 1. |
| 3 | Validation split is TEST_A_UNSEEN_SPEAKER | **PASS** | L139: val_split defaults to "TEST_A_UNSEEN_SPEAKER". Dataset confirms 18 samples. |
| 4 | Empty validation raises | **PASS** | L184: val_loader.assert_nonempty("validation") raises RuntimeError. |
| 5 | Train/val speakers are disjoint | **PASS** | L187-192: overlap check raises on non-empty. Dataset: TRAIN=SPK_001-008, VAL=SPK_009-010, overlap=set(). |
| 6 | Random seed exists | **PASS** | L140: seed=42. L176-177: torch.manual_seed + np.random.seed. |
| 7 | Parameter count is dynamic | **PASS** | L200: sum(p.numel()...). L337: metadata uses param_count variable. |
| 8 | Checkpoint saving computes SHA-256 | **PASS** | L328: _sha256_of_file(save_p). L120-125: reads and hashes file. |
| 9 | Run metadata is written | **PASS** | L334-364: JSON sidecar with all required fields including training_input/deployment_input/input_contract_status. |
| 10 | IRM is not falsely described as active loss | **PASS** | L233-234: Comment says "diagnostic reference ONLY". Loss at L261: alpha*l1_loss+(1-alpha)*si_loss. IRM not in loss. |
| 11 | metrics.py cannot silently fabricate PESQ/STOI | **PASS** | compute_pesq raises RuntimeError if not PESQ_AVAILABLE. compute_stoi raises RuntimeError if not PYSTOI_AVAILABLE. Surrogates are separate functions. |

### Files changed:
- src/ai/train.py -- replaced with PH0.7 patched version
- src/evaluation/metrics.py -- replaced with PH0.7 patched version

### Remaining risk:
- center= resolution is critical and deferred to Gate 2.

---

## PH0.7 GATE 2 -- DSP CONTRACT

**STATUS: FAIL -- Train/deployment STFT mismatch detected. center= inconsistency confirmed.**

### STFT Configuration Comparison:

| Parameter | Training | CausalEngine | HybridPipeline |
|-----------|----------|--------------|----------------|
| frame_size | 512 | **256** | 512 |
| hop_size | 256 | **128** | 256 |
| center | **True** (default) | N/A (manual) | N/A (manual) |
| FFT bins | 257 | **129** | 257 |
| STFT library | torch.stft | np.fft.rfft | np.fft.rfft |

### OLA Normalization Comparison:

| Path | Method | Normalization |
|------|--------|---------------|
| Training | torch.istft (COLA) | PyTorch internal |
| CausalEngine | overlap_buf += recon | **NONE** (attenuated by ~0.75) |
| HybridPipeline | output += recon*window; norm += window^2; output /= norm | window^2 division |

**OLA MATCH: NO -- three different reconstruction methods**

### Phase Convention:
MATCH: YES -- all preserve noisy/input phase, magnitude-only enhancement

### Critical Findings:
1. Training uses center=True (PyTorch default). Deployment uses manual FFT (no centering). center=True pads frame_size/2 zeros on each side -- this is a causality violation per the project's own causality verifier.
2. CausalEngine frame=256, Training frame=512. The Conv2D model is spatially invariant so it can handle different freq-bin counts, but the spectral resolution differs.
3. CausalEngine OLA does NOT apply window normalization. Hann^2 with 50% overlap sums to 0.75 -- output is attenuated.

### Action Required:
- center=False MUST be added to train.py (per master handoff, non-negotiable)
- frame_size/hop_size mismatch is a design decision requiring user approval
- OLA normalization inconsistency must be resolved canonically

### Action Taken:
Applying center=False to train.py. NOT changing frame_size/hop_size -- that is a design decision.

---

## PH0.7 GATE 3 -- METRIC PROVENANCE

**STATUS: PASS**

```
PESQ_AVAILABLE: True   (pesq 0.0.4)
PYSTOI_AVAILABLE: True (pystoi 0.4.1)
Python: 3.13.5
NumPy: 2.3.4
Torch: 2.9.0
```

Real PESQ and STOI are available. Official evaluation is NOT blocked.
Historical PESQ=1.50 / STOI=0.896: REQUIRES REVALIDATION.

---

## PH0.7 GATE 4 -- TEST SUITE

**STATUS: FAIL -- test_08b exceeds 8 ms budget even in isolation**

### Full suite run (77 other tests concurrent):
```
77 passed, 1 failed in 151.67s
test_08b: P95 = 9.37 ms > 8.00 ms
```

### Isolated run (test_08 only, no other tests):
```
1 failed in 37.66s
test_08b: P95 = 8.93 ms > 8.00 ms
```

### Analysis:
- Isolation reduced P95 from 9.37ms to 8.93ms (0.44ms improvement)
- But 8.93ms still exceeds the 8.00ms budget by 0.93ms
- This is NOT contention flakiness -- it is a genuine budget exceedance
- PH0.6 reported P95 = 6.61ms on the same machine -- the difference may be due to system state, background processes, or library version changes
- Test 8a (streaming mechanism correctness) PASSES in both runs

### Per master handoff:
- The 8ms assertion is NOT weakened
- The failure is documented honestly
- This does NOT block training (test 8 is a latency benchmark, not a correctness test)
- The latency issue is separate from the training pipeline validation

### Remaining risk:
- Laptop latency is marginal -- P95 is within 1ms of the budget
- Real-time compliance on Pi 4 remains untested

---

## PH3 Resolution of Test 8 (September 8, 2026):
- Root cause diagnosed: `CausalStreamingEngine` had defaulted to `TinyEnhancerWrapper` (uncompiled PyTorch eager CPU execution, taking ~4.9 ms AI inference alone).
- Fix implemented: Added `TinyEnhancerONNXWrapper` (ONNX Runtime CPUExecutionProvider) to `src/ai/tiny_enhancer.py` and configured `CausalStreamingEngine` to use it when `models/E2_causal.onnx` is present.
- Result: Test 8 now achieves **P50 = 2.49 ms / P95 = 3.83 ms <= 8.00 ms** (PASS).
- Full regression suite status: **91/91 tests pass (100% clean)**.
- Boundary: Real-time embedded compliance on physical Raspberry Pi 4 remains unmeasured and pending physical bring-up (PH3.1).

---

## PH3.5 Long-Duration Stress & Numerical Validation (September 8, 2026):
- Status: **FROZEN & ACCEPTED**.
- 12 adversarial streaming suites executed with 100% pass rate:
  - 60-second continuous streaming: P50 = 3.07 ms, P95 = 3.74 ms, P99 = 3.91 ms, Max = 6.63 ms <= 8.00 ms budget.
  - Zero non-finite propagation, reset determinism bit-exact, silence stability verified, overdrive and reference delay skew handled.
- Regression suite: **96/96 tests pass**.
- Software expansion frozen. Dual-system architectural distinction frozen:
  - System A (Verified): Primary -> VSS-NLMS -> E2 Causal ONNX -> Comm Speech.
  - System B (Proposed): Reference -> FxNLMS -> Speaker -> Ear Cavity -> Error Mic -> Adaptation.
  - Two-Rate Supervisory Coupling (Proposed): AI frame governor (8.0 ms) -> physical ANC loop.

---

## PH4-SIM Simulated Physical ANC & Secondary-Path Validation (September 9, 2026):
- Status: **COMPLETE & VALIDATED [PASS]**.
- Attribution: **SIMULATED PHYSICAL CONTROL EXPERIMENT (Host CPU)**.
- Electro-acoustic plant models implemented:
  - Primary path P(z): 8-sample acoustic propagation delay + 4th-order low-pass transmission loss (fc = 800 Hz).
  - Secondary path S(z): 3-sample DAC/ADC delay + speaker diaphragm electro-mechanics + ear-cup cavity resonance (f0 = 300 Hz, Q = 1.6).
  - Perturbed plant S_leak(z): Glasses frame cushion break with bass compliance loss below 400 Hz.
  - Parametric uncertainty matrix: L2 modeling error across 0%, 5%, 10%, 20%, 30%, 50%.
- 10 Core Control Investigations Answered:
  1. Convergence: Converges in 12.5 ms; achieves +93 dB on pure sinusoids (100-500 Hz).
  2. Secondary-Path Phase: Standard NLMS fails (-15.10 dB amplification); FxNLMS achieves +73.98 dB across all phase angles.
  3. Plant Uncertainty: Robust up to 50% magnitude mismatch (+18.39 dB cancellation) without divergence.
  4. Seal Perturbation: Cushion leak collapses cancellation from +19.88 dB to -5.66 dB without divergence, proving necessity of supervisory detection.
  5. Stability Boundary: Maximum stable step size mu_max = 0.10; divergence occurs within < 3 ms at mu >= 0.50.
  6. Noise Profiles: Stable across tonal (+30.04 dB), pink (+14.82 dB), engine (+14.71 dB), rotor (+19.54 dB), and gunfire (+15.47 dB).
  7. Control Bandwidth: Core band (50-700 Hz) achieves deep active cancellation; steep roll-off above 800 Hz (+13.07 dB at 1000 Hz).
  8. Sensor SNR: Cancellation strictly bounded by sensor noise floor (Delta SPL_max approx SNR_sensor).
  9. Supervisory Coupling: Two-rate governor freezes adaptation (50 hops frozen) during user speech, protecting voice preservation.
  10. Transport Delay: Physical cancellation requires total latency <= 0.5 ms (<= 8 samples); latency >= 1.5 ms collapses into noise amplification.
- Total Regression Suite: **100/100 tests pass (100% clean)**.
- Evidence Boundary: Strictly designates control simulation on host CPU; zero false physical ear-cup acoustic claims.

---

## PH4.1 Simulation Realism & Robustness (September 9, 2026):
- Status: **FROZEN AS SIMULATION MILESTONE**.
- Verdict Classification:
  - Software regression: 🟢 Strong (104/104 tests pass)
  - Simulation robustness: 🟢 Complete
  - Coherence experiment: 🟡 Good (approaches upper bound under adopted model; reference coherence is a dominant physical limiter)
  - Plant uncertainty: 🟡 Multi-dimensional (phase/delay much more critical than amplitude)
  - Actuator constraints: 🟢 Modeled (clipping degrades cancellation and generates harmonic distortion/THD)
  - 94 dB interpretation: 🟢 Correctly reframed as idealized mathematical simulation ceiling
  - Imperfect environment range: 🟡 Qualified (cancellation ranged from 2.02 to 10.12 dB across evaluated disturbances; sensitivity evidence, not physical headset prediction)
  - Physical ANC evidence: 🔴 None (frozen; transitioning to physical plant)
- Strategic Architecture Decision:
  - Simulation rabbit hole permanently closed. No further simulation expansions unless physical experiment uncovers a specific question.
  - Strict system boundary preserved:
    - **System A (Verified):** Dual-mic speech enhancement (`Primary + Reference -> VSS-NLMS -> E2 Causal ONNX -> Enhanced Speech`).
    - **System B (Physical ANC):** `Reference -> FxNLMS -> Speaker -> Ear Cavity -> Error Mic -> FxNLMS adaptation`.
    - E2 model acts strictly as an AI supervisor for parameter adaptation; it is **not** the acoustic controller itself.

---

## PH5 Physical Acoustic Loop Bring-Up: Stages PH5.1 – PH5.6 (September 9, 2026):
- Status: **PH5.1 PASSED | PH5.2 FAILED (CURRENT PATH) | PH5.3 MEASURED (INTERPRETATION CORRECTED) | PH5.4 FAILED (BUFFER JITTER PROVEN) | PH5.5 BASIC BASELINE ONLY | PH5.6 FAILED (ACTUATOR THD 196.8%) | PH5.7–PH5.10 STRICTLY BLOCKED**.
- Master Project Status Classification:
  ```text
  PH0–PH3       Digital/AI Pipeline                  🟢 ACCEPTED
  PH4.1         Robust Simulation                    🟢 FROZEN AS SIMULATION MILESTONE
  PH5.1         Physical Audio I/O                   🟢 PASSED (Hardware streaming active)
  PH5.2         Physical Transport Latency           🔴 FAILED (174 ms host path unsuitable for ANC)
  PH5.3         Plant Identification                 🟡 MEASUREMENT OBTAINED (Interpretation corrected)
  PH5.4         Repeatability                        🔴 FAILED (Buffer jitter proven via alignment study)
  PH5.5         Passive Baseline                     🟡 PROVISIONAL (Requires outside vs inside earcup ratio)
  PH5.6         Open-Loop Anti-Noise                 🔴 FAILED (Actuator THD = 196.8% at 200 Hz)
  PH5.7         Physical FxNLMS Adaptation           ⛔ STRICTLY BLOCKED
  PH5.8         Broadband ANC Sweep                  ⛔ STRICTLY BLOCKED
  PH5.9         Physical Validation Campaign         ⛔ STRICTLY BLOCKED
  PH5.10        E2 AI Supervisor on Physical Loop    ⛔ STRICTLY BLOCKED
  ```
- Execution Modality: **Physical Hardware Audio Loop** (Realtek Audio Codec on Host Platform).
- Hardware Audio I/O:
  - Active Input Device: `Microphone Array (Realtek(R) Audio)`
  - Active Output Device: `Speaker / Headphone (Realtek(R) Audio)`
  - Test Harness: `src/hardware/duplex_audio.py`, `src/hardware/latency_loopback.py`, `src/hardware/secondary_path_measurer.py`, `src/hardware/repeatability_verifier.py`, `src/hardware/open_loop_tester.py`
  - Automated Runner: `scripts/run_ph5_physical_bringup.py`
  - Recovery Protocol Document: `results/ph5_physical/PH5_RECOVERY_PLAN.md`
- Hard Gate Measurement Results:
  | Gate Step | Description | Target / Requirement | Measured Result | Status |
  | :--- | :--- | :--- | :--- | :--- |
  | **PH5.1** | Hardware Audio I/O | Duplex DAC/ADC stream open | Full-duplex active on host soundcard | 🟢 **PASS** |
  | **PH5.2** | Physical Transport Latency | Target: sub-20 ms (< 5 ms loop) | **174.32 ms P50** (MME) / **91.06 ms P50** (WASAPI) — Host OS path rejected for ANC | 🔴 **CURRENT PATH REJECTED** |
  | **PH5.3** | Secondary Path $S(z)$ Identification | Farina swept-sine deconvolution | $\hat{s}(t)$ extracted (128 taps), IR SNR: **30.15 dB**. End-to-end transport + cavity delay: **173.62 ms** | 🟡 **PRELIMINARY MEASURED** |
  | **PH5.4** | $S(z)$ Repeatability | Aligned $R \ge 0.95$, $\Delta G \le 0.5$ dB | Raw: Mean $R = 0.3154$. **Aligned: Mean $R = 0.9244$, Min $R = 0.8525$** (Apparatus rejected) | 🔴 **APPARATUS REJECTED** |
  | **PH5.5** | Passive Acoustic Baseline | External vs Internal earcup attenuation | Background RMS: $0.005472$ (provisional); full $A_{\text{passive}}(f)$ required with external noise | 🟡 **BASELINE INCOMPLETE** |
  | **PH5.6** | Open-Loop Anti-Noise Test | Transducer pickup, polarity, THD | SNR: **+19.80 dB**, Polarity: **FAIL**, THD: **196.8%** (Laptop actuator rejected) | 🔴 **ACTUATOR REJECTED** |
- Critical Scientific Insights & Hard Gate Audit:
  1. **Strict Terminology Correction for 173.62 ms Delay:**
     - 173.62 ms is **NOT** the physical acoustic secondary path delay inside an ear cavity (which is $\approx 0.087\text{ ms}$).
     - It is formally designated as: **"Measured end-to-end audio transport + transducer/cavity path delay under the current host audio configuration."**
     - This empirically confirms that host PC audio is fundamentally unsuitable as the physical ANC control path, conclusively justifying the dedicated embedded I²S architecture.
  2. **Repeatability Separation & Empirical Delay-Alignment Study (PH5.4):**
     - Measured relative peak arrival offsets across trials: `[0, -10, -10, -16, -18]` samples at 16 kHz.
     - **Explicit Alignment Proof:** Re-extracting FIRs centered on actual detected peak offsets recovered mean pairwise correlation to **$R_{\text{aligned}} = 0.9244$ (min $R_{\text{aligned}} = 0.8525$)**, with zero negative correlations.
     - **Metric Separation:** We formally separate **plant repeatability** (Target: $R_{\text{aligned}} \ge 0.95$) from **transport determinism** ($R_{\text{raw}}$ and $\sigma_t$). Raw $R \ge 0.95$ is not a universal hard requirement for the acoustic plant.
     - **Scientific Boundary:** Timing offset explains a large portion of the poor raw correlation, but not all of the remaining gap to 0.95 (which includes transducer thermal settling and acoustic coupling).
     - Because $R_{\text{aligned}} = 0.9244 < 0.95$ and the actuator failed independently, FxNLMS remains strictly **BLOCKED**.
  3. **Actuator Replacement Mandated (PH5.6 / PH5-R1):**
     - Laptop speaker produced **THD = 196.8%** at 200 Hz. Officially designated as a **diagnostic fixture, not an ANC actuator**.
     - Actuator Specification: **40 mm 32Ω high-excursion dynamic headphone driver** (linear at 100–700 Hz, $\text{THD} \le 0.5\%$).
  4. **Integrated Single-Codec Audio Architecture (PH5-R2):**
     - Split DAC/ADC modules (PCM5102A + INMP441) are explicitly rejected due to asynchronous clock domains.
     - Selected: **Waveshare WM8960 Audio Codec** on Raspberry Pi 4B (unified 12.288 MHz PLL, shared MCLK/BCLK/LRCLK, simultaneous stereo sampling).
     - Exact channel mapping: `In-L = Reference Mic`, `In-R = Error Mic`, `Out-L = Anti-Noise Actuator`.
  5. **Three-Experiment Recovery Milestone Gate:**
     FxNLMS (PH5.7) will be unlocked only after the new apparatus passes:
     - **Exp A (Electrical Loopback):** $t_{\text{ADC/DAC}} < 5\text{ ms}$, $\sigma_t \le 0.05\text{ ms}$ ($0.8\text{ sample at } 16\text{ kHz}$) across 20 trials.
     - **Exp B (Acoustic Impulse Response):** Recover causal $S(z)$ with $\tau_{\text{acoustic}} < 0.5\text{ ms}$.
     - **Exp C (20-Trial Repeatability):** Measure $\mu_t, \sigma_t \le 0.05\text{ ms}$ ($0.8\text{ sample at } 16\text{ kHz}$), $R_{\text{raw}}, R_{\text{aligned}} \ge 0.95, \Delta G \le 0.5\text{ dB}, \Delta\phi \le 5^\circ$.
  6. **Pre-Registered Architectural Specification:**
     The long-term intelligence architecture (PH6 State Estimator $\to$ PH7 Supervisor $\to$ PH8 Residual Chain $\to$ PH9 MoE) is formally pre-registered in [`docs/ARCHITECTURE_HYPOTHESIS.md`](file:///f:/SIH%202026/docs/ARCHITECTURE_HYPOTHESIS.md), including state vector $\mathbf{z}_t$, 5-mode state machine, pre-registered hypotheses H1–H4, and explicit MoE kill criteria. Code modifications for PH6–PH9 remain frozen until PH5 passes.
