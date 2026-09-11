# Architectural Hypothesis: Hierarchical Noise-Aware Hybrid ANC & Supervisory Controller
**Document Version:** 1.0 (September 9, 2026)  
**Authoritative Status:** PRE-REGISTERED RESEARCH & ENGINEERING SPECIFICATION  
**Target Phases:** PH6 (State Estimator) $\to$ PH7 (Supervisor) $\to$ PH8 (Residual Chain) $\to$ PH9 (MoE Investigation)  
**Prerequisite Hard Gate:** PH5 (Physical Actuator & Deterministic I²S Audio Loop) must pass before code execution of PH6–PH9.

---

## 1. Executive Summary & Core Architectural Separation

This specification defines the pre-registered architecture for integrating acoustic intelligence with adaptive active noise control. Rather than attempting to train a monolithic neural network to directly generate physical anti-noise, or using a brittle classifier that hard-switches adaptive filters, the system decouples into **two distinct physical jobs coordinated by an acoustic state supervisor**:

```text
                         ┌──────────────────────┐
                         │      AUDIO INPUT     │
                         └──────────┬───────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  │                                   │
                  ▼                                   ▼
           🎤 REFERENCE MIC                     🎤 PRIMARY MIC
           (Acoustic Shell)                     (Near-Mouth / Headset)
                  │                                   │
                  │                                   │
                  └──────────────┬────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │  NOISE/STATE ESTIMATOR  │
                    │                         │
                    │  Acoustic State Vector  │
                    │          z_t            │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   SUPERVISORY GOVERNOR   │
                    │                         │
                    │ Hysteresis (N frames)   │
                    │ Parameter Interpolation │
                    │ Confidence Fallback     │
                    │ Impulsive Gate/Freeze   │
                    └────────────┬────────────┘
                                 │
                    Operating Parameters Only (μ, λ, L, freeze)
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │        FxNLMS           │
                    │    FAST PHYSICAL ANC    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                         🔊 ANC SPEAKER
                         (40mm 32Ω Driver)
                                 │
                                 ▼
                         ┌─────────────┐
                         │  EAR CAVITY │
                         └──────┬──────┘
                                │
                                ▼
                         🎤 ERROR MIC
                                │
                                └──────────► FxNLMS Adaptation Loop


       PRIMARY SIGNAL ───────────────────────────────────┐
                                                          ▼
                                                 ┌────────────────┐
                                                 │  E2_causal AI  │
                                                 │  Residual SE   │
                                                 └───────┬────────┘
                                                         │
                                                         ▼
                                                   🗣️ CLEAN SPEECH
```

### The Two Non-Negotiable Core Principles
1. **Job A (Physical Acoustic ANC) $\ne$ Job B (Digital Speech Enhancement):**
   - **Job A:** Cancels acoustic sound pressure inside the physical ear-cup cavity via sample-by-sample FxNLMS (`Ref Mic -> FxNLMS -> Speaker -> Cavity -> Error Mic`).
   - **Job B:** Preserves intelligible communication speech from the primary microphone by applying neural spectral masking to the residual (`Primary Mic -> VSS-NLMS -> E2 Causal ONNX -> Enhanced Speech`).
2. **Supervision Routes Parameters, Never Waveforms:**
   - The physical control path has **exactly one physical FxNLMS controller**. Multiple parallel controllers adapting simultaneously against the same error microphone induce cross-filter competition and acoustic instability.
   - The supervisor interpolates **controller operating parameters** ($\mu, \lambda, L$, freeze fraction), **never anti-noise waveforms**.
   - Soft Mixture-of-Experts (MoE) weighting ($Y = \sum w_i Y_i$) is restricted strictly to perceptual AI spectral masks (PH9), never physical anti-noise.

---

## 2. Dual-Channel Acoustic State Vector ($z_t$)

The classifier does not emit discrete equipment labels. It continuously estimates the **acoustic state of the environment** at frame cadence (8.0 ms hop) as a 9-dimensional state vector:

$$\mathbf{z}_t = \big[ p_T,\, p_M,\, p_B,\, p_I,\, p_U,\, p_S,\, C,\, E,\, I \big]^T$$

Where:
| Element | Variable | Mathematical Range | Channel Provenance | Description |
| :---: | :---: | :---: | :---: | :--- |
| 1 | $p_T$ | $[0.0, 1.0]$ | Reference Channel | Probability of low-frequency tonal / quasi-stationary noise |
| 2 | $p_M$ | $[0.0, 1.0]$ | Reference Channel | Probability of modulated / periodic non-stationary noise (rotor/RPM) |
| 3 | $p_B$ | $[0.0, 1.0]$ | Reference Channel | Probability of broadband stochastic / diffuse noise (wind, roar) |
| 4 | $p_I$ | $[0.0, 1.0]$ | Primary & Reference | Probability of impulsive / ballistic transient event |
| 5 | $p_U$ | $[0.0, 1.0]$ | Reference Channel | Probability of unknown / out-of-distribution acoustic condition |
| 6 | $p_S$ | $[0.0, 1.0]$ | Primary Channel | Probability of active near-end voice speech |
| 7 | $C$ | $[0.0, 1.0]$ | Cross-Channel | Magnitude-squared coherence $C_{xy}(f)$ across $100\text{–}1000\text{ Hz}$ |
| 8 | $E$ | $\mathbb{R}_{\ge 0}$ (dBFS) | Reference Channel | Estimated disturbance power / RMS level |
| 9 | $I$ | $\mathbb{R}_{\ge 0}$ | Primary Channel | Temporal impulsiveness measure (kurtosis / crest factor) |

### Probability Simplex Constraint
The noise regime probabilities form a proper probability distribution:
$$\sum_{k \in \{T, M, B, I, U\}} p_k = 1.0$$
While speech probability $p_S$ is orthogonal (speech may co-occur with any noise regime).

---

## 3. Physical Acoustic Regime Taxonomy vs Equipment Labels

To avoid overfitting to synthetic audio dataset names (e.g. "T-90", "BMP-2", "Dhruv", "INSAS"), the taxonomy classifies **physical acoustic behaviors**:

| Regime | Physical Mechanism | Acoustic Hallmarks | Adaptive Challenge | Controller Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Tonal ($T$)** | Resonant cavities, rotational engine firing, 50/60 Hz mains | Sharp spectral peaks, high harmonicity, low spectral flatness ($< 0.15$) | High eigenvalue spread, slow convergence | Nominal $\mu$, standard FIR length ($L=64$), minimal leakage |
| **Modulated ($M$)** | Rotor blade vortex interaction, engine acceleration / RPM slew | Time-varying fundamental ($f_0(t)$), periodic AM/FM sidebands | Plant and disturbance non-stationarity | Fast tracking $\mu_{\text{max}}$, short filter length ($L=32$), normalized energy tracking |
| **Broadband ($B$)** | Turbulent air flow, exhaust hiss, diffuse tracked-vehicle roar | High spectral flatness ($> 0.50$), low autocorrelation | Lower theoretical cancellation ceiling | Moderate $\mu$, higher leakage $\lambda$ to prevent weight drift |
| **Impulsive ($I$)** | Gunshot muzzle blast, artillery impact, hatch slam | Heavy-tailed distribution (kurtosis $\kappa > 6.0$), sub-5 ms rise time | Gradient explosion, weight divergence in standard FxLMS | **Instant adaptation freeze** ($\mu \to 0$), Huber M-estimate clipping, analog diode protection |
| **Unknown ($U$)** | Sensor detachment, acoustic seal break, novel jammer | Low classifier confidence ($\max p_k < \tau_{\text{conf}}$) | High risk of destructive anti-noise amplification | **Safe conservative fallback:** low $\mu$, moderate leakage, neutral residual E2 |

---

## 4. Supervisory Governor & 5-Mode State Machine

The supervisor maps the state vector $\mathbf{z}_t$ into controller operating modes:

```text
                  ┌──────────────────────────────────────────────┐
                  │                 STATE VECTOR                 │
                  │ z_t = [p_T, p_M, p_B, p_I, p_U, p_S, C, E, I]│
                  └──────────────────────┬───────────────────────┘
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
                 [p_I > τ_impulse]              [p_S > τ_speech]
                         │                               │
                 (Yes)   ▼                       (Yes)   ▼
                    ┌─────────┐                     ┌─────────┐
                    │ MODE 4  │                     │ MODE 1  │
                    │ IMPULSE │                     │ QUIET / │
                    │ FREEZE  │                     │ SPEECH  │
                    └─────────┘                     └─────────┘
                         │ (No)                          │ (No)
                         └───────────────┬───────────────┘
                                         │
                                         ▼
                               [max(p_T,p_M,p_B) < τ_conf]
                                         │
                                 (Yes)   ▼
                                    ┌─────────┐
                                    │ MODE 5  │
                                    │ UNKNOWN │
                                    │FALLBACK │
                                    └─────────┘
                                         │ (No)
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
                 [p_T >= p_M]                    [p_M > p_T]
                         │                               │
                         ▼                               ▼
                    ┌─────────┐                     ┌─────────┐
                    │ MODE 2  │                     │ MODE 3  │
                    │ TONAL / │                     │MODULATED│
                    │STATION- │                     │ NON-    │
                    │  ARY    │                     │STATION- │
                    └─────────┘                     │  ARY    │
                                                    └─────────┘
```

### 4.1 Operating Parameter Matrix
| Mode | State Trigger | Step Size $\mu$ | Leakage $\lambda$ | Filter Length $L$ | Error Mic Weight | Speech Protection |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mode 1: QUIET / SPEECH** | $p_S \ge 0.70$ or $E < -45\text{ dBFS}$ | $0.05 \times \mu_0$ (very conservative) | $1.0 \times 10^{-4}$ | 64 taps | Normal | High priority (suppression attenuated) |
| **Mode 2: TONAL / STATIONARY** | $p_T \ge \max(p_M, p_B)$ and $p_S < 0.70$ | $1.00 \times \mu_0$ (nominal) | $1.0 \times 10^{-5}$ | 128 taps | Normal | Neutral |
| **Mode 3: MODULATED / DYNAMIC** | $p_M > p_T$ and $p_S < 0.70$ | $2.50 \times \mu_0$ (fast tracking) | $5.0 \times 10^{-5}$ | 64 taps | Elevated | Neutral |
| **Mode 4: IMPULSIVE PROTECTION** | $p_I \ge 0.50$ or $\kappa > 6.0$ | **$0.00$ (Instant Freeze)** | $0.0$ | Frozen | Diode clipped | Transient clamp (limiter engaged) |
| **Mode 5: UNKNOWN FALLBACK** | $\max(p_T, p_M, p_B) < 0.50$ or $C < 0.40$ | $0.20 \times \mu_0$ (safe fallback) | $2.0 \times 10^{-4}$ | 64 taps | Normal | Generalist E2 fallback |

### 4.2 Hysteresis & Transition Smoothing
To prevent mode chatter when operating on regime boundaries:
1. **$N$-Frame Dominance Hysteresis:** A proposed new mode must maintain strict probability dominance for at least $N = 4\text{ consecutive frames}$ ($32\text{ ms}$) before a state transition is acknowledged. Exception: **Mode 4 (Impulse)** triggers instantly with $N=1$ frame.
2. **Convex Parameter Blending:** When transitioning between active regimes (e.g. Mode 2 $\to$ Mode 3), operating parameters blend smoothly over $M = 8\text{ frames}$ using a linear ramp:
   $$\mu(n) = (1 - \alpha_n)\mu_{\text{old}} + \alpha_n \mu_{\text{new}}, \quad \alpha_n = \frac{n}{M}$$
   $$\lambda(n) = (1 - \alpha_n)\lambda_{\text{old}} + \alpha_n \lambda_{\text{new}}$$

---

## 5. Pre-Registered Hypotheses (H1 – H4)

These four hypotheses must be evaluated using controlled, non-cherry-picked test sets before the architecture can be considered validated:

### Hypothesis H1: Regime-Aware Supervisory Robustness
> **H1 Statement:** Acoustic-regime-aware supervisory control improves active acoustic cancellation stability and/or speech preservation relative to a fixed-parameter FxNLMS baseline under concatenated mixed sequences without manual tuning.

- **Baseline System (Control):** Fixed-parameter FxNLMS ($\mu = \text{const}$, $\lambda = \text{const}$) coupled with frozen baseline E2.
- **Treatment System:** FxNLMS governed by the 5-mode Supervisory Governor driven by $\mathbf{z}_t$.
- **Test Corpus:** 10 concatenated sequences (60 seconds each) transitioning across:
  $$\text{Stationary Engine (15s)} \longrightarrow \text{Impulsive Gunshot (2s)} \longrightarrow \text{Clean Speech (10s)} \longrightarrow \text{Rotor Modulated (15s)} \longrightarrow \text{Wind Buffet (18s)}$$
- **Falsification Criteria:** H1 is rejected if:
  1. Mean acoustic cancellation improvement is $< 2.0\text{ dB}$ across the full sequence, OR
  2. Clean speech attenuation exceeds $-1.5\text{ dB}$ during speech segments, OR
  3. Divergence count under impulsive transients is $> 0$.

---

### Hypothesis H2: Hysteresis & Parameter Smoothing vs Hard Switching
> **H2 Statement:** $N$-frame dominance hysteresis and parameter interpolation significantly reduce transition artifacts and spectral discontinuities compared to instantaneous argmax regime switching.

- **Comparison Conditions:**
  - **Condition A (Hard Switching):** Instantaneous argmax mode switching without hysteresis or parameter blending.
  - **Condition B (Proposed):** $N=4$ frame dominance hysteresis + $M=8$ frame parameter interpolation.
- **Evaluation Metrics:**
  - Spectral Flux Discontinuity ($\Delta\text{SF}$) concentrated at mode transition frames:
    $$\text{SF}(t) = \sum_k \big(|X_t(k)| - |X_{t-1}(k)|\big)^2$$
  - Output RMS envelope jump ratio: $\Delta\text{RMS} = 20\log_{10}(\text{RMS}_{t} / \text{RMS}_{t-1})$.
  - Objective Speech Quality (STOI / PESQ) on speech segments spanning a mode switch.
- **Falsification Criteria:** H2 is rejected if Condition B does not demonstrate at least a **$30\%$ reduction** in transition spectral flux discontinuity over Condition A.

---

### Hypothesis H3: Edge Computational Budget Compatibility
> **H3 Statement:** The combined computational cost of feature extraction, state estimation, and supervisory governance consumes less than $10\%$ of the 8.0 ms streaming hop budget on host CPU, and sustains real-time execution on Raspberry Pi 4B.

- **Target Metrics:**
  - Host CPU (Intel Ultra 7): $t_{\text{estimator}} + t_{\text{supervisor}} \le 0.80\text{ ms}$ (P95).
  - Embedded Edge (Raspberry Pi 4B @ 1.5 GHz): $t_{\text{total}} = t_{\text{DSP}} + t_{\text{estimator}} + t_{\text{supervisor}} + t_{\text{E2}} \le 6.5\text{ ms}$ (P95) ($\ge 18\%$ headroom).
- **Falsification Criteria:** H3 is rejected if the supervisor + classifier adds more than $0.80\text{ ms}$ host processing latency.

---

### Hypothesis H4: Confidence Calibration & Out-of-Distribution Safety
> **H4 Statement:** Under novel or out-of-distribution acoustic disturbances, the state estimator's unknown probability $p_U$ increases above threshold $\tau_{\text{conf}}$, reliably engaging conservative fallback (Mode 5) rather than confidently assigning an incorrect specialist.

- **Test Evaluation:**
  - In-Distribution Set: Vehicle engines, helicopter rotors, wind, gunfire (procedural training distributions).
  - Out-of-Distribution Set: Factory siren sweeps, marine sonar pings, synthetic frequency chirps, acoustic feedback squeal.
- **Falsification Criteria:** H4 is rejected if a novel/OOD disturbance produces $p_U < 0.50$ while any known regime receives $> 0.70$ probability (i.e. if the classifier fails to acknowledge uncertainty and confidently forces novel noise into an incorrect specialist).

---

## 6. Pre-Registered Kill Criteria for PH9 (Mixture of Experts)

The transition to specialist neural enhancement heads (PH9 MoE) is **not guaranteed**. It will be permanently abandoned under the following pre-registered conditions:

```text
                               PH8 EVALUATION
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
    [E2 achieves >= +9.0 dB SI-SDR          [E2 exhibits systematic,
     across all regimes & < 1.0 dB           regime-specific failures
     speech attenuation]                     on held-out TEST_B/C]
                 │                                       │
                 ▼                                       ▼
     PERMANENTLY KILL PH9 MoE                 PROCEED TO PH9 MoE
     (Retain single E2 generalist)            (Shared encoder + 5 specialist heads)
                                                         │
                                                         ▼
                                              PH9 TRAINING & AUDIT
                                                         │
                                 ┌───────────────────────┴───────────────────────┐
                                 ▼                                               ▼
                    [MoE fails to beat E2 by >= 1.0 dB              [MoE achieves >= +1.0 dB SI-SDR
                     SI-SDR, or exceeds 1.5x latency]                improvement within <= 1.25x latency]
                                 │                                               │
                                 ▼                                               ▼
                       PERMANENTLY KILL PH9                           ADOPT MoE SPECIALISTS
```

### Explicit Kill Rules
1. **Performance Ceiling Rule:** If the existing baseline `E2_causal` achieves $\ge +9.0\text{ dB}$ SI-SDR on held-out test data across all regimes with $\le 1.0\text{ dB}$ clean speech attenuation, **PH9 MoE is cancelled before training begins**.
2. **Margin Rule:** If trained, the MoE architecture must outperform `E2_causal` by at least **$+1.0\text{ dB}$ mean SI-SDR** across `TEST_B` and `TEST_C`. A delta $< 1.0\text{ dB}$ is considered marginal and triggers immediate abandonment in favor of the simpler single-model architecture.
3. **Latency Cap Rule:** MoE inference time must not exceed **$1.25\times$** baseline E2 inference latency on target hardware.

---

## 7. Multi-Phase Execution Roadmap & Prerequisite Gates

```text
================================================================================
PH5: PHYSICAL PLANT BRING-UP (CURRENT BLOCKER)
  ├── PH5-R1: 40mm 32Ω headphone driver mounted in circumaural ear-cup
  ├── PH5-R2: Waveshare WM8960 Audio HAT + Pi 4B (unified MCLK/BCLK/LRCLK domain)
  └── PH5.7-A: Exp A (Loopback < 5ms) + Exp B (S(z) τ < 0.5ms) + Exp C (Aligned R >= 0.95)
================================================================================
                                     │
                         (Passes Hard Gate Only)
                                     ▼
================================================================================
PH6: DUAL-CHANNEL NOISE STATE ESTIMATOR
  ├── Extend src/dsp/noise_regime_detector.py to dual-channel input
  ├── Output continuous state vector z_t = [p_T, p_M, p_B, p_I, p_U, p_S, C, E, I]
  └── Validate confidence calibration & unknown noise detection (H4)
================================================================================
                                     │
                                     ▼
================================================================================
PH7: 5-MODE SUPERVISORY CONTROLLER
  ├── Refactor src/dsp/supervisory_coupling.py into 5-mode state machine
  ├── Implement N=4 frame dominance hysteresis & convex parameter interpolation
  └── Validate H1 (robustness on mixed sequences) and H2 (discontinuity reduction)
================================================================================
                                     │
                                     ▼
================================================================================
PH8: RESIDUAL COUPLING & SYSTEM INTEGRATION
  ├── Execute three-way ablation: FxNLMS-only vs E2-only vs FxNLMS+E2
  └── Evaluate E2 residual performance against Pre-Registered Kill Criteria
================================================================================
                                     │
                     (Only if Kill Criteria not met)
                                     ▼
================================================================================
PH9: SPARSE MIXTURE OF EXPERTS (EXPLORATORY / CONDITIONAL)
  ├── Shared causal convolutional encoder + 5 specialist mask heads
  └── Enforce +1.0 dB margin rule vs baseline E2_causal
================================================================================
```

---

## 8. Commitments & Integrity Guardrails

1. **Code Freeze Enforced:** No source code modifications or unit test alterations relating to PH6–PH9 will take place until PH5 physical bring-up satisfies the Hard Gate.
2. **No Monolithic Filters:** The system will never combine acoustic cancellation and speech enhancement into an unstructured end-to-end black box.
3. **No Equipment Labels:** The intelligence layer is physically grounded in acoustic behavior (tonal, modulated, broadband, impulsive), never military equipment brand names.
4. **Safety Defaults to Conservative:** When in doubt ($p_U > \tau_{\text{conf}}$ or $C < 0.40$), the supervisor reduces adaptation gain rather than taking aggressive risks.
