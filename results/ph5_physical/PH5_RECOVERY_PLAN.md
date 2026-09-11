# PH5 Recovery Plan: Physical Acoustic Plant & Deterministic Audio Path
**Document Version:** 2.0 (September 9, 2026)  
**Status:** AUTHORITATIVE HARDWARE RECOVERY SPECIFICATION  
**Hard Gate Rule:** FxNLMS Adaptation (PH5.7) and AI Supervisory Coupling (PH5.10) remain strictly **BLOCKED** until all criteria below are experimentally satisfied.

---

## 1. Executive Summary & Hard Gate Diagnostic Findings

During the initial physical hardware bring-up (PH5.1–PH5.6), the Hard Gate successfully prevented premature closed-loop adaptation by exposing two fatal hardware apparatus constraints:

1. **Measured End-to-End Audio Transport Delay: 173.6–174.3 ms**
   - **Diagnosis:** This latency is **not** the acoustic secondary-path propagation delay inside an ear cavity (which is physical distance $/ c \approx 0.03\text{ m} / 343\text{ m/s} \approx 0.087\text{ ms}$). It is the **end-to-end audio transport + driver/OS buffer pipeline delay** of the host Windows/MME/WASAPI software stack.
   - **Implication:** The host Windows PC audio path is definitively unsuitable as the real-time ANC control path. The 174 ms measurement serves as empirical justification for the dedicated embedded I²S architecture.
   
2. **Inter-Run Waveform Repeatability & Timing Jitter (PH5.4):**
   - **Empirical Measurement:** Raw pairwise correlation yielded mean $R = 0.3154$ and min $R = -0.6849$.
   - **Causal Verification via Sample Alignment:** Relative peak arrival offsets across trials were measured as `[0, -10, -10, -16, -18]` samples at 16 kHz. Re-extracting impulse responses centered on the actual detected peak for each trial recovered mean correlation to **$R_{\text{aligned}} = 0.9244$ (min $R_{\text{aligned}} = 0.8525$)**, eliminating all negative correlations.
   - **Scientific Boundary:** Timing offset explains a large portion of the poor raw correlation. However, it does not explain 100% of the remaining gap between 0.9244 and 0.95, which also incorporates secondary acoustic reflections, micro-transducer thermal settling, and acoustic seal coupling.
   - **Metric Separation:** We strictly separate **plant repeatability** ($R_{\text{aligned}} \ge 0.95$) from **transport determinism** ($R_{\text{raw}}$ and $\sigma_t$). Raw $R \ge 0.95$ is not a universal hard requirement for the acoustic plant.

3. **Actuator Non-Linearity & THD Failure (PH5.6):**
   - **Empirical Measurement:** Driving a 200 Hz pure tone on the internal laptop speaker produced **THD = 196.8%**, with heavy harmonic distortion spilling into 1000, 1600, 1800, and 2000 Hz.
   - **Diagnosis:** Miniature laptop micro-speakers have an acoustic high-pass cutoff and chassis resonance around 800–1000 Hz. They cannot linearly execute large cone excursions at 200 Hz.
   - **Directive:** The laptop speaker is officially designated as a **diagnostic fixture, not an ANC actuator**. No algorithm tuning (changing $\mu$, filter length, or retraining E2) will be conducted on this actuator.

---

## 2. Master Project Status Classification

```text
PH0–PH3       Digital/AI Pipeline                  🟢 ACCEPTED
PH4.1         Robust Simulation                    🟢 FROZEN AS SIMULATION MILESTONE
PH5.1         Physical Audio I/O                   🟢 PASSED (Hardware streaming active)
PH5.2         Physical Transport Latency           🔴 FAILED (174 ms host path rejected)
PH5.3         Plant Identification                 🟡 MEASUREMENT OBTAINED (Interpretation corrected)
PH5.4         Repeatability                        🔴 FAILED (Current apparatus rejected; buffer jitter proven)
PH5.5         Passive Baseline                     🟡 PROVISIONAL (Requires outside vs inside earcup ratio)
PH5.6         Open-Loop Anti-Noise                 🔴 FAILED (Laptop actuator rejected; THD = 196.8%)

PH5-R1        Actuator Replacement                 🔄 IN PROGRESS (40mm 32Ω headphone driver specified)
PH5-R2        Deterministic I²S Path               🔄 IN PROGRESS (Single-codec WM8960 topology specified)

PH5.7         Physical FxNLMS Adaptation           ⛔ STRICTLY BLOCKED
PH5.8         Broadband ANC Sweep                  ⛔ STRICTLY BLOCKED
PH5.9         Physical Validation Campaign         ⛔ STRICTLY BLOCKED
PH5.10        E2 AI Supervisor on Physical Loop    ⛔ STRICTLY BLOCKED
```

---

## 3. Hardware Architecture & Channel Wiring Specification (PH5-R1 & PH5-R2)

### 3.1 Architectural Selection: Single-Codec Synchronous Clock Domain (WM8960)
Split ADC/DAC architectures (e.g. PCM5102A DAC + INMP441 MEMS microphone) are **explicitly rejected** because independent breakout boards lack a unified master clock (MCLK), introducing inter-channel sample drift and phase skew.

We standardize on the **Waveshare WM8960 Stereo Audio HAT** mounted directly onto the Raspberry Pi 4B GPIO header:
- **Locked Clock Domain:** Single on-board PLL generates 12.288 MHz MCLK. Both ADC channels and both DAC channels run on the exact same BCLK and LRCLK crystal domain.
- **Simultaneous Stereo Sampling:** Reference and error channels are sampled on the exact same clock edge ($0.00\text{ }\mu\text{s}$ inter-channel skew).
- **Integrated Driver:** High-fidelity headphone amplifier ($40\text{ mW}$ into $32\,\Omega$, $\text{THD} < 0.05\%$).

### 3.2 Exact Channel Allocation Across Systems
| System | Physical Port | Pin / Terminal | Channel Function | Acoustic Location |
| :--- | :--- | :--- | :--- | :--- |
| **System B (Physical ANC)** | **WM8960 In-L** | `LINPUT1 / LINPUT2` | **Reference Microphone** | External ear-cup shell facing noise field |
| **System B (Physical ANC)** | **WM8960 In-R** | `RINPUT1 / RINPUT2` | **Error Microphone** | Inside ear cavity 5 mm from ear canal |
| **System B (Physical ANC)** | **WM8960 Out-L**| `HP_L / SPK_L` | **Anti-Noise Actuator** | 40 mm 32Ω driver inside ear-cup |
| **System B (Physical ANC)** | **WM8960 Out-R**| `HP_R / SPK_R` | Reserved / Monitor | Diagnostic / line monitor |
| **System A (Comms Speech)** | Dedicated Mic Port | USB Audio / Boom Port | Primary Voice Mic | Close-talk boom mic near mouth |

---

## 4. The Three Decisive Recovery Experiments (PH5.7-A)

Before unlocking FxNLMS, execute only these three targeted experiments on the new apparatus:

### Experiment A: Pure Electrical Loopback ($t_{\text{ADC/DAC}}$)
```text
WM8960 DAC Out ──(Direct 3.5mm Shielded Cable)──► WM8960 ADC In
```
- **Excitation:** Calibrated 50 ms linear chirp ($300\text{ Hz} \to 3000\text{ Hz}$).
- **Measurements across 20 trials:**
  - Round-trip electrical latency: $\mu_t$ (Target: $< 5\text{ ms}$).
  - Latency jitter: $\sigma_t$ (Target: $\le 0.05\text{ ms}$ / $0.8\text{ sample at } 16\text{ kHz}$).
  - Peak cross-correlation: $R_{\text{electrical}} \ge 0.99$.

### Experiment B: Acoustic Impulse Response & Transfer Function $S(z)$
```text
WM8960 DAC Out ──► 40mm Driver ──► Ear Cavity ──► Error Mic ──► WM8960 ADC In
```
- **Excitation:** Farina logarithmic swept-sine ($50\text{ Hz} \to 1000\text{ Hz}$, $1.0\text{ s}$ duration) at nominal drive level ($-6\text{ dBFS}$).
- **Analysis:**
  - Deconvolve using time-reversed inverse sweep.
  - Separate linear impulse response $\hat{s}(t)$ from harmonic distortion.
  - Extract 128-tap causal FIR model $\hat{S}(z)$.
  - Measure true electro-acoustic latency: $\tau_{\text{acoustic}} = t_{\text{total}} - t_{\text{ADC/DAC}}$ (Target: $< 0.5\text{ ms}$).
  - Document magnitude $|S(f)|$ and unwrapped phase $\angle S(f)$ across $100\text{–}700\text{ Hz}$.

### Experiment C: 20-Trial Repeatability & Stability Campaign
Execute the exact same Farina swept-sine stimulus 20 consecutive times without altering the acoustic seal or volume.

Record and evaluate the 6 mandatory metrics:
| Metric | Description | Target Requirement | Purpose |
| :--- | :--- | :--- | :--- |
| **$\mu_t$** | Mean acoustic arrival time | Consistent with Exp B ($\pm 0.1\text{ ms}$) | Baseline transport latency |
| **$\sigma_t$** | Delay standard deviation | $\le 0.05\text{ ms}$ ($0.8\text{ sample at } 16\text{ kHz}$) | Transport timing determinism |
| **$R_{\text{raw}}$** | Raw unaligned correlation | Diagnostic report | End-to-end clock locking |
| **$R_{\text{aligned}}$** | Delay-aligned correlation | **$\ge 0.95$ (Authoritative Hard Gate)** | Physical acoustic plant repeatability |
| **$\Delta G$** | Inter-run magnitude variation | $\le 0.5\text{ dB}$ across $100\text{–}700\text{ Hz}$ | Transducer & coupling stability |
| **$\Delta\phi$** | Inter-run phase variation | $\le 5.0^\circ$ across $100\text{–}700\text{ Hz}$ | Secondary-path phase stability |

---

## 5. Gate Exit Decision

FxNLMS closed-loop adaptation (PH5.7) will be unlocked **if and only if**:
1. Electrical loopback latency $\mu_t \le 10\text{ ms}$ with $\sigma_t \le 0.05\text{ ms}$.
2. Plant repeatability achieves **$R_{\text{aligned}} \ge 0.95$** across 20 trials.
3. Actuator THD at 200 Hz is $\le 2.0\%$ at nominal drive level.
4. Magnitude variation $\Delta G \le 0.5\text{ dB}$ and phase variation $\Delta\phi \le 5^\circ$.
