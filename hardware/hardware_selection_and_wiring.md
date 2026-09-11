# PS 26052: Physical Hardware Selection & Wiring Architecture (PH5-R1 / PH5-R2)
**Document Version:** 1.0 (September 9, 2026)  
**Authoritative Status:** ACTIVE HARDWARE DESIGN SPECIFICATION  
**Target Milestone:** PH5-R1 (Actuator Replacement) & PH5-R2 (Synchronous I²S Audio Interface)

---

## 1. Architectural Topology: Integrated Single-Clock Codec (Option A)

### Why Split Modules Were Rejected
Using separate I²S DAC breakout boards (e.g. PCM5102A) alongside separate digital I²S MEMS microphones (e.g. INMP441):
1. **Clock Domain Asynchrony:** Even when sharing BCLK and LRCLK, independent PCB modules lack a unified low-jitter Master Clock (MCLK). Differences in internal decimation/interpolation digital filters produce inter-channel sample drift and phase jitter.
2. **Raspberry Pi I²S Constraint:** The Raspberry Pi 40-pin GPIO header provides only a single physical I²S peripheral (PCM). Splitting bidirectional full-duplex traffic across multiple non-synchronized breakout boards frequently causes ALSA buffer underflow (`XRUN`).

### Adopted Architecture: Integrated Wolfson/Cirrus WM8960 Audio Codec
We select the **Waveshare WM8960 Stereo Audio HAT** (or ReSpeaker 2-Mics Pi HAT) directly mounted onto the Raspberry Pi 4B GPIO header:
- **Unified Clock Domain:** On-board PLL generates 12.288 MHz Master Clock (MCLK). ADC and DAC operate from the **exact same crystal oscillator, same BCLK, and same LRCLK**.
- **Synchronous Stereo ADC:** Samples both input channels on the exact same clock edge (inter-channel sampling skew = 0.00 µs).
- **Integrated Transducer Drive:** On-board Class-D speaker amplifier (1W into 8Ω) and high-fidelity stereo headphone driver (40mW into 32Ω, THD < 0.05%), eliminating external breadboard amplifiers.

```text
                               WAVESHARE WM8960 AUDIO CODEC
                    ┌──────────────────────────────────────────────────┐
                    │                                                  │
Reference Mic ─────►│ In-L (Left ADC)  ──┐                             │
(Acoustic Shell)    │                    │                             │
                    │                    ├── Stereo I²S DMA Buffer ───►│───► Raspberry Pi 4
Error Mic ─────────►│ In-R (Right ADC) ──┘   (Ch0: Ref, Ch1: Err)      │     (sample-level FxNLMS)
(Ear Cavity)        │                                                  │
                    │                    ┌── Shared Master Clock ◄─────│◄─── (Synchronous PLL)
                    │                    │   (BCLK, LRCLK, MCLK)       │
Anti-Noise Actuator◄│ Out-L (HP/Amp L) ◄─┴── I²S DAC Stream ◄──────────│◄─── Anti-Noise Signal y(n)
(40mm Ear Driver)   │                                                  │
                    └──────────────────────────────────────────────────┘
```

---

## 2. Transducer & Actuator Selection (PH5-R1)

### The Failure of the Diagnostic Laptop Actuator
- The host laptop speaker failed open-loop testing with **THD = 196.8%** at 200 Hz.
- Micro-laptop speakers have an acoustic high-pass cutoff around 800–1000 Hz and cannot produce the linear cone excursion required for 100–500 Hz anti-noise.

### Selected Low-Frequency ANC Actuator
| Component | Part / Specification | Performance Characteristics | Function |
| :--- | :--- | :--- | :--- |
| **Acoustic Actuator** | **40 mm Dynamic Headphone Driver Unit** (Peerless / TDK / Boke 40mm 32Ω) | • Frequency Range: **20 Hz – 20,000 Hz**<br>• Impedance: **32 Ω ± 15%**<br>• Sensitivity: **105 dB/mW ± 3 dB**<br>• Linear Excursion ($X_{\max}$): $\ge 0.5\text{ mm}$<br>• $\text{THD} \le 0.5\%$ at 94 dB SPL (100–1000 Hz) | Mounted inside the ear-cup cavity to generate anti-noise acoustic pressure. |
| **Error Microphone** | **Calibrated Electret / MEMS Capsule** (WM8960 Left onboard or Knowles SPU0410HR5H-PB capsule) | • Frequency Range: **50 Hz – 10,000 Hz**<br>• Sensitivity: $-42\text{ dBV/Pa}$<br>• SNR: $\ge 62\text{ dBA}$<br>• Acoustic Overload Point: $\ge 120\text{ dB SPL}$ | Mounted internally inside ear cavity, positioned 5 mm from ear canal entrance. |
| **Reference Microphone**| **Low-Noise Capsule** (WM8960 Right onboard or Knowles capsule) | • Sensitivity: $-42\text{ dBV/Pa}$<br>• Matched phase response to error mic | Mounted on external ear-cup shell facing ambient acoustic disturbance. |
| **Acoustic Ear-Cup Fixture** | Closed-Back Circumaural Headphone Shell (e.g. modified 3M Peltor / Sennheiser HD206 shell) | • Cavity Volume: **$35\text{ cm}^3$ nominal**<br>• Memory-foam cushion for repeatable acoustic seal<br>• Rigid acoustic baffle isolating reference mic from cavity | Provides physical acoustic attenuation and controlled secondary acoustic path. |

---

## 3. Exact Channel Allocation & System Separation

To prevent confusion between **System A (Speech Enhancement)** and **System B (Physical Acoustic ANC)**, channel roles are frozen:

### System B: Physical Acoustic ANC (PH5 Milestone Focus)
| Hardware Port | Signal Name | System Role | Acoustic Location |
| :--- | :--- | :--- | :--- |
| **WM8960 In-L (Ch 0)** | $x(n)$ | **Reference Microphone** | External shell facing incoming combat/engine noise |
| **WM8960 In-R (Ch 1)** | $e(n)$ | **Error Microphone** | Inside ear-cup cavity facing ear canal |
| **WM8960 Out-L (Ch 0)**| $y(n)$ | **Anti-Noise Actuator** | Inside ear-cup directed at eardrum |
| **WM8960 Out-R (Ch 1)**| — | Reserved / Passthrough Monitoring | Diagnostic line-out |

### System A: Communications Speech Enhancement (Previously Verified in PH2–PH3)
When System A is operated:
- **Ch 0 (Speech Mic):** Close-talk boom microphone facing mouth.
- **Ch 1 (Noise Reference):** External reference microphone.
- **Output:** Clean communication speech routed to the user's communication channel.

---

## 4. Raspberry Pi 4B to WM8960 Pinout & Wiring Specification

The WM8960 Audio HAT mates directly onto the standard 40-pin GPIO header of the Raspberry Pi 4B. The physical pin connections are:

| Raspberry Pi Pin # | BCM GPIO | Pin Name | Direction | WM8960 Function | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pin 1** | — | **3.3V Power** | Output from Pi | Analog VDD | Clean 3.3V logic & analog rail |
| **Pin 2** | — | **5.0V Power** | Output from Pi | SPKVSS / Amp Power | 5V rail for Class-D speaker amplifier |
| **Pin 6** | — | **Ground (GND)** | Common | Ground | Digital & analog reference plane |
| **Pin 3** | GPIO 2 | **I²C SDA** | Bidirectional | Control Data | Codec configuration (ALSA mixer registers) |
| **Pin 5** | GPIO 3 | **I²C SCL** | Bidirectional | Control Clock | Codec configuration clock (100 kHz) |
| **Pin 12** | GPIO 18 | **PCM CLK (BCLK)**| Output from Codec / Pi | I²S Bit Clock | Bit clock for audio data (e.g. 1.024 MHz) |
| **Pin 35** | GPIO 19 | **PCM FS (LRCLK)**| Output from Codec / Pi | I²S Word Select | Frame sync (Left/Right clock @ 16 kHz or 48 kHz) |
| **Pin 38** | GPIO 20 | **PCM DIN** | Input to Pi | I²S ADC Data | Stereo ADC stream (`[Ch0: Ref, Ch1: Err]`) |
| **Pin 40** | GPIO 21 | **PCM DOUT** | Output from Pi | I²S DAC Data | Anti-noise DAC output stream (`y(n)`) |

---

## 5. Ear-Cup Internal Transducer Wiring Schematic

```text
               CLOSED-BACK EAR-CUP CAVITY (~35 cm³)
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  [Reference Mic Capsule]                                         │
│       │ (Shielded Twisted Pair)                                  │
│       ▼                                                          │
│   To WM8960 Left Mic Input (Pins LINPUT1/LINPUT2)                │
│                                                                  │
│  ══════════════════ ACOUSTIC BAFFLE WALL ═════════════════════   │
│                                                                  │
│  [40mm 32Ω Actuator] ◄──── Twisted Pair ◄── WM8960 Headphone Out │
│       │ (Acoustic Emission)                                      │
│       ▼                                                          │
│  [Ear-Cup Cavity Air Volume: S(z)]                               │
│       │                                                          │
│       ▼                                                          │
│  [Error Mic Capsule] ─────► Twisted Pair ──► WM8960 Right Mic In │
│       │                                      (Pins RINPUT1/R2)   │
│       ▼                                                          │
│  [Artificial Ear / Canal Opening]                                │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 6. Execution Roadmap for Recovery Milestone

1. **Physical Assembly:** Solder the 40mm 32Ω driver into the ear-cup baffle. Route shielded twisted-pair wiring to the WM8960 terminal blocks.
2. **Execute Experiment A (Pure Electrical Loopback):**
   - Connect line-out to line-in via direct cable.
   - Run calibrated chirp cross-correlation across 20 trials.
   - Record $\mu_t$ and $\sigma_t$ (Target: $t < 5\text{ ms}$, $\sigma_t \le 0.05\text{ ms}$ / $0.8\text{ sample at } 16\text{ kHz}$).
3. **Execute Experiment B (Secondary Path $S(z)$ Identification):**
   - Place ear-cup on test coupler.
   - Run Farina swept-sine ($50 \to 1000\text{ Hz}$).
   - Extract $\hat{s}(t)$, $|S(f)|$, $\angle S(f)$.
4. **Execute Experiment C (20-Run Repeatability):**
   - Run 20 identical acoustic sweeps without altering the fixture.
   - Compute:
     - Timing determinism: $R_{\text{raw}}$ and $\sigma_t \le 0.05\text{ ms}$ ($0.8\text{ sample at } 16\text{ kHz}$).
     - Plant repeatability: **$R_{\text{aligned}} \ge 0.95$**.
     - Magnitude variation: $\Delta G \le 0.5\text{ dB}$.
     - Phase variation: $\Delta\phi \le 5^\circ$.
5. **Only if all pass:** Unlock PH5.7 (FxNLMS at very low $\mu$).
