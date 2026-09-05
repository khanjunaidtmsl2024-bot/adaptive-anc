# Bill of Materials (BOM): Stage-1 Hardware Prototype
## DRDO Problem Statement 26052 • Smart India Hackathon 2026
**Authoritative Status:** VERIFIED ENGINEERING SPECIFICATION  
**Target Architecture:** Dual-Mic Hardware Frontend + Laptop Compute Brain  
**Total Estimated Budget:** ₹2,750 INR (< ₹3,000 Ceiling)  

---

## 1. Procurement & Component Breakdown

| Item # | Component | Specifications | Primary Function | Source / Vendor | Approx. Cost (INR ₹) | Status |
|---|---|---|---|---|---|---|
| **HW-01** | **INMP441 MEMS Mic (x2)** | Omnidirectional digital MEMS, I²S output, 61 dBA SNR, 120 dB SPL AOP, 24-bit I²S, 3.3V | Dual-mic array (Primary + Reference) | [Robu.in](https://robu.in) / [Quartzcomponents](https://quartzcomponents.com) | ₹440 (₹220 x 2) | Required |
| **HW-02** | **I²S to USB Bridge / Stereo ADC** | PCM2902 / PCM1808 or ESP32-S3 USB-JTAG bridge (24-bit, 48kHz / 16kHz stereo I²S) | Synchronized 2-channel audio capture into laptop | [Robu.in](https://robu.in) / Amazon.in | ₹650 | Required |
| **HW-03** | **Monitor Headphones** | Over-ear closed-back wired headset with 3.5mm TRS jack (e.g. Sennheiser HD206 or standard tactical headset) | Real-time enhanced speech monitoring | Amazon.in / Lab Stock | ₹850 | Available |
| **HW-04** | **Dual Portable Test Speakers (x2)** | 3W 4Ω magnetic shielded mini speakers (Speaker A: Speech, Speaker B: Noise) | Acoustic test rig sound projection | Robu.in / Amazon.in | ₹450 | Required |
| **HW-05** | **Breadboard & Jumper Wire Kit** | 400-point solderless breadboard + 40-pin Dupont wires (M-M, M-F) | Microphone mounting & clock distribution | Quartzcomponents | ₹160 | Required |
| **HW-06** | **Acoustic Baffle & Spacing Rig** | 15 cm rigid acrylic/3D-printed acoustic spacer with foam isolation | Eliminates structural acoustic cross-bleed | Local Fab / 3D Print | ₹200 | In-House |
| **TOTAL** | | | | | **₹2,750 INR** | **Within Budget** |

---

## 2. Stage-1 Hardware Rationale

### Why Laptop Compute Brain First?
As established in Section 18 of the project transfer handoff:
1. **Prevents Premature Hardware Lock-in:** Expensive edge AI SBCs (Jetson Orin Nano ₹45,000 or Raspberry Pi 5 ₹8,000) should never be purchased before the exact model memory footprint and compute FLOPs are profiled.
2. **Deterministic Profiling:** The laptop allows deep profiling of STFT latency, NLMS divergence, and impulse recovery without edge thermal throttling.
3. **Synchronized Front-End Proof:** Proves that two-channel physical audio acquisition works without buffer underruns before porting C++ runtimes to embedded microcontrollers.

---

## 3. Alternative MEMS Option: InvenSense ICS-43434
If higher acoustic overload headroom is required for loud combat environments:
- **Part:** InvenSense ICS-43434 (I²S digital microphone)
- **Acoustic Overload Point (AOP):** 120 dB SPL
- **SNR:** 65 dBA (4 dB cleaner noise floor than INMP441)
- **Cost:** ~₹320 INR per breakout module
- **Drop-in Compatible:** Identical BCLK, LRCLK/WS, and SD pinout to INMP441.
