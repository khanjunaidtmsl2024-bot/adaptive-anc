# PS 26052 - REFERENCE LIBRARY
## Research, Algorithms, Hardware, Standards - Organized by Topic

---

## ARCHITECTURE REFERENCES

### DeepFilterNet2 (Primary Model)
- Paper: Schrotter et al., 2022, IWAENC. 107 citations.
- Key result: RTF 0.04 on notebook Core-i5 (published, not our measurement)
- Architecture: ERB filter banks + depthwise separable convolutions + deep filtering
- Sample rate: Likely 48 kHz default (VERIFY before freezing audio path)
- Repo: https://github.com/Rikorose/DeepFilterNet
- Paper: https://arxiv.org/abs/2205.05474

### DCCRN (Comparison Model)
- Paper: Hu et al., Interspeech 2020 DNS Challenge winner
- Architecture: Complex-valued CRN with phase-aware processing
- Successor: DCCRN+ (Lv et al. 2021)
- Distilled: Distil-DCCRN (Han et al. 2024) - 30% params, better PESQ

### Conv-TasNet (Ultra-Low-Latency Alternative)
- Causal variant: 3 ms algorithmic latency
- Tradeoff: Weaker quality for stationary noise

---

## ALGORITHM REFERENCES

### NLMS Adaptive Filter
- Update: w[n+1] = w[n] + mu * e[n] * x_vec[n] / (eps + ||x_vec[n]||^2)
- Typical mu: 0.1 to 0.5
- Filter length: 64 to 256 taps
- Config A: d[n]=primary, x[n]=reference, e[n]=d[n]-w^T*x[n]



### Reference-Microphone Speech Leakage (CRITICAL RISK)

**The most immediate engineering risk in the entire architecture.**

If the reference microphone contains too much target speech, the adaptive stage learns: "speech is correlated noise — cancel it."

**Verification procedure:**
1. Record speech-only → check reference channel level
2. Record noise-only → check reference channel level
3. Measure coherence between primary and reference
4. Run NLMS with speech in reference → check if speech is suppressed
5. If leakage is high → controller must reduce/freeze adaptive cancellation

**Relevant research:** Valin's hybrid DSP/DL paper includes VAD output that forces the recurrent net to learn speech-vs-noise discrimination. This is directly applicable to our controller design.

**Scite-verified papers on this topic:**
- Ali, van Waterschoot & Moonen (2018) — GSC for hearing devices with external mic
- Zhou et al. (2020) — Real-time dual-microphone speech enhancement
- Cheong et al. (2024) — Postfilter for dual-channel speech enhancement

### Spectral Subtraction
- Good for: Stationary noise, simple implementation
- Bad for: Musical noise artifacts
- Use: Classical baseline for ablation table

---

## DATASET REFERENCES

### DNS Challenge
- Repo: https://github.com/microsoft/DNS-Challenge
- Clean speech: ~500 hours, 2150 speakers
- Noise: 180+ hours, 150 classes

### MAD - Military Audio Dataset
- Paper: Nature Scientific Data, 2024
- Repo: https://github.com/kaen2891/military_audio_dataset
- Content: 8075 samples, gunshot/shelling/helicopter/vehicle/fighter

### C3GD - Gunshot Dataset
- Paper: https://arxiv.org/abs/2606.18135
- Content: Clean gunshot recordings across calibers

### NOISEX-92 (Unseen Test Only)
- NEVER train on this. Held-out generalization test only.

---

## HARDWARE REFERENCES

### MEMS Microphones
- INMP441: I2S, ~, 61 dB SNR
- SPH0645: I2S, ~, 65 dB SNR

### Jetson Power Modes
- AGX Orin: 15W/30W/50W/MAXN
- Always state which power mode you benchmarked under

---

## STANDARDS REFERENCES

### MIL-STD-1474E
- Steady-state: 85 dBA at ear
- Impulse: 140 dB(P) peak

### ITU-T P.862 (PESQ)
- Withdrawn 2024, but PS 26052 explicitly names it
- Modes: NB (8 kHz) and WB (16 kHz)

---

## COMMERCIAL LANDSCAPE

### 3M PELTOR ComTac
- 18 dB noise cancellation, NRR 23-34 dB
- MAP modes: Pre-set gain profiles, manually selected
- Gap: Classical analog/DSP, not AI-driven, no automatic adaptation

---


## SCITE-VERIFIED HYBRID ANC REFERENCES (added from Scite verification)

### Kwon, Kim & Park (2022) — LSTM + FxLMS for Diesel-Engine Radiation Noise
- **Paper:** Active Noise Reduction with Filtered LMS Improved by LSTM for Radiation Noise of Diesel Engine
- **Source:** Applied Sciences, 12(20), 10248
- **Relevance:** Direct LSTM + FxLMS hybrid for engine noise — closest to our Config A architecture
- **Key finding:** LSTM improves FxLMS convergence speed and noise-cancellation for nonlinear broadband noise
- **Priority:** MUST READ for hybrid architecture validation
- **In bank:** Not yet downloaded (IEEE/MDPI access needed)

### Multichannel / Beamforming References (Scite-verified)
- **MVDR / GSC / MWF beamforming** — directly relevant to two-mic prototype
- **Ali, van Waterschoot & Moonen (2018)** — GSC for hearing devices with external mic
- **Shimada et al. (2019)** — Unsupervised multichannel NMF beamforming
- **Zhou et al. (2020)** — Real-time dual-microphone speech enhancement
- **Saric et al. (2021)** — Low-cost MVDR noise reduction
- **Cheong et al. (2024)** — Postfilter for dual-channel speech enhancement
- **Architecture implication:** Second mic opens multichannel processing branch (GSC: fixed beamformer + blocking matrix + adaptive noise canceller)
- **Priority:** SHOULD READ for V1 hardware architecture

### VAD / Speech Protection (Scite-verified)
- **Valin hybrid DSP/DL paper** includes VAD output — forces recurrent net to learn speech-vs-noise discrimination
- **Implication:** Controller can use speech confidence to control suppression aggressiveness
- **Priority:** SHOULD READ for controller design

### Kalman-Based Enhancement (Scite-verified)
- **Roy, Nicolson & Paliwal (2020)** — Deep Learning-Based Kalman Filter for Speech Enhancement
- **Relevance:** DL estimates quantities for Kalman filter under varied noise — supports separate impulse/noise-estimation branch
- **Priority:** RESEARCH ONLY for V2 contingency

### Construction-Site / Impact-Noise ANC (Scite-verified)
- **Deep learning-based ANC on construction sites (2023)** — addresses transient/high-frequency impact noise including jackhammers
- **Relevance:** Closest to impulsive defence noise problem
- **Status:** Requires ScienceDirect access

### Vehicle-Engine Hybrid DL + FxLMS (Scite-verified)
- **Multi-channel ANC with DL secondary-path estimation + nFxLMS (2025)** — real vehicle engine noise, non-stationary acceleration
- **Relevance:** AI for secondary-path modelling + adaptive controller
- **Status:** Requires ScienceDirect access

### Speech-Preserving ANC (Scite-verified)
- **Park et al. (2026)** — Deep learning speech-preserving ANC in reverberant environments
- **Relevance:** Explicitly addresses conventional ANC suppressing speech with noise
- **Architecture:** CRN/LSTM + complex spectral mapping + speech-retention loss
- **In bank:** Yes (30_Speech_Preserving_ANC_2026.pdf)

---

## PAPER VERIFICATION LOG

| Paper | Status | Notes |
|-------|--------|-------|
| 20_Deep_Complex_UNet | FIXED | Old file was misidentified (unrelated content). Replaced with correct arXiv 1903.03107 (5.16 MB). Old file backed up as 20_Deep_Complex_UNet_OLD_MISIDENTIFIED.pdf |
| All other papers | VERIFIED | Checked via file size and PDF header in previous session |

---

## FUTURE RESEARCH (Phase 2 Parking Lot)
- Neural beamforming
- Learned controller network
- Physical FxLMS with secondary speaker
- Mamba/SSM models
- Complex-domain masking
- Multi-loss composite training
- Custom PCB / accelerator

---

*Add entries as new research is encountered. V1 scope frozen per Contract.*

---

## FAILURE INJECTION PLAN (from V6 §97)

The team should deliberately create failures to test robustness.

| ID | Fault | Expected Behaviour |
|----|-------|-------------------|
| F1 | Disconnect reference mic | Adaptive stage freezes/reduces safely |
| F2 | Overload microphone | Protection mode activates |
| F3 | AI process stops | Fallback/bypass engages |
| F4 | Artificially increase inference time | Runtime monitor detects overload |
| F5 | Drop audio frames | Event logged, recovery visible |
| F6 | Feed speech into reference mic | Reference-health logic reduces adaptive aggression |
| F7 | Impulse during speech | Controlled response without prolonged speech destruction |

**Rule:** Run all 7 failure tests before the final demo. Record results in Evidence Log.

---

## FULL SYSTEM DEBUG TELEMETRY (from V6 §115)

The runtime should expose these signals for debugging:



**Rule:** If audio is bad, check these signals first. Never start debugging by changing the neural network.

---

## HARDWARE BRING-UP ORDER (from V6 §91)

Never connect everything at once. Follow this sequence:



---

## ELECTRICAL DEBUGGING (from V6 §92)

If audio is bad, check in this order:

1. Power supply
2. Ground
3. Cable
4. Connector
5. Sample clock
6. I2S format
7. Channel order
8. Gain
9. Clipping
10. Driver
11. Buffer
12. Software

**Never start by changing the neural network.**

---

## VERIFIED RESEARCH ANCHORS (from V6 §127)

These are confirmed facts, not hypotheses:

1. **DNS Challenge repo** provides dataset-generation scripts, clean speech/noise/RIR resources, unit tests, and provenance information.

2. **DeepFilterNet repo** documents precompiled deep-filter path as accepting 48 kHz WAV files. Audio-rate policy must be based on exact installed version.

3. **DeepFilterNet2 paper** reports RTF 0.04 on notebook Core-i5 under stated conditions. This is feasibility evidence, not our performance.

**Rule:** Never quote a literature number as a project result.

---

*Add entries as new research is encountered. V1 scope frozen per Contract.*
