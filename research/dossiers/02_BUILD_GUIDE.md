# PS 26052 — BUILD GUIDE
## 30-Day Execution Plan for V1 Prototype
### Read the relevant phase each day. Do not skip gates.

**Document:** 02_BUILD_GUIDE.md  
**Contract:** 01_V1_CONTRACT.md (the authority — read it first)  
**Evidence:** 04_EVIDENCE_LOG.md (record everything here)

---

## How to Use This Document

1. Read the Contract (01) on Day 1. Understand what V1 is.
2. Each day, read the relevant phase in this guide.
3. Follow the phase structure: Objective → Knowledge → Build → Experiment → Measure → Deliver → Gate.
4. Do NOT start the next phase until the current gate is signed.
5. Record every decision and result in the Evidence Log (04).

---

## FIRST-WEEK ACTION PLAN (Days 1-7)

### Day 1 — Environment Setup
- Install Python 3.10+, Git, VS Code
- Create repository with: `src/`, `tests/`, `configs/`, `data/`, `models/`, `results/`, `docs/`, `hardware/`
- Create virtual environment, install dependencies
- Run signal inspection script on any audio file
- Read Contract sections 1-5
- **GATE:** Environment reproducible; first waveform inspected

### Day 2 — SNR and Dataset Basics
- Implement SNR calculation
- Create 20 clean/noisy pairs using the dataset generator
- Verify requested vs actual SNR
- Start evaluation CLI
- **GATE:** 20 valid pairs + metric sanity check

### Day 3 — Evaluation Harness
- Finish evaluation harness (SNR/STOI/PESQ/SI-SNR)
- Run spectral subtraction baseline
- Create results.csv
- Start STFT lab
- **GATE:** A baseline number exists before AI

### Day 4 — STFT/iSTFT
- Complete manual STFT/iSTFT implementation
- Check NOLA reconstruction
- Build mic→headphone passthrough if hardware arrived
- **GATE:** Near-perfect reconstruction; basic audio path

### Day 5 — NLMS
- Implement LMS and NLMS from scratch
- Run convergence/divergence experiments
- Log μ/tap effects
- **GATE:** NLMS convergence plot

### Day 6 — Two-Mic Acquisition
- Connect two microphones
- Verify channel identity and synchronization
- Test reference correlation
- **GATE:** 60-second stable dual-channel recording

### Day 7 — Weekly Review
- Run all gates
- Record decisions in Evidence Log
- Decide whether to proceed to AI integration
- **GATE:** Week-1 gate signed; no unresolved P0 issue

---

## PHASE 0 — Environment and Engineering Foundations
**Day 1 | Owner: Team Lead**

### Objective
Create a reproducible development environment and refresh audio/electrical fundamentals.

### Knowledge
- Python virtual environments; Git; directory structure
- Sampling, Nyquist frequency, PCM, bit depth, float vs int16
- RMS, peak, dB, dBFS, SNR, crest factor, clipping, headroom
- Microphone sensitivity, frequency response, self-noise, saturation
- Basic Linux process/thread concepts

### What to Build
- Repository with standard structure
- Pinned `requirements.txt` and reproducibility README
- Signal inspection script (prints sample rate, channels, duration, peak, RMS, clipping count)

### Experiments
1. Generate 1 kHz sine wave at 48 kHz, 16 kHz, 8 kHz
2. Deliberately undersample and observe aliasing
3. Convert float audio to int16 and back
4. Clip waveform and calculate dBFS and distortion

### Measurements
- Sample rate and channel count
- RMS/peak/dBFS
- Clipping percentage
- Environment hash

### Deliverables
- Repository initialized
- `requirements.txt`
- `signal_inspection.py`
- First lab notebook entry

### Pass Gate
Every member can explain Nyquist, dBFS, clipping, and the difference between sample rate and bit depth.

### Common Failures
- Silent assumptions about sample rate
- Integer overflow
- Wrong normalization
- Mixing dB SPL with dBFS

---

## PHASE 1 — DSP Laboratory: FFT, STFT, iSTFT and Classical Baselines
**Days 2-3 | Owner: DSP Lead**

### Objective
Understand frequency-domain machinery deeply enough to debug a neural enhancement pipeline without treating STFT as a black box.

### Knowledge
- DFT/FFT, frequency bins, spectral leakage, window functions
- Frame length, hop size, overlap-add, NOLA/COLA concepts
- Complex spectrum: magnitude and phase
- Spectral subtraction and its musical-noise limitations

### What to Build
- `stft_manual.py` — manual FFT analysis for a frame
- `istft_manual.py` — manual overlap-add reconstruction
- `spectral_subtraction.py` — classical baseline
- Use SciPy only as reference after manual version works

### Experiments
1. Reconstruct clean sine wave through STFT/iSTFT
2. Measure reconstruction error for several window/hop combinations
3. Add broadband noise at known SNR
4. Implement spectral subtraction and listen to artifacts

### Measurements
- Reconstruction error
- NOLA check
- Input/output SNR
- Spectral subtraction ΔSNR
- Artifact notes

### Deliverables
- `stft_manual.py`, `istft_manual.py`, `spectral_subtraction.py`
- Plots of magnitude/phase
- Lab report

### Pass Gate
Near-perfect reconstruction for a valid configuration; every member can explain why an invalid overlap creates modulation/gaps.

### Common Failures
- Wrong FFT scaling
- Window normalization
- Phase mishandling
- Off-by-one frame indexing
- Incorrect overlap-add

---

## PHASE 2 — Adaptive Filtering: LMS → NLMS → Controlled Reference Cancellation
**Days 4-5 | Owner: DSP Lead**

### Objective
Build and break adaptive filters before integrating them with AI.

### Knowledge
- LMS and NLMS equations; step-size; convergence/divergence
- Reference/primary correlation and why reference quality matters
- Filter length, delay, coefficient initialization
- Difference between reference-mic noise cancellation and physical FxLMS ANC

### What to Build
- `nlms.py` — LMS and NLMS from scratch using NumPy
- Synthetic primary/reference signals with known correlated noise
- Speech leakage added to reference
- Frozen Config A signal flow implementation

### Experiments
1. Set μ too large → observe divergence
2. Set μ too small → measure convergence time
3. Decorrelate reference → measure failure
4. Introduce speech leakage → observe speech cancellation
5. Compare NLMS-only against raw baseline

### Measurements
- MSE curve
- Convergence time
- Output SNR
- Speech distortion
- Reference correlation/coherence

### Deliverables
- `nlms.py`
- Synthetic test suite
- Reference-quality experiment results
- Config A decision log entry

### Pass Gate
Team can predict whether a change in μ/taps/reference quality should improve or destabilize the filter before running it.

### Common Failures
- Wrong sign convention
- Tap alignment error
- Reference delay mismatch
- Speech leakage into reference
- Integer/float type errors

---

## PHASE 3 — Audio Hardware and Two-Microphone Acquisition
**Days 6-7 | Owner: Hardware Lead**

### Objective
The first physical milestone is not AI. It is reliable, synchronized two-channel audio acquisition.

### Candidate Low-Cost Hardware
- Microphones: INMP441 (I2S, ~$2 each) or SPH0645
- Dev board: Raspberry Pi 5 or laptop (Phase 1-2)
- Audio interface: built-in sound card or USB adapter for two channels
- Headset: any 3.5mm or USB headset for output

### Knowledge
- I2S: bit clock, word select/LRCLK, serial data, master/slave clocking
- Digital MEMS microphone output and channel selection
- Synchronized two-channel capture
- Linux audio devices, ALSA/PortAudio, device enumeration
- Grounding, power noise, cable management, physical placement

### Bring-Up Sequence
1. Power and identify one microphone
2. Record 10-second signal, inspect waveform
3. Connect second microphone
4. Verify channel identity (speak/tap near only one mic)
5. Record 60 seconds with both channels
6. Check for dropouts, clipping, drift, sample-rate mismatch
7. Only after stable capture, mount in repeatable geometry

### Pass Gate
60-second dual-channel recording with zero unexplained dropouts. Both channels have known identity and stable sample rate. Test log contains hardware revision, wiring, and capture settings.

---

## PHASE 4 — Real-Time Audio Systems
**Days 8-9 | Owner: Streaming Lead**

### Objective
Create a streaming audio pipeline before adding AI.

### Knowledge
- Audio callbacks and why they must stay lightweight
- Ring buffers, producer/consumer threading, backpressure
- Frame size, buffering, algorithmic latency
- Loopback measurement and P50/P95/P99 latency

### What to Build
- streaming_skeleton.py — mic to headphones passthrough
- Add fixed delay, then simple DSP filter
- Timestamp input/output blocks, calculate latency statistics
- latency_logger.py — P50/P95/P99/max/underruns

### Experiments
1. Change buffer sizes and record latency
2. Introduce CPU load and observe underruns
3. Stop/restart input device and verify fault handling
4. Measure click-to-output loopback latency

### Measurements
- P50/P95/P99 latency
- Max latency
- Underruns/overruns
- Frame drops

### Pass Gate
Stable 10-minute streaming run with no unexplained glitches at the selected operating point.

### Common Failures
- Blocking callback
- Python GC/GIL pauses
- Queue starvation
- Buffer mismatch
- Device clock mismatch

---

## PHASE 5 — Dataset Engineering
**Days 10-11 | Owner: ML Lead**

### Objective
Build the reproducible noisy-clean dataset pipeline before serious model training.

### Dataset Starter Stack
- Clean speech: DNS Challenge clean set (~500 hours)
- Defence noise: MAD dataset (gunshot, shelling, helicopter, vehicle, fighter)
- Impulsive: C3GD gunshot recordings or synthetic impulses
- Unseen test: NOISEX-92 (babble, f16, factory) — NEVER train on this
- RIRs: MIT IR Survey or OpenAIR

### Mixing Formula
alpha = sqrt(Ps / (Pn * 10^(SNR_dB/10)))
x[n] = s[n] + alpha * n[n]

### Pass Gate
Train and test have no shared speaker/source. Requested SNR achieved within tolerance.

---

## PHASE 6 — ML/PyTorch Fundamentals
**Days 12-13 | Owner: ML Lead**

### Objective
Make the EE team capable of reading, training, and debugging neural audio models.

### What to Build
- tiny_model.py — small spectrogram mask CNN/GRU
- Training config, checkpoint save/load, deterministic inference

### Pass Gate
Every AI member can explain every tensor entering and leaving their model.

---

## PHASE 7 — DeepFilterNet Benchmark and Fine-Tuning
**Days 14-16 | Owner: ML Lead**

### CRITICAL — Do This First
1. Install DeepFilterNet in clean environment
2. Verify exact sample rate (likely 48 kHz, NOT 16 kHz)
3. Run pretrained inference on known noisy audio
4. Record version, commit, checkpoint
5. Measure CPU/RAM/inference time on YOUR hardware
6. Run evaluation harness BEFORE any fine-tuning

### Pass Gate
Reproducible pretrained baseline exists. If deployment blocked, switch to compact CRN fallback.

---

## PHASE 8 — Controller and Hybrid Integration
**Days 17-18 | Owner: Systems Lead**

### Objective
Combine NLMS, AI and deterministic state control only after each block works independently.

### What to Build
- controller.py — pure deterministic controller
- Thresholds from calibration experiments (NOT arbitrary claims)
- State transition logging

### Pass Gate
Controller never makes speech worse. Every threshold traceable to a measurement.

---

## PHASE 9 — Formal Hybrid Architecture Validation
**Day 19 | Owner: ML Lead + DSP Lead**

### Mandatory Ablation Matrix
| System | Raw | NLMS-only | AI-only | NLMS→AI | NLMS→AI+Controller |
|--------|-----|-----------|---------|---------|-------------------|
| Stationary noise | | | | | |
| Non-stationary noise | | | | | |
| Impulsive noise | | | | | |
| Mixed noise | | | | | |
| Unseen noise | | | | | |

**Same test clips across all systems. No cherry-picking.**

### Pass Gate
If hybrid is not consistently beneficial, remove it from claimed performance. Negative results are acceptable; unsupported claims are not.

---

## PHASE 10 — Embedded Deployment Ladder
**Days 20-22 | Owner: Hardware Lead**

### Deployment Ladder
1. PyTorch reference (laptop)
2. PyTorch to ONNX on laptop
3. Compare outputs (numerical validation)
4. ONNX to TensorRT on target (if available)
5. FP16 benchmark
6. Optional INT8
7. End-to-end streaming on target
8. 30-minute stability test

### Pass Gate
Deployed output functionally equivalent within tolerance and meets real-time budget.

---

## PHASE 11 — Physical Prototype Integration
**Days 23-25 | Owner: Hardware Lead**

### What to Build
- Mount primary and reference mics
- Connect capture hardware, processing unit, headset output
- Add bypass/A-B control

### Experiments
1. Record raw noisy speech
2. Run AI-only
3. Run NLMS→AI
4. Run controller
5. Repeat at several mic positions
6. Run impulse protection at safe levels

### Pass Gate
10-minute controlled run produces stable output and repeatable measurements.

---

## PHASE 12 — Evaluation Laboratory
**Day 26 | Owner: ML Lead**

### What to Build
- evaluate.py — evaluation CLI
- Validate files before scoring
- Write results to CSV with config hash
- Generate summary plots

### Pass Gate
Same command reproduces same numbers within numerical tolerance.

---

## PHASE 13 — Impulsive-Noise Subsystem
**Day 27 | Owner: DSP Lead**

### What to Build
- Safe synthetic impulse library
- Detector using crest factor + spectral flux
- Calibrated thresholds from local recordings
- Hysteresis and recovery logic

### Pass Gate
Detector stable enough to trigger protection without state chatter.

---

## PHASE 14 — Hardware/Software Debugging
**Day 28 | Owner: All**

### Fault Injection Tests
1. Unplug reference mic
2. Force wrong sample rate
3. Force AI timeout
4. Fill processing queue
5. Introduce clipping
6. Restart audio device
7. Kill/restart processing thread

### Pass Gate
Every member can diagnose at least one software and one hardware failure without AI help.

---

## PHASE 15 — Controlled Acoustic Test Rig
**Day 29 | Owner: Hardware Lead**

### What to Build
- Mark fixed mic and speaker positions
- Record geometry and gain settings
- Use repeatable noise playback files

### Pass Gate
Repeated trials within predefined tolerance.

---

## PHASE 16 — Full System Acceptance
**Day 30 | Owner: Team Lead**

### Acceptance Tests
1. Stationary noise test
2. Non-stationary test
3. Impulsive test
4. Mixed test
5. Unseen noise test
6. 10-30 minute stability run
7. Fault-injection test

### Freeze Before Testing
- Software commit
- Model/checkpoint
- Hardware wiring

### Pass Gate
All critical tests pass or have documented limitation. No demo claim exceeds the evidence.

---

*Reference: 01_V1_CONTRACT.md (scope) | 03_REFERENCE_LIBRARY.md (knowledge) | 04_EVIDENCE_LOG.md (results)*

---

## EXPERIMENT NAMING AND DEPENDENCY GRAPH (from V6 §105-106)

Every experiment gets a unique ID. Never jump ahead. The dependency graph shows what must be done first.

### Experiment IDs

| ID | Name | Phase | Owner |
|----|------|-------|-------|
| E001 | Audio sanity | 0 | All |
| E002 | SNR mixer | 5 | ML Lead |
| E003 | STFT reconstruction | 1 | DSP Lead |
| E004 | Spectral subtraction | 1 | DSP Lead |
| E005 | LMS | 2 | DSP Lead |
| E006 | NLMS | 2 | DSP Lead |
| E007 | Dual-mic capture | 3 | Hardware Lead |
| E008 | Streaming passthrough | 4 | Streaming Lead |
| E009 | DeepFilterNet offline | 7 | ML Lead |
| E010 | AI streaming | 7 | ML Lead |
| E011 | AI fine-tuning | 7 | ML Lead |
| E012 | Impulse detection | 13 | DSP Lead |
| E013 | NLMS + AI hybrid | 9 | ML + DSP |
| E014 | Controller | 8 | Systems Lead |
| E015 | Embedded deployment | 10 | Hardware Lead |
| E016 | Long-run stability | 14 | All |
| E017 | Final ablation | 16 | All |

### Dependency Graph



### Rules
- Do not skip experiments
- Each experiment produces: input, procedure, result, decision
- Record every result in 04_EVIDENCE_LOG.md
- A failed experiment is still an engineering result

---

## THE 6 EXPERIMENTS THAT MATTER MORE THAN ANY PAPER

These six questions are worth more to your SIH project than another ten generic speech-enhancement papers. Answer them with measurements, not literature.

### Experiment C1: Does the AI model run causally in your real audio pipeline?
- Install DeepFilterNet2, run on live mic input
- Measure actual RTF, latency, CPU, RAM
- Verify causal processing (no future frames)
- **Pass:** RTF < 1.0 on target hardware with measured latency
- **Fail:** Model requires non-causal lookahead exceeding latency budget

### Experiment C2: Does the reference microphone actually contain useful noise correlation?
- Record: speech-only, noise-only, speech+noise
- Measure: coherence between primary and reference
- Measure: speech leakage into reference
- **Pass:** Reference has high noise correlation, low speech leakage
- **Fail:** Reference contains more speech than noise → adaptive stage will cancel speech

### Experiment C3: Does adaptive processing improve results or damage speech?
- Compare: AI-only vs NLMS→AI on same test set
- Measure: SNR, STOI, PESQ, SI-SNR for both
- **Pass:** NLMS→AI beats AI-only on at least 2 metrics without degrading others
- **Fail:** NLMS→AI damages speech → remove NLMS from V1

### Experiment C4: What happens during clipping and impulsive noise?
- Feed impulse at various amplitudes
- Check: does system protect speech? recovery time? artifacts?
- **Pass:** Recovery < 500 ms, no prolonged speech destruction
- **Fail:** System crashes or produces dangerous output

### Experiment C5: Does the system generalize to defence noises it never trained on?
- Train on subset, test on held-out noise classes
- Report: seen vs unseen noise performance separately
- **Pass:** Unseen noise performance within 2 dB of seen noise
- **Fail:** >5 dB degradation on unseen → model is overfitting to training noise

### Experiment C6: What is the true end-to-end latency?
- Physical measurement: impulse in → peak out
- Report: P50, P95, P99, max
- **Pass:** P95 < 40 ms
- **Fail:** P95 > 40 ms → simplify pipeline

### Rule
These 6 experiments are the minimum evidence required before any SIH presentation. No amount of literature reading replaces these measurements.

