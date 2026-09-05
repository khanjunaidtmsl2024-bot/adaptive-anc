# PS 26052 — ADAPTIVE-DEFENCE ANC
## AI/ML-Enabled Adaptive Noise Cancellation / Speech Enhancement for Defence Communication

**SIH 2026 · DRDO · Department of Defence Production / iDEX · Hardware · Smart Vehicles**

**Master Working Document — Research + Learning + Build + Test + Validation + SIH Defence**

**Compiled:** 24 August 2026  
**Document role:** Single working reference for the team  
**Current project state:** Research/knowledge layer complete; V1 frozen for execution; measured performance not yet established.

---

# 0. READ THIS FIRST

This document is the consolidated working reference for the project.

It combines the strongest and most current material from the project's research documents, including:

- the supplied SIH Problem Statement 26052;
- deep technical research;
- the architecture and model-selection studies;
- dataset and training strategy;
- embedded deployment strategy;
- hybrid AI + adaptive filtering design;
- learning curriculum for the six-member engineering team;
- first-week and 30-day execution plans;
- evaluation and ablation framework;
- fail-safe and runtime-control design;
- hardware strategy;
- SIH demonstration plan;
- judge questions;
- decision and evidence governance.

The project has reached a **freeze point**: the knowledge layer is sufficiently organized to begin execution. From this point onward, the default activity is **build → test → measure → decide**, not another open-ended research loop.

> **Core rule:** A paper's result is not our result. A proposed architecture is not a proven architecture. A target is not an achievement. Only documented experiments produce project evidence.

---

# 1. PROJECT IDENTITY

## 1.1 Official problem statement

| Field | Value |
|---|---|
| Problem Statement ID | **26052** |
| Title | **To develop an AI/ML-enabled adaptive noise cancellation (ANC) system that effectively suppresses stationary, non-stationary, and impulsive defence noises while maintaining high speech intelligibility and real-time performance on embedded hardware.** |
| Organization | DRDO |
| Department | Department of Defence Production / iDEX |
| Category | Hardware |
| Theme | Smart Vehicles |
| Context | Defence and mission-critical communication |

The supplied problem statement identifies acoustic disturbances including:

- gunshots;
- artillery fire;
- helicopter rotor noise;
- armoured-vehicle sound;
- emergency sirens;
- other stationary, non-stationary and impulsive environmental disturbances.

The requested system combines AI/ML noise suppression with adaptive filtering and requires a path from data generation through training, evaluation, real-time inference, embedded deployment and live microphone/headset integration.

---

# 2. WHAT THE SIH PROBLEM ACTUALLY REQUIRES

The problem statement can be translated into five major engineering obligations.

| SIH requirement | What our prototype must eventually prove | Evidence required |
|---|---|---|
| Scalable dataset pipeline | Controlled clean/noisy mixtures covering defence-noise conditions | Dataset manifest, mixer, sample pairs, verified SNRs, leakage-safe splits |
| AI/ML noise suppression | A justified model with measured enhancement performance | Model version, checkpoint, configuration, test results |
| Training framework | Reproducible training/fine-tuning with appropriate losses | Config, seed, loss curves, checkpoint, evaluation |
| Real-time inference | Streaming operation with measured latency and stability | End-to-end latency, P50/P95/P99, underruns, CPU/RAM/power |
| Live prototype | Primary/reference microphones + processor + headset/communication output | Working hardware, wiring, live demonstration, measurements |

## 2.1 SIH-stated performance targets

The supplied PS names:

- **SNR > 15 dB**
- **STOI > 0.85**
- **PESQ > 2.5**
- low real-time latency

These are **project acceptance targets**, not current achievements.

Every final result must state the test condition and input SNR. Do not quote a literature benchmark as though it were our result.

---

# 3. THE MOST IMPORTANT TERMINOLOGY DECISION

## 3.1 Speech enhancement vs physical acoustic ANC

The word ANC in the problem statement must not cause the team to make an incorrect technical claim.

### Speech enhancement

A microphone captures:

\[
x(t)=s(t)+n(t)
\]

or, with reverberation,

\[
x(t)=s(t)*h(t)+n(t)
\]

where:

- \(s(t)\) = desired speech;
- \(n(t)\) = environmental noise;
- \(h(t)\) = room/acoustic impulse response;
- \(x(t)\) = observed microphone signal.

The system estimates an enhanced speech signal \(\hat{s}(t)\).

### Physical acoustic ANC

Physical ANC normally creates an anti-noise signal through a secondary acoustic path and requires modelling/control of that path. A full physical feedforward ANC experiment therefore introduces additional hardware and control problems.

### V1 terminology

For V1, use:

> **AI-driven speech enhancement with a reference-microphone adaptive noise-cancellation stage in a real-time communication chain.**

Do **not** claim:

- certified hearing protection;
- military deployment readiness;
- broadband physical anti-noise ANC;
- full FxLMS/secondary-path ANC;

unless those capabilities are separately implemented and experimentally validated.

### Phase-2 physical ANC

Full physical feedforward ANC remains a future branch requiring:

- secondary speaker/anti-noise actuator;
- error microphone;
- secondary-path model;
- acoustic control loop;
- stability testing;
- acoustic validation.

---

# 4. WHERE THE PROJECT IS RIGHT NOW

## 4.1 Current state

| Layer | Status |
|---|---|
| Problem definition | **FROZEN** |
| Engineering interpretation | **FROZEN** |
| V1 architecture | **FROZEN FOR EXECUTION** |
| Research/literature layer | **SUBSTANTIALLY COMPLETE** |
| Learning roadmap | Planned |
| Dataset strategy | Defined; acquisition/exact subset still an execution task |
| Evaluation framework | Defined; measurements still unknown |
| Hardware strategy | Low-cost first; expensive edge compute conditional |
| Deployment path | Laptop → export → edge profiling |
| AI model quality on our data | **UNKNOWN** |
| NLMS benefit | **UNKNOWN** |
| Microphone/reference performance | **UNKNOWN** |
| End-to-end latency | **UNKNOWN** |
| Embedded performance | **UNKNOWN** |
| Final SIH metrics | **NOT YET MEASURED** |
| Final novelty wording | **PROVISIONAL; must follow evidence** |

## 4.2 What this means

We are **not** at the point where we can honestly say:

- "our model achieves STOI > 0.85";
- "our system achieves SNR > 15 dB";
- "AI + NLMS is better";
- "the prototype is real-time";
- "the system works on Jetson";
- "our two-microphone setup is optimal."

Those are experimental questions.

### Current phase

**PHASE 0 → PHASE 1 TRANSITION**

We are moving from:

> research and architecture definition

to:

> implementation and measurement.

---

# 5. THE CORE ENGINEERING PRINCIPLE

The winning system is not the model with the highest offline PESQ.

The winning system is the system that:

1. preserves speech;
2. suppresses difficult noise;
3. handles stationary and rapidly changing noise;
4. survives impulsive events;
5. works causally or near-causally;
6. operates within embedded resource limits;
7. produces reproducible measurements;
8. works in a physical microphone/headset chain;
9. fails safely;
10. can be defended honestly before judges.

---

# 6. V1 SYSTEM CONTRACT — FREEZE

## 6.1 High-level signal flow

```text
                  ┌──────────────────────┐
PRIMARY MIC ────► │ Audio acquisition   │
                  │ + conditioning       │
                  └──────────┬───────────┘
                             │
                             ▼
                       STFT / features
                             │
                             ▼
                  ┌──────────────────────┐
REFERENCE MIC ───►│ Reference / NLMS    │
                  │ adaptive stage       │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ AI speech            │
                  │ enhancement          │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Runtime controller  │
                  │ + impulse handling  │
                  │ + fail-safe logic   │
                  └──────────┬───────────┘
                             │
                             ▼
                       iSTFT / output
                             │
                             ▼
                     DAC / audio codec
                             │
                             ▼
                     Headset / comms
```

### Important V5 architecture decision

The current V5 execution contract freezes the **reference-microphone NLMS stage before the AI model** as the first configuration to test.

This is a test configuration, not a claim that it will ultimately win.

The exact runtime signal definitions must be explicit:

- \(d[n]\): desired/primary-channel signal presented to the adaptive stage;
- \(x[n]\): reference input;
- \(y[n]\): adaptive-filter estimate;
- \(e[n]\): residual/error signal.

The team must measure whether this placement actually improves the final system.

## 6.2 Alternative topology

A post-AI residual adaptive stage may be tested later if the experiment shows that it is more useful.

Do not change topology because it looks theoretically cleaner. Change it because a controlled experiment gives evidence.

---

# 7. SAMPLE-RATE POLICY

Do **not** hard-code 16 kHz merely because it is convenient.

The V1 sample rate must be compatible end-to-end with:

1. the selected AI model;
2. the audio device;
3. the evaluation metrics;
4. the streaming implementation;
5. the deployment runtime.

### Required procedure

1. Inspect the exact model/repository version.
2. Record supported sample rates.
3. Inspect the audio interface.
4. Test device/model compatibility.
5. Freeze the selected rate in the experiment configuration.
6. Avoid unnecessary resampling.
7. If a future narrowband communication/radio path is required, use a dedicated compatible model/path.

The earlier generic "16 kHz first" rule is therefore an option, not a universal DeepFilterNet requirement.

---

# 8. AI MODEL STRATEGY

## 8.1 Candidate families

| Model/family | Role |
|---|---|
| LMS | Classical educational baseline |
| NLMS | Practical adaptive baseline |
| RNNoise-style model | Lightweight neural reference |
| Compact CRN | Causal STFT-domain candidate |
| DCCRN | Complex-domain comparison |
| DeepFilterNet2/3 | Primary real-time candidate |
| DTLN | Lightweight fallback |
| Conv-TasNet | Low-latency time-domain comparison |
| FullSubNet / related sub-band models | Quality/reference comparison |

## 8.2 Current primary direction

**DeepFilterNet2/compatible DeepFilterNet path** is the leading candidate because its research was specifically directed toward efficient real-time speech enhancement and embedded feasibility.

The DeepFilterNet2 paper describes:

- STFT-domain processing;
- ERB-domain envelope enhancement;
- complex-domain deep filtering;
- reduced temporal-buffer requirements;
- efficient convolutional architecture;
- real-time operation on Raspberry Pi-class hardware.

These are feasibility references, not promises for our hardware.

## 8.3 Model selection rule

Do not select by reputation.

Select by:

\[
\text{quality} \;+\; \text{speech preservation} \;+\; \text{robustness} \;+\; \text{latency} \;+\; \text{memory} \;+\; \text{power}
\]

subject to the actual prototype constraints.

---

# 9. WHAT IS ACTUALLY NOVEL

## 9.1 Not novel by itself

Do not claim novelty for:

- DeepFilterNet itself;
- DCCRN itself;
- LMS/NLMS itself;
- STFT itself;
- a generic neural denoiser;
- a generic noise-cancelling headset;
- "AI noise cancellation" as a phrase.

## 9.2 Defensible system-level contribution

The stronger contribution is:

> **A defence-oriented, experimentally validated AI/DSP communication pipeline designed for highly dynamic and impulsive defence-like acoustic environments, combining defence-specific data construction, explicit noise-condition stratification, real-time speech enhancement, reference-aware adaptive processing where experimentally useful, runtime protection/fail-safe logic, unseen-noise validation, and measured embedded deployment.**

This claim is provisional.

The final claim must be updated after ablation.

If NLMS does not help, remove it from the novelty claim.

If the impulse controller does not help, narrow the claim.

If the chosen model fails under unseen noise, report that limitation.

---

# 10. DATASET ENGINEERING

## 10.1 Dataset philosophy

The dataset is not a side task.

It is one of the central engineering problems.

The model must encounter:

### Stationary noise

Examples:

- engine-like machinery;
- continuous vehicle noise;
- steady mechanical noise.

### Non-stationary noise

Examples:

- rotor-like noise;
- changing vehicle noise;
- sirens;
- wind;
- sudden noise transitions.

### Impulsive noise

Examples:

- gunshot-like events;
- shelling/artillery-like events;
- controlled short high-energy transients;
- synthetic impulses where appropriate.

### Mixed conditions

Examples:

- vehicle + wind;
- rotor + speech + impulse;
- machinery + siren;
- changing background + impulsive event.

---

# 11. DATA SOURCES

The research documents identify the following starter sources.

| Resource | Purpose |
|---|---|
| DNS Challenge | Large clean-speech/noise/RIR foundation |
| MAD Military Audio Dataset | Defence-specific military acoustic events |
| C3GD / gunshot-specific material | Controlled impulsive-event diversity |
| NOISEX-92 | Useful held-out/generalization noise reference |
| RIR datasets | Reverberation and room realism |

### Important

Record for every source:

- retrieval date;
- license/provenance;
- source URL;
- class;
- sample rate;
- duration;
- any preprocessing.

Do not mix data sources without recording provenance.

---

# 12. DATASET MIXING

For basic additive mixing:

\[
x[n]=s[n]+\alpha n[n]
\]

where \(\alpha\) is selected to achieve the target SNR.

For target SNR:

\[
SNR_{dB}
=
10\log_{10}
\left(
\frac{P_s}{P_n}
\right)
\]

The mixer must verify the **actual resulting SNR**, not assume that the requested value was achieved.

## 12.1 Suggested initial SNR coverage

Research planning considered ranges extending from very noisy conditions to clean conditions, with example targets such as:

- -10 dB;
- -5 dB;
- 0 dB;
- 5 dB;
- 10 dB;
- 15 dB;
- 20 dB.

The exact distribution must be frozen in the dataset configuration before final testing.

## 12.2 Impulse mixing

Do not treat a gunshot/impulse like a stationary noise file with a single global SNR.

Instead randomize:

- event onset;
- event amplitude;
- event duration;
- number of events;
- distance/acoustic level where simulation is appropriate;
- background condition;
- speech overlap.

---

# 13. DATA AUGMENTATION

Use controlled augmentation such as:

- gain changes;
- reverberation;
- RIR convolution;
- microphone response variation;
- random noise mixing;
- sudden onset/offset;
- clipping simulation;
- speed/pitch changes where appropriate;
- mixed-noise conditions.

The target speech should remain the clean reference where the experiment is intended to reconstruct undistorted speech.

---

# 14. DATA SPLIT AND LEAKAGE CONTROL

The project must prevent:

- speaker leakage;
- noise-recording leakage;
- room/RIR leakage where relevant;
- duplicate or near-duplicate clips across splits;
- training on final test noise.

Minimum principle:

```text
TRAIN
  speakers A...
  noise recordings A...
  rooms A...

VALIDATION
  different speakers/noise subset

TEST
  unseen speakers
  unseen recordings
  held-out noise
  stress/transition/impulse cases
```

A strong final test should contain genuinely unseen noise conditions.

---

# 15. DATASET FOLDER STRUCTURE

Recommended working structure:

```text
adaptive-anc/
├── README.md
├── LICENSES/
├── configs/
│   ├── audio.yaml
│   ├── dataset.yaml
│   ├── training.yaml
│   └── evaluation.yaml
│
├── data/
│   ├── raw/
│   │   ├── clean_speech/
│   │   ├── general_noise/
│   │   ├── defence_noise/
│   │   ├── impulses/
│   │   └── rir/
│   │
│   ├── processed/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   └── manifests/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── src/
│   ├── audio/
│   ├── dsp/
│   ├── adaptive/
│   ├── models/
│   ├── streaming/
│   ├── controller/
│   └── evaluation/
│
├── experiments/
│   ├── E001/
│   ├── E002/
│   └── ...
│
├── models/
│   ├── checkpoints/
│   ├── exported/
│   └── optimized/
│
├── logs/
│   ├── runtime/
│   ├── training/
│   └── hardware/
│
├── reports/
│   ├── figures/
│   ├── tables/
│   └── drafts/
│
└── docs/
    ├── decisions/
    ├── lab-notes/
    └── dataset-cards/
```

---

# 16. SIGNAL-PROCESSING FOUNDATION

The team must understand the signal path before relying on an AI model.

## 16.1 Sampling

Continuous audio becomes discrete samples:

\[
x[n]=x(nT_s)
\]

where:

\[
T_s=\frac{1}{f_s}
\]

Understand:

- sampling frequency;
- Nyquist frequency;
- aliasing;
- quantization;
- PCM;
- ADC;
- DAC;
- clipping;
- dynamic range.

## 16.2 dB

For power:

\[
dB=10\log_{10}\frac{P_2}{P_1}
\]

For amplitude ratios:

\[
dB=20\log_{10}\frac{A_2}{A_1}
\]

The team must be able to explain why these differ.

---

# 17. FFT AND STFT

## 17.1 FFT

FFT provides an efficient computation of the discrete Fourier transform.

The important engineering interpretation is:

- time domain → frequency representation;
- frequency bins represent spectral components;
- phase contains timing/phase information.

## 17.2 STFT

For streaming speech, use overlapping short-time frames.

A conceptual STFT form is:

\[
X(m,k)=
\sum_n
x[n]w[n-mH]e^{-j2\pi kn/N}
\]

where:

- \(w[n]\) = window;
- \(H\) = hop size;
- \(N\) = FFT size;
- \(m\) = frame index;
- \(k\) = frequency-bin index.

The team must understand:

- windowing;
- overlap;
- hop size;
- FFT size;
- magnitude;
- phase;
- reconstruction;
- iSTFT;
- NOLA/overlap-add conditions;
- algorithmic latency.

---

# 18. ADAPTIVE FILTERING

## 18.1 LMS baseline

LMS is the educational baseline.

Generic update:

\[
w(n+1)=w(n)+\mu e(n)x(n)
\]

where:

- \(w(n)\) = filter coefficient vector;
- \(\mu\) = step size;
- \(e(n)\) = error;
- \(x(n)\) = reference vector.

The team must understand:

- convergence;
- divergence;
- step-size effect;
- filter length;
- correlated reference;
- uncorrelated reference;
- speech leakage.

## 18.2 NLMS

Normalized LMS:

\[
w(n+1)=
w(n)+
\frac{\mu}{\epsilon+\|x(n)\|^2}
e(n)x(n)
\]

NLMS is the practical first adaptive-filter baseline because normalization reduces sensitivity to input-power variation.

## 18.3 What NLMS must prove

Do not assume NLMS improves the system.

Measure:

- convergence;
- residual error;
- ΔSNR;
- speech distortion;
- behaviour with reference leakage;
- behaviour under changing noise;
- behaviour under impulse events.

---

# 19. REFERENCE MICROPHONE

The reference microphone is not automatically a "clean noise microphone."

It should ideally:

- capture correlated environmental noise;
- contain less desired speech;
- remain synchronized with the primary path.

Problems include:

- speech leakage;
- reflections;
- microphone mismatch;
- phase/time offset;
- poor physical placement;
- low correlation with the primary noise.

Therefore measure:

- inter-channel delay;
- coherence/correlation;
- noise-only correlation;
- speech leakage;
- channel stability.

If the reference becomes unreliable, the controller should reduce/freeze adaptive cancellation.

---

# 20. IMPULSIVE-NOISE SUBSYSTEM

Impulsive events are short, broadband and high-energy.

Initial detector features may include:

- short-time energy;
- peak amplitude;
- crest factor;
- spectral energy;
- spectral change.

A small classifier can be considered later if the simple detector is insufficient.

## 20.1 Runtime modes

A useful initial state concept:

| Mode | Behaviour |
|---|---|
| Speech-dominant | Conservative suppression |
| Stationary | Stronger adaptive suppression |
| Dynamic | AI-dominant behaviour |
| Impulse/protection | Rapid response, conservative adaptation, recovery |

## 20.2 Hardware warning

If an impulse clips the microphone or ADC, the lost information cannot be reconstructed by AI.

Therefore measure:

- microphone headroom;
- gain;
- ADC range;
- clipping;
- limiter behaviour;
- overload recovery.

---

# 21. AI TRAINING STRATEGY

## 21.1 Recommended progression

Do not immediately build the most complex model.

Progress:

1. classical baseline;
2. simple AI baseline;
3. pretrained DeepFilterNet benchmark;
4. controlled fine-tuning;
5. defence-specific augmentation;
6. hybrid integration;
7. ablation;
8. embedded optimization.

## 21.2 Representation

Primary candidates:

- STFT magnitude/mask;
- complex spectral representation;
- time-domain waveform.

The current leading direction is efficient STFT/complex-aware enhancement.

## 21.3 Losses

The SIH statement names:

- SI-SNR;
- L1;
- L2;
- perceptual loss.

Possible training combinations may include:

\[
L =
\lambda_1L_{SI-SNR}
+
\lambda_2L_{spectral}
+
\lambda_3L_{perceptual}
\]

Weights must be experimentally justified.

Do not select weights because they "look standard."

## 21.4 Important caution

PESQ/STOI are evaluation metrics and are not automatically appropriate as naïve direct training losses.

Perceptual proxies/differentiable approximations may be investigated, but subjective listening must remain part of validation.

---

# 22. EVALUATION FRAMEWORK

## 22.1 Core metrics

| Metric | Purpose |
|---|---|
| Input SNR | Defines starting condition |
| Output SNR | Noise-suppression result |
| ΔSNR | Improvement |
| SI-SNR / SI-SDR | Reconstruction quality |
| STOI | Speech intelligibility |
| PESQ | Speech quality; explicitly requested by PS |
| POLQA | Optional modern speech-quality complement |
| WER / word accuracy | Communication usefulness |
| Latency | Real-time feasibility |
| CPU/GPU | Compute burden |
| RAM | Memory burden |
| Model size | Deployment burden |
| Power | Portable/embedded feasibility |
| Failure rate | Robustness |
| Subjective listening | Human-perceived artifacts |

## 22.2 Required reporting

Always report:

```text
Scenario
Input SNR
Output SNR
ΔSNR
STOI
PESQ
SI-SNR/SI-SDR
Latency
CPU/GPU
RAM
Power
Failure notes
```

Never report only an overall average.

---

# 23. TEST MATRIX

| Scenario | Example | Required measurements |
|---|---|---|
| Stationary | Engine/machinery | SNR, STOI, PESQ |
| Non-stationary | Rotor/siren/vehicle | STOI, PESQ, latency |
| Impulsive | Short high-energy event | Recovery time, STOI, SI-SDR |
| Transition | Noise changes during speech | Adaptation time, intelligibility |
| Mixed | Vehicle + wind + impulse | Full metric set |
| Unseen | Held-out noise | Generalization |
| Resource stress | Worst-case streaming | Latency, CPU, RAM, power |

---

# 24. MANDATORY BASELINE/ABLATION

The hybrid system must earn its complexity.

At minimum compare:

| ID | System |
|---|---|
| A | Unprocessed noisy speech |
| B | Spectral subtraction |
| C | LMS |
| D | NLMS |
| E | AI-only |
| F | AI + adaptive stage |
| G | AI + controller/impulse handling |
| H | Final V1 |

The final report should show not merely which system has the highest score, but **why each block exists**.

---

# 25. MANDATORY ABLATION TABLE

Maintain a table like:

| Configuration | SNR | ΔSNR | STOI | PESQ | SI-SDR | Latency | CPU | RAM | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Noisy input | — | — | — | — | — | — | — | — | Baseline |
| Spectral subtraction | — | — | — | — | — | — | — | — | |
| NLMS | — | — | — | — | — | — | — | — | |
| AI only | — | — | — | — | — | — | — | — | |
| AI + NLMS | — | — | — | — | — | — | — | — | |
| AI + controller | — | — | — | — | — | — | — | — | |
| Final V1 | — | — | — | — | — | — | — | — | |

### Rule

No cherry-picking.

Use the same test protocol for all configurations.

---

# 26. REAL-TIME ENGINEERING

The actual latency is:

\[
L_{total}
=
L_{ADC}
+
L_{buffer}
+
L_{algorithmic}
+
L_{inference}
+
L_{adaptive}
+
L_{reconstruction}
+
L_{DAC}
\]

A low neural-network inference time does **not** prove low end-to-end latency.

## 26.1 Measure

- P50;
- P95;
- P99;
- maximum;
- underruns;
- overruns;
- dropped frames;
- CPU;
- RAM;
- GPU where relevant;
- power;
- temperature.

## 26.2 Streaming rules

Use:

- causal/near-causal processing;
- streaming state;
- preallocated buffers;
- stable frame scheduling;
- timestamps;
- deterministic logging.

Do not use future audio without counting its look-ahead as latency.

---

# 27. SOFTWARE STACK

The exact stack is allowed to evolve, but the current reference path is:

### Development

- Python;
- PyTorch;
- NumPy;
- SciPy;
- audio I/O library appropriate to the platform;
- Git;
- experiment configuration files.

### Model/runtime

- PyTorch during development;
- ONNX where export is validated;
- ONNX Runtime for portable inference testing;
- TensorRT where NVIDIA deployment is justified and validated.

### Research/reference repositories

- DeepFilterNet;
- DNS Challenge;
- MAD;
- PyTorch documentation;
- ONNX Runtime;
- NVIDIA TensorRT.

---

# 28. HARDWARE STRATEGY

## 28.1 Cost discipline

Do not start by purchasing the most expensive hardware.

The correct ladder is:

```text
Existing laptop
      ↓
Offline algorithm validation
      ↓
Laptop streaming / HIL proof
      ↓
Low-cost dual-mic audio prototype
      ↓
Measured compute requirement
      ↓
Smallest suitable edge target
      ↓
Optimization
      ↓
Physical demonstrator
```

## 28.2 Low-cost PoC

| Layer | Low-cost PoC | Deployment path |
|---|---|---|
| Acquisition | 2 × synchronized/I²S MEMS microphones or suitable audio interface | Higher-quality synchronized mic/codec |
| Compute | Existing laptop | Jetson/ARM/DSP selected from measurement |
| Adaptive DSP | CPU implementation | Optimized target implementation |
| AI | Laptop PyTorch | ONNX/TensorRT/other validated runtime |
| Output | Consumer headset | Communication headset/interface |
| Enclosure | Simple prototype enclosure | Rugged enclosure |
| Test rig | Small speaker + fixed geometry | Representative acoustic environment |

## 28.3 Jetson

The SIH statement names NVIDIA Jetson AGX Orin 64 GB or similar hardware.

That does **not** mean the team must immediately buy one.

Use it when the measured model/streaming requirements justify it.

---

# 29. AUDIO HARDWARE REQUIREMENTS

Characterize:

- microphone SNR;
- sensitivity;
- frequency response;
- dynamic range;
- overload;
- ADC headroom;
- gain;
- synchronization;
- channel identity;
- latency.

The physical prototype should eventually contain:

```text
Primary microphone
Reference microphone
Audio interface / ADC / I²S
Embedded processor
DAC / audio codec
Headset
Power system
Telemetry/debug interface
Enclosure
```

---

# 30. CONTROLLER AND FAIL-SAFE

A mission-critical communication system should not assume the AI is always healthy.

## 30.1 State concept

```text
             AI healthy
                 │
                 ▼
        AI enhancement active
                 │
        confidence decreases
                 ▼
      conservative enhancement
                 │
       AI inference failure
                 ▼
          DSP fallback
                 │
       hardware/audio fault
                 ▼
       safe bypass / comms
```

Additional conditions:

- clipping → protection;
- reference unreliable → reduce/freeze adaptive cancellation;
- impulse detected → protected mode;
- runtime overload → degrade safely;
- output fault → transparent bypass where possible.

## 30.2 Logging

Every runtime event should record:

- timestamp;
- state;
- detector status;
- confidence;
- clipping;
- latency;
- CPU/GPU;
- reference health;
- fallback events.

---

# 31. EMBEDDED DEPLOYMENT

## 31.1 Deployment ladder

```text
PyTorch model
     ↓
Offline validation
     ↓
Export
     ↓
ONNX / compatible runtime
     ↓
FP32 benchmark
     ↓
FP16 benchmark
     ↓
INT8 if justified
     ↓
Edge profiling
     ↓
Streaming hardware test
```

## 31.2 Optimization options

Only after profiling:

- quantization;
- pruning;
- reduced context;
- sub-band processing;
- smaller architecture;
- optimized operators;
- buffer optimization;
- memory reuse.

Never optimize blindly.

---

# 32. PHYSICAL TEST RIG

Build a repeatable acoustic setup.

Control:

- source geometry;
- speech source;
- noise source;
- microphone position;
- headset position;
- room;
- source level;
- test sequence.

The final demo should not depend on an uncontrolled random environment.

---

# 33. GRAND DEMO SEQUENCE

A proposed sequence:

1. raw speech + engine-like noise;
2. enhanced speech;
3. changing rotor/siren-like noise;
4. impulsive event;
5. mixed noise;
6. show live output;
7. show waveform/spectrogram if useful;
8. show SNR/STOI/PESQ results;
9. show latency;
10. show CPU/RAM;
11. show physical microphone/headset/processor.

The dashboard supports the hardware.

The dashboard is not the product.

---

# 34. LEARNING CURRICULUM

The six-member engineering team must collectively understand:

| Area | Must know |
|---|---|
| Digital audio | Sampling, PCM, dB, ADC/DAC, dynamic range |
| DSP | FFT, STFT, windows, filters, phase |
| Adaptive filtering | LMS, NLMS, convergence, step size |
| Speech processing | Spectrum, voiced/unvoiced, harmonics |
| AI/ML | CNN/RNN/LSTM/CRN, masks, causal inference |
| Data engineering | SNR mixing, RIR, augmentation, leakage |
| Evaluation | SNR, SI-SNR, STOI, PESQ, WER |
| Real-time | Buffering, causality, latency |
| Embedded AI | ONNX, TensorRT, quantization, profiling |
| Hardware | MEMS mic, I²S, DAC, codec, power |

---

# 35. LEARNING GATES

## G0 — Audio foundations

Must explain:

- sampling;
- Nyquist;
- PCM;
- dB;
- clipping;
- ADC/DAC.

Artifact:

- waveform inspection;
- sample-rate test.

## G1 — STFT

Must implement:

- framing;
- window;
- FFT;
- STFT;
- iSTFT.

Artifact:

- reconstruction test.

## G2 — LMS/NLMS

Must implement:

- LMS;
- NLMS;
- convergence tests.

Artifact:

- convergence plot;
- residual comparison.

## G3 — Audio acquisition

Must capture:

- primary;
- reference;
- synchronized channels.

Artifact:

- dual-channel recording.

## G4 — Streaming

Must achieve:

- stable frame scheduling;
- no persistent underruns;
- measured latency.

Artifact:

- streaming log.

## G5 — Dataset

Must produce:

- clean/noisy pairs;
- verified SNR;
- metadata;
- leakage-safe splits.

Artifact:

- manifest.

## G6 — AI baseline

Must run:

- one lightweight AI model;
- reproducible inference.

Artifact:

- first AI metric table.

## G7 — Hybrid

Must compare:

- AI-only;
- AI + adaptive stage.

Artifact:

- ablation.

## G8 — Embedded

Must measure:

- latency;
- RAM;
- CPU/GPU;
- power.

Artifact:

- hardware benchmark.

---

# 36. FIRST-WEEK EXECUTION PLAN

## Day 1 — Environment + audio inspection

Tasks:

- install Python;
- install Git;
- create repository;
- create environment;
- run signal inspection;
- inspect a WAV file;
- read the core project contract.

Gate:

> Environment is reproducible and the team has inspected an actual waveform.

---

## Day 2 — SNR + first mixtures

Tasks:

- implement SNR;
- generate approximately 20 clean/noisy pairs;
- verify requested vs actual SNR;
- begin evaluation CLI.

Gate:

> 20 valid mixtures and a metric sanity check exist.

---

## Day 3 — Evaluation + classical baseline

Tasks:

- finish evaluation harness;
- run spectral subtraction;
- create `results.csv`;
- start STFT lab.

Gate:

> At least one baseline number exists before AI work.

---

## Day 4 — STFT/iSTFT + passthrough

Tasks:

- complete STFT/iSTFT;
- verify reconstruction;
- check overlap/add/NOLA conditions;
- if hardware exists, test microphone → output passthrough.

Gate:

> Near-perfect reconstruction and a basic audio path.

---

## Day 5 — LMS/NLMS

Tasks:

- implement LMS;
- implement NLMS;
- test convergence/divergence;
- vary step size;
- vary filter length;
- log results.

Gate:

> NLMS convergence evidence exists.

---

## Day 6 — Two-microphone acquisition

Tasks:

- acquire primary/reference channels;
- verify channel identity;
- check synchronization;
- estimate correlation/coherence;
- record noise-only and speech+noise conditions.

Gate:

> Stable dual-channel recording for at least 60 seconds.

---

## Day 7 — Review + freeze

Tasks:

- run all gates;
- record decisions;
- list unresolved issues;
- decide whether AI integration is justified.

Gate:

> Week-1 gate signed with no unresolved P0 issue.

---

# 37. 30-DAY EXECUTION PLAN

| Days | Must be true |
|---|---|
| 1–3 | Environment, repository, audio inspection, first mixtures, metric sanity |
| 4–7 | STFT/iSTFT, spectral baseline, streaming passthrough, two-mic capture |
| 8–10 | LMS/NLMS + reference-quality experiments |
| 11–14 | Tiny neural model + DeepFilterNet benchmark |
| 15–18 | Fine-tuning/data pipeline + automated evaluation |
| 19–22 | Streaming AI + controller + impulse detector |
| 23–25 | Hybrid ablation + physical acoustic rig |
| 26–28 | Embedded deployment/profiling if hardware is justified/available |
| 29–30 | Acceptance tests, demo freeze, video, report, judge rehearsal |

---

# 38. 60/90-DAY EXTENSION

## Days 31–60

- stabilize AI baseline;
- benchmark candidate models;
- improve defence-noise augmentation;
- add impulsive training;
- establish unseen-noise test set;
- complete hybrid ablation;
- improve streaming stability;
- begin embedded export.

## Days 61–90

- edge optimization;
- quantization;
- microphone/codec integration;
- power measurement;
- physical acoustic rig;
- enclosure;
- robustness testing;
- final evidence package.

These are extension targets, not permission for scope drift.

---

# 39. KILL / PIVOT CRITERIA

| Gate | Kill criterion | Action |
|---|---|---|
| End Day 3 | No reproducible audio/metric environment | Fix environment before AI |
| End Day 7 | No stable streaming path | Stop model work; debug audio I/O |
| End Day 10 | NLMS cannot converge on synthetic tests | Do not integrate; return to algorithm lab |
| End Day 14 | AI does not beat noisy baseline on held-out test | Change data/model before hardware expansion |
| Before embedded | Runtime far outside latency budget | Switch to smaller causal model |
| Before demo freeze | NLMS provides no consistent gain | Remove it from core contribution |
| Before final demo | Impulse controller harms speech | Use conservative bypass/protection and narrow claim |

---

# 40. ACCEPTANCE CRITERIA

V1 should not be called complete until it can demonstrate:

### Functional

- stationary-noise scenario;
- non-stationary scenario;
- impulsive scenario;
- mixed scenario;
- live microphone input;
- headset/output path.

### Quantitative

- SNR target assessed;
- STOI target assessed;
- PESQ target assessed;
- latency measured;
- CPU/RAM measured;
- model size recorded;
- power measured where possible.

### Experimental

- noisy baseline;
- classical baseline;
- AI-only;
- hybrid;
- ablation;
- unseen-noise evaluation.

### Engineering

- streaming;
- no persistent underruns;
- runtime logging;
- fail-safe/fallback;
- reproducible configuration.

### Demonstration

- repeatable acoustic setup;
- live demo;
- measured evidence;
- known limitation;
- clear next step.

---

# 41. EVIDENCE FRAMEWORK

Every important statement must receive one of three labels.

## VERIFIED RESEARCH

Supported by a primary paper, official documentation or other high-confidence source.

Example:

> DeepFilterNet2 demonstrated real-time operation under the conditions stated in its paper.

This proves the paper's result, not ours.

## ENGINEERING TARGET

A target we have chosen.

Examples:

- end-to-end latency target;
- SNR target;
- model-size target.

Targets are not achievements.

## HYPOTHESIS

A statement we need to test.

Examples:

- AI + NLMS will outperform AI-only;
- impulse detection will improve recovery;
- a particular microphone placement will improve cancellation.

The experiment decides.

---

# 42. SOURCE-OF-TRUTH HIERARCHY

When sources disagree, use:

1. current official SIH/DRDO problem statement;
2. official standards/manufacturer documentation;
3. peer-reviewed primary research;
4. official repositories/reference implementations;
5. secondary articles/tutorials;
6. team assumptions and engineering targets.

Never silently replace the problem statement with a paper.

---

# 43. DOCUMENT / DATA / CODE GOVERNANCE

## Master document

This file is the planning/reference baseline until explicitly revised.

## Experiment records

Store separately.

Never overwrite old results.

## Architecture changes

Record:

- decision;
- date;
- alternatives;
- evidence;
- owner;
- reason;
- affected experiments;
- revisit condition.

## Code truth

Track:

- Git commit;
- environment;
- configuration;
- dataset version.

## Model truth

Track:

- checkpoint;
- model version;
- preprocessing;
- postprocessing;
- configuration.

## Dataset truth

Track:

- source;
- provenance;
- retrieval date;
- mixing parameters;
- split;
- manifest.

## Measured results

Only documented experiments count.

---

# 44. EXPERIMENT RECORD TEMPLATE

Create one record per experiment.

```text
Experiment ID:
Date:
Owner:
Objective:

Hypothesis:

Configuration:
- model:
- sample rate:
- FFT:
- hop:
- filter:
- controller:
- dataset version:

Input conditions:
- noise:
- input SNR:
- duration:
- microphone setup:

Measurements:
- output SNR:
- ΔSNR:
- STOI:
- PESQ:
- SI-SNR/SI-SDR:
- latency P50:
- latency P95:
- latency P99:
- CPU:
- RAM:
- power:

Observed failures:

Interpretation:

Decision:
- keep
- modify
- reject
- defer

Next experiment:
```

---

# 45. DECISION LOG TEMPLATE

| ID | Date | Decision | Alternatives | Evidence | Owner | Revisit condition |
|---|---|---|---|---|---|---|
| D001 | | | | | | |
| D002 | | | | | | |
| D003 | | | | | | |

Do not make important architecture decisions only in chat.

---

# 46. DATASET CARD TEMPLATE

```text
Dataset version:
Date:
Owner:

Sources:
Licenses:
Retrieval dates:

Clean speech:
Noise:
Defence noise:
Impulse data:
RIR:

Sample rate:
SNR distribution:
Augmentation:

Train speakers:
Validation speakers:
Test speakers:

Held-out noise:
Leakage checks:

Manifest location:

Known limitations:
```

---

# 47. MODEL CARD TEMPLATE

```text
Model name:
Version:
Commit:
Checkpoint:

Architecture:
Sample rate:
Frame size:
Hop:
Look-ahead:

Parameters:
Model size:

Training dataset:
Validation dataset:
Loss:

Inference runtime:
CPU/GPU:
RAM:

Measured metrics:

Known failure cases:
```

---

# 48. HARDWARE TEST RECORD

```text
Device:
Firmware/OS:
Audio interface:
Microphone model:
Reference microphone:
Headset:

Sample rate:
Bit depth:
Buffer size:

Gain:
Input level:
Clipping observed:

P50 latency:
P95 latency:
P99 latency:
Maximum latency:

CPU:
RAM:
GPU:
Power:
Temperature:

Underruns:
Overruns:
Dropped frames:

Notes:
```

---

# 49. FAILURE-DEBUGGING TREE

## No output

Check:

1. audio device;
2. sample rate;
3. channel count;
4. permissions;
5. buffer;
6. process state.

## Output is distorted

Check:

1. clipping;
2. gain;
3. normalization;
4. iSTFT reconstruction;
5. model output range;
6. DAC.

## Speech disappears

Check:

1. mask strength;
2. loss;
3. reference leakage;
4. NLMS adaptation;
5. controller thresholds;
6. training data.

## Noise remains

Check:

1. input SNR;
2. noise type;
3. reference correlation;
4. model generalization;
5. adaptive filter convergence;
6. latency.

## Live system glitches

Check:

1. buffer size;
2. scheduling;
3. inference time;
4. memory allocation;
5. CPU contention;
6. device driver;
7. synchronization.

---

# 50. AI USE RULE

AI tools may help with:

- explaining equations;
- debugging;
- generating boilerplate;
- checking syntax;
- suggesting experiments;
- organizing documentation.

The team must still understand:

- the signal flow;
- the equations;
- the code;
- the data;
- the metrics;
- the measurements;
- the failure modes.

A student who cannot explain the model or reproduce the experiment cannot credibly defend the system.

---

# 51. TEAM STRUCTURE — SIX MEMBERS

| Role | Ownership |
|---|---|
| DSP lead | STFT, adaptive filtering, signal chain, latency |
| AI/ML lead | Model, training, loss, quantization |
| Embedded lead | Streaming, buffering, deployment |
| Audio hardware lead | Microphones, ADC/DAC, headset, power |
| Validation lead | Dataset, metrics, experiments |
| Systems/demo lead | Integration, telemetry, fail-safe, demo, report |

AI/ML and Embedded should overlap heavily because model export → runtime → hardware is a high-risk boundary.

---

# 52. REQUIREMENT TRACEABILITY

| PS requirement | Subsystem | Test/evidence |
|---|---|---|
| Stationary noise | AI + adaptive stage | Engine/machinery test |
| Non-stationary noise | Causal AI + controller | Rotor/siren/vehicle test |
| Impulsive noise | Impulse handling + protected adaptation | Controlled transient test |
| Dataset pipeline | Mixer + metadata | Manifest + SNR verification |
| AI suppression | Selected model | Checkpoint + metrics |
| Adaptive filtering | NLMS | Reference-mic experiment |
| Low latency | Streaming engine | E2E latency |
| Embedded | Edge runtime | CPU/RAM/power |
| Primary/reference mic | Audio hardware | Channel integrity/coherence |
| Headset | End-to-end output | Live demo |
| Speech quality | Evaluation | STOI/PESQ/SI-SNR |
| Robustness | Held-out tests | Unseen-noise results |

---

# 53. SIH DEFENCE PACKAGE

The final submission should eventually contain:

1. problem definition;
2. motivation;
3. technical gap;
4. architecture;
5. dataset pipeline;
6. training pipeline;
7. adaptive filtering;
8. impulse handling;
9. real-time design;
10. embedded deployment;
11. hardware;
12. evaluation protocol;
13. results;
14. ablation;
15. limitations;
16. cost;
17. future work;
18. demo evidence.

---

# 54. THREE-MINUTE DEMO

### 0:00–0:20

State the problem:

> Speech communication is degraded by stationary, changing and impulsive defence-like noise.

### 0:20–0:45

Show:

- primary microphone;
- reference microphone;
- processor;
- headset.

Explain their roles.

### 0:45–1:15

Demonstrate:

- noisy speech;
- AI-enhanced speech.

### 1:15–1:45

Demonstrate:

- hybrid path;
- response to changing/impulsive noise.

### 1:45–2:15

Show:

- SNR;
- STOI;
- PESQ;
- baseline comparison.

### 2:15–2:40

Show:

- live latency;
- CPU/RAM;
- edge hardware.

### 2:40–3:00

State:

- cost;
- strongest measured result;
- strongest limitation;
- next deployment step.

---

# 55. JUDGE QUESTIONS

## Q1. Why do you call this ANC?

Answer:

> The PS uses the ANC terminology, but our V1 is specifically a real-time communication enhancement system combining AI speech enhancement with a reference-microphone adaptive noise-cancellation stage. Full physical broadband ANC is a separate secondary-path control problem and is outside the V1 claim unless experimentally implemented.

## Q2. What does AI do that NLMS cannot?

Answer:

> NLMS adapts to correlated reference noise but does not inherently model complex nonlinear speech/noise structure. The AI model estimates a learned speech-preserving enhancement representation. We prove the division of labour through the AI-only versus AI+NLMS ablation.

## Q3. Why not just use DeepFilterNet?

Answer:

> DeepFilterNet is our strong starting/reference model, not our novelty. The project contribution is the defence-oriented dataset, impulsive and rapidly changing conditions, adaptive/reference processing, runtime control, unseen-noise validation and measured embedded deployment.

## Q4. Why two microphones?

Answer:

> The primary microphone contains speech plus noise. A suitably placed reference microphone can capture correlated environmental noise with less desired speech, giving the adaptive stage additional information. We explicitly measure reference leakage and coherence rather than assuming the reference is perfect.

## Q5. What happens during an impulse?

Answer:

> The runtime controller detects transient conditions and moves into a conservative/protection mode, limiting unsafe adaptive behaviour and prioritizing speech preservation and rapid recovery.

## Q6. What is your latency?

Never answer from a paper.

Answer with your measured:

- P50;
- P95;
- P99;
- maximum;
- end-to-end measurement method.

## Q7. How did you avoid leakage?

Answer:

> Speakers, noise recordings and relevant environmental conditions are separated between train/validation/test. Held-out noise is used to test generalization.

## Q8. What is your strongest baseline?

Answer using the actual ablation table.

## Q9. What is your biggest failure?

Have one.

A credible system has limitations.

## Q10. Is this military-ready?

Correct answer:

> No. This is a student research prototype validating the architecture and measurable engineering principles. A fielded system would require qualification, environmental testing, EMC, safety, communications integration, human-factor validation and appropriate ruggedization.

---

# 56. WHAT NOT TO DO

Do not:

- start with a giant model;
- buy expensive hardware before measuring compute;
- build a custom PCB before the signal chain works;
- build a dashboard before the audio works;
- evaluate only stationary noise;
- use only one weak baseline;
- train and test on overlapping noise;
- report literature scores as project scores;
- call laptop inference "embedded real time";
- claim physical ANC without a physical acoustic control loop;
- add transformers/beamforming/multiple microphones just because they sound advanced;
- optimize one metric while speech quality collapses;
- hide failure cases;
- repeatedly redesign the architecture because of new papers.

---

# 57. SCOPE FREEZE — IN

## V1 IN

- primary microphone;
- reference microphone;
- model-compatible audio rate;
- STFT/appropriate feature path;
- reference-mic NLMS configuration;
- lightweight AI enhancement;
- runtime controller;
- impulse detection/protection;
- headset output;
- offline evaluation;
- streaming evaluation;
- controlled acoustic test;
- embedded profiling when justified;
- fail-safe/fallback;
- ablation.

---

# 58. SCOPE FREEZE — OUT / DEFERRED

Unless a measured failure creates a reason to add them:

- full physical broadband ANC;
- custom PCB;
- large transformer architecture;
- unnecessary beamforming;
- many microphones;
- military codec integration;
- certified hearing protection;
- ruggedized field product;
- production-grade EMC qualification;
- environmental qualification;
- full radio integration.

These are not forbidden forever.

They are **not allowed to silently expand V1**.

---

# 59. PHASE-2 PARKING LOT

Possible future work:

1. physical FxLMS;
2. secondary-path modelling;
3. additional microphones;
4. beamforming;
5. multimodal sensing;
6. military codec integration;
7. advanced impulse classifier;
8. larger/sub-band architectures;
9. quantization/pruning research;
10. rugged enclosure;
11. field-representative vehicle/aircraft testing;
12. communications-unit integration.

Each item requires a reason:

> **What unresolved requirement or measured failure does this solve?**

If there is no answer, it stays parked.

---

# 60. CURRENT PROJECT DASHBOARD

## Completed

- [x] Problem statement captured
- [x] Requirements translated into engineering obligations
- [x] ANC vs speech-enhancement terminology boundary
- [x] Architecture candidates researched
- [x] Primary AI direction identified
- [x] Dataset strategy defined
- [x] Evaluation framework defined
- [x] Embedded strategy defined
- [x] Hybrid hypothesis defined
- [x] Fail-safe concept defined
- [x] Learning curriculum defined
- [x] First-week execution plan defined
- [x] 30/60/90 roadmap defined
- [x] Acceptance criteria defined
- [x] Judge-question framework defined
- [x] Scope freeze established
- [x] Evidence hierarchy established

## Not yet proven

- [ ] Dataset actually generated and validated
- [ ] SNR pipeline verified
- [ ] STFT/iSTFT implementation verified
- [ ] LMS convergence verified
- [ ] NLMS convergence verified
- [ ] Two-mic hardware verified
- [ ] Reference correlation measured
- [ ] Streaming path verified
- [ ] AI baseline benchmarked
- [ ] Defence-noise fine-tuning completed
- [ ] AI vs NLMS ablation completed
- [ ] Impulse detector validated
- [ ] End-to-end latency measured
- [ ] Embedded runtime measured
- [ ] SNR target achieved
- [ ] STOI target achieved
- [ ] PESQ target achieved
- [ ] Final novelty claim earned
- [ ] Final demo frozen

---

# 61. THE PROJECT'S NEXT STEP — EXACTLY

Do not start another research pass.

Start **G0/G1**.

### Immediate objective

Build the first verified audio/DSP artifact:

```text
WAV / microphone
      ↓
sampling inspection
      ↓
SNR calculation
      ↓
noise mixing
      ↓
STFT
      ↓
iSTFT
      ↓
reconstructed audio
      ↓
verification
```

### First concrete outputs

By the end of the first implementation block, the repository should contain:

```text
01_signal_inspection
02_snr_mixer
03_stft_istft
04_spectral_baseline
05_lms
06_nlms
07_evaluation
```

Then proceed to:

```text
08_dual_mic_capture
09_streaming_passthrough
10_ai_baseline
11_hybrid
12_controller
13_embedded
14_final_ablation
```

---

# 62. THE OPERATING LOOP FROM NOW ON

Every work session should end with at least one of:

- a working experiment;
- a measured number;
- a debug finding;
- a design decision;
- a validated artifact;
- a rejected hypothesis;
- a reproducible failure.

If a session produces none of these, it is probably research drift.

The permanent workflow is:

```text
LEARN
  ↓
UNDERSTAND
  ↓
IMPLEMENT
  ↓
TEST
  ↓
MEASURE
  ↓
DEBUG
  ↓
DECIDE
  ↓
FREEZE
  ↓
INTEGRATE
  ↓
VALIDATE
  ↓
DEFEND
```

---

# 63. FINAL ENGINEERING POSITION

We started with a broad SIH problem statement.

We now have:

- a defined engineering interpretation;
- a frozen V1 architecture;
- a model-selection strategy;
- a dataset strategy;
- an evaluation protocol;
- a learning curriculum;
- a hardware/deployment ladder;
- a runtime safety concept;
- a test matrix;
- an ablation plan;
- a first-week execution plan;
- a 30-day build plan;
- a scope-control mechanism;
- a judge-defence framework.

But we do **not** yet have experimental proof of the final performance.

That is intentional.

The project is now at the point where **evidence must replace discussion**.

---

# 64. FINAL NON-NEGOTIABLE RULES

1. **Never confuse literature performance with project performance.**
2. **Never claim a hypothesis as a result.**
3. **Never change V1 architecture without evidence.**
4. **Never hide an important failure.**
5. **Never sacrifice speech intelligibility for a pretty noise-suppression number.**
6. **Never call offline inference real-time.**
7. **Never call a software denoiser physical ANC.**
8. **Never buy hardware before the measured requirement justifies it.**
9. **Never let a new paper automatically expand scope.**
10. **Every major claim must have a measurement or a source.**
11. **Every architecture change must enter the decision log.**
12. **Every final metric must be reproducible.**
13. **The physical demo must support the measured engineering evidence.**
14. **The final novelty claim must be written after the ablation, not before it.**
15. **The project moves forward by building and measuring, not by accumulating pages.**

---

# 65. RESOURCE REGISTER

## DeepFilterNet

Repository:

https://github.com/Rikorose/DeepFilterNet

DeepFilterNet2 paper:

https://arxiv.org/abs/2205.05474

## DNS Challenge

https://github.com/microsoft/DNS-Challenge

## Military Audio Dataset

https://github.com/kaen2891/military_audio_dataset

MAD Figshare collection:

https://doi.org/10.6084/m9.figshare.c.7001919

## C3GD descriptor

https://arxiv.org/abs/2606.18135

## PyTorch

https://docs.pytorch.org/tutorials/

## ONNX Runtime

https://onnxruntime.ai/

## NVIDIA TensorRT

https://docs.nvidia.com/deeplearning/tensorrt/

## SciPy

https://docs.scipy.org/doc/scipy/

---

# 66. RESEARCH REFERENCES ALREADY USED IN THE PROJECT

Key reference families include:

- DeepFilterNet / DeepFilterNet2;
- DCCRN;
- Conv-TasNet;
- DTLN;
- FullSubNet / sub-band enhancement;
- RNNoise;
- Microsoft DNS Challenge;
- Military Audio Dataset;
- gunshot-specific corpora;
- NOISEX-92;
- STOI;
- PESQ / ITU-T P.862;
- POLQA / ITU-T P.863;
- SI-SNR / SI-SDR;
- perceptual speech-enhancement losses;
- embedded speech-enhancement literature;
- real-time audio/embedded deployment literature.

Use primary sources for numerical claims whenever possible.

---

# 67. VERSION CONTROL

**Current compiled master:** `PS_26052_ADAPTIVE_DEFENCE_ANC_MASTER.md`

**Status:** Execution baseline

**Date:** 24 August 2026

**Next revision rule:**

A revision is justified when:

- a major experiment changes the architecture;
- a measured failure forces a redesign;
- a new requirement appears;
- a hardware constraint changes;
- a validated result changes the novelty claim.

When revising:

```text
Version
Date
Change
Reason
Evidence
Affected experiments
Owner
Approval
```

Do not silently overwrite the history.

---

# 68. APPENDIX — QUICK ONE-PAGE TEAM CHECKLIST

## Before coding

- [ ] Read PS contract
- [ ] Read V1 scope
- [ ] Understand terminology boundary
- [ ] Know current evidence status

## Before AI

- [ ] SNR mixer works
- [ ] STFT/iSTFT works
- [ ] Evaluation harness works
- [ ] Classical baseline exists

## Before hybrid

- [ ] LMS works
- [ ] NLMS works
- [ ] Reference channel verified
- [ ] AI baseline measured

## Before hardware

- [ ] Streaming works
- [ ] Latency measured
- [ ] Model runtime profiled
- [ ] Hardware requirement known

## Before final demo

- [ ] Ablation complete
- [ ] Unseen-noise test complete
- [ ] Impulse test complete
- [ ] Latency measured
- [ ] CPU/RAM/power measured
- [ ] Known failure documented
- [ ] Demo repeatable
- [ ] Judge answers rehearsed

---

# END OF MASTER DOCUMENT

## Current instruction to the team

> **Stop expanding the research scope. Start executing G0/G1.**
>
> The next meaningful project milestone is not another architecture discussion.
>
> It is the first reproducible audio/DSP experiment.



---

# 69. V6 COMPLETION AUDIT — WHAT WAS MISSING AND IS NOW ADDED

This section exists because a "complete reference document" must distinguish three different kinds of completeness:

1. **Knowledge completeness** — what the team needs to learn.
2. **Build-plan completeness** — how the team will turn that knowledge into software/hardware.
3. **Evidence completeness** — what must actually be measured before the project can claim success.

The project can be complete in categories 1 and 2 while category 3 is necessarily unfinished until the team runs experiments.

The earlier master had most of the research and planning, but the following implementation details needed to be made explicit:

- exact starter software environment;
- exact experiment IDs and dependency order;
- concrete dataset acquisition/provenance workflow;
- dataset generator specification;
- evaluation-harness specification;
- signal alignment rules;
- explicit metric-validity rules;
- exact NLMS experiment contract;
- reference-microphone calibration experiment;
- impulse-library construction protocol;
- controller input definitions and initial calibration procedure;
- streaming architecture;
- real-time latency measurement method;
- hardware selection decision tree;
- low-cost prototype bill of materials categories;
- wiring and electrical checks;
- embedded bring-up sequence;
- model-selection benchmark matrix;
- pretrained → fine-tune → export sequence;
- unit/integration/system-test hierarchy;
- failure-injection plan;
- reproducibility protocol;
- security/safety/data-provenance controls;
- acceptance-test procedure;
- final evidence-package structure;
- explicit "do not claim" list;
- continuation/abandonment gates.

These are now part of the V6 master.

---

# 70. THE MASTER STATUS — IMPORTANT DISTINCTION

## 70.1 Planning/reference status

The following are now fully specified:

- [x] PS requirements
- [x] system interpretation
- [x] V1 architecture
- [x] terminology boundary
- [x] model-selection strategy
- [x] dataset strategy
- [x] data-mixing mathematics
- [x] leakage controls
- [x] DSP curriculum
- [x] adaptive-filter curriculum
- [x] AI/ML curriculum
- [x] streaming architecture
- [x] embedded deployment path
- [x] hardware architecture
- [x] evaluation framework
- [x] ablation plan
- [x] controller/fail-safe plan
- [x] test matrix
- [x] debugging hierarchy
- [x] experiment logging
- [x] requirement traceability
- [x] SIH demo strategy
- [x] judge-question strategy
- [x] scope control
- [x] future roadmap
- [x] kill criteria
- [x] implementation artifact plan
- [x] hardware bring-up plan
- [x] software bring-up plan
- [x] dataset starter protocol
- [x] evaluation harness specification
- [x] NLMS mathematical contract
- [x] impulse-test protocol
- [x] latency measurement protocol
- [x] reproducibility protocol

## 70.2 Engineering execution status

These cannot be checked until the team actually performs them:

- [ ] actual dataset generated
- [ ] actual evaluation harness validated
- [ ] actual STFT implementation verified
- [ ] actual NLMS convergence measured
- [ ] actual microphone channels verified
- [ ] actual streaming path verified
- [ ] actual AI inference measured
- [ ] actual fine-tuning completed
- [ ] actual hybrid benefit measured
- [ ] actual impulse detector calibrated
- [ ] actual controller thresholds validated
- [ ] actual embedded deployment completed
- [ ] actual SNR/STOI/PESQ achieved
- [ ] actual latency achieved
- [ ] actual power measured
- [ ] actual final demo validated

**This is not a document deficiency. These boxes are empirical by definition.**

---

# 71. V6 ARCHITECTURE CONTRACT

## 71.1 V1 system

```text
PRIMARY MIC
    │
    ├───────────────► raw primary stream
    │
    ▼
reference-aware adaptive preprocessing
    ▲
    │
REFERENCE MIC
    │
    ▼
NLMS CONFIG A
    │
    ▼
AI SPEECH ENHANCEMENT
    │
    ▼
DETERMINISTIC CONTROLLER
    │
    ├── impulse protection
    ├── AI-health monitoring
    ├── reference-health monitoring
    └── fallback logic
    │
    ▼
iSTFT / output reconstruction
    │
    ▼
DAC / audio output
    │
    ▼
HEADSET / COMMUNICATION OUTPUT
```

## 71.2 Important architectural rule

Config A is the first hybrid experiment:

\[
d[n] = \text{primary microphone}
\]

\[
x[n] = \text{reference microphone}
\]

\[
y[n] = \hat n[n]
\]

\[
e[n] = d[n]-y[n]
\]

and:

\[
e[n]\rightarrow AI
\]

The AI-only baseline uses:

\[
d[n]\rightarrow AI
\]

This creates a direct and fair comparison.

## 71.3 Why this is an experiment

The system does not assume that NLMS-before-AI will be superior.

The measured comparison determines whether the adaptive stage remains in the final system.

---

# 72. EXACT DATASET PIPELINE

## 72.1 Pipeline

```text
Source registration
      ↓
License/provenance check
      ↓
Audio integrity check
      ↓
Speaker/source split
      ↓
Noise/event classification
      ↓
RIR registration
      ↓
Mixture generation
      ↓
Measured SNR verification
      ↓
Clipping check
      ↓
Metadata generation
      ↓
Train/validation/test manifest
      ↓
Dataset checksum/version
```

## 72.2 Metadata minimum

Every generated clip must record:

```text
clip_id
clean_source_id
speaker_id
noise_source_id
noise_class
impulse_id
rir_id
target_snr_db
measured_snr_db
gain_speech
gain_noise
sample_rate
duration
peak_level
clipping
split
dataset_version
generator_version
timestamp
```

## 72.3 Dataset versioning

Use:

```text
DATASET_V001
DATASET_V002
...
```

Never silently regenerate an old dataset with changed random seeds or parameters.

---

# 73. DATASET GENERATOR — IMPLEMENTATION SPECIFICATION

The generator must support:

### Inputs

- clean speech;
- noise;
- optional impulse;
- optional RIR;
- target SNR;
- output duration;
- sample rate;
- random seed.

### Processing

1. load audio;
2. validate finite values;
3. convert to mono if required;
4. resample only under an explicit configuration;
5. normalize according to a documented policy;
6. choose speech segment;
7. choose noise segment;
8. scale noise;
9. insert impulse if requested;
10. convolve RIR if requested;
11. mix;
12. calculate actual SNR;
13. detect clipping;
14. reject or rescale according to the frozen policy;
15. write WAV;
16. write metadata.

### Reproducibility

Given:

```text
same source IDs
same parameters
same seed
same generator version
```

the generated clip must be reproducible.

---

# 74. DATASET QUALITY TESTS

The generator must have automated tests.

### Test D1

Target 10 dB SNR.

Pass if:

\[
|SNR_{measured}-10| < \delta
\]

where \(\delta\) is defined in the experiment configuration.

### Test D2

No NaN/Inf.

### Test D3

No accidental clipping unless clipping is the intended augmentation.

### Test D4

Clean/noisy duration match.

### Test D5

Sample-rate consistency.

### Test D6

No train/test speaker overlap.

### Test D7

No train/test source-recording overlap.

### Test D8

No augmentation leakage.

### Test D9

Manifest references existing files.

### Test D10

Dataset checksum matches recorded version.

---

# 75. EVALUATION HARNESS — COMPLETE SPECIFICATION

The evaluation system must never be a notebook-only calculation.

## 75.1 Command-level concept

```text
evaluate.py
    --clean clean.wav
    --noisy noisy.wav
    --enhanced enhanced.wav
    --config evaluation.yaml
```

## 75.2 Processing order

```text
load
 ↓
validate
 ↓
sample-rate check
 ↓
channel check
 ↓
finite-value check
 ↓
time alignment
 ↓
optional calibrated delay compensation
 ↓
metric calculation
 ↓
CSV row
 ↓
aggregate
 ↓
report
```

## 75.3 Alignment

This is critical.

Clean and estimate must refer to the same temporal content.

If a system introduces known algorithmic delay:

1. measure it;
2. document it;
3. align for quality metrics only when the metric protocol permits;
4. separately report the real end-to-end delay.

Never "hide" latency by shifting files and then reporting zero system latency.

## 75.4 Output files

```text
results/
├── per_clip.csv
├── per_condition.csv
├── aggregate.csv
├── failure_cases.csv
├── latency.csv
└── plots/
```

## 75.5 Required aggregate views

- overall;
- by input SNR;
- by noise class;
- by speaker;
- by unseen noise;
- by impulse/non-impulse;
- by model;
- by hardware platform.

---

# 76. METRIC IMPLEMENTATION RULES

## 76.1 SNR

Use the known clean reference in synthetic experiments.

For a clean target \(s\) and estimate \(\hat{s}\):

\[
n_{res}[n]=\hat{s}[n]-s[n]
\]

\[
SNR_{est}=10\log_{10}
\frac{P_s}{P_{res}}
\]

Also report input SNR.

## 76.2 STOI

Use a validated implementation and document:

- package;
- version;
- sample-rate mode;
- any resampling;
- signal alignment.

Do not write a fake STOI approximation.

## 76.3 PESQ

Use a validated implementation and the valid sample-rate mode.

Record:

- implementation;
- version;
- mode;
- sample rate;
- alignment method.

Do not silently resample.

## 76.4 SI-SNR/SI-SDR

Document the exact definition used.

Different implementations can differ in centering/scaling conventions.

## 76.5 DNSMOS / subjective metrics

Use as supplementary evidence, not as a replacement for the SIH-stated metrics.

## 76.6 WER/WAcc

If speech-to-text evaluation is available, use it as an additional communication-oriented metric.

---

# 77. MODEL-SELECTION MATRIX — COMPLETE DECISION FRAMEWORK

Do not freeze model choice from literature alone.

| Candidate | Primary purpose | Quality potential | Compute | Streaming risk | V1 role |
|---|---|---:|---:|---:|---|
| Spectral subtraction | Classical baseline | Low–medium | Very low | Low | Mandatory baseline |
| LMS | Adaptive baseline | Low–medium | Very low | Low | Mandatory baseline |
| NLMS | Adaptive baseline | Medium | Very low | Low | Mandatory hybrid experiment |
| RNNoise-like | Lightweight neural | Medium | Very low | Low | Optional baseline |
| Compact CRN | Own trainable model | Medium–high | Low | Low–medium | Learning/fallback |
| DCCRN-style | Complex-domain model | High | Medium | Medium | Fallback |
| DeepFilterNet2/compatible | Real-time enhancement | High | Low–medium | Medium | Primary benchmark |
| DTLN | Lightweight enhancement | Medium–high | Low–medium | Medium | Fallback |
| Conv-TasNet | Time-domain reference | High | Medium–high | Medium | Research comparison |
| FullSubNet family | Quality/reference | High | Medium–high | Medium | Quality ceiling/reference |

### Decision gate

Keep a candidate only if it passes:

1. quality;
2. robustness;
3. latency;
4. memory;
5. streaming;
6. deployment.

---

# 78. DEEPFILTERNET DECISION — IMPORTANT CURRENT FACT

The official DeepFilterNet repository currently documents the precompiled `deep-filter` command as supporting **48 kHz WAV input**. The V1 project therefore must not continue using a generic 16 kHz assumption for the DeepFilterNet path. The exact installed version/backend must be checked before freezing the audio rate. citeturn0search4

The DeepFilterNet2 research paper reports a real-time factor of 0.04 on a notebook Core-i5 CPU under the paper's stated conditions. This is feasibility evidence, not our performance. citeturn0academia24

### Required experiment

```text
DFN-E001
Check exact repository version
Check model package
Check input sample rate
Run offline inference
Measure:
- processing time
- audio delay
- memory
- output quality
```

Only after this experiment can the team freeze the exact DFN configuration.

---

# 79. AI TRAINING PIPELINE — FULL

```text
Dataset manifest
      ↓
DataLoader
      ↓
preprocessing
      ↓
model
      ↓
forward pass
      ↓
enhanced signal
      ↓
loss calculation
      ↓
backpropagation
      ↓
optimizer
      ↓
gradient clipping/check
      ↓
validation
      ↓
checkpoint
      ↓
experiment log
```

## 79.1 Training records

Every training run must store:

```text
run_id
git_commit
dataset_version
model_version
configuration
random_seed
optimizer
learning_rate
batch_size
epochs
loss weights
GPU/CPU
training duration
best checkpoint
validation metrics
```

---

# 80. FINE-TUNING STRATEGY

Do not immediately fine-tune on a huge defence dataset.

### Stage A

Pretrained model → local inference.

### Stage B

Tiny defence subset → verify training code.

### Stage C

Small controlled fine-tune → evaluate.

### Stage D

Expanded training set → compare.

### Stage E

Held-out defence/noise test.

### Stage F

Unseen noise stress test.

### Stage G

Embedded deployment.

If fine-tuning makes general speech quality worse while improving a narrow noise category, investigate catastrophic specialization before accepting it.

---

# 81. AI FAILURE MODES

The team must explicitly test:

- speech attenuation;
- musical noise;
- robotic speech;
- transient smearing;
- consonant loss;
- over-suppression;
- under-suppression;
- unseen-noise failure;
- clipping;
- state-reset artifacts;
- frame-boundary artifacts;
- long-stream drift;
- high CPU load;
- model crash.

---

# 82. STREAMING ENGINE — IMPLEMENTATION CONTRACT

## 82.1 Thread model

```text
Audio callback
     │
     ▼
Input ring buffer
     │
     ▼
Processing thread
     │
     ├── STFT
     ├── NLMS
     ├── AI
     ├── controller
     └── iSTFT
     │
     ▼
Output ring buffer
     │
     ▼
Audio callback / output device
```

## 82.2 Callback rules

The callback should not:

- load models;
- allocate large arrays repeatedly;
- perform disk I/O;
- print continuously;
- block on locks;
- perform long inference.

## 82.3 Streaming state

Maintain state for:

- STFT overlap;
- iSTFT overlap;
- AI recurrent/state buffers;
- NLMS tap history;
- controller state;
- output buffering.

---

# 83. LATENCY MEASUREMENT — EXACT METHOD

Use a physical loopback or timestamped impulse method.

### Test

1. Generate a sharp click.
2. Send it into the acoustic/electrical input.
3. Record input and output.
4. detect corresponding peaks.
5. calculate delay.
6. repeat at least multiple times.
7. calculate P50/P95/P99/max.

Measure separately:

```text
input-device latency
buffer latency
algorithmic/lookahead latency
inference latency
output-device latency
total latency
```

### Rule

A benchmark saying "real-time factor < 1" is not enough.

The prototype must report actual end-to-end latency.

---

# 84. REAL-TIME FACTOR

Define:

\[
RTF=\frac{T_{processing}}{T_{audio}}
\]

Interpretation:

- \(RTF<1\): processing is faster than audio duration;
- \(RTF=1\): real-time boundary;
- \(RTF>1\): cannot sustain real-time operation under the measured configuration.

RTF alone does not capture buffering or I/O latency.

---

# 85. CONTROLLER — FROM PLACEHOLDER TO CALIBRATABLE SYSTEM

The controller inputs are:

```text
speech_confidence
impulse_score
reference_quality
clipping_flag
AI_health
runtime_load
```

## 85.1 Initial state machine

```text
NORMAL
  │
  ├── impulse → IMPULSE_PROTECT
  ├── poor reference → REF_UNRELIABLE
  ├── low speech confidence → CONSERVATIVE
  ├── AI fault → DSP_FALLBACK
  ├── clipping → AUDIO_PROTECT
  └── otherwise → NORMAL
```

## 85.2 Threshold policy

The thresholds in the controller are **initial calibration parameters**, not literature truths.

For every threshold:

1. choose an initial value;
2. label it as a hypothesis;
3. evaluate on a calibration set;
4. optimize on validation only;
5. freeze;
6. test on untouched data.

Never tune thresholds directly on the final test set.

---

# 86. IMPULSE DETECTOR — CALIBRATION PROCEDURE

## Features

Start with:

- normalized short-time energy;
- peak/RMS ratio;
- crest factor;
- spectral flux;
- spectral flatness.

## Calibration dataset

Create:

```text
IMPULSE_NEGATIVE/
  engine
  rotor
  speech
  wind
  siren

IMPULSE_POSITIVE/
  synthetic bursts
  impact sounds
  legally usable transient recordings
  controlled high-energy transients
```

## Evaluation

Measure:

- precision;
- recall;
- F1;
- false-positive rate;
- false-negative rate;
- detection delay.

## Safety rule

If false positives cause frequent suppression of ordinary speech/noise, simplify the detector.

---

# 87. REFERENCE-MIC CALIBRATION

Test candidate positions.

For each:

1. record noise-only;
2. record speech-only;
3. record speech+noise;
4. calculate coherence;
5. calculate speech leakage;
6. run NLMS;
7. measure output SNR;
8. inspect speech artifacts.

A candidate is not selected simply because correlation is high.

Select based on:

\[
\text{noise correlation}
+
\text{speech preservation}
+
\text{stability}
+
\text{physical practicality}
\]

---

# 88. TWO-MIC SYNCHRONIZATION

The team must establish:

- same sample clock or known relationship;
- fixed channel order;
- time alignment;
- gain consistency;
- channel naming.

Test:

```text
inject impulse
↓
record both channels
↓
cross-correlate
↓
estimate delay
↓
repeat
```

Record delay and variance.

---

# 89. HARDWARE DECISION TREE

Do not buy Jetson first.

```text
Can laptop run offline model?
        │
        YES
        ▼
Can laptop sustain streaming?
        │
        YES
        ▼
Can low-cost audio hardware provide 2 channels?
        │
        YES
        ▼
Measure model compute requirement
        │
        ├── low enough → low-cost edge candidate
        │
        └── too high → stronger edge compute
```

The edge device is selected **after measurement**, not by prestige.

---

# 90. LOW-COST PROTOTYPE BILL OF MATERIALS — CATEGORIES

A low-cost proof-of-concept can be constructed from:

| Item | Purpose | Selection rule |
|---|---|---|
| 2-channel microphone interface OR synchronized digital mic pair | Primary + reference | Stable synchronized capture |
| Existing laptop | AI/DSP compute | Use first |
| Wired headphones | Output | Low latency |
| Small powered speaker | Noise/speech source | Controlled rig |
| Cables | Interconnect | Reliable |
| USB power/data | Hardware | Stable |
| Breadboard/prototype board | Interconnect | Temporary only |
| Mounting material | Mic geometry | Repeatable |
| Multimeter | Electrical debugging | Mandatory |
| Optional logic analyzer | I²S debugging | College lab if available |

### Cost philosophy

The first prototype should prove the engineering, not imitate the final defence product.

A low-cost proof-of-concept is legitimate if the report clearly shows:

```text
PoC hardware
     ↓
measured compute requirement
     ↓
engineering scaling path
     ↓
target embedded hardware
```

---

# 91. HARDWARE BRING-UP ORDER

Never connect everything at once.

### H1

Power-only test.

### H2

Processor boots.

### H3

Audio device enumerates.

### H4

One microphone records.

### H5

Second microphone records.

### H6

Channels are synchronized.

### H7

Headphone output works.

### H8

Mic → output passthrough.

### H9

DSP in loop.

### H10

AI in loop.

### H11

Hybrid path.

### H12

Controller/fail-safe.

### H13

Full acoustic test.

---

# 92. ELECTRICAL DEBUGGING CHECKLIST

If audio is bad:

1. power supply;
2. ground;
3. cable;
4. connector;
5. sample clock;
6. I²S format;
7. channel order;
8. gain;
9. clipping;
10. driver;
11. buffer;
12. software.

Never start by changing the neural network.

---

# 93. EMBEDDED BRING-UP CHECKLIST

Before claiming edge deployment:

- [ ] OS verified
- [ ] device tree/audio device verified
- [ ] microphone input verified
- [ ] output verified
- [ ] Python/native runtime verified
- [ ] model runtime installed
- [ ] offline inference verified
- [ ] streaming inference verified
- [ ] CPU measured
- [ ] RAM measured
- [ ] temperature measured
- [ ] power measured
- [ ] long-run stability tested
- [ ] no persistent underruns
- [ ] recovery behaviour tested

---

# 94. ONNX EXPORT GATE

Before export:

- save known input;
- run PyTorch;
- save output.

After export:

- run ONNX;
- compare output.

Measure:

\[
E=\|y_{PyTorch}-y_{ONNX}\|
\]

Use an appropriate numerical tolerance.

If the outputs differ unexpectedly:

- inspect preprocessing;
- dynamic axes;
- operator support;
- state handling;
- tensor layout;
- data type.

Do not proceed to TensorRT until ONNX is validated.

---

# 95. QUANTIZATION GATE

Only quantize after FP32 is stable.

Compare:

| Runtime | Quality | Latency | RAM | Power |
|---|---|---|---|---|
| FP32 | | | | |
| FP16 | | | | |
| INT8 | | | | |

Accept quantization only if:

- quality loss is acceptable;
- latency improves meaningfully;
- stability remains acceptable.

---

# 96. UNIT → INTEGRATION → SYSTEM TEST

## Unit tests

Test:

- SNR;
- STFT;
- iSTFT;
- NLMS;
- detector;
- controller;
- preprocessing.

## Integration tests

Test:

- dataset → model;
- model → streaming;
- microphone → DSP;
- DSP → AI;
- controller → runtime.

## System tests

Test:

- full live microphone path;
- stationary noise;
- non-stationary noise;
- impulse;
- mixed noise;
- failure injection;
- long-duration run.

---

# 97. FAILURE-INJECTION PLAN

The team should deliberately create failures.

### F1

Disconnect reference microphone.

Expected:

> adaptive stage freezes/reduces safely.

### F2

Overload microphone.

Expected:

> protection mode.

### F3

AI process stops.

Expected:

> fallback/bypass.

### F4

Artificially increase inference time.

Expected:

> runtime monitor detects overload.

### F5

Drop audio frames.

Expected:

> event logged and recovery behaviour visible.

### F6

Feed speech into reference microphone.

Expected:

> reference-health logic reduces adaptive aggression.

### F7

Impulse during speech.

Expected:

> controlled impulse response without prolonged speech destruction.

---

# 98. LONG-RUN STABILITY TEST

The system must run continuously for a defined duration.

Record:

- CPU;
- RAM;
- temperature;
- power;
- latency;
- underruns;
- overruns;
- state transitions;
- crashes.

A 20-second successful demo is not evidence of long-run stability.

---

# 99. HUMAN LISTENING TEST

Objective metrics are insufficient.

Create a small controlled listening protocol.

Listeners compare:

- noisy;
- classical;
- AI-only;
- hybrid;
- final.

Ask about:

- speech clarity;
- background noise;
- artifacts;
- robotic sound;
- transient handling;
- overall preference.

For a student project this need not be a formal clinical study. It is engineering validation.

Do not make medical claims from it.

---

# 100. SPEECH COMMUNICATION TEST

A particularly useful system metric is whether speech remains understandable.

Possible procedure:

1. prepare known speech phrases;
2. mix noise;
3. run each system;
4. perform ASR or human transcription;
5. calculate word accuracy/error;
6. compare with STOI/PESQ.

This helps answer:

> "Can the user actually understand the message?"

rather than only:

> "Did the waveform score improve?"

---

# 101. SAFETY BOUNDARY

This prototype is not:

- hearing-protection certification;
- medical equipment;
- tactical radio certification;
- ballistic protection;
- EMC-qualified military equipment;
- environmental-qualified military equipment.

Future deployment would require appropriate:

- acoustic safety;
- EMC;
- environmental;
- electrical;
- human-factor;
- communications;
- reliability;
- certification/qualification testing.

The SIH prototype demonstrates engineering feasibility, not field qualification.

---

# 102. DATA / LICENSE / LEGAL CONTROL

For every external audio source:

- identify owner;
- record license;
- record source URL;
- record retrieval date;
- preserve license text where permitted;
- record whether commercial/public demonstration is permitted.

Do not put an unverified copyrighted recording into the final public demo.

Use synthetic or clearly licensed audio when provenance is uncertain.

---

# 103. CYBER / SOFTWARE SUPPLY-CHAIN CONTROL

For the final build:

- pin dependencies;
- record versions;
- keep hashes where practical;
- do not run random copied scripts;
- inspect external repositories before executing;
- keep secrets out of Git;
- maintain a requirements/environment file;
- archive the exact model checkpoint.

---

# 104. REPRODUCIBLE ENVIRONMENT

Minimum:

```text
Python version
OS version
GPU/CPU
CUDA version if applicable
PyTorch version
NumPy version
SciPy version
audio I/O package
metric package versions
model repository commit
dataset version
configuration file
```

Preferred:

- lockfile;
- container where practical;
- Git commit;
- model checksum.

---

# 105. EXPERIMENT NAMING

Use:

```text
E001_AUDIO_SANITY
E002_SNR_MIX
E003_STFT_RECON
E004_SPECTRAL_SUB
E005_LMS
E006_NLMS
E007_DUAL_MIC
E008_STREAM_PASS
E009_DFN_OFFLINE
E010_AI_STREAM
E011_AI_FINETUNE
E012_IMPULSE_DETECT
E013_NLMS_AI
E014_CONTROLLER
E015_EMBEDDED
E016_LONG_RUN
E017_FINAL_ABLATION
```

Every experiment produces:

- input;
- procedure;
- result;
- decision.

---

# 106. EXPERIMENT DEPENDENCY GRAPH

```text
E001
 │
 ├── E002
 │    └── E003
 │         └── E004
 │
 ├── E005
 │    └── E006
 │
 ├── E007
 │    └── E013
 │
 ├── E008
 │    └── E010
 │
 └── E009
      └── E011
           └── E013
                └── E014
                     └── E015
                          └── E016
                               └── E017
```

Do not jump to E017.

---

# 107. FIRST IMPLEMENTATION ARTIFACTS

The first repository milestone must contain:

```text
src/
  dsp/
    stft.py
    istft.py
    spectral_subtraction.py

  adaptive/
    lms.py
    nlms.py

  data/
    mixer.py
    manifest.py

  evaluation/
    metrics.py
    evaluate.py

  streaming/
    buffers.py
    pipeline.py

  controller/
    controller.py
    impulse.py

tests/
  test_snr.py
  test_stft.py
  test_istft.py
  test_nlms.py
  test_mixer.py
```

---

# 108. MINIMUM CODE QUALITY RULES

Every function must have:

- clear input;
- clear output;
- unit convention;
- sample-rate assumption if applicable;
- shape/channel convention;
- error behaviour.

Avoid:

- hidden global state;
- magic constants;
- undocumented resampling;
- silent clipping;
- silent channel swapping.

---

# 109. CONFIGURATION — NO MAGIC NUMBERS

Put tunable values in configuration:

```yaml
audio:
  sample_rate: VERIFY
  channels: 2
  dtype: float32

stft:
  fft_size: VERIFY
  hop_size: VERIFY
  window: hann

nlms:
  taps: 128
  mu: 0.2
  eps: 1.0e-8

controller:
  impulse_threshold: CALIBRATE
  reference_threshold: CALIBRATE
  speech_threshold: CALIBRATE

evaluation:
  target_snrs_db: [-10, -5, 0, 5, 10, 15, 20]
```

Values marked `VERIFY` or `CALIBRATE` are not final until experiments determine them.

---

# 110. WHAT THE TEAM MUST LEARN BEFORE AI

Every member should be able to answer:

### Audio

- What is sampling?
- Why does aliasing happen?
- What is dBFS?
- What is clipping?
- What is dynamic range?

### DSP

- What does FFT do?
- Why use STFT?
- Why overlap frames?
- What is phase?
- Why can iSTFT fail?

### Adaptive filtering

- What is LMS?
- Why does NLMS normalize?
- What makes an adaptive filter diverge?
- Why is reference quality critical?

If a member cannot answer these, they should not be debugging the AI pipeline yet.

---

# 111. WHAT THE TEAM MUST LEARN FOR AI

Every AI member must understand:

- tensors;
- batch/channel/time dimensions;
- train/validation/test;
- forward pass;
- loss;
- gradient;
- optimizer;
- learning rate;
- overfitting;
- checkpointing;
- inference vs training;
- causal processing;
- stateful inference;
- preprocessing consistency.

---

# 112. WHAT THE TEAM MUST LEARN FOR EMBEDDED

Every embedded member must understand:

- Linux processes;
- threads;
- scheduling;
- audio devices;
- buffers;
- I/O latency;
- CPU/RAM profiling;
- power;
- temperature;
- model runtime;
- ONNX;
- TensorRT if used;
- deployment packaging.

---

# 113. WHAT THE TEAM MUST LEARN FOR HARDWARE

The hardware team must understand:

- microphone sensitivity;
- ADC;
- DAC;
- I²S;
- USB audio;
- clocking;
- channel synchronization;
- grounding;
- power integrity;
- gain staging;
- clipping;
- connector integrity.

---

# 114. WHAT THE TEAM MUST LEARN FOR DEBUGGING

The fundamental debugging rule is:

> **Change one layer at a time.**

When the system fails, isolate:

```text
Physical
  ↓
Driver
  ↓
Raw audio
  ↓
DSP
  ↓
Adaptive filter
  ↓
AI
  ↓
Controller
  ↓
Output
```

Do not simultaneously modify model, sample rate, buffer size and hardware wiring.

---

# 115. FULL SYSTEM DEBUG MODE

The runtime should optionally expose:

```text
input RMS
reference RMS
input clipping
reference clipping
reference coherence
NLMS mode
NLMS mu
AI processing time
AI health
controller state
output RMS
output clipping
buffer fill
underrun count
overrun count
temperature
CPU
RAM
```

This turns debugging from guessing into instrumentation.

---

# 116. FINAL ACCEPTANCE TEST — PROCEDURE

## AT-01 Stationary

- fixed speech;
- engine-like noise;
- multiple SNRs;
- all baselines;
- metrics.

## AT-02 Non-stationary

- changing rotor/siren/vehicle-like noise;
- abrupt transition;
- measure adaptation and latency.

## AT-03 Impulsive

- controlled impulse;
- speech overlap;
- measure detection/recovery.

## AT-04 Unseen noise

- held-out noise family;
- no training exposure;
- report generalization.

## AT-05 Hardware

- live two-mic;
- headset;
- edge compute;
- latency/resource.

## AT-06 Failure

- reference failure;
- AI failure;
- clipping;
- buffer overload.

## AT-07 Long run

- sustained operation;
- stability metrics.

---

# 117. ACCEPTANCE SCORECARD

| Requirement | Pass condition | Evidence |
|---|---|---|
| Dataset | reproducible generator | versioned dataset |
| Stationary noise | measured improvement | metric table |
| Non-stationary | measured intelligibility/latency | metric table |
| Impulsive | measured recovery | impulse report |
| AI | held-out improvement | model report |
| Adaptive | justified by ablation | A/B table |
| Real time | stable stream | latency log |
| Hardware | live two-mic system | video/photos/log |
| Embedded | measured resource use | benchmark |
| Speech | STOI/PESQ/SI-SNR | evaluation |
| Robustness | unseen noise | generalization report |
| Safety/fallback | failure tests | failure log |
| Reproducibility | code/data/config trace | repository |
| SIH demo | repeatable | rehearsal record |

---

# 118. FINAL GO / NO-GO MATRIX

### GO only if:

- [ ] audio chain stable;
- [ ] DSP baseline works;
- [ ] dataset generator verified;
- [ ] evaluation harness verified;
- [ ] AI baseline beats or meaningfully improves the noisy baseline on the agreed test protocol;
- [ ] hybrid stage is mathematically valid;
- [ ] hybrid stage has measured benefit OR is honestly removed;
- [ ] streaming is stable;
- [ ] hardware works;
- [ ] required categories are tested;
- [ ] results are reproducible;
- [ ] demo is repeatable.

### NO-GO / PIVOT if:

- AI cannot beat the noisy baseline;
- streaming cannot sustain real time;
- microphone system is unreliable;
- hybrid stage consistently damages speech;
- final claims depend on hidden offline processing;
- the team cannot reproduce its own metrics.

---

# 119. SIH CLAIMS MATRIX

Before writing any slide, classify every statement.

| Claim | Evidence class |
|---|---|
| Paper reports RTF 0.04 | Published evidence |
| PS target STOI > 0.85 | Requirement |
| Our model STOI = 0.87 | Team measurement |
| AI+NLMS is better | Only after ablation |
| Edge runtime is real-time | Only after measured E2E latency |
| Prototype is military-ready | **Do not claim** |
| Prototype demonstrates engineering feasibility | Allowed if evidence supports it |

---

# 120. FINAL COST STRATEGY

The cost argument must be engineering-based.

Do not say:

> "We made it cheap."

Say:

> "We separated the proof-of-concept hardware from the deployment hardware. The PoC validates the signal-processing, AI and real-time architecture at low cost. The measured compute, audio-interface and latency requirements define the migration path to a more capable embedded platform."

This is stronger because it demonstrates engineering judgement.

---

# 121. DEPLOYMENT MIGRATION PATH

```text
V1
Low-cost acoustic PoC
       ↓
V1.1
Better synchronized audio hardware
       ↓
V1.2
Measured edge platform
       ↓
V2
Physical ANC / secondary-path experiment
       ↓
V3
Communication-unit integration
       ↓
V4
Ruggedization + EMC/environmental qualification
       ↓
Field evaluation
```

No stage should be skipped because it sounds impressive.

---

# 122. FINAL DOCUMENT COMPLETENESS CHECKLIST

## Research

- [x] Problem interpretation
- [x] State of art
- [x] Model candidates
- [x] DSP
- [x] adaptive filtering
- [x] speech enhancement
- [x] defence noise
- [x] impulsive noise
- [x] datasets
- [x] losses
- [x] metrics
- [x] embedded AI
- [x] real-time systems
- [x] hardware
- [x] physical ANC boundary
- [x] deployment roadmap

## Learning

- [x] foundations
- [x] STFT
- [x] adaptive filters
- [x] audio hardware
- [x] streaming
- [x] dataset engineering
- [x] ML
- [x] AI enhancement
- [x] integration
- [x] embedded
- [x] debugging

## Implementation

- [x] repository structure
- [x] configuration
- [x] dataset pipeline
- [x] evaluation harness specification
- [x] NLMS contract
- [x] controller contract
- [x] streaming contract
- [x] hardware bring-up
- [x] embedded deployment
- [x] model export
- [x] quantization gate
- [x] test hierarchy

## Testing

- [x] unit tests
- [x] integration tests
- [x] system tests
- [x] stationary
- [x] non-stationary
- [x] impulsive
- [x] mixed
- [x] unseen noise
- [x] failure injection
- [x] long-run
- [x] latency
- [x] resource
- [x] subjective listening

## Evidence

- [x] source hierarchy
- [x] provenance
- [x] versioning
- [x] experiment IDs
- [x] decision log
- [x] model card
- [x] dataset card
- [x] hardware record
- [x] traceability
- [x] SIH evidence pack

## SIH

- [x] demo sequence
- [x] judge questions
- [x] novelty boundaries
- [x] cost strategy
- [x] deployment roadmap
- [x] limitations
- [x] go/no-go criteria

---

# 123. WHAT IS STILL IMPOSSIBLE TO "COMPLETE" INSIDE A DOCUMENT

The following cannot honestly be completed by writing more pages:

1. actual SNR;
2. actual STOI;
3. actual PESQ;
4. actual AI generalization;
5. actual NLMS improvement;
6. actual microphone coherence;
7. actual streaming latency;
8. actual embedded resource usage;
9. actual power consumption;
10. actual failure rate;
11. actual user preference;
12. actual SIH demonstration reliability.

These are **experimental outputs**.

Therefore:

> **The document is now complete as the project's research, learning, build, testing and evidence plan. The prototype itself is deliberately not declared complete.**

That distinction is non-negotiable.

---

# 124. FINAL OPERATING CONTRACT FOR THE SIX-MEMBER TEAM

From this point onward:

### Research question

> "What do we need to know to make the next engineering decision?"

### Learning question

> "Can every relevant member explain and reproduce this?"

### Build question

> "What artifact proves we implemented it?"

### Testing question

> "What measurement can falsify our assumption?"

### Debugging question

> "Which layer is actually failing?"

### SIH question

> "What evidence proves the claim to a judge?"

### Scope question

> "Does this feature solve a measured problem in V1?"

If the answer is no, park it.

---

# 125. FINAL MASTER STATUS — V6

**Research/reference:** COMPLETE ENOUGH TO EXECUTE

**Learning curriculum:** COMPLETE

**Build plan:** COMPLETE

**Testing plan:** COMPLETE

**Hardware bring-up plan:** COMPLETE

**AI/ML integration plan:** COMPLETE

**Embedded deployment plan:** COMPLETE

**Evaluation/evidence plan:** COMPLETE

**SIH demonstration plan:** COMPLETE

**Future research roadmap:** COMPLETE

**Prototype:** NOT YET BUILT

**Measured results:** NOT YET AVAILABLE

**Next action:** EXECUTE E001_AUDIO_SANITY

---

# 126. THE ONLY CORRECT NEXT MOVE

Do not create V7 because you found another model.

Do not redesign V1 because a new paper looks exciting.

Do not buy expensive edge hardware because it looks more professional.

Do not build a dashboard before the audio path works.

Do not start with fine-tuning.

Start here:

```text
E001_AUDIO_SANITY
        ↓
E002_SNR_MIX
        ↓
E003_STFT_RECON
        ↓
E004_SPECTRAL_SUB
        ↓
E005_LMS
        ↓
E006_NLMS
        ↓
E007_DUAL_MIC
        ↓
E008_STREAM_PASS
        ↓
E009_DFN_OFFLINE
        ↓
...
```

The project is now **plan-complete**.

The next unknowns belong to the laboratory.

---

# 127. VERIFIED EXTERNAL RESEARCH ANCHORS

The Microsoft DNS Challenge repository provides dataset-generation scripts, clean speech/noise/RIR resources, unit tests for the synthesizer, and provenance/license information. It also documents the scale of the full dataset, which is far larger than a student team needs for an initial subset. citeturn0search0turn0search5

The DeepFilterNet official repository currently documents the precompiled `deep-filter` path as accepting 48 kHz WAV files, so the project's audio-rate policy must be based on the exact installed version rather than an inherited generic 16 kHz assumption. citeturn0search4

DeepFilterNet2's primary paper reports a real-time factor of 0.04 on a notebook Core-i5 CPU under its experimental conditions and explicitly positions the method for real-time embedded speech enhancement. This is a feasibility reference and must not be presented as a measured project result. citeturn0academia24

---

# 128. END OF V6 MASTER

**Project principle:**

> Build the smallest experimentally valid system that proves the engineering, measures its limits, and provides a credible migration path toward the deployment system described by PS 26052.

**Evidence beats complexity.**

**Measurement beats assumption.**

**Engineering beats presentation.**

**The team learns; the tools assist.**

**The prototype earns the claims.**
