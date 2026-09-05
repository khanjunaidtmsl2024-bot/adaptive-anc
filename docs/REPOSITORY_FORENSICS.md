# REPOSITORY FORENSICS AUDIT REPORT
**Project**: SIH 2026 Hardware Edition — Problem Statement SIH26052  
**Title**: AI/ML-Enabled Adaptive Noise Cancellation System for Defence and Mission-Critical Communication  
**Date**: September 5, 2026  
**Auditor**: Lead Autonomous Engineering Agent  
**Evidence Policy**: STRICT (VERIFIED | ENGINEERING TARGET | HYPOTHESIS | BLOCKED)

---

## 1. Executive Summary & Current System Reality

| Dimension | README / Paper Claim | Current Code Reality | Evidence Label |
| :--- | :--- | :--- | :--- |
| **System Identity** | "Adaptive Active Noise Cancellation (ANC)" | **Dual-Microphone Electrical Adaptive Noise Cancellation + Post-DSP AI Speech Enhancement (Config A)**. Not acoustic ANC (no physical secondary path $S(z)$, error mic, or speaker feedback loop). | `VERIFIED` |
| **Stage 1 DSP** | Real-time dual-mic adaptive filter | `NLMSFilter` in `src/dsp/nlms.py` is functional and tested. However, early benchmarks in `baseline_comparison.csv` showed SNR degradation due to unaligned primary/reference signals and lack of leakage gating. | `VERIFIED` |
| **Stage 2 AI Model** | "TinyComplexEnhancerV3 / DeepFilterNet" | `TinyEnhancerNet` is a **4-layer 2D ConvNet** estimating spectral magnitude mask $M(f,t) \in [0, 1]$. Exact parameter count = **9,569** (37.38 KB in FP32). Not a complex-domain residual network yet. | `VERIFIED` |
| **Latency** | "<8 ms / <10 ms real-time" | STFT analysis window is 512 samples (32 ms @ 16 kHz). 128-sample hop is 8 ms, but analysis window + buffering creates ~24–28 ms total algorithmic latency. Sub-8 ms end-to-end is currently an `ENGINEERING TARGET`, not measured reality. | `VERIFIED` |
| **Dataset & Defence Noise** | "Military defence noise dataset" | Currently uses synthetic mathematical simulations (formant speech + diesel harmonics + rotor harmonics + ballistic shockwave transients). Real Military Audio Dataset (MAD) recordings are not yet merged. | `VERIFIED` |
| **Dataset Splitting** | "Rigorous generalization protocol" | `artifacts/01_dataset_generator.py` randomly picks noise classes (`choice(['stationary', 'non_stationary', 'impulsive'])`) and splits cyclically by mix index, causing speaker and recording leakage across train/val/test. Needs complete replacement with disjoint Test A/B/C. | `VERIFIED` |
| **Hardware Platform** | "Raspberry Pi 4 + WM8960 Audio HAT" | Emulated offline and desktop simulation ready. Hardware deployment scripts and JIT/INT8 quantized models generated; physical hardware verification is pending physical device connection. | `BLOCKED` (Hardware physical access) |

---

## 2. Complete Component Audit Matrix

| Component | File Path | Status | Evidence | Problem / Root Cause | Required Engineering Action |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Adaptive NLMS Filter** | `src/dsp/nlms.py` | `WORKING` | `pytest tests/test_nlms.py` passes. Sample-by-sample and block modes verified. | Lacks dynamic variable step-size (VSS) and speech leakage gating. | Implement VSS-NLMS and cross-spectral leakage detector in `src/dsp/`. |
| **Spectral Subtraction** | `src/dsp/spectral_subtraction.py` | `WORKING` | `pytest tests/test_baselines_and_integrations.py` passes. | Classical baseline only; musical noise at aggressive subtraction factors. | Keep as comparative baseline `CONFIG_1`. |
| **Wiener Filter** | `src/dsp/wiener.py` | `WORKING` | `pytest tests/test_baselines_and_integrations.py` passes. Decision-directed a priori SNR tracking. | Musical noise residual at low SNR (<0 dB). | Keep as comparative baseline `CONFIG_1`. |
| **Robust Kalman Filter** | `src/dsp/kalman.py` | `WORKING` | Huber M-estimator impulse handling verified. | State-space formulation computationally heavier than NLMS. | Keep for impulsive bench comparison. |
| **Acoustic FxLMS** | `src/dsp/fxlms.py` | `RESEARCH ONLY` | Code exists with synthetic secondary path FIR filter. | Not connected to physical speaker or secondary path $S(z)$. True acoustic ANC is future work. | Isolate from electrical cancellation path; clearly demarcate acoustic ANC vs electrical dual-mic ANC. |
| **AI Neural Enhancer** | `src/ai/tiny_enhancer.py` | `WORKING` | 9,569 parameters (Conv2d: 1->16->32->16->1). PyTorch CPU forward pass verified. | Magnitude mask only; does not correct phase. Receptive field is local $3\times 3$. | Benchmark against DTLN / causal complex models. |
| **Edge Export & Quant** | `src/ai/export_onnx.py` | `WORKING` | Produces TorchScript JIT (57.9 KB) and native dynamic INT8 (42.3 KB). | `onnxscript` not installed in env, so TorchScript & PyTorch INT8 are primary edge targets. | Maintain TorchScript/INT8 deployment path for embedded targets. |
| **Ichigo Integration Bridge** | `src/integrations/ichigo_bridge.py` | `WORKING` | `python -m src.integrations.ichigo_bridge` runs and confirms +0.039 STOI gain in hybrid over raw. | Hardcoded 10,417 parameter count in dictionary fallback instead of true 9,569. Used Config B (Post-AI) in one method. | Update dictionary to 9,569 params; standardize pipeline to canonical Config A (Pre-AI NLMS). |
| **Hybrid Pipeline** | `src/pipeline/hybrid_chain.py` | `PARTIALLY WORKING` | Runs Config A (NLMS -> AI). | Lines 82-83 use non-overlapping rectangular chunking `frames = trimmed.reshape(n_frames, hop)` instead of windowed overlap-add STFT. | Replace with stateful causal overlap-add STFT engine from `src/streaming/stft_engine.py`. |
| **Fallback & Safety Controller** | `src/pipeline/fallback_controller.py` | `WORKING` | Detects crest factor > 5.0, kurtosis > 10.0, energy ratio > 10.0. Clamps and crossfades. | Rule-based thresholding without adaptive background statistics. | Add adaptive running variance tracking for impulse detection. |
| **Ring Buffer** | `src/streaming/ring_buffer.py` | `WORKING` | Thread-safe circular buffer with wrap-around pointers. 100% test pass. | None. | Ready for real-time audio callback. |
| **Streaming STFT Engine** | `src/streaming/stft_engine.py` | `WORKING` | Overlap-add synthesis with Hanning window verified in unit tests. | Operates block-by-block; needs state persistence across streaming hops. | Integrate into unified causal streaming loop. |
| **Live Audio Stream** | `src/streaming/live_stream_audio.py` | `WORKING` | Simulates multi-channel streaming audio callback with latency measurement. | Synthetic generator input rather than physical soundcard ALSA/I2S. | Connect to real hardware ALSA backend when deployed on Raspberry Pi. |
| **Objective Metrics Engine** | `src/evaluation/metrics.py` | `WORKING` | Computes SNR, SI-SNR, STOI, PESQ. | `pystoi` and `pesq` external libraries fall back to correlation/surrogate if not installed. | Ensure robust calculation and record whether native or surrogate was used. |
| **Artifact 1: Dataset Gen** | `artifacts/01_dataset_generator.py` | `BROKEN` | Generates audio files if run. | 1. Unconditionally requires `soundfile` (`sys.exit(1)`). 2. Randomly chooses `noise_class` regardless of source audio. 3. Cyclic split causes data leakage. | Replace with `src/dataset/generate.py`, `splits.py`, `provenance.py` implementing Test A/B/C. |
| **Artifact 2: Eval Harness** | `artifacts/02_evaluation_harness.py` | `PARTIALLY WORKING` | Script exists for batch CSV generation. | Requires `soundfile`. Hardcoded evaluation criteria. | Refactor into `src/evaluation/evaluate.py`. |
| **Artifact 3: Streaming Skeleton** | `artifacts/03_streaming_skeleton.py` | `PARTIALLY WORKING` | Demonstrates dual-channel callback. | Reports `algorithmic_latency_ms = frame_size / sample_rate` as total latency (misleading). | Use full latency breakdown from `LATENCY_CONTRACT.md`. |
| **Artifact 4: Adaptive Filter Lab** | `artifacts/04_adaptive_filter_lab.py` | `WORKING` | Generates Config A/B/C benchmarks. | Output `baseline_comparison.csv` showed negative SNR gain due to uncalibrated primary/reference coupling. | Fix reference path modeling and add delay alignment. |

---

## 3. The 5 Major Engineering Contradictions Identified

1. **The "ANC" vs "Speech Enhancement" Contradiction**:
   - The repository frequently calls itself "AI-based Active Noise Cancellation".
   - *Reality*: The code implements dual-microphone reference cancellation in the digital domain (electrical cancellation) followed by single-channel spectral enhancement. It does not cancel sound waves in the air using an acoustic actuator and error microphone.
   - *Resolution*: Strictly classify the system as **"Hybrid Adaptive Electrical Noise Cancellation + AI Residual Speech Enhancement"**.

2. **The Latency Contradiction (8 ms Hop vs 8 ms Latency)**:
   - Documentation claims sub-8 ms latency.
   - *Reality*: A 512-sample STFT window at 16 kHz spans 32 ms. A 128-sample hop is 8 ms, but the window requires at least 16–32 ms of audio buffering.
   - *Resolution*: Formalize a 6-component latency budget in `docs/LATENCY_CONTRACT.md`. Total E2E budget is 20–28 ms for 512/256 framing; sub-10 ms requires 128/64 framing with retrained causal models.

3. **The Dataset Taxonomy Flaw**:
   - `01_dataset_generator.py` chose `noise_class = choice(['stationary', 'non_stationary', 'impulsive'])` randomly.
   - *Reality*: A recording of gunfire could be labeled stationary, and engine rumble could be labeled impulsive.
   - *Resolution*: Enforce directory-driven and metadata-driven taxonomy (`src/dataset/`):
     - `stationary`: engine idle, generator hum, HVAC.
     - `non_stationary`: rotor blade variation, vehicle acceleration, wind.
     - `impulsive`: gunfire, artillery shockwave, blast transients.

4. **The Split Contamination Flaw**:
   - The generator assigned splits cyclically: `index / total < 0.7 -> train`.
   - *Reality*: Different chunks from the same speaker and the same noise recording appeared in both training and test sets.
   - *Resolution*: Implement disjoint protocols:
     - **Test A**: Unseen speakers (speakers 009–010 held out).
     - **Test B**: Unseen noise recordings (noise audio files held out).
     - **Test C**: Unseen noise categories (entire noise class held out).

5. **The Rectangular Framing Bug in `hybrid_chain.py`**:
   - Lines 82-83 reshaped audio into non-overlapping blocks without windowing before rfft.
   - *Reality*: Produces high-frequency boundary clicks at every block edge.
   - *Resolution*: Route all STFT processing through `src/streaming/stft_engine.py` with Hanning windowing and proper 50% overlap-add reconstruction.

---

## 4. Current System Reality Check

```
+---------------------------------------------------------------------------------------------------+
|                                 ACTUAL WORKING SYSTEM PIPELINE                                   |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  Primary Mic:  d[n] = s[n] + v[n]   (Speech + Ambient Acoustic Noise)                            |
|  Reference Mic: x[n] = v_ref[n]      (Correlated Environmental Noise)                             |
|                                                                                                   |
|         Primary d[n]                     Reference x[n]                                           |
|              |                                 |                                                  |
|              v                                 v                                                  |
|         [ADC / AFE]                       [ADC / AFE]                                             |
|              |                                 |                                                  |
|              v                                 v                                                  |
|              |                     [Cross-Correlation Delay Est.]                                 |
|              |                                 |                                                  |
|              |                                 v                                                  |
|              |                      x_aligned[n] = x[n - delta]                                   |
|              |                                 |                                                  |
|              |                                 v                                                  |
|              |                     [Leakage / Coherence Gate]                                     |
|              |                     (Freezes/attenuates mu if speech)                             |
|              |                                 |                                                  |
|              +-------------------+             v                                                  |
|                                  |      [NLMS / VSS-NLMS]                                         |
|                                  |             |                                                  |
|                                  |             v y[n] (Estimated Noise)                           |
|                                  v             |                                                  |
|                                  +----(-)------+                                                  |
|                                         |                                                         |
|                                         v e[n] = d[n] - y[n] (Electrical Residual)                |
|                                         |                                                         |
|                          +--------------+--------------+                                          |
|                          |                             |                                          |
|                          v                             v                                          |
|               [Regime & Impulse Detector]     [Causal Overlap-Add STFT]                           |
|               (Crest factor, Spectral flux)            |                                          |
|                          |                             v                                          |
|                          |                    [TinyEnhancer / AI]                                 |
|                          |                    (Spectral mask M(f,t))                              |
|                          |                             |                                          |
|                          |                             v                                          |
|                          |                    [Stateful Overlap iSTFT]                            |
|                          |                             |                                          |
|                          +-------------(+)-------------+                                          |
|                                         |                                                         |
|                                         v                                                         |
|                            [Speech Preserver / Limiter]                                           |
|                                         |                                                         |
|                                         v                                                         |
|                                Enhanced Audio Out s^[n]                                           |
+---------------------------------------------------------------------------------------------------+
```

---

## 5. Audit Action Plan
1. Produce `docs/LATENCY_CONTRACT.md` establishing realistic latency budgets.
2. Build `src/dataset/` to replace flawed `artifacts/01_dataset_generator.py` with disjoint Test A/B/C.
3. Fix rectangular framing in `src/pipeline/hybrid_chain.py` by integrating stateful windowed STFT.
4. Implement `src/dsp/vss_nlms.py`, `delay_alignment.py`, `leakage_detector.py`, `impulse_protection.py`.
5. Execute the first verified experimental baseline matrix and write `results/csv/baseline_results.csv`.
