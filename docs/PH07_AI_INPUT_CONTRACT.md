# PH0.7 AI Input Contract

## A. What exactly enters TinyEnhancer at deployment?

From `src/streaming/causal_engine.py` L155-171:

```
x_AI[n] = STFT(window * impulse_protect(NLMS(primary[n], reference[n])))
```

Concretely:
1. `primary[n]` and `reference[n]` enter the VSS-NLMS adaptive filter
2. NLMS output: `dsp_out = primary - NLMS_estimate` (the residual)
3. Impulse protection is applied to `dsp_out`
4. The result is windowed (Hann), transformed via `np.fft.rfft`
5. Magnitude spectrogram enters TinyEnhancer

The AI model sees the **NLMS residual magnitude spectrum**, not the raw primary mic.

## B. What exactly enters TinyEnhancer during training?

From `src/ai/train.py` L220-225:

```python
primary_t = torch.from_numpy(primary).float()
noisy_stft = torch.stft(primary_t, frame_size, hop_size, window=window,
                         return_complex=True, center=False)
noisy_mag = noisy_stft.abs()
noisy_input = noisy_mag.unsqueeze(0).unsqueeze(0)  # -> model
```

The AI model sees the **raw primary mic magnitude spectrum** during training.

**This is a mismatch.**

## C. Should training input become NLMS residual?

### Option 1: Raw primary (current)

- **Pro**: Simple, no dependency on NLMS implementation at training time
- **Con**: Model learns to suppress all noise directly from noisy speech. At deployment, NLMS has already removed correlated noise, so the model sees a different spectral distribution. The model is trained on a problem it never faces in production.

### Option 2: NLMS residual (recommended)

- **Pro**: Training distribution matches deployment distribution. The model learns to handle exactly what it will see: speech + residual non-stationary noise that NLMS could not remove.
- **Con**: Requires running NLMS at training time, adding a dependency. NLMS behavior varies with filter length, step size, reference quality -- training must simulate realistic NLMS conditions.
- **Implementation**: For each training sample, run the same `VSSNLMSFilter` on `(primary, reference)` to produce `nlms_residual`, then feed `nlms_residual` as the noisy input.

### Option 3: Mixed curriculum

- **Pro**: Model sees both distributions, potentially more robust
- **Con**: Complicated schedule, harder to reproduce, may not converge better than Option 2

### Recommendation: Option 2

The model should be trained on NLMS residuals because that is what it will process at deployment. A model trained on raw noisy speech learns to suppress noise patterns that NLMS will have already removed -- wasted capacity. The residual after NLMS contains speech + non-linear/non-stationary remnants that are specifically the AI's job.

## D. What remains the target?

**Clean speech magnitude spectrum** (for L1 loss) and **clean speech waveform** (for SI-SDR loss).

This does NOT change between options. The target is always the clean source. What changes is the input: raw primary vs NLMS residual.

## E. How should NLMS variation be represented?

During training with Option 2, the NLMS parameters should vary to make the model robust:

| Parameter | Range | Rationale |
|-----------|-------|-----------|
| filter_length | 32-128 | Production uses 64, but variation prevents overfitting to one specific NLMS behavior |
| step_size (mu) | 0.01-0.1 | Regime adaptation changes mu at deployment |
| Reference quality | Clean ref, leaked ref, delayed ref | Real reference mics pick up speech leakage |
| NLMS convergence | Fully converged, partially converged, fresh | Model must handle startup transients |

## F. Should NLMS be frozen during training?

**Yes, NLMS should be run as a fixed preprocessing step, not trained jointly.**

Rationale:
- NLMS is already a mature adaptive filter with well-understood convergence. Joint training provides no benefit and makes optimization harder.
- The training pipeline should be: `(primary, reference) -> frozen NLMS -> residual -> AI model -> enhanced`
- NLMS parameters can be varied across samples (as described in E) but the NLMS filter itself is not a trainable PyTorch module.

## Implementation Plan

1. Before training, precompute NLMS residuals for all training samples (or compute on-the-fly)
2. Store residuals alongside clean/primary/reference in the dataset
3. Train TinyEnhancer on `(nlms_residual_mag, clean_mag)` pairs
4. Validate on TEST_A_UNSEEN_SPEAKER using the same NLMS preprocessing
5. The metadata sidecar must record `training_input = "NLMS_RESIDUAL"` and the NLMS configuration used

## Status

**NOT YET IMPLEMENTED.** This document records the design decision.
The current baseline training will still use raw primary (Option 1) as the first controlled experiment. Option 2 will be Experiment E after the baseline is established.
