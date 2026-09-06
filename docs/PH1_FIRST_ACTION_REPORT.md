# PH1 FIRST-ACTION REPORT — Pre-Training Gate
**SIH26052 Adaptive AI/ML ANC · 2026-09-06 · Status: DSP CONTRACT VERIFIED — TRAINING NOT STARTED**

Per the PH1 master prompt (FIRST ACTION section): this report is produced **before any
training job is launched**. Training has NOT started. Nothing below is a measured model
quality result; evidence tiers are marked per item.

---

## 1. Authoritative production path (verified in code)

```
CausalStreamingEngine  (src/streaming/causal_engine.py)
  -> reference-mic adaptive cancellation (VSS-NLMS / NLMS, Numba-accelerated)
  -> impulse protection controller
  -> STFT(frame=256, hop=128, hann, center=False)
  -> AI speech enhancement (TinyEnhancerWrapper)
  -> WOLA overlap-add streaming output
```
Evidence tier: **VERIFIED MEASUREMENT** (read from source + exercised by tests).

## 2. Git / environment snapshot (at time of report)

| Item | Value |
|---|---|
| Git commit | `5d821a7` (HEAD) — PH0.7/PH1 freeze work committed on `main` (see §2a) |
| Frame / hop | 256 / 128 samples (16 kHz) — deployment; training now also 256/128 (Gate 2 fix applied) |
| center | `False` everywhere (train.py torch.stft/istft + deployment np.fft) |
| Runtime | Python 3.11, torch 2.14.0+cpu (CUDA: **not available**), numpy 2.4.6, numba 0.67.0 |
| Host | Windows 10, 8 CPU threads, 17.1 GB RAM (no GPU, no Raspberry Pi/Jetson attached) |

## 2a. Post-freeze audit (go-condition re-verification)

Audit items 1–4 of the PH1 go-condition were re-run against the committed state:

1. **Backend provenance (item 1) — FIXED & RECORDED.** `ph06_headroom_benchmark.py`
   now writes backend metadata into `ph06_headroom_profile.csv`
   (`numba_version=0.67.0, numpy_version=2.4.6, nlms_backend=numba,
   impulse_backend=numba`, engine classes `VSSNLMSFilterFast` /
   `ImpulseProtectionControllerFast`). The timings in that CSV are therefore
   proven to be genuinely Numba-backed, not Python-fallback timings.
2. **Latency reproducibility (item 2) — NOT REPRODUCED as a stable PASS.** Seven
   isolated 1000-hop runs gave P95 ∈ [4.90, 14.52] ms (median ≈ 6.9 ms;
   representative series: 8.33, 8.18, 4.90, 6.77, 6.90, 14.52, 7.87). The
   previously committed single run (P95 = 5.54 ms, +2.47 headroom) was one
   quiet-moment draw on a shared, load-varying host. Verdict: the 8 ms gate is
   **marginal on this host** (some runs FAIL), and a stable pass must be shown on
   a quiet/isolated machine (or the edge target host) before it is called proven.
   The gate itself is NOT relaxed. AI-stage P95 (2.8–10.9 ms across runs) is the
   dominant contributor; DSP P95 stayed 0.64–1.05 ms.
3. **Impulse-test wording (item 3) — RESOLVED.** Measured: a unit impulse exits
   the engine at peak **0.9500** because the impulse-protection controller's hard
   output limiter caps at 0.95 (hearing/DAC fail-safe, by design) BEFORE the
   STFT/AI/WOLA chain. WOLA itself is unity-gain: a sub-limiter impulse (amp 0.5)
   reconstructs at 0.5000 (<1e-3 error, no dispersion) and sine/constant signals
   reconstruct at <1e-4. The test now asserts the limiter ceiling (0.95 ± 1e-3)
   and a separate unity-gain impulse below it, instead of the old ±0.06
   "near-unity" tolerance that papered over the mechanism.
4. **Parameter counts (item 4) — RECONCILED (geometry-dependent).** CRN-Micro =
   723,801 and DTLN = 775,939 parameters under the FROZEN contract geometry
   (frame 256 → freq_bins 129); they are 986,457 / 989,315 only under the legacy
   512-frame geometry (freq_bins 257). The earlier ~724K/~776K figures match the
   contract geometry; the 986K/989K figures come from `phase4_benchmark`/wrapper
   defaults (freq_bins=257, frame=512). See §4. Neither architecture is "at the
   1M edge" under the deployed contract — a point worth stating plainly.

## 3. Gates run and results (all VERIFIED MEASUREMENT)

| Gate | Result | Evidence |
|---|---|---|
| Deterministic reconstruction (WOLA) | **PASS** — boundary zero, exact 1-hop (128 smp / 8.0 ms) lag, sub-limiter impulse unity gain (0.5→0.5, <1e-3, no dispersion), sine/constant max err < 1e-4. Unit impulse is capped at 0.95 by the impulse-protection output limiter (fail-safe by design, NOT a reconstruction error) | `tests/test_streaming.py::test_causal_streaming_engine_deterministic_reconstruction` |
| Full test suite | **79/79 PASS** | `python -m pytest -q` |
| Isolated streaming latency (1000 steady-state hops) | **MARGINAL on this host — NOT reproducible as stable PASS.** 7 runs: P95 4.90–14.52 ms (median ≈ 6.9). Backend recorded as Numba in CSV. See §2a.2 | `results/csv/ph06_headroom_profile.csv` |
| PH0.5 DSP backends | Numba engaged (0.67.0); NLMS 87.9×, impulse 29.2× vs Python (isolated) | `results/csv/ph05_speedup_summary.csv` |

Latency note (Phase 17 discipline): the 8 ms gate is **per-hop processing time**, NOT
end-to-end latency. Algorithmic latency of the streaming path is frame_size = 16 ms
plus buffering; the two are reported separately in all PH1 outputs.

## 4. Model inventory (parameter counts computed dynamically, never hard-coded)

Parameter counts are GEOMETRY-DEPENDENT and must be stated with their geometry:

| Model | Params @ contract (frame 256 / bins 129) | Params @ legacy (frame 512 / bins 257) | fp32 @ contract | Status in repo |
|---|---:|---:|---:|---|
| TinyEnhancer | **9,569** | 9,569 (geometry-independent) | 37.4 KiB | Implemented (`src/ai/tiny_enhancer.py`) — deployed Stage 2 |
| CRN-Micro | **723,801** | 986,457 | 2.76 MiB | Implemented (`src/ai/crn.py`) — untrained, random init |
| DTLN | **775,939** | 989,315 | 2.96 MiB | Implemented (`src/ai/dtln.py`) — untrained, random init |
| DeepFilterNet2 | — | — | — | **NOT PRESENT.** No code, no adapter, no checkpoint. Papers only (`research/papers/`). Requires external integration (`deepfilternet` package) before any DFN2 job can run. |

Notes:
- 9,569 (not the stale 10,417) is confirmed as TinyEnhancer's true count.
- **Geometry mismatch to fix in the adapter layer (Phase 19 / Step 8):** the
  CRN/DTLN wrappers and `src/evaluation/phase4_benchmark.py` default to the
  legacy geometry (`freq_bins=257`, `frame_size=512`). Every PH1 instantiation
  must override to `freq_bins=129` / `frame_size=256` so parameter counts and
  behaviour match the frozen deployment contract. Until that override exists,
  wrapper-default counts (986K/989K) do not describe the deployed system.
- Under the contract geometry CRN/DTLN sit ~0.72M/~0.78M — comfortably under the
  1M budget (Phase 5), though still ~75× TinyEnhancer.

## 5. Dataset / manifests (Phase 3 resolution)

- Base manifest: `data/v4/metadata/metadata_v4.csv` — **110 rows, SHA-256
  `83a84a08…d78ff3`, verified unchanged** (freeze script asserts this hash).
- Split definitions (enforced in code by `DatasetLoader` + speaker-disjointness check):
  - **TRAIN** = SPK_001..SPK_008 (56 rows)
  - **TEST_A_UNSEEN_SPEAKER** = SPK_009..SPK_010 only, never in training (18 rows)
  - **TEST_B_UNSEEN_NOISE_REC** = held-out noise recordings, speakers SPK_001..008
  - **TEST_C_UNSEEN_NOISE_CATEGORY** = impulsive defence category, held-out class (20 rows)
- **TEST_B gap found and closed:** base TEST_B had only 2 noise classes (tank engine =
  STATIONARY, helicopter rotor = PERIODIC_ROTOR) against the required ≥ 8.
  `experiments/scripts/ph1_testb_extension.py` procedurally generated the 6 missing
  classes with fixed seeds: **drone/UAV, wind, mechanical/armoured, tactical siren,
  gunshot transient, artillery broadband** — 192 new samples
  (6 classes × 8 speakers × 4 SNRs: −5/0/+5/+10 dB), plus recorded per-sample
  parameters (speaker ID, noise ID, SNR target **and** achieved, leakage 0.15,
  delay samples, gain, impulse positions in samples, per-sample seed).
- Generation is deterministic and verified: **rerun reproduces byte-identical audio
  (DETERMINISM: PASS), --verify re-checks every file hash (PASS), 0 hash mismatches,
  0 clipped files, 0 non-finite samples, achieved-SNR self-report errors 0/192.**
  Impulsive mixes avoid clipping by mixture scaling; the honestly measured
  `actual_snr_db` is recorded (drift up to ~6 dB vs target on impulsive classes is
  stated, not hidden).
- Canonical frozen manifest: `data/v4/metadata/metadata_v4_extended.csv` (302 rows,
  SHA-256 `8f895a92…00e7`); full provenance in
  `experiments/manifests/ph1_frozen_canonical_manifest.json` (SHA-256 of manifest +
  all 614 referenced audio files).
- **Provenance label (binding):** PROCEDURAL / SYNTHETIC DEFENCE-NOISE.
  This suite is **NOT real battlefield data**. Any future real recordings go to a
  separate `TEST_C_REAL_DEFENCE` and are never mixed into synthetic results.

## 6. Execution readiness

| Item | Value |
|---|---|
| Model implementations available now | TinyEnhancer, CRN-Micro, DTLN (inference + training paths) |
| Models requiring external integration | DeepFilterNet2 (absent from repo) |
| Planned parallel jobs (primary matrix) | 4 models × 4 front ends = 16, + noisy/NLMS/VSS-NLMS-only baselines (mandatory) |
| Compute available | 8 CPU threads, 17 GB RAM, no CUDA — jobs run serial-to-2× parallel; CPU-only training pace is the main schedule risk |
| Supervisor | `scripts/ph1_supervisor.py` — **not yet written** (Phase 13) |
| Experiment contract | **frozen**: `configs/ph1_experiment_contract.yaml` (v1) |

## 7. Blockers / risks (honest list)

1. **No GPU** — 30-epoch training of 3 models × 16 configs on 8 CPU threads will be
   slow; expect the RESIDUAL_AI leg (which must run NLMS per sample) to dominate.
2. **DeepFilterNet2 integration** is an open work item (package + 16 kHz resampling
   contract + streaming adapter). No DFN2 numbers can exist before that lands.
3. **PyTorch AI stage latency** P95 ≈ 2.8–10.9 ms across runs on this host —
   DSP (P95 ≈ 0.6–1.0 ms) is safely inside the budget but the AI stage puts the
   integrated 8 ms gate at the margin; it is NOT reproducible as a stable pass on
   this shared host (§2a.2). Phase 17 reports DSP/AI/combined separately (it does).
4. **CRN/DTLN parameter budget** under the contract geometry is ~0.72M/~0.78M
   (not ~0.99M, which is the legacy-geometry figure) — under budget but still
   ~75× TinyEnhancer; keep the documented channel reduction as an option.
5. Impulsive SNR drift (up to ~6 dB under clipping-avoidance scaling) is recorded
   honestly but narrows the low-SNR impulsive test points.

## 8. What training is and is NOT allowed to change

**Allowed (per master prompt):**
- Batch size (auto from memory), optimizer = AdamW @ 1e-4, max 30 epochs,
  early stopping on validation SI-SDR, gradient clipping 5.0.
- Common hyperparameters in stage 1; model-specific tuning ONLY for top 2–3
  finalists, never against TEST_A/TEST_B.
- Loss variants (baseline → +complex STFT → +multi-res STFT → +SI-SDR weighting)
  only as controlled ablations on the top candidate(s).

**NOT allowed:**
- Any change to the frozen DSP contract (16 kHz / 256 / 128 / Hann / 50% /
  center=False / WOLA) — including "reverting to 512/256 because legacy training
  code uses it".
- Any change to NLMS algorithm, impulse-protection logic, metrics, or the 8 ms
  latency gate to make a candidate pass.
- Replacing real PESQ/STOI with surrogates; failing evaluation if unavailable.
- Regenerating TEST_A/TEST_B/TEST_C silently; training on any test split;
  converting paper numbers, host-PC latency, or synthetic-noise results into
  claims about real systems.
- Tuning until a chosen model wins (Phase 10 violation).

## 9. Compliance with EXECUTION ORDER (Phases 30)

| Step | Status |
|---|---|
| 1 Inspect repository | DONE (this report) |
| 2 Fix WOLA/reconstruction | Already fixed (PH0.7); verified by tests |
| 3 Reconstruction tests | PASS |
| 4 Full test suite | 79/79 PASS |
| 5 Latency benchmark | MARGINAL — not reproducible as stable PASS on this host (§2a.2); backend now recorded in CSV |
| 6 Freeze experiment contract | DONE — `configs/ph1_experiment_contract.yaml` |
| 7 Freeze dataset manifests | DONE — canonical + hashes (Sec. 5) |
| 8 Build model adapters | PARTIAL — TinyEnhancer/CRN/DTLN exist; DFN2 missing |
| 9 Inference verification (no training) | PENDING (next session) |
| 10–20 Baselines → matrix → rank → report | PENDING (not started, per FIRST ACTION stop rule) |

## 10. STOP

Training has not started. Per the master prompt, the next permitted actions are
model-adapter completion (DFN2 integration decision) and inference verification —
still before any training job.
