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
| Git commit | `cb5f4c4` — feat(ph0.6): evaluation integrity investigation and 1000-hop headroom benchmark |
| Frame / hop | 256 / 128 samples (16 kHz) — deployment; training now also 256/128 (Gate 2 fix applied) |
| center | `False` everywhere (train.py torch.stft/istft + deployment np.fft) |
| Runtime | Python 3.11, torch 2.14.0+cpu (CUDA: **not available**), numpy 2.4.6, numba 0.67.0 |
| Host | Windows 10, 8 CPU threads, 17.1 GB RAM (no GPU, no Raspberry Pi/Jetson attached) |

## 3. Gates run and results (all VERIFIED MEASUREMENT)

| Gate | Result | Evidence |
|---|---|---|
| Deterministic reconstruction (WOLA) | **PASS** — boundary zero, exact 1-hop (128 smp / 8.0 ms) lag, unity impulse peak, sine/constant max err < 1e-4 | `tests/test_streaming.py::test_causal_streaming_engine_deterministic_reconstruction` |
| Full test suite | **79/79 PASS** | `python -m pytest -q` |
| Isolated streaming latency (1000 steady-state hops) | **PASS, +2.47 ms headroom** — P95 within 8 ms gate; P99 7.52 ms; 1.0% of hops > 8 ms | `results/csv/ph06_headroom_profile.csv` |
| PH0.5 DSP backends | Numba engaged (0.67.0); NLMS 87.9×, impulse 29.2× vs Python (isolated) | `results/csv/ph05_speedup_summary.csv` |

Latency note (Phase 17 discipline): the 8 ms gate is **per-hop processing time**, NOT
end-to-end latency. Algorithmic latency of the streaming path is frame_size = 16 ms
plus buffering; the two are reported separately in all PH1 outputs.

## 4. Model inventory (parameter counts computed dynamically, never hard-coded)

| Model | Params (dynamic) | fp32 size | Status in repo |
|---|---:|---:|---|
| TinyEnhancer | **9,569** | 37.4 KB | Implemented (`src/ai/tiny_enhancer.py`) — deployed Stage 2 |
| CRN-Micro | **986,457** | 3.85 MB | Implemented (`src/ai/crn.py`) — untrained, random init |
| DTLN | **989,315** | 3.86 MB | Implemented (`src/ai/dtln.py`) — untrained, random init |
| DeepFilterNet2 | — | — | **NOT PRESENT.** No code, no adapter, no checkpoint. Papers only (`research/papers/`). Requires external integration (`deepfilternet` package) before any DFN2 job can run. |

Note: 9,569 (not the stale 10,417) is confirmed as TinyEnhancer's true count.
CRN/DTLN are near the 1M budget edge — Phase 5 says "prefer significantly below 1M
if quality is retained"; their channel configuration may need a documented reduction
before the campaign, as a controlled design decision (not silently).

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
3. **PyTorch AI stage latency** p95 ≈ 5–11 ms on this host — the 8 ms *processing*
   gate holds for DSP but integrated per-hop processing sits at the edge; Phase 17
   must report DSP/AI/combined latency separately (it does).
4. **CRN/DTLN parameter budget** sits at ~0.99M — at, not below, the 1M target.
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
| 5 Latency benchmark | PASS (+2.47 ms headroom) |
| 6 Freeze experiment contract | DONE — `configs/ph1_experiment_contract.yaml` |
| 7 Freeze dataset manifests | DONE — canonical + hashes (Sec. 5) |
| 8 Build model adapters | PARTIAL — TinyEnhancer/CRN/DTLN exist; DFN2 missing |
| 9 Inference verification (no training) | PENDING (next session) |
| 10–20 Baselines → matrix → rank → report | PENDING (not started, per FIRST ACTION stop rule) |

## 10. STOP

Training has not started. Per the master prompt, the next permitted actions are
model-adapter completion (DFN2 integration decision) and inference verification —
still before any training job.
