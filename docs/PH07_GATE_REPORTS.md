# PH0.7 Gate Reports

## PH0.7 GATE 1 -- CODE INTEGRITY

**STATUS: PASS (with one item deferred to Gate 2)**

### Checked:

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 1 | train.py has center=False on every torch.stft | **DEFERRED to G2** | center= is NOT explicitly set. PyTorch default is center=True. Docstring note 6 explicitly flags this as pending Gate 2 resolution. |
| 2 | torch.istft uses causally consistent convention | **DEFERRED to G2** | L259: torch.istft inherits default center=True. Must be resolved with item 1. |
| 3 | Validation split is TEST_A_UNSEEN_SPEAKER | **PASS** | L139: val_split defaults to "TEST_A_UNSEEN_SPEAKER". Dataset confirms 18 samples. |
| 4 | Empty validation raises | **PASS** | L184: val_loader.assert_nonempty("validation") raises RuntimeError. |
| 5 | Train/val speakers are disjoint | **PASS** | L187-192: overlap check raises on non-empty. Dataset: TRAIN=SPK_001-008, VAL=SPK_009-010, overlap=set(). |
| 6 | Random seed exists | **PASS** | L140: seed=42. L176-177: torch.manual_seed + np.random.seed. |
| 7 | Parameter count is dynamic | **PASS** | L200: sum(p.numel()...). L337: metadata uses param_count variable. |
| 8 | Checkpoint saving computes SHA-256 | **PASS** | L328: _sha256_of_file(save_p). L120-125: reads and hashes file. |
| 9 | Run metadata is written | **PASS** | L334-364: JSON sidecar with all required fields including training_input/deployment_input/input_contract_status. |
| 10 | IRM is not falsely described as active loss | **PASS** | L233-234: Comment says "diagnostic reference ONLY". Loss at L261: alpha*l1_loss+(1-alpha)*si_loss. IRM not in loss. |
| 11 | metrics.py cannot silently fabricate PESQ/STOI | **PASS** | compute_pesq raises RuntimeError if not PESQ_AVAILABLE. compute_stoi raises RuntimeError if not PYSTOI_AVAILABLE. Surrogates are separate functions. |

### Files changed:
- src/ai/train.py -- replaced with PH0.7 patched version
- src/evaluation/metrics.py -- replaced with PH0.7 patched version

### Remaining risk:
- center= resolution is critical and deferred to Gate 2.

---

## PH0.7 GATE 2 -- DSP CONTRACT

**STATUS: FAIL -- Train/deployment STFT mismatch detected. center= inconsistency confirmed.**

### STFT Configuration Comparison:

| Parameter | Training | CausalEngine | HybridPipeline |
|-----------|----------|--------------|----------------|
| frame_size | 512 | **256** | 512 |
| hop_size | 256 | **128** | 256 |
| center | **True** (default) | N/A (manual) | N/A (manual) |
| FFT bins | 257 | **129** | 257 |
| STFT library | torch.stft | np.fft.rfft | np.fft.rfft |

### OLA Normalization Comparison:

| Path | Method | Normalization |
|------|--------|---------------|
| Training | torch.istft (COLA) | PyTorch internal |
| CausalEngine | overlap_buf += recon | **NONE** (attenuated by ~0.75) |
| HybridPipeline | output += recon*window; norm += window^2; output /= norm | window^2 division |

**OLA MATCH: NO -- three different reconstruction methods**

### Phase Convention:
MATCH: YES -- all preserve noisy/input phase, magnitude-only enhancement

### Critical Findings:
1. Training uses center=True (PyTorch default). Deployment uses manual FFT (no centering). center=True pads frame_size/2 zeros on each side -- this is a causality violation per the project's own causality verifier.
2. CausalEngine frame=256, Training frame=512. The Conv2D model is spatially invariant so it can handle different freq-bin counts, but the spectral resolution differs.
3. CausalEngine OLA does NOT apply window normalization. Hann^2 with 50% overlap sums to 0.75 -- output is attenuated.

### Action Required:
- center=False MUST be added to train.py (per master handoff, non-negotiable)
- frame_size/hop_size mismatch is a design decision requiring user approval
- OLA normalization inconsistency must be resolved canonically

### Action Taken:
Applying center=False to train.py. NOT changing frame_size/hop_size -- that is a design decision.

---

## PH0.7 GATE 3 -- METRIC PROVENANCE

**STATUS: PASS**

```
PESQ_AVAILABLE: True   (pesq 0.0.4)
PYSTOI_AVAILABLE: True (pystoi 0.4.1)
Python: 3.13.5
NumPy: 2.3.4
Torch: 2.9.0
```

Real PESQ and STOI are available. Official evaluation is NOT blocked.
Historical PESQ=1.50 / STOI=0.896: REQUIRES REVALIDATION.

---

## PH0.7 GATE 4 -- TEST SUITE

**STATUS: FAIL -- test_08b exceeds 8 ms budget even in isolation**

### Full suite run (77 other tests concurrent):
```
77 passed, 1 failed in 151.67s
test_08b: P95 = 9.37 ms > 8.00 ms
```

### Isolated run (test_08 only, no other tests):
```
1 failed in 37.66s
test_08b: P95 = 8.93 ms > 8.00 ms
```

### Analysis:
- Isolation reduced P95 from 9.37ms to 8.93ms (0.44ms improvement)
- But 8.93ms still exceeds the 8.00ms budget by 0.93ms
- This is NOT contention flakiness -- it is a genuine budget exceedance
- PH0.6 reported P95 = 6.61ms on the same machine -- the difference may be due to system state, background processes, or library version changes
- Test 8a (streaming mechanism correctness) PASSES in both runs

### Per master handoff:
- The 8ms assertion is NOT weakened
- The failure is documented honestly
- This does NOT block training (test 8 is a latency benchmark, not a correctness test)
- The latency issue is separate from the training pipeline validation

### Remaining risk:
- Laptop latency is marginal -- P95 is within 1ms of the budget
- Real-time compliance on Pi 4 remains untested

