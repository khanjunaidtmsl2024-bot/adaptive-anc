# 06 — REBUILD GUIDE: Teammate ANC Repo (`ichigo137/anc`)

**Purpose:** Complete, file-by-file explanation of the teammate's repository so any team member can understand it and rebuild it without him.
**Source repo:** https://github.com/ichigo137/anc  (public · owner "Pabitra Roy" · created 30 Aug 2026)
**Document compiled:** 4 Sep 2026
**Scope note:** The URL `ichigo137/ancn` does not exist; this document covers `ichigo137/anc`.

---

## 0. TL;DR — what this repo is

A **single-channel, offline speech-enhancement experiment** (noise-reduction only): 10 phone-recorded clean clips → mixed with synthetic noise (white / pink / 50–150 Hz hum / synthetic impulses) at 6 SNRs → tiny CNN trained in the STFT domain → noisy files enhanced → SNR / SI-SDR / STOI / correlation measured.

**It is NOT an ANC system** and does **not** touch most of PS 26052 yet:

- no reference microphone / two-mic processing
- no adaptive filter (LMS/NLMS/FxLMS)
- no controller / impulse protection
- no streaming / latency work
- no defence noise (helicopter/engine/siren/gunshot)
- no PESQ, no ONNX/quantization/edge work

Treat it as the **single-channel AI baseline block** (roughly E004/E009 territory of the V1 plan), not the final system.

---

## 1. How this guide was verified (do not skip)

Every claim was checked against the live repository on 4 Sep 2026, not chat memory:

| Check | Method |
|---|---|
| Full repo cloned locally | `git clone -b dev`; `origin/main` also fetched |
| Every text file read | all 14 `src/*.py` files + both `.gitignore` files |
| Duplicate detection | `sha256sum` over every `src` file |
| Checkpoint contents | `torch.load` of all 5 `.pt` files (keys + config printed) |
| Checkpoint ↔ architecture | `load_state_dict` test against both model classes |
| Audio formats | Python `wave` probe of `clean/`, `clean_raw/`, `noisy/`, `noisy_test/` |
| File counts / tree | full `find` listing of the `dev` tree |
| main vs dev | `git diff --name-status origin/main HEAD` + `md5sum` comparisons |
| Commits | GitHub API + local git log |

Anything marked *(unverified)* could not be determined from the repo alone and needs the teammate.

---

## 2. Repository identity and branch map

- No README, no LICENSE, no `requirements.txt`, no docs anywhere in the repo.
- Working tree ≈ **789 MB** — almost all of it WAV audio.
- Three branches:

| Branch | Points at | Meaning |
|---|---|---|
| `main` | `ef1a4da` "initial" (30 Aug) | Frozen at the very first commit. Stale. |
| `test` | `ef1a4da` (same commit) | Empty experiment branch. No extra work. |
| `dev` | `085ee89` (1 Sep) | **The real current work — always analyse/run this branch.** |

**Team rule:** clone with `git clone -b dev https://github.com/ichigo137/anc`; ask the teammate to merge `dev` → `main` (or delete the ghost branches) so nobody analyses a stale copy again.

---

## 3. Commit history (`dev`)

| Commit | Date | Message | Effect (verified by diff/file state) |
|---|---|---|---|
| `ef1a4da` | 30 Aug | initial | baseline state (what `main`/`test` still show) |
| `db12a06` | 30 Aug | baseline v1 speech enhancement evaluation | first trainer/eval outputs |
| `3aee986` | 30 Aug | Add V2 complex-domain speech enhancement experiment | `train_v2.py`, `inference_v2.py`, `evaluate_v2.py`, `tiny_enhancer_v2.pt` added |
| `fde2f4f` | 1 Sep | modified train.py | `train.py` moved to log-domain + SNR-aware loss |
| `bbb04f3` | 1 Sep | modified infer.py | `inference.py` moved to chunked 4-s processing |
| `085ee89` | 1 Sep | modified train.py (dev HEAD) | final retrain + regenerated evaluation outputs |

`dev` is **5 commits ahead of main**. On `dev`, `evaluate.py` ≡ `evaluate_v1.py` and `train_backup_v1.py` ≡ `train_broken_edit.py` are byte-identical duplicates (checksum-verified) — a code-hygiene problem, not different experiments.

---

## 4. Complete file tree (`dev` HEAD)

```
anc/  (~789 MB)
├── .gitignore                     git ignore rules (root)
├── dataset/
│   ├── clean/         11 files    clean targets: speech_001.wav … speech_010.wav + speech_001.m4a (16 kHz mono)
│   ├── clean_raw/     10 files    original phone recordings (48 kHz stereo; see §6) — NOT used by any script
│   ├── noise/          0 files    folder exists (code references it) but is EMPTY — no real noise committed
│   ├── noisy/        192 files    training mixtures: speech_001…008 × 4 noise types × 6 SNRs
│   └── noisy_test/    48 files    held-out test mixtures: speech_009 & speech_010 × 4 noise × 6 SNRs
├── models/             5 files    PyTorch checkpoints (see §9)
├── output/
│   ├── *.wav                      stale + current enhanced results
│   └── evaluation/    48 files    current batch evaluation outputs (speech_009/010)
└── src/ (14 .py + 1 .gitignore)   all code — detailed in §8
```

**Corrected facts vs earlier chat analysis:** `src/` has **14 Python files, not 16**; `noisy/` is **192 files, not ~240**; `noisy_test/` is **48 files covering speech_009 AND speech_010**, not 24.

---

## 5. The pipeline in one picture

```
[phone recordings]  →  clean_raw/  (48 kHz stereo, messy headers)
        │  (manual conversion — no script committed)
        ▼
 clean/  (16 kHz mono WAV)        ← clean target for training
        │
        ▼  src/generate_dataset.py        (no seed → NOT reproducible)
 clean + synthetic noise @ SNR −5…20 dB  →  noisy/  (192 train files)
        │
        ▼  src/train.py  (or train_v1.py, train_v2.py)
 noisy/ → 4-s chunks → STFT → tiny CNN → mask/spectrum estimate
        │
        ▼  best-val checkpoint → models/*.pt
        │
        ▼  src/inference.py / _v1 / _v2
 noisy_test file → enhance → output/*_enhanced.wav
        │
        ▼  src/evaluate.py / evaluate_v2.py   (calls inference via subprocess)
 clean vs noisy vs enhanced → SNR, SI-SDR, STOI, correlation → printed report
        (results are console-printed only — never saved to a file)
```

---

## 6. Data inventory (verified)

### 6.1 Folders and true contents

| Folder | Files | Composition | Notes |
|---|---|---|---|
| `clean/` | 11 | speech_001.wav…010.wav + **speech_001.m4a** (leftover) | 16 kHz, mono, 16-bit; durations ≈ 6.1 s (001) to 61.2 s (010) |
| `clean_raw/` | 10 | speech_001…010 | speech_001 = 16 kHz mono 6.1 s; speech_002–010 = **48 kHz stereo** phone recordings |
| `noise/` | 0 | — | generator expects real noise here (`*.wav`) but nothing was ever committed |
| `noisy/` | 192 | 8 speakers × 4 noise types × 6 SNR | the training universe |
| `noisy_test/` | 48 | speakers **009 & 010** × 4 noise × 6 SNR | true unseen-speaker test set (see §6.3) |

SNR levels used everywhere: **−5, 0, 5, 10, 15, 20 dB**. Noise types: `hum`, `impulsive`, `pink`, `white` (all synthetic).

### 6.2 `clean_raw/` header problem — important

Several raw files (e.g., `speech_002.wav`, `speech_003.wav`) have **corrupt length headers**: Python's `wave` module reads them as ~6.2 hours long, while file size corresponds to ~20–60 s of real audio. The audio content is fine (librosa/soundfile read it); the header metadata is unreliable — never trust header lengths on these files.

### 6.3 The test split — what it actually proves

- Training mixtures live in `noisy/` (speakers 1–8). `train.py` further splits these 8 speakers 80/20 by recording ID → ~6 train speakers, 2 validation speakers. **No leakage** within `noisy/`.
- The evaluator tests `noisy_test/` = speakers **9 and 10**, never seen in training → genuine **unseen-speaker** test. Good.
- But the noise **types and generators are identical** between train and test (same recipe, same SNR grid) → this is **NOT an unseen-noise / robustness test**. The "impulsive" clips are synthetic decaying bursts, not gunshots/impacts (no defence relevance yet).

### 6.4 Naming convention = metadata

```
speech_009__hum__snr-5.wav   →  speaker=speech_009 · noise=hum · target SNR=−5 dB
```

All scripts parse filenames with `.split("__")` — fragile by design but consistent.

---

## 7. Model generations at a glance (all offline, non-causal)

| Generation | Files | Input representation | Output | Loss | Status on `dev` |
|---|---|---|---|---|---|
| V1 (magnitude mask) | `train_v1.py`, `inference_v1.py` | linear magnitude | IRM mask | 0.2·MSE(mask) + 0.8·MAE(mask×noisy − clean) | legacy |
| Current ("V3" loss, log-domain) | `train.py`, `inference.py` | log1p magnitude | mask on log spectrum | 0.20·MSE(mask) + 0.75·MAE(recon) + 0.05·identity | **default path** |
| V2 (complex domain) | `train_v2.py`, `inference_v2.py`, `evaluate_v2.py` | real+imag STFT channels | estimated complex spectrum | 0.75·L1(complex) + 0.25·L1(magnitude) | **code present, checkpoint broken — see §9** |

The "V3" name appears only inside `train.py`'s comments; there is no `train_v3.py`.

---

## 8. `src/` — file-by-file detail

### 8.1 `src/generate_dataset.py` (219 lines) — dataset builder

**Libraries:** `pathlib`, `numpy`, `soundfile` (write), `librosa` (read/resample). No torch.

- **Constants:** `SR = 16000`, `SNR_LEVELS = [-5, 0, 5, 10, 15, 20]`.
- `load_audio(path)` → `librosa.load(sr=16000, mono=True)`, float32, **peak-normalised** (`audio /= max|audio|`).
- `match_length(audio, length)` → random crop if longer; `np.tile` + trim if shorter.
- `add_noise(clean, noise, snr_db)` — the SNR math is **correct**: `P_noise_desired = P_clean / 10^(SNR/10)`; `scale = sqrt(P_noise_desired / P_noise)`; `noisy = clean + scale·noise`; if peak > 0.99 rescale to 0.99 (clipping guard).
- `generate_synthetic_noise(length, type)`:
  - `white` → `np.random.normal(0,1)`
  - `pink` → FFT-shape white noise by `1/sqrt(f)` (approximate)
  - `hum` → `0.7·sin(2π·50·t) + 0.3·sin(2π·100·t) + 0.15·sin(2π·150·t)` (electrical/mechanical hum)
  - `impulsive` → small Gaussian background + random exponentially-decaying bursts (width ≤ 80 samples, decay `/15`, amplitude 0.5–1.0). **Synthetic clicks, not real transients.**
- **Main loop:** for every clean file → for each noise (real if present, else 4 synthetic) → 6 SNRs → writes `PCM_16` WAV to `noisy/`.
- **Critical defect: NO random seed** → rerunning produces a *different* dataset (violates V6 §72–73 reproducibility). It would also write all 10 speakers (240 files); the committed state has 8 speakers in `noisy/` because 48 files were later moved into `noisy_test/` manually — the move is not encoded in any script.
- `dataset/noise/` is read but **empty** → the "real noise" feature was never actually used.

### 8.2 Trainers

**Shared behaviour (all trainers):** read `noisy/`; map each file to its clean partner by prefix; trim to equal length; take a 4-second chunk (random start in train, **center crop in validation**); zero-pad if shorter; STFT with `n_fft=512, hop=128, win=512` (≈ 257 freq bins × ~501 time frames per chunk); 3×3 padded CNNs; split by **recording ID** (leakage-free); `SEED = 42`; save only the best-validation checkpoint as a **dict** (`model_state_dict` + config) — never a raw model.

#### `src/train.py` (589 lines) — CURRENT default trainer
**Libraries:** `pathlib`, `random`, `numpy`, `librosa`, `torch`, `torch.nn`, `torch.utils.data`.

- **Hyperparameters:** 16 kHz · n_fft 512 · hop 128 · win 512 · 4-s chunks · **30 epochs** · batch 4 · **lr 1e-3** · Adam · seed 42.
- **Dataset** parses SNR from the filename (`stem.split("__snr")[-1]`) and returns 4 tensors: noisy log-magnitude, target mask, clean log-magnitude, SNR value.
- **Front end:** magnitude → `log1p` → noisy & clean divided by `max(noisy_log)+1e-8`. Target mask = `clip(clean_log/(noisy_log+1e-8), 0, 1)` — a **log-domain ratio mask**, not the classic linear IRM.
- **Model `TinyEnhancer`** (the whole repo's workhorse — **9,569 params**): `Conv2d(1→16) → ReLU → Conv2d(16→32) → ReLU → Conv2d(32→16) → ReLU → Conv2d(16→1) → Sigmoid`; mask ∈ [0,1].
- **Loss** (commented "V3" in the file — the interesting part):
  - `mask_loss` = MSE(predicted_mask, target_mask) × **0.20**
  - `magnitude_loss` = mean|predicted_mask·noisy − clean| (L1 reconstruction) × **0.75**
  - `identity_loss` = mean((predicted_mask − 1)²) × **0.05**, weighted per sample by `clamp((SNR−10)/10, 0, 1)` — at high input SNR the model is pushed to leave speech alone (**anti over-suppression heuristic**); ≤10 dB → no penalty, ≥20 dB → full penalty.
- **Quirks:** duplicated dead code (two `best_val_loss = inf` blocks, double "Starting training…" print); saves to `models/tiny_enhancer.pt`.

#### `src/train_v1.py` (516 lines) — V1, magnitude IRM
Same shell as `train.py` (chunking / STFT / split / seeds / save location). **Differences:** raw **linear** magnitude (no log1p), normalized by `noisy_mag.max()`; target = linear IRM clipped [0,1]; dataset returns only 3 items (no SNR); loss = **0.2·MSE(mask) + 0.8·MAE(mask×noisy_mag − clean_mag)** — no identity term. Legacy — superseded by `train.py`.

#### `src/train_backup_v1.py` (483 lines) ≡ `src/train_broken_edit.py` (byte-identical, checksum `d0b2d616…`)
Earliest surviving trainer variant (single MSE `criterion` — plain regression, no 0.2/0.8 weighting), saves to `tiny_enhancer.pt`. Legacy snapshot — **do not run**; two identical copies are still committed.

#### `src/train_v2.py` (738 lines) — V2, complex-domain
- **Hyperparameters:** **40 epochs** · batch 4 · **lr 5e-4** · Adam · gradient clipping `max_norm=5.0` · seed 42.
- **Front end:** complex STFT; noisy & clean divided by the same scale `max|noisy_stft|+1e-8`; then **stacked as 2 channels [real, imag]** → input `[2, 257, ~501]`; target = clean complex spectrum in the same form.
- **Model `TinyComplexEnhancer`** (**75,746 params**): `Conv2d(2→32)+BN+ReLU` → 4 `ResidualBlock(32)` with **dilation 1, 2, 4, 8** (residual `x + block(x)`; each block = 2× dilated conv+BN with ReLU between) → `Conv2d(32→2)` → **tanh** (output channels = predicted real/imag).
- **Loss:** `0.75·L1(pred−target)` on the complex plane + `0.25·L1(|pred|−|target|)` on magnitudes.
- **Save:** rich dict with `model_type="TinyComplexEnhancer"`, `input_channels=2`, `output_channels=2`, `best_val_loss` + usual STFT keys → `models/tiny_enhancer_v2.pt`.
- **⚠ Critical:** the format this script writes does **not** match the file currently at `models/tiny_enhancer_v2.pt` (see §9). A fresh run of `train_v2.py` is required to produce a valid V2 checkpoint.

### 8.3 Inference scripts

**Shared contract:** load checkpoint dict → rebuild model class → `load_state_dict` → `model.eval()` → read config (`sr`, `n_fft`, `hop`, `win`, `chunk_seconds`) from the checkpoint → CLI `[in, out]` → save PCM WAV at the same sample rate with a 0.99 clip guard.

#### `src/inference.py` (399 lines) — current path (matches `train.py`)
Splits audio into **4-s chunks** (padding the last), processes each chunk independently: STFT → `log1p` magnitude → normalize by the *chunk's own* `max+1e-8` (consistent with training) → CNN → mask → `estimated_clean_log = mask × noisy_log` (unnormalised) → `expm1` → magnitude; **reuses the noisy phase**; iSTFT per chunk; concatenate; trim to the original length.
**Known weak spot:** chunks are normalized/processed independently with no overlap or cross-fade → possible level discontinuities at 4-s boundaries (offline-quality concern only — no real-time claim).

#### `src/inference_v1.py` (276 lines) — legacy whole-file path
No chunking: STFT whole file → normalize magnitude by max → CNN mask → **mask × original (unnormalised) magnitude** → noisy phase → single iSTFT. Matches V1 training.

#### `src/inference_v2.py` (340 lines) — complex path (matches `train_v2.py`)
Whole-file complex STFT → normalize by `max|STFT|` → stack [real, imag] → `TinyComplexEnhancer` → tanh → **scale restored** → iSTFT (length = input).
**⚠ Broken as committed:** it builds `TinyComplexEnhancer`, but the committed `tiny_enhancer_v2.pt` holds `TinyEnhancer` (mask) weights → `load_state_dict` raises `RuntimeError` on the first layer (verified by direct load test, §9). The script cannot run until a valid V2 checkpoint exists.

### 8.4 Evaluation scripts

**Shared contract (all three are the same file except the inference subprocess target):** for every file in `noisy_test/` → find its clean partner → run the inference script via `subprocess.run([sys.executable, INFERENCE_SCRIPT, noisy, enhanced])` → load clean/noisy/enhanced at 16 kHz → truncate to the common length → compute metrics → per-file prints → overall averages → breakdowns by noise type and SNR level.

**Metrics implemented locally (only STOI is a library call):**
- `snr_db(clean, x)` = `10·log10(P_clean / P_(x−clean))` — residual noise = difference from the clean reference.
- `si_sdr(reference, estimate)` — DC removed, estimate projected onto reference, then `10·log10(P_target/P_residual)` (correct SI-SDR construction).
- `speech_intelligibility(...)` = `pystoi.stoi(clean, enhanced, 16000, extended=False)` inside try/except → silently returns 0.0 on any failure.
- `correlation(a, b)` = numpy correlation coefficient.

**Outputs:** console tables only. **Metrics are never written to a file** — no CSV/log exists in the repo, so numeric results from past runs are lost to history (real evidence-management gap).

| File | Inference script it calls | Status on `dev` |
|---|---|---|
| `evaluate.py` (655 lines) | `src/inference.py` (current mask model) | runs (needs librosa + pystoi installed) |
| `evaluate_v1.py` (655 lines) | same — **byte-identical** to `evaluate.py` (checksum `b55f353f…`) | duplicate |
| `evaluate_v2.py` (656 lines) | `src/inference_v2.py` | **broken end-to-end** — `inference_v2.py` can't load the committed checkpoint |

Note: both evaluators write to the **same** `output/evaluation/` folder with identical filenames — running one overwrites the other's outputs. Provenance of the committed `output/evaluation/*.wav` files is therefore ambiguous without the teammate.

### 8.5 Tooling

#### `src/benchmark_gpu.py` (140 lines) — CUDA throughput check
Re-declares `TinyEnhancer`, builds a synthetic batch `[4, 1, 257, 501]` (the 4-s STFT shape), runs 20 warm-up + 100 timed train steps (Adam + MSE), prints step time, steps/s and `torch.cuda.max_memory_allocated`.
- **Answers:** "how fast can this machine *train* this net?" — it does **not** measure enhancement latency.
- **CPU-only machines will crash** (it calls `torch.cuda.*` unconditionally after a cpu fallback). GPU machine only.

### 8.6 Broken / legacy files and config files

#### `src/test_model.py` (48 lines) — genuinely broken, delete
- Does `model = torch.load(...)` then `model.eval()` — but the checkpoint is a **dict**, so `.eval()` fails.
- Feeds a raw 1-D waveform tensor into a 2-D CNN expecting `[1, 1, F, T]`.
- Writes to `dataset/enhanced/`, which doesn't exist.
- Confirmed experiment debris.

#### `.gitignore` (root) and `src/.gitignore`
- Root: standard Python ignores (`.venv/`, `__pycache__`, IDE/OS files, `.env`, logs, `*.egg-info/`). Nothing algorithm-related.
- `src/.gitignore`: single line `/.venv` (a stray virtualenv once lived inside `src/`).

---

## 9. `models/` — checkpoint verification table (verified 4 Sep 2026)

All five are torch zip archives, loadable with `torch.load(weights_only=False)`; all store the plain dict `{model_state_dict, sample_rate:16000, n_fft:512, hop_length:128, win_length:512, chunk_seconds:4}` — except `tiny_enhancer_baseline.pt`, which **lacks** the `chunk_seconds` key.

| File | Size | md5 (12 chars) | Tensors | Loads into `TinyEnhancer` (mask, 9.5K)? | Loads into `TinyComplexEnhancer` (75.7K)? |
|---|---|---|---|---|---|
| `tiny_enhancer.pt` | 42,337 B | `2331c4d3c854` | 8 | OK | RuntimeError |
| `tiny_enhancer_baseline.pt` | 42,209 B | `2dfcb110e424` | 8 | OK | RuntimeError |
| `tiny_enhancer_exp1.pt` | 42,209 B | `d83e65165295` | 8 | OK | RuntimeError |
| `tiny_enhancer_v1.pt` | 42,337 B | `1f8516169bc9` | 8 | OK | RuntimeError |
| `tiny_enhancer_v2.pt` | 42,337 B | `6bccb06e1a71` | 8 | OK | RuntimeError |

**Headline finding — the V2 checkpoint is mislabelled.** `train_v2.py` saves a *rich* dict (`model_type="TinyComplexEnhancer"`, `input_channels`, `output_channels`, `best_val_loss`) and only ever writes complex weights (75,746 params, 65 tensors, keys like `input_layer.0.weight`). The file at `models/tiny_enhancer_v2.pt` instead holds the **plain 6-key dict with 8 mask-CNN tensors** (`network.0.weight` …) — exactly the format `train.py` writes. Verified by direct `load_state_dict` test:

> **`inference_v2.py` + `evaluate_v2.py` are broken against the committed `tiny_enhancer_v2.pt`. The true V2 complex checkpoint is missing or was overwritten. Fix: re-run `train_v2.py` (or recover the original file), then confirm the new `tiny_enhancer_v2.pt` loads into `TinyComplexEnhancer` before trusting any V2 result.**

Second consequence: **none of the 5 checkpoints contains complex-domain weights** → there are *zero* committed results from the V2 complex model. Every enhanced WAV in `output/` comes from the 9.5K-param mask CNN (`train.py`/`inference.py` family). *(Which exact checkpoint produced the current `output/evaluation/*.wav` files is unverified — filenames don't record it.)*

Parameter counts (computed from the repo's own class definitions): `TinyEnhancer` = **9,569**; `TinyComplexEnhancer` = **75,746**.

---

## 10. `output/` inventory and provenance gap

| Path | What it is |
|---|---|
| `output/enhanced.wav`, `output/test.wav` | stale leftovers from early single-file runs |
| `output/v2_test.wav` (dev only) | leftover V2 single-file experiment |
| `output/speech_009__hum__snr-5_enhanced.wav` | root-level leftover of one current-path run |
| `output/evaluation/` (48 files) | batch outputs for speech_009 + speech_010 × 24 conditions |

Provenance gaps: no script logs which model/checkpoint/commit produced which WAV, and evaluation metric tables were only ever printed to stdout. Rebuilding = regenerating both audio and metrics.

---

## 11. What is genuinely good (keep and reuse)

1. **Leakage-free splitting by original recording ID** — and the held-out `noisy_test/` (speakers 9–10) is truly unseen. Most student repos get this wrong; this one gets it right.
2. **Correct SNR mixing math + clipping guard** in `generate_dataset.py` (power-based scaling, 0.99 peak rescale).
3. **Train/inference normalization consistency in the current path** (`train.py` ↔ `inference.py`, both per-4-s-window on `log1p` magnitude) — the #1 silent killer of mask models, handled properly.
4. **Checkpoints store their own STFT config** — model/config drift is much less likely.
5. **SI-SDR implemented correctly** (projection onto reference, DC removal); STOI via the validated `pystoi` package.
6. **The SNR-aware identity loss** in `train.py` is a thoughtful over-suppression guard — worth keeping.
7. **The V2 complex-domain code path** (dilated-residual complex estimator) is the right *direction* for the PS (phase-aware enhancement) — it just needs a valid checkpoint.

---

## 12. What is broken / missing / against the plan

| # | Issue | Evidence | Consequence / fix |
|---|---|---|---|
| B1 | `tiny_enhancer_v2.pt` ≠ complex model | load test (§9) | V2 pipeline dead as committed; re-run `train_v2.py` |
| B2 | Generator has no seed | code read (§8.1) | datasets not reproducible; add seed + versioning (V6 §72–73) |
| B3 | `clean_raw` → `clean` conversion has no script | no such file in repo | cannot reproduce `clean/`; ask teammate or rebuild |
| B4 | `clean_raw/` headers corrupt | wave probe (§6.2) | don't trust header lengths; use soundfile/librosa |
| B5 | `evaluate.py`/`evaluate_v2.py` write to the same folder | code read (§8.4) | results overwrite each other; separate output dirs per model |
| B6 | Metrics never saved to file | code read (§8.4) | past numbers lost; add CSV + run-id (V6 §75) |
| B7 | No PESQ (PS-required metric) | repo-wide absence | add validated PESQ (16 kHz mode) |
| B8 | `test_model.py` broken; 4 duplicate/legacy files | checksums (§8.6) | delete; keep one trainer per generation |
| B9 | No `requirements.txt`, no README | repo root | write both (torch, librosa, soundfile, numpy, pystoi + versions) |
| B10 | No real or defence noise | `dataset/noise/` empty | next data task: helicopter/engine/siren/impulse bank (V6 §72) |
| B11 | No adaptive filter / two-mic / streaming / controller / latency / edge | repo-wide absence | the entire rest of the V1 roadmap (E005→E017) |

---

## 13. Rebuild recipe (do this to own the repo)

### 13.1 Environment
```
Python 3.11 (team machine verified: Python 3.11.9, torch 2.14.0+cpu)
pip install torch librosa soundfile numpy pystoi scipy
```
(The teammate may have used a GPU + different versions — see §14 Q4.)

### 13.2 Reproduce the existing pipeline, in order
```bash
git clone -b dev https://github.com/ichigo137/anc
cd anc

# 1) Dataset — add a fixed seed to generate_dataset.py first (else output differs)
python src/generate_dataset.py            # regenerates noisy/*.wav from clean/*.wav

# 2) Train the default log-domain mask model
python src/train.py                       # → models/tiny_enhancer.pt (best val loss)

# 3) Enhance one test file
python src/inference.py dataset/noisy_test/speech_009__hum__snr-5.wav output/rebuild_check.wav

# 4) Full evaluation on the 48 held-out clips (needs pystoi)
python src/evaluate.py                    # prints per-file + breakdowns — CAPTURE the output

# 5) V2 complex experiment — only valid after retraining:
python src/train_v2.py                    # → models/tiny_enhancer_v2.pt (rich dict, complex weights)
python src/inference_v2.py dataset/noisy_test/speech_009__hum__snr-5.wav output/rebuild_v2.wav
python src/evaluate_v2.py
```
**Acceptance check after retraining V2:** `tiny_enhancer_v2.pt` must load into `TinyComplexEnhancer` (65 tensors, `input_layer.*` keys). Verify with a 3-line `load_state_dict` test before trusting any V2 number.

### 13.3 Expected artifacts after a full run
`noisy/` (240 files if all 10 speakers are present — move speakers 009/010 into `noisy_test/` to match the committed layout), `models/tiny_enhancer.pt`, `output/evaluation/*_enhanced.wav` (48), console metric tables.

### 13.4 What you cannot rebuild from the repo alone
- `clean_raw/` → `clean/` conversion (no script; §B3)
- original recording hardware/conditions (§14 Q1)
- the real V2 checkpoint contents (mislabelled; §9)
- past numeric evaluation results (never saved; §B6)

---

## 14. Questions only the teammate can answer

1. **Recording setup:** what device/app recorded `clean_raw/` (48 kHz stereo), and why do several files have corrupt length headers? Is `speech_001.m4a` the original of `speech_001.wav`?
2. **Whose voice / data rights:** who is speaking in the 10 clips? Is the team OK using these recordings (consent / licensing) in a public SIH demo?
3. **Checkpoint truth:** which run produced which `.pt`? In particular — why does `tiny_enhancer_v2.pt` contain the *mask* CNN weights instead of the complex model's? Was the real V2 checkpoint overwritten, or was V2 never successfully trained?
4. **Environment & numbers:** what Python/torch/GPU did he train on? What were the final train/val losses and the printed evaluation numbers (SNR Δ / SI-SDR Δ / STOI Δ) per generation? He should paste the console output — it was never saved.

---

## 15. Corrections to earlier chat analyses

| Earlier claim | Corrected fact (verified) |
|---|---|
| "`src/` has 16 files" | **14** `.py` files on `dev` (+ 1 `.gitignore`) |
| "`train.py` ≡ `train_v1.py`" | true on `main` only; on `dev`, `train.py` was rewritten (log-domain + identity loss) and the two differ |
| "`noisy/` ~240 files" | **192** files (speakers 1–8 only) |
| "`noisy_test/` = 24 files, speech_009 only" | **48** files — speech_009 **and** speech_010 |
| "checkpoints span model families incl. complex V2" | **all 5 checkpoints are the 9.5K-param mask CNN**; no complex checkpoint exists on disk |
| "`evaluate_v1.py` is the older evaluator" | it is **byte-identical** to `evaluate.py` (a copy, not an older version) |
| "`test_model.py` broken only on `main`" | broken on `dev` too |
| "repo is small" | working tree ≈ **789 MB**, audio-dominated |

---

## 16. How this maps to the V1 roadmap (next steps)

This repo already delivers, in miniature, the foundation for these plan experiments — treat it as the starting scaffold:

- **E001/E002 (audio sanity + SNR mixing):** `generate_dataset.py` exists but needs the defence-noise bank, a seed, metadata CSV and measured-SNR verification (V6 §72–74).
- **E003/E004 (STFT + spectral baseline):** not present — spectral subtraction is still to be built; this CNN is not a classical baseline.
- **E009 (offline AI):** `train.py`/`inference.py` + `evaluate.py` = a working offline single-channel AI baseline on synthetic noise. Its numbers (once captured) become the "our baseline" row in the model-selection matrix (V6 §77).
- **E005–E007, E010–E017 (adaptive, dual-mic, streaming, controller, edge):** nothing here — that is the work still ahead.

**Do not** present this repo as the PS 26052 solution at the Sept 12 internal hackathon. Its honest position: "single-channel AI speech-enhancement baseline, working end-to-end, trained on synthetic noise; two-mic adaptive + defence-noise + real-time layers to follow (measured next)."

---

*End of 06_REBUILD_GUIDE — generated by verification against the live repository (`dev` @ 085ee89), 4 Sep 2026.*
