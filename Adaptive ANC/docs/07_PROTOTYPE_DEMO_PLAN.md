# 07 — PROTOTYPE DEMO PLAN (Judge-Facing)

**Purpose:** Execution plan to turn V3 evidence into the best representable prototype — understandable in 30 s, believable in 90 s, impressive in 7 minutes — while proving the team can do AI + ML + software + hardware integration.
**Dates:** Written 5 Sep · Finalised 5 Sep (freeze). Internal hackathon 12 Sep · SIH portal 20 Sep.
**Status:** FROZEN — no further revisions. Execute D2 → D1. Next artefact: `results_v3_test.csv`.
**Related docs:** 01_V1_CONTRACT, 04_EVIDENCE_LOG, 05_DEFENCE_PACKAGE, 06_REBUILD_GUIDE.

---

## THE ONE IDEA

The demo must tell a story that only an engineering team could tell, backed by numbers they measured themselves:

```
PROBLEM (physics, our home turf)
   → DATASET (controlled, measured-SNR verified, taxonomy)
   → MODEL (complex residual, speech-preserving)
   → LIVE SYSTEM (mic → AI → speaker, telemetry visible)
   → ROADMAP (edge path, honest claims)
```

**Central message (said out loud in Act 4):**

> "We are not claiming a finished military ANC product. We are demonstrating a measured hybrid adaptive electrical noise-cancellation and AI speech-enhancement architecture — and building the acoustic loop around it."

Never present targets as measurements. Never fake the adaptive stage.

---

## PART 0 — THE DEMO EXPERIENCE (7 minutes)

### ACT 1 (0:00–1:30) The problem, in your home territory
- Play 10 s of comms audio in synthetic rotor-like noise (label it honestly in
  the slide: **"synthetic rotor-like noise"** or **"DEMAND vehicle/transport
  recording"** — never imply defence provenance the dataset doesn't have).
- Play an impulse burst. "Now what happened to the comms?"
- Show the frozen V1 architecture (Config A, from 01_V1_CONTRACT):

```
             CONFIG A — ELECTRICAL PATH

Reference Mic
     │
     │ x[n]
     ▼
┌──────────────┐
│    NLMS      │
│ Controller   │
└──────┬───────┘
       │ y[n] = wᵀx[n]
       │
       ▼
Primary Mic ──────────────┐
d[n] = speech + noise    │
                         ▼
                  e[n] = d[n] − y[n]
                         │
                         ▼
                   AI Enhancer
                         │
                         ▼
                 Enhanced Speech
```

- One line: "Two microphones, one physics idea (Widrow 1975), one neural network for what the physics can't remove."
- Credibility anchor: "N252-093 is a U.S. Navy SBIR topic targeting ambient acoustic noise cancellation in the electrical signal of a military boom microphone, <10 ms latency. Multiple Phase I awards were made in 2025–26. We reproduce that principle with open, measured engineering."
- Why this team: "We're electrical engineers — the analog chain, the mics, gain staging, grounding, and I²S clocks are OUR language."
- **Always describe the architecture as:** "A hybrid adaptive electrical noise-cancellation and AI speech-enhancement architecture." Do not say "AI-powered adaptive ANC" until the acoustic loop exists.

### ACT 2 (1:30–4:00) The engineering, with numbers
- Dataset pipeline: canonical clean → float-WAV mixing → measured-SNR verification (target −5 dB → measured −5.0000 dB) → speaker-disjoint AND noise-disjoint test splits.
- The model: complex STFT → residual network (trainable parameters: `sum(p.numel() for p in model.parameters() if p.requires_grad)` — report the printed number, not a hand-written one), why residual + SNR-aware identity loss protects speech.
- THE TABLE (held-out speakers AND held-out noise recordings):

| Noise | Category | Input SNR | Noisy STOI → Enhanced STOI | Noisy PESQ → Enhanced PESQ |
|---|---|---|---|---|
| (real numbers from results_v3_test.csv) |

- Report: mean Δ, median Δ, worst Δ, pass rate. If a single clip degrades badly, say so — "we found and are investigating one failure mode" is engineering strength.
- Band-attenuation demo: integrate PSD around 45–55 Hz before/after; say: "Power in the 50-Hz band was reduced by X dB." That is engineering because it IS engineering.

### ACT 3 (4:00–6:00) The live system
- Live: mic → AI enhancement → speaker, waveform + spectrogram updating, telemetry strip showing: hop (128 samples = 8 ms), processing time per block, RTF, buffer fill. The screen reads "LATENCY: hop 8 ms · algorithmic X ms · end-to-end Y ms (see D9)" — the hop is never displayed as latency.
- Switch noise sources live (synthetic white / hum / impulsive / DEMAND vehicle).
- NLMS stage: show the highest maturity gate actually passed (see G1–G6 ladder in Part 3) with a label on screen. If live with 2 USB mics (G3): convergence curve, reference coherence, adaptation gating when reference contains speech leakage. If not yet live: show the NLMS simulation convergence plot and say exactly "adaptive stage validated in simulation (G2); live integration in progress." Never fake it.
- Architecture panel: each block (STFT → NLMS → AI → iSTFT) lights up with its measured processing time.

### ACT 4 (6:00–7:00) The roadmap
- Pi 4: laptop numbers → Pi numbers → ONNX → INT8 (see the toolchain analysis: FP32 borderline on A72, INT8 comfortable after quantization).
- Latency story: "N252-093 demands <10 ms end-to-end. Our current framing has 16 ms lookahead. D9 is our Pareto experiment — here is the quality-vs-latency-vs-compute tradeoff table we produced." Show the Pareto plot.
- Cost strategy: "PoC proves the signal chain; measured compute defines the migration path."
- Close: "What you saw is the measured subsystem. The acoustic loop is the next integration. The claim we make today is the evidence, not the finished product."

---

## PART 1 — WHAT MUST BE TRUE BY 12 SEP (deliverables D1–D9)

| # | Deliverable | Proof of completion | Who |
|---|---|---|---|
| D1 | **results_v3_test.csv** — held-out speakers AND held-out noise, per-file: sample_id, speaker_id, speech_source, noise_id, noise_source, noise_category, noise_recording, target_snr_db, measured_snr_db, in_out_snr, si_sdr, stoi, pesq, pass_snr, pass_stoi, pass_pesq | CSV committed; **GO gate (see below)** | R2+R5 |
| D2 | **DATASET_V004** — noise taxonomy (synthetic / DEMAND / defence*), recursive DEMAND glob + 16 kHz mono downmix, noise-recording-disjoint + noise-category-disjoint test, full metadata (see provenance spec below) | CSV + regenerated set | R2 |
| D3 | **Ablation + failure analysis** — mask-CNN vs V3 on same held-out; worst-file report | table + failure paragraph | R5 |
| D4 | **Canonical streaming demo** — block streaming (128-sample hop), honest telemetry | rehearsed ×3 | R4 |
| D5 | **NLMS simulation** — leakage-gated adaptation; convergence, misadjustment, residual, speech distortion; live if mics arrive | plot + clip | R1 |
| D6 | **Rig** — mics, headset, labeled noise sources, gain staging verified | working setup | R3 |
| D7 | **Pitch pack** — slides + one-pager | rehearsal record | R6 |
| D8 | **Fail-safe pack** — precomputed clips, rehearsal video, second laptop | fallback <10 s | R6 |
| D9 | **Latency Pareto experiment** — framing configs (512/128 baseline, 512/64 same-model, 256/128 retrained, 256/64 retrained, 128/64 retrained, causal retrained); measure per config: algorithmic latency (window, lookahead, hop), compute (inference ms, RTF, CPU%, memory), quality (ΔSNR, ΔSTOI, ΔPESQ); produce Pareto plot | table + tradeoff plot; answers "can we approach <10 ms?" | R1+R4 |

### D1 GO/NO-GO gate (evening of 6 Sep)

```
PASS iff ALL:
  mean ΔSNR  > 0
  mean ΔSTOI > 0
  mean ΔPESQ > 0
  median ΔSTOI > 0
  worst  ΔSTOI > −0.03    ← no catastrophic degradation
  worst  ΔPESQ > −0.10
  pass-rate change ≤ 5 pts on the 3 SIH targets

FAIL → STOP. Investigate data / model / baseline before anything else.
```

### D2 noise-disjoint split rules (precise definitions)

```
TEST A — Unseen speaker:
  speakers 009–010 held out; all noise recordings/type used
  (tests speaker generalisation)

TEST B — Unseen noise recording:
  2 DEMAND scenes + 1 synthetic class held out of training mixtures
  (tests generalisation to new acoustic environments)

TEST C — Unseen noise category:
  Test B recordings from a category NOT in training
  (e.g., t
  (e.g., train: white/pink/hum/transport; test: impulsive/rotor)
  (strongest generalisation claim; label honestly)

### D2 full metadata provenance

Every row in metadata_v4.csv:

```
sample_id, speech_id, speaker_id, speech_source,
noise_id, noise_source, noise_category, noise_recording,
noise_segment_start, noise_segment_end,
target_snr_db, measured_snr_db,
sample_rate, duration_s, seed,
split, test_set,          <- which generalisation test (A/B/C/none)
generator_version, timestamp
```

---

## PART 2 - DAY-BY-DAY EXECUTION

**Roles:** R1 DSP/adaptive, R2 ML/data, R3 hardware, R4 software, R5 metrics, R6 narrative.

| Day | Focus | Tasks |
|---|---|---|
| 5 Sep | Freeze | Canonical model = tiny_enhancer_v3_controlled.pt. Archive 7 others. R6 writes claims matrix. R3 hardware check. |
| 6 Sep AM | D2 | R2: V004 dataset (recursive DEMAND, taxonomy, held-out splits, full metadata). |
| 6 Sep PM | D1 | R2+R5: held-out eval -> results_v3_test.csv. |
| 6 Sep EVE | GO/NO-GO | D1 gate passed? proceed. FAIL -> stop, investigate. |
| 7 Sep | D3 | Ablation + failure analysis; 4 killer visuals. R1 writes NLMS sim (leakage-gated). |
| 8 Sep | D4 | R4: block streaming demo (128-sample hop, honest telemetry). R3: rig test. |
| 9 Sep | D5 + D9 | R1: NLMS sim (highest gate: G1 or G2). R1+R4: latency Pareto experiment. Pi 4 best effort. |
| 10 Sep | D6 | Rig verified; integration begins. |
| 11 Sep | Rehearsal | Dress rehearsal x2; kill-test; record video. |
| 12 Sep | Hackathon | Execute 7-min script. |
| 13-19 Sep | Portal | Evidence pack: abstract, N252-093 novelty, architecture diagram, evidence table, latency Pareto, roadmap, cost. |
| 20 Sep | Submit | Done. |

---

## PART 3 - NLMS MATURITY LADDER (demo shows highest gate passed)

```
G1  NLMS mathematical simulation        (clean + noise refs, no mic)
G2  Offline two-signal NLMS simulation   (two reference files, measured convergence)
G3  Live two-microphone electrical path  (USB mics -> laptop -> e[n] output)
G4  Measured cancellation               (SNR before/after NLMS on live signal)
G5  Acoustic speaker loop               (V2 future - FxLMS + secondary path)
G6  Embedded real-time ANC              (V2 future - Pi4/edge)
```

**Demo protocol:** label the highest gate reached on screen. If G2: "NLMS validated in offline simulation." If G3: "Live two-mic electrical path." Do NOT claim G4+ unless measured.

---

## PART 3 (cont.) - ENGINEERING-DEPTH CHECKLIST

1. **Measurement discipline** - measured-SNR verification, float-WAV, speaker+noise disjoint splits, PESQ at 16 kHz with documented alignment. -> "We verify waveforms, not labels."

2. **Model depth** - complex residual, identity loss, dilated receptive fields, gradient clipping; parameter count resolved programmatically. -> "We know what we trained and why."

3. **THE MANTRA: hop != latency** - 128-sample hop = 8 ms; latency = buffering + window/lookahead (16 ms at 512/center) + model processing + iSTFT + output buffering + DAC. D9 provides the Pareto evidence; the demo never displays the hop as latency.

4. **Adaptive theory** - Widrow reference-mic principle, NLMS normalization, mu stability, adaptation gating from reference leakage/double-talk (not naive VAD freeze). Measure: leakage, convergence, misadjustment, residual, speech distortion. -> "This is signal theory - our home."

5. **Generalization matrix** - two axes:

| | NOISE seen | NOISE unseen |
|---|---|---|
| SPEAKER seen | train | robustness |
| SPEAKER unseen | Test A: unseen speaker | Test C: unseen speaker + unseen noise |
| NOISE CATEGORY unseen | - | Test B/C: unseen noise class |

Test C is the strongest claim - label honestly (never call synthetic impulsive "gunshot").

6. **Hardware physics** - mic placement, gain staging, clipping guard, grounding, I2S clock sync, why USB mics are a legitimate PoC. -> "Electrical is our base."

7. **Generalisation honesty** - Defence noise is the stated next step, not the current evidence. DEMAND is transport/environmental, not military.

---

## PART 4 - CLAIMS MATRIX

| You may say | You may NOT say |
|---|---|
| V3 improved STOI/PESQ on held-out speakers AND held-out noise (only with CSV) | We achieve STOI > 0.85, PESQ > 2.5 |
| Model has [printed] trainable parameters | Lightweight enough for any edge device |
| Hybrid adaptive electrical noise-cancellation and AI enhancement (Config A); acoustic ANC is V2 | AI-powered adaptive ANC / acoustic cancellation |
| N252-093 is a Navy SBIR topic for boom-mic electrical noise cancellation; Phase I awards 2025-26 | The Navy is still funding this / Nobody has built anything |
| NLMS validated in simulation (G2) / live electrical path (G3) (whichever is true) | Our system does acoustic ANC |
| DEMAND committed; V004 mixes it in; defence-class noise is the next step | Trained on defence noise |
| Hop 8 ms; algorithmic X ms; end-to-end Y ms (D9 measured) | Sub-10 ms system (until D9 proves it) |
| Impulse robustness tested with synthetic impulsive noise | Gunshot/artillery robustness |

---

## PART 5 - THE SINGLE RISK

The demo lives or dies on D1.

- Numbers strong -> lead with them.
- Numbers weak -> shift to improvement deltas and let the GO/NO-GO gate trigger investigation.
- Table must exist by evening of 6 Sep. Everything else is presentation.

**Do-not list:**
- Do not display target numbers as measured results.
- Do not display 8 ms as latency.
- Do not call it real-time ANC - it is a measured speech-enhancement + electrical-cancellation demonstration.
- Do not draw AI->anti-noise->speaker for V1 - V1 is Config A (electrical); acoustic ANC is V2.
- Do not use helicopter-ish or imply defence provenance the dataset doesn't have.
- Do not build more UI before the numbers exist.
- Do not claim Pi 4 works until RTF/memory are measured on the Pi.

---

## END OF 07 - STATUS (FROZEN)

- [ ] D2 DATASET_V004 - due 6 Sep AM
- [ ] D1 results_v3_test.csv + GO/NO-GO - due 6 Sep evening
- [ ] D3 ablation + failure analysis - due 7 Sep
- [ ] D4 streaming demo - due 8 Sep
- [ ] D5 NLMS simulation (highest gate G1-G3) - due 9 Sep
- [ ] D9 latency Pareto experiment - due 9 Sep
- [ ] D6 rig - due 10 Sep
- [ ] D7 pitch pack - due 11 Sep
- [ ] D8 fail-safe pack + rehearsal x2 - due 11 Sep
- [ ] Hackathon - 12 Sep
- [ ] Portal pack - 13-19 Sep, submit 20 Sep
