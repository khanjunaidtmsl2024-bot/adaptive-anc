# Evaluation-Integrity Audit — Hostile Sweep Results

**PS 26052 · Adaptive Defence ANC · 2026-09-06**
**Scope:** every `try/except` site and numeric-fallback pattern in `src/`, `experiments/scripts/`, `scripts/`
that could put a value into a benchmark CSV/JSON. 111 Python files scanned; 45 `except` lines across 25 files;
**zero bare `except:`; zero except-bodies that assign a numeric literal.**

**House rule (enforced):** a failed metric computation must never land as a valid-looking number.
It becomes `float("nan")` plus a printed/logged reason (or an explicit error field that propagates
fail-safe into derived flags — Python NaN comparisons are `False`, so "not degraded"/"robust"
conclusions must treat NaN as degraded). Import guards that set honest `FLAG = False` are the
acceptable exception and are listed as SAFE.

---

## CRITICAL / HIGH — fixed

| # | File:line | Issue | Severity | Fix (commit) | Validation |
|---|-----------|-------|----------|--------------|------------|
| 1 | `src/evaluation/ai_ablation_and_pesq_audit.py:62-68` (`calculate_pesq_wb`) | PESQ exception returned fixed `1.0` (bottom of the valid scale → reads as a real catastrophic score) | CRITICAL | `float("nan")` + stderr log (`a79150f`) | Forced too-short input → `nan` + "Buffer needs to be at least 1/4 of a second long" on stderr |
| 2 | `src/evaluation/vss_nlms_sweep.py:179-182` (`out_stoi`) | STOI exception → `0.0` written into sweep record (`stoi: round(0.0,4)`) | HIGH | `float("nan")` + stderr log (`a79150f`) | Forced-failure probe → `nan` in record |
| 3 | `src/evaluation/leakage_breakdown_sweep.py:165-169` (`out_stoi`) | STOI exception → `0.0` written into sweep record | HIGH | `float("nan")` + stderr log + `stoi_error` field (`a79150f`) | Forced-failure probe → `stoi_error=True`, `stoi=nan` |
| 4 | `src/evaluation/leakage_breakdown_sweep.py` (`is_degraded` / `alpha_crit`) | `(in_stoi - out_stoi) > 0.10` with NaN is `False` → failed measurement silently read as "not degraded" → false "Robust across all alphas" | HIGH | NaN propagates into `stoi_error`/`is_degraded` (fail-safe) (`a79150f`, `247ccf0`) | Forced-failure probe → 12/12 rows `is_degraded=True`, `alpha_crit=0.00` |
| 5 | `src/evaluation/ai_ablation_and_pesq_audit.py:~194` (raw `pystoi.stoi`) | Uncaught STOI exception crashed the whole 5-way ablation mid-sweep, leaving the previous CSV stale on disk | HIGH | `float("nan")` + stderr log (`247ccf0`) | Forced-failure probe → 15/15 records `stoi=nan`, run completes |
| 6 | `src/evaluation/leakage_breakdown_sweep.py:~102` (`in_stoi`) | Uncaught input-side STOI exception crashed the entire sweep | HIGH | `float("nan")` + stderr log; `in_stoi_failed` propagates into every row's `stoi_error`/`is_degraded` (`247ccf0`) | Forced-failure probe → sweep completes, 12/12 fail-safe |

## MEDIUM — fixed

| # | File:line | Issue | Severity | Fix (commit) | Validation |
|---|-----------|-------|----------|--------------|------------|
| 7 | `src/integrations/ichigo_bridge.py` checkpoint load | Load failure printed a warning but exposed no programmatic state → batch consumer could score a random-weight model | MEDIUM | `checkpoint_loaded` flag (`a79150f`); this pass: `checkpoint_path` (loaded path or `None`), `checkpoint_sha256` (SHA-256 of loaded file bytes or `None`), `weights_status` (`TRAINED_CHECKPOINT` / `RANDOM_INITIALIZATION`) — all in `inspect_model()` | Missing-path test asserts `RANDOM_INITIALIZATION` / `None` / `None`; corrupt-file probe in `247ccf0` pass on TinyEnhancer |
| 8 | `src/streaming/live_stream_audio.py` `run_live`/`run_simulation` | Return dict had no mode key → simulated run indistinguishable from hardware run in JSON | MEDIUM | `mode`/`is_simulated` (`a79150f`); this pass: full schema `execution_mode` (`SIMULATION`/`HARDWARE`), `hardware_available`, `hardware_target` (`Raspberry Pi 4`), `hardware_model`, `audio_interface` (`WM8960`) via `_execution_provenance()` | Tests assert simulation schema + hardware schema (unit, no device needed) |
| 9 | `src/evaluation/vss_nlms_sweep.py:55-62` (`measure_convergence_time`) | Buffer too short to measure → returned `0.0` ms ("instant convergence"), flattering NLMS in the mu/filter-length ranking CSV | MEDIUM | `float("nan")` + stderr note (`247ccf0`) | Probe: short buffer → `nan` + note; normal buffer → numeric, no note |
| 10 | `src/ai/tiny_enhancer.py:66-68` checkpoint load | `except: pass` → silent random weights fed plausible PESQ/STOI numbers | MEDIUM | stderr warning + `checkpoint_loaded` flag (`247ccf0`) | Corrupt-file probe → flag `False` + warning |

## WARNING / left as-is (reasoned)

| # | File:line | Issue | Verdict | Reason |
|---|-----------|-------|---------|--------|
| 11 | `src/evaluation/statistical_validation.py` (3× raw `pystoi.stoi`) | STOI exception crashes the run loudly **before any CSV is written** — no fabricated value, but a mid-run crash leaves the previous CSV stale | LEFT-AS-IS | Honest (loud) failure; fail-safe hardening needs a NaN policy threaded through scipy t-tests/Wilcoxon/bootstrap — a design change, deferred per mission scope. Next-pass candidate. |
| 12 | `src/dsp/noise_regime_detector.py:117` (`return 0.0`) | Excess kurtosis for frames < 4 samples returns `0.0` | SAFE | `0.0` is the *neutral* excess-kurtosis value (Gaussian); benign default direction, not an evaluation artifact |

## SAFE — house style, no change

| # | Site | Why safe |
|---|------|----------|
| 13 | Import guards: `TORCH_AVAILABLE`, `NUMBA_AVAILABLE`, `sf = None`, `psutil = None`, `SOUNDDEVICE_AVAILABLE`, `PESQ_AVAILABLE`, `STOI_AVAILABLE` (src/ai, src/dsp, src/dataset, src/evaluation, src/streaming) | Honest `FLAG=False` semantics; consumers gate on the flag |
| 14 | `src/evaluation/metrics.py` | Loud raises; surrogate/error fields stamped; `_pkg_version` degrades to `"not_installed"` string |
| 15 | `src/evaluation/ph05_pesq_ablation.py:218-236` | `NaN` + `pesq_error: str(ex)` recorded |
| 16 | `src/evaluation/ph06_integrity_check.py:402-408` | `NaN` + `WARNING: ... metric failed` |
| 17 | `src/evaluation/ph06_headroom_benchmark.py:41-49` | Version probes degrade to `"unknown"`/`"unavailable"` strings, never numbers |
| 18 | `src/ai/export_onnx.py:67-117` | Printed reason at every stage; per-format exported flags tracked |
| 19 | `src/audio_io/sources.py:159`, `sinks.py:77` | Re-raise as `HardwareUnavailableError` (loud, documented) |
| 20 | `src/integrations/ichigo_bridge.py`, `src/streaming/live_stream_audio.py` import guards | Honest flag semantics (see 13) |

## Repository-wide numeric scan (this pass)

- 111 Python files scanned; 0 bare `except:`; 0 except-bodies assigning `0.0`/`1.0`/arrays.
- Two literal `return 0.0` sites found outside except blocks: #9 (fixed) and #12 (SAFE, above).

## Bottom line

No failed metric computation can currently land as a valid-looking numerical score in any
benchmark CSV/JSON in this repository. Remaining integrity work is the `statistical_validation.py`
NaN policy (item 11) — a design decision, not a bug fix.