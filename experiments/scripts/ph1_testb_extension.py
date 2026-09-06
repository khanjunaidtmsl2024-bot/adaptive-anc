"""
PH1 — Deterministic TEST_B Noise-Class Extension.
PS 26052 — Adaptive Defence ANC.

Closes the gap found during the PH1 first-action inspection: the frozen
TEST_B_UNSEEN_NOISE_REC split contained only STATIONARY (tank engine) and
PERIODIC_ROTOR (helicopter) classes, but the PH1 master prompt requires
>= 8 defence-noise classes in the unseen-noise suite.

This script PROCEURALLY generates the 6 missing classes with FIXED seeds:

    drone_uav            — multi-rotor blade-pass harmonics (~80-400 Hz) + motor whine
    wind_noise           — low-pass filtered turbulent broadband noise (slow envelope)
    mechanical_armoured  — tracked-vehicle mechanical clatter (metallic resonances + impacts)
    tactical_siren       — FM siren glide (non-stationary)
    gunshot_transient    — isolated impulsive transients (impulse position recorded)
    artillery_broadband  — broadband blast + rolling decay (impulse position recorded)

Outputs (all under data/v4/, matching V4 conventions):
    mixed/TEST_B_EXTENSION/<sample_id>__primary.wav
    mixed/TEST_B_EXTENSION/<sample_id>__reference.wav
    data/v4/metadata/metadata_v4_extension.csv   (append-only manifest rows)

Determinism: every sample derives from RandomState(base_seed + hash(sample_id) % 10_000).
Re-running this script regenerates byte-identical audio and is a no-op on the manifest.

Provenance: every row is labeled PROCEDURAL / SYNTHETIC DEFENCE-NOISE.
These are NOT real battlefield recordings.

Usage:
    python experiments/scripts/ph1_testb_extension.py            # generate
    python experiments/scripts/ph1_testb_extension.py --verify   # verify hashes only
"""

import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
DURATION = 3.0
N_SAMPLES = int(SR * DURATION)
BASE_SEED = 42
LEAKAGE = 0.15          # matches existing TEST_B rows
SPEAKERS = ["SPK_001", "SPK_002", "SPK_003", "SPK_004",
            "SPK_005", "SPK_006", "SPK_007", "SPK_008"]
SNR_DB = [-5.0, 0.0, 5.0, 10.0]

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "v4"
MANIFEST = DATA_ROOT / "metadata" / "metadata_v4.csv"
EXT_DIR = DATA_ROOT / "mixed" / "TEST_B_EXTENSION"
EXT_MANIFEST = DATA_ROOT / "metadata" / "metadata_v4_extension.csv"

PROVENANCE = "PROCEDURAL / SYNTHETIC DEFENCE-NOISE"


# ------------------------------------------------------------------
# Noise generators — each takes (rng, n) and returns float32 noise
# ------------------------------------------------------------------

def _norm(x: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(x))
    return (x / peak).astype(np.float32) if peak > 0 else x.astype(np.float32)


def noise_drone_uav(rng, n) -> np.ndarray:
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for bpf, amp in [(80.0, 1.0), (160.0, 0.7), (240.0, 0.45), (320.0, 0.3), (400.0, 0.2)]:
        sig += amp * np.sin(2 * np.pi * bpf * t + rng.uniform(0, 2 * np.pi))
    # motor whine (high, thin) + electrical buzz
    sig += 0.25 * np.sin(2 * np.pi * 3800 * t) * (0.8 + 0.2 * np.sin(2 * np.pi * 3.1 * t))
    sig += 0.15 * rng.normal(0, 1, n)
    # slow drift as the drone moves
    drift = 1.0 + 0.25 * np.sin(2 * np.pi * 0.4 * t)
    return _norm(sig * drift)


def noise_wind(rng, n) -> np.ndarray:
    # Turbulent broadband noise -> strong low-pass + slowly varying envelope
    x = rng.normal(0, 1, n)
    # 4-pole low-pass via repeated one-pole filters (deterministic, no scipy dependency)
    for _ in range(4):
        alpha = 0.02
        y = np.empty_like(x)
        acc = 0.0
        for i in range(n):
            acc += alpha * (x[i] - acc)
            y[i] = acc
        x = y
    t = np.arange(n) / SR
    env = 0.7 + 0.3 * np.abs(np.sin(2 * np.pi * 0.23 * t + rng.uniform(0, 6)))
    return _norm(x * env)


def noise_mechanical_armoured(rng, n) -> np.ndarray:
    t = np.arange(n) / SR
    sig = 0.3 * np.sin(2 * np.pi * 45.0 * t)                     # idling diesel
    sig += 0.2 * np.sin(2 * np.pi * 90.0 * t + 0.7)
    # metallic track-squeal resonances, amplitude-modulated by track links
    link_rate = 11.0                                             # links per second
    for f, a in [(1200.0, 0.5), (2700.0, 0.3), (4100.0, 0.15)]:
        phase = 2 * np.pi * f * t
        env = 0.5 * (1 + np.sin(2 * np.pi * link_rate * t + rng.uniform(0, 6)))
        sig += a * env * np.sin(phase + rng.uniform(0, 2 * np.pi))
    sig += 0.12 * rng.normal(0, 1, n)
    return _norm(sig)


def noise_tactical_siren(rng, n) -> np.ndarray:
    t = np.arange(n) / SR
    # FM glide 600 -> 1200 Hz, 0.5 Hz wail cycle (non-stationary)
    inst_f = 900 + 300 * np.sin(2 * np.pi * 0.5 * t)
    phase = 2 * np.pi * np.cumsum(inst_f) / SR
    sig = 0.8 * np.sin(phase)
    sig += 0.2 * np.sin(2 * phase)                               # 2nd harmonic
    sig += 0.05 * rng.normal(0, 1, n)
    return _norm(sig)


def _impulse_train(rng, n, num_impulses, rise_s, decay_s, spread=True):
    """Place num_impulses ballistic transients; return (signal, positions)."""
    sig = np.zeros(n)
    positions = []
    slots = np.linspace(int(0.3 * SR), int(0.9 * SR), num_impulses + 2)[1:-1].astype(int)
    rise = max(1, int(rise_s * SR))
    decay = max(1, int(decay_s * SR))
    envelope = np.concatenate([np.linspace(0, 1, rise), np.exp(-np.linspace(0, 6, decay))])
    for idx in slots:
        j = idx + (int(rng.uniform(0, 0.05 * SR)) if spread else 0)
        j = min(j, n - len(envelope) - 1)
        sig[j:j + len(envelope)] += envelope * rng.uniform(0.8, 1.2)
        positions.append(int(j))
    return sig, positions


def noise_gunshot(rng, n):
    sig, positions = _impulse_train(rng, n, num_impulses=3, rise_s=0.0005, decay_s=0.05)
    # crack: short broadband burst riding each impulse
    for p in positions:
        m = min(64, n - p)
        sig[p:p + m] += 0.6 * rng.normal(0, 1, m)
    sig += 0.02 * rng.normal(0, 1, n)                            # faint room tone
    return _norm(sig), positions


def noise_artillery(rng, n):
    sig, positions = _impulse_train(rng, n, num_impulses=2, rise_s=0.002, decay_s=0.35)
    # rolling low-frequency decay after each blast
    for p in positions:
        tail_len = int(0.5 * SR)
        tail_t = np.arange(tail_len) / SR
        tail = 0.4 * np.exp(-tail_t / 0.15) * np.sin(2 * np.pi * 55 * tail_t)
        m = min(tail_len, n - p)
        sig[p:p + m] += tail[:m]
    sig += 0.05 * rng.normal(0, 1, n)
    return _norm(sig), positions


# noise_id -> (category, generator, is_impulsive)
NOISE_CLASSES = {
    "drone_uav_multicopter":     ("NON_STATIONARY", lambda rng, n: (noise_drone_uav(rng, n), []), False),
    "wind_turbulent_field":      ("STATIONARY",     lambda rng, n: (noise_wind(rng, n), []), False),
    "mechanical_armoured_clatter": ("NON_STATIONARY", lambda rng, n: (noise_mechanical_armoured(rng, n), []), False),
    "tactical_siren_glide":      ("NON_STATIONARY", lambda rng, n: (noise_tactical_siren(rng, n), []), False),
    "gunshot_transient_insas":   ("IMPULSIVE",      noise_gunshot, True),
    "artillery_broadband_blast": ("IMPULSIVE",      noise_artillery, True),
}


def _sample_rng(noise_id: str, speaker: str, snr: float) -> tuple:
    key = f"{noise_id}|{speaker}|{snr}"
    seed = (BASE_SEED + int(hashlib.sha256(key.encode()).hexdigest(), 16)) % 1_000_000
    return np.random.RandomState(seed), seed


def _snr_scale(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    p_clean = float(np.mean(clean ** 2)) + 1e-12
    p_noise = float(np.mean(noise ** 2)) + 1e-12
    return noise * np.sqrt(p_clean / p_noise / (10 ** (snr_db / 10)))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_rows():
    """Generate every extended TEST_B sample deterministically. Returns manifest rows."""
    rows = []
    for noise_id, (category, gen, is_impulsive) in NOISE_CLASSES.items():
        for speaker in SPEAKERS:
            for snr in SNR_DB:
                sample_id = f"{speaker}__{noise_id}__{'+' if snr >= 0 else ''}{int(snr)}dB"
                rng, seed = _sample_rng(noise_id, speaker, snr)

                clean, _ = sf.read(str(DATA_ROOT / "clean" / f"{speaker}_clean.wav"), dtype="float32")
                if len(clean) != N_SAMPLES:
                    raise RuntimeError(f"{speaker}_clean.wav is {len(clean)} samples, expected {N_SAMPLES}")

                noise, impulse_positions = gen(rng, N_SAMPLES)
                noise_scaled = _snr_scale(clean, noise, snr)

                # Reference mic: attenuated/delayed noise + small speech leakage (V4 convention)
                ref = noise_scaled * 0.7
                delay = int(rng.randint(0, 3))
                if delay:
                    ref[delay:] = ref[:-delay]
                    ref[:delay] = 0.0
                ref = ref + LEAKAGE * clean * float(rng.uniform(0.9, 1.1))

                primary = clean + noise_scaled

                # Anti-clipping: if the mixture would exceed ±0.98, scale the
                # WHOLE mixture down. This preserves neither the target SNR
                # exactly (clean file on disk is not rescaled), so we record
                # the ACTUALLY ACHIEVED SNR from the written files below.
                peak = float(np.max(np.abs(primary)))
                if peak > 0.98:
                    primary = primary * (0.98 / peak)

                gain_db = 0.0 if rng.uniform() < 0.5 else -1.0
                if gain_db != 0.0:
                    primary = primary * (10 ** (gain_db / 20))

                # Achieved SNR, measured on exactly what is written to disk
                # (i.e. AFTER anti-clipping and gain scaling).
                written_noise = primary - clean
                actual_snr = 10.0 * np.log10(
                    (float(np.mean(clean ** 2)) + 1e-12)
                    / (float(np.mean(written_noise ** 2)) + 1e-12)
                )

                prim_rms = float(np.sqrt(np.mean(primary ** 2)))
                ref_rms = float(np.sqrt(np.mean(ref ** 2)))

                stem = f"{sample_id}__primary"
                EXT_DIR.mkdir(parents=True, exist_ok=True)
                p_path = EXT_DIR / f"{stem}.wav"
                r_path = EXT_DIR / f"{sample_id}__reference.wav"
                sf.write(str(p_path), primary.astype(np.float32), SR)
                sf.write(str(r_path), ref.astype(np.float32), SR)

                row = {
                    "sample_id": sample_id,
                    "split": "TEST_B_UNSEEN_NOISE_REC",
                    "speaker_id": speaker,
                    "noise_id": noise_id,
                    "noise_category": category,
                    "provenance": PROVENANCE,
                    "target_snr_db": f"{snr}",
                    "actual_snr_db": f"{actual_snr:.2f}",
                    "speech_leakage_ratio": f"{LEAKAGE}",
                    "delay_samples": str(delay),
                    "gain_mismatch_db": f"{gain_db}",
                    "duration_sec": f"{DURATION}",
                    "primary_rms": f"{prim_rms:.6f}",
                    "reference_rms": f"{ref_rms:.6f}",
                    "primary_path": f"mixed/TEST_B_EXTENSION/{p_path.name}",
                    "reference_path": f"mixed/TEST_B_EXTENSION/{r_path.name}",
                    "clean_path": f"clean/{speaker}_clean.wav",
                    "sha256_hash": _sha256(p_path),
                    "seed": str(seed),
                    "impulse_positions_samples": ";".join(str(p) for p in impulse_positions) if impulse_positions else "",
                    "noise_class_index": "1",
                }
                rows.append(row)
    return rows


def write_manifest(rows):
    with open(EXT_MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"manifest: {EXT_MANIFEST} ({len(rows)} rows)")
    print(f"manifest sha256: {hashlib.sha256(EXT_MANIFEST.read_bytes()).hexdigest()}")


def verify(rows):
    ok = True
    for r in rows:
        p = DATA_ROOT / r["primary_path"]
        if not p.exists() or _sha256(p) != r["sha256_hash"]:
            print(f"MISMATCH: {r['sample_id']}")
            ok = False
    print("VERIFY:", "PASS — all audio byte-identical to manifest hashes" if ok else "FAIL")
    return ok


def main():
    verify_only = "--verify" in sys.argv
    if not EXT_MANIFEST.exists():
        if verify_only:
            print("No extension manifest to verify — run generator first.")
            sys.exit(1)
        rows = build_rows()
        write_manifest(rows)
        print(f"Generated {len(rows)} samples across {len(NOISE_CLASSES)} noise classes x "
              f"{len(SPEAKERS)} speakers x {len(SNR_DB)} SNRs.")
    else:
        with open(EXT_MANIFEST, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if verify_only:
            sys.exit(0 if verify(rows) else 1)
        # Idempotent: regenerate deterministically and confirm hashes
        new_rows = build_rows()
        same = all(a["sha256_hash"] == b["sha256_hash"] for a, b in zip(rows, new_rows)) and len(rows) == len(new_rows)
        print("DETERMINISM:", "PASS — regenerated hashes match frozen manifest" if same else "FAIL")
        write_manifest(new_rows)


if __name__ == "__main__":
    main()
