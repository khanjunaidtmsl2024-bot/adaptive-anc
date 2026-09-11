import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate
import csv
import os

FILE1 = "usb1_48k.wav"
FILE2 = "usb2_48k.wav"

FS_EXPECTED = 48000

# Analysis settings
WINDOW_SEC = 1.0
STEP_SEC = 0.5
MAX_LAG = 500


# ============================================================
# LOAD
# ============================================================

fs1, x1 = wavfile.read(FILE1)
fs2, x2 = wavfile.read(FILE2)

print("=" * 60)
print("       DUAL USB SYNCHRONIZATION ANALYSIS")
print("=" * 60)
print()

print("USB #1:")
print("  Sample rate :", fs1)
print("  Samples     :", len(x1))
print()

print("USB #2:")
print("  Sample rate :", fs2)
print("  Samples     :", len(x2))
print()

if fs1 != FS_EXPECTED or fs2 != FS_EXPECTED:
    print("WARNING: sample rate is not 48000 Hz")

# Convert to float
x1 = x1.astype(np.float64)
x2 = x2.astype(np.float64)

# Equal length
N = min(len(x1), len(x2))
x1 = x1[:N]
x2 = x2[:N]

print("Common samples :", N)
print("Duration       :", N / FS_EXPECTED, "seconds")
print()


# ============================================================
# BASIC LEVELS
# ============================================================

rms1 = np.sqrt(np.mean(x1 ** 2))
rms2 = np.sqrt(np.mean(x2 ** 2))

print("RMS USB #1 :", rms1)
print("RMS USB #2 :", rms2)

if rms1 > 0 and rms2 > 0:
    gain_ratio = rms2 / rms1
    gain_db = 20 * np.log10(gain_ratio)

    print("Approx gain difference (USB2 - USB1):",
          round(gain_db, 3), "dB")

print()


# ============================================================
# NORMALIZED CROSS CORRELATION
# ============================================================

def find_delay(a, b, max_lag):

    a = a - np.mean(a)
    b = b - np.mean(b)

    denom = np.sqrt(np.sum(a*a) * np.sum(b*b))

    if denom == 0:
        return None, 0

    corr = correlate(a, b, mode="full")

    center = len(a) - 1

    start = center - max_lag
    end = center + max_lag + 1

    corr_window = corr[start:end]

    lag_values = np.arange(-max_lag, max_lag + 1)

    idx = np.argmax(np.abs(corr_window))

    best_lag = lag_values[idx]

    best_corr = corr_window[idx] / denom

    return best_lag, best_corr


# ============================================================
# GLOBAL TEST
# ============================================================

print("=" * 60)
print("GLOBAL CORRELATION")
print("=" * 60)

lag, corr = find_delay(x1, x2, MAX_LAG)

if lag is not None:

    delay_ms = lag / FS_EXPECTED * 1000

    print("Best lag       :", lag, "samples")
    print("Delay          :", round(delay_ms, 4), "ms")
    print("Correlation    :", round(corr, 5))

    print()

    if abs(corr) >= 0.90:
        print("RESULT: VERY HIGH CORRELATION")
    elif abs(corr) >= 0.70:
        print("RESULT: GOOD CORRELATION")
    elif abs(corr) >= 0.50:
        print("RESULT: MODERATE CORRELATION")
    else:
        print("RESULT: LOW CORRELATION")

print()


# ============================================================
# WINDOWED DELAY / DRIFT ANALYSIS
# ============================================================

print("=" * 60)
print("WINDOWED DELAY / CLOCK DRIFT ANALYSIS")
print("=" * 60)

window = int(WINDOW_SEC * FS_EXPECTED)
step = int(STEP_SEC * FS_EXPECTED)

results = []

position = 0

while position + window <= N:

    a = x1[position:position + window]
    b = x2[position:position + window]

    lag, corr = find_delay(a, b, MAX_LAG)

    time_sec = (position + window / 2) / FS_EXPECTED

    if lag is not None:

        delay_ms = lag / FS_EXPECTED * 1000

        results.append([
            time_sec,
            lag,
            delay_ms,
            corr
        ])

        print(
            f"t={time_sec:6.2f}s | "
            f"lag={lag:5d} samples | "
            f"delay={delay_ms:8.4f} ms | "
            f"corr={corr: .5f}"
        )

    position += step


# ============================================================
# SAVE CSV
# ============================================================

csv_file = "dual_usb_sync_results.csv"

with open(csv_file, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "time_sec",
        "lag_samples",
        "delay_ms",
        "correlation"
    ])

    writer.writerows(results)


# ============================================================
# DRIFT ESTIMATE
# ============================================================

if len(results) >= 3:

    times = np.array([r[0] for r in results])
    lags = np.array([r[1] for r in results])

    # Linear fit: lag = slope*time + intercept
    slope, intercept = np.polyfit(times, lags, 1)

    print()
    print("=" * 60)
    print("CLOCK DRIFT ESTIMATE")
    print("=" * 60)

    print("Lag slope :", slope, "samples/second")

    # Convert to ppm
    # relative rate error approximately slope / sample_rate
    drift_ppm = slope / FS_EXPECTED * 1e6

    print("Estimated relative clock drift:",
          round(drift_ppm, 3), "ppm")

    print()

    first_lag = lags[0]
    last_lag = lags[-1]

    print("First measured lag :", first_lag, "samples")
    print("Last measured lag  :", last_lag, "samples")
    print("Lag change         :", last_lag - first_lag, "samples")

    print()

    if abs(last_lag - first_lag) <= 1:
        print("PRELIMINARY: VERY STABLE RELATIVE TIMING")
    elif abs(last_lag - first_lag) <= 5:
        print("PRELIMINARY: SMALL TIMING DRIFT")
    elif abs(last_lag - first_lag) <= 20:
        print("PRELIMINARY: SIGNIFICANT TIMING DRIFT")
    else:
        print("PRELIMINARY: LARGE TIMING DRIFT")

print()
print("Results saved to:")
print(os.path.abspath(csv_file))
print()
print("=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)