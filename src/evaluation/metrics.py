"""
Objective Speech Quality & Intelligibility Metric Engine.
PS 26052 — Adaptive Defence ANC.

Metrics:
1. SNR (Signal-to-Noise Ratio, dB)
2. SI-SNR (Scale-Invariant Signal-to-Noise Ratio, dB)
3. STOI (Short-Time Objective Intelligibility, 0.0 to 1.0)
4. PESQ (Perceptual Evaluation of Speech Quality, ITU-T P.862, -0.5 to 4.5)

PH0.7 EVALUATION INTEGRITY CONTRACT
------------------------------------
compute_pesq() / compute_stoi() NEVER silently substitute an approximation.
If the real library is unavailable, they raise. This is intentional: a
fabricated number that looks identical to a real one is worse than a loud
crash, because a crash gets fixed and a fabricated number gets published.

If you explicitly want an approximate value (e.g. for a quick local sanity
check with no internet access to install packages), call
compute_pesq_surrogate() / compute_stoi_surrogate() directly. These are
separate, differently-named functions so nobody can call the real metric
name and accidentally get an approximation back. Every result they produce
is stamped with metric_source so it can never be mistaken for the real
thing downstream (in a CSV, a report, or a README table).
"""

import sys
import importlib.metadata as importlib_metadata
from typing import Dict, Any
import numpy as np

# Optional external metrics libraries
try:
    from pystoi import stoi as calc_stoi
    PYSTOI_AVAILABLE = True
except ImportError:
    PYSTOI_AVAILABLE = False

try:
    from pesq import pesq as calc_pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False


def _pkg_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "not_installed"


def get_metric_provenance() -> Dict[str, Any]:
    """
    Returns the exact evaluation environment so every experiment can record
    which metric implementations (real or surrogate) and package versions
    produced its numbers. Call this once per experiment run and save the
    result alongside the CSV/JSON output.
    """
    return {
        "python_version": sys.version.split()[0],
        "numpy_version": _pkg_version("numpy"),
        "torch_version": _pkg_version("torch"),
        "pesq_available": PESQ_AVAILABLE,
        "pesq_version": _pkg_version("pesq") if PESQ_AVAILABLE else None,
        "pystoi_available": PYSTOI_AVAILABLE,
        "pystoi_version": _pkg_version("pystoi") if PYSTOI_AVAILABLE else None,
    }


def compute_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Conventional segmental / global Signal-to-Noise Ratio (dB)."""
    length = min(len(clean), len(noisy))
    c = clean[:length]
    n = noisy[:length] - c
    p_signal = float(np.mean(c ** 2)) + 1e-12
    p_noise = float(np.mean(n ** 2)) + 1e-12
    return float(10.0 * np.log10(p_signal / p_noise))


def compute_si_snr(reference: np.ndarray, estimated: np.ndarray) -> float:
    """Scale-Invariant Signal-to-Noise Ratio (SI-SNR / SI-SDR, dB)."""
    length = min(len(reference), len(estimated))
    ref = reference[:length] - np.mean(reference[:length])
    est = estimated[:length] - np.mean(estimated[:length])

    ref_energy = float(np.sum(ref ** 2)) + 1e-12
    scale = float(np.dot(est, ref)) / ref_energy
    s_target = scale * ref
    e_noise = est - s_target

    p_target = float(np.sum(s_target ** 2)) + 1e-12
    p_noise = float(np.sum(e_noise ** 2)) + 1e-12
    return float(10.0 * np.log10(p_target / p_noise))


def compute_stoi(clean: np.ndarray, enhanced: np.ndarray, sample_rate: int = 16000) -> float:
    """
    Real Short-Time Objective Intelligibility score (0.0 to 1.0). Target: >0.85.

    Raises RuntimeError if pystoi is not installed. Does NOT fall back to an
    approximation — use compute_stoi_surrogate() if you explicitly want one.
    """
    if not PYSTOI_AVAILABLE:
        raise RuntimeError(
            "pystoi is not installed — refusing to substitute a correlation-based "
            "approximation for real STOI. Run: pip install pystoi\n"
            "If you explicitly want an approximate value, call "
            "compute_stoi_surrogate() instead (it is a different function, "
            "on purpose, and its output is stamped metric_source='STOI_SURROGATE')."
        )

    length = min(len(clean), len(enhanced))
    return float(calc_stoi(clean[:length], enhanced[:length], sample_rate, extended=False))


def compute_pesq(clean: np.ndarray, enhanced: np.ndarray, sample_rate: int = 16000) -> float:
    """
    Real PESQ score (ITU-T P.862, wideband/narrowband). Target: >2.5.

    Raises RuntimeError if pesq is not installed. Does NOT fall back to an
    approximation — use compute_pesq_surrogate() if you explicitly want one.
    """
    if not PESQ_AVAILABLE:
        raise RuntimeError(
            "pesq is not installed — refusing to substitute a fabricated "
            "SI-SNR-derived proxy for real PESQ. Run: pip install pesq\n"
            "If you explicitly want an approximate value, call "
            "compute_pesq_surrogate() instead (it is a different function, "
            "on purpose, and its output is stamped metric_source='PESQ_SURROGATE')."
        )

    length = min(len(clean), len(enhanced))
    mode = "wb" if sample_rate >= 16000 else "nb"
    return float(calc_pesq(sample_rate, clean[:length], enhanced[:length], mode))


def compute_stoi_surrogate(clean: np.ndarray, enhanced: np.ndarray) -> Dict[str, Any]:
    """
    EXPLICIT APPROXIMATION — not real STOI. Raw Pearson correlation between
    clean and enhanced waveforms, clipped to [0, 1]. No auditory model, no
    windowing, no perceptual weighting. Only use this when pystoi cannot be
    installed and you need a rough, clearly-labeled sanity signal.

    Returns a dict (not a bare float) so the surrogate label travels with
    the number wherever it's logged.
    """
    length = min(len(clean), len(enhanced))
    r = np.corrcoef(clean[:length], enhanced[:length])[0, 1]
    return {
        "value": float(np.clip(r, 0.0, 1.0)),
        "metric_source": "STOI_SURROGATE",
        "metric_real": False,
        "method": "pearson_correlation",
    }


def compute_pesq_surrogate(clean: np.ndarray, enhanced: np.ndarray) -> Dict[str, Any]:
    """
    EXPLICIT APPROXIMATION — not real PESQ. Linear function of SI-SNR with
    no perceptual/auditory basis. Only use this when pesq cannot be
    installed and you need a rough, clearly-labeled sanity signal.

    Returns a dict (not a bare float) so the surrogate label travels with
    the number wherever it's logged.
    """
    si = compute_si_snr(clean, enhanced)
    proxy = 1.0 + (si + 5.0) * (3.5 / 25.0)
    return {
        "value": float(np.clip(proxy, 1.0, 4.5)),
        "metric_source": "PESQ_SURROGATE",
        "metric_real": False,
        "method": "linear_si_snr_proxy",
    }


def evaluate_all_metrics(
    clean: np.ndarray,
    enhanced: np.ndarray,
    sample_rate: int = 16000,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Computes all primary DRDO PS 26052 metrics in a single call.

    strict=True (default): raises immediately if pesq/pystoi aren't
        installed. This is the only mode that should ever be used for
        official results, reports, or README claims.
    strict=False: falls back to the explicit surrogate functions instead
        of raising, but every returned field is stamped so a surrogate
        result can never be mistaken for a real one downstream. Use this
        only for quick local iteration, never for reported numbers.
    """
    length = min(len(clean), len(enhanced))
    c = clean[:length]
    e = enhanced[:length]

    snr_val = compute_snr(c, e)
    si_snr_val = compute_si_snr(c, e)

    if strict:
        stoi_val = compute_stoi(c, e, sample_rate)
        pesq_val = compute_pesq(c, e, sample_rate)
        stoi_source, pesq_source = "STOI", "PESQ"
        stoi_real, pesq_real = True, True
    else:
        if PYSTOI_AVAILABLE:
            stoi_val = compute_stoi(c, e, sample_rate)
            stoi_source, stoi_real = "STOI", True
        else:
            surrogate = compute_stoi_surrogate(c, e)
            stoi_val, stoi_source, stoi_real = surrogate["value"], surrogate["metric_source"], False

        if PESQ_AVAILABLE:
            pesq_val = compute_pesq(c, e, sample_rate)
            pesq_source, pesq_real = "PESQ", True
        else:
            surrogate = compute_pesq_surrogate(c, e)
            pesq_val, pesq_source, pesq_real = surrogate["value"], surrogate["metric_source"], False

    meets_snr = snr_val >= 15.0
    meets_stoi = stoi_val >= 0.85
    meets_pesq = pesq_val >= 2.50

    return {
        "snr_db": round(snr_val, 2),
        "si_snr_db": round(si_snr_val, 2),
        "stoi": round(stoi_val, 3),
        "pesq": round(pesq_val, 2),
        "sample_rate": sample_rate,
        "stoi_source": stoi_source,
        "pesq_source": pesq_source,
        "stoi_real": stoi_real,
        "pesq_real": pesq_real,
        "compliance": {
            "snr_passed": meets_snr,
            "stoi_passed": meets_stoi,
            "pesq_passed": meets_pesq,
            "all_targets_passed": meets_snr and meets_stoi and meets_pesq,
        },
    }


class AudioMetrics:
    """Class wrapper providing static access to all objective evaluation metrics."""
    calculate_snr = staticmethod(compute_snr)
    calculate_si_snr = staticmethod(compute_si_snr)
    calculate_stoi = staticmethod(compute_stoi)
    calculate_pesq = staticmethod(compute_pesq)
    calculate_stoi_surrogate = staticmethod(compute_stoi_surrogate)
    calculate_pesq_surrogate = staticmethod(compute_pesq_surrogate)
    evaluate_all = staticmethod(evaluate_all_metrics)
    get_provenance = staticmethod(get_metric_provenance)
