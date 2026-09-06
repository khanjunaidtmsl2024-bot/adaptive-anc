#!/usr/bin/env python3
"""
ANC X-RAY DIAGNOSTIC DASHBOARD & EXPERIMENT BUNDLE EXPORTER
Smart India Hackathon (SIH) 2026 - Problem Statement ID: 26052
DRDO / Department of Defence Production / iDEX - Smart Vehicles

Generates publication-grade diagnostic plots for the 8-stage hybrid ANC signal chain:
1. Time-domain waveform stack (Clean, Noisy, NLMS noise estimate, NLMS residual, AI enhanced)
2. Tri-spectrogram comparison (Noisy Input, NLMS Residual, AI Output)
3. NLMS convergence curve (error power e^2(t), mu(t))
4. Selected adaptive filter coefficients w_i(t) over time
5. Continuous Delta_SNR(t) trajectory
6. AI time-frequency suppression mask heatmap M(t, f)
7. Comparative speech-quality bar chart (SNR, Delta_SNR, SI-SDR, STOI, PESQ)
8. Streaming latency scatter & budget threshold profile (P50, P95, P99, MAX)
Master multi-panel figure: anc_xray_full_panel.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import soundfile as sf
from scipy import signal

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    matplotlib = None
    plt = None
    gridspec = None

# Scientific palette
PALETTE = {
    "clean": "#10b981",       # Emerald green
    "noisy": "#ef4444",       # Rose red
    "nlms_est": "#f59e0b",    # Amber
    "residual": "#8b5cf6",    # Purple
    "enhanced": "#06b6d4",    # Cyan
    "budget": "#dc2626",      # Bright red
    "bg_dark": "#0f172a",     # Slate dark
    "grid": "#334155",        # Muted slate
}


def compute_spectrogram_db(x: np.ndarray, sr: int = 16000, nperseg: int = 512, noverlap: int = 384):
    """Computes log-magnitude spectrogram in dB with floor at -80 dBFS."""
    f, t, Sxx = signal.spectrogram(x, fs=sr, nperseg=nperseg, noverlap=noverlap, window="hann")
    Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-12))
    return f, t, Sxx_db


def plot_waveform_stack(
    clean: Optional[np.ndarray],
    primary: np.ndarray,
    noise_estimate: np.ndarray,
    residual: np.ndarray,
    enhanced: np.ndarray,
    sr: int = 16000,
    out_path: Optional[str] = None,
) -> plt.Figure:
    """Plots 5-track time-domain waveform stack with aligned time axis."""
    n_samples = len(primary)
    t = np.arange(n_samples) / sr

    fig, axes = plt.subplots(5, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("ANC X-Ray: Time-Domain Waveform Evolution Across Pipeline Stages", fontsize=13, fontweight="bold")

    tracks = [
        ("Clean Speech s[n]", clean if clean is not None else np.zeros_like(primary), PALETTE["clean"]),
        ("Primary Mic d[n] = s[n] + n[n]", primary, PALETTE["noisy"]),
        ("NLMS Noise Estimate y[n]", noise_estimate, PALETTE["nlms_est"]),
        ("NLMS Residual e[n] = d[n] - y[n]", residual, PALETTE["residual"]),
        ("AI Enhanced Speech s_hat[n]", enhanced, PALETTE["enhanced"]),
    ]

    for ax, (title, data, color) in zip(axes, tracks):
        lim = min(len(t), len(data))
        ax.plot(t[:lim], data[:lim], color=color, linewidth=0.8, alpha=0.85)
        ax.set_ylabel(title, fontsize=8, fontweight="semibold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(-1.05, 1.05)

    axes[-1].set_xlabel("Time (seconds)", fontsize=10, fontweight="semibold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    return fig


def plot_spectrogram_comparison(
    noisy: np.ndarray,
    residual: np.ndarray,
    enhanced: np.ndarray,
    sr: int = 16000,
    out_path: Optional[str] = None,
) -> plt.Figure:
    """Plots tri-spectrogram comparison (Noisy vs NLMS Residual vs AI Enhanced)."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("ANC X-Ray: Time-Frequency Spectral Evolution (0–8 kHz)", fontsize=13, fontweight="bold")

    stages = [
        ("1. Primary Noisy Input d[n]", noisy),
        ("2. NLMS Residual e[n] (Acoustic Filtering)", residual),
        ("3. AI Output s_hat[n] (Spectral Masking)", enhanced),
    ]

    vmin, vmax = -60.0, 10.0
    pcm = None

    for ax, (title, data) in zip(axes, stages):
        f, t, Sxx_db = compute_spectrogram_db(data, sr=sr)
        pcm = ax.pcolormesh(t, f, Sxx_db, shading="gouraud", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_ylabel("Frequency (Hz)", fontsize=9, fontweight="semibold")
        ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
        ax.set_ylim(0, sr // 2)

    axes[-1].set_xlabel("Time (seconds)", fontsize=10, fontweight="semibold")
    cbar = fig.colorbar(pcm, ax=axes.ravel().tolist(), orientation="vertical", pad=0.02, aspect=25)
    cbar.set_label("Magnitude (dBFS)", fontsize=9)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    return fig


def plot_nlms_convergence(
    error_power: np.ndarray,
    mu_trajectory: Optional[np.ndarray] = None,
    sr: int = 16000,
    hop_size: int = 128,
    out_path: Optional[str] = None,
) -> plt.Figure:
    """Plots NLMS error power convergence curve in dB and variable step-size mu(t)."""
    fig, ax1 = plt.subplots(figsize=(10, 4.5))

    t = np.arange(len(error_power)) * (hop_size / sr)
    err_db = 10 * np.log10(np.maximum(error_power, 1e-12))

    color_err = PALETTE["residual"]
    ax1.plot(t, err_db, color=color_err, linewidth=1.2, label="Error Power (dBFS)")
    ax1.set_xlabel("Time (seconds)", fontsize=10, fontweight="semibold")
    ax1.set_ylabel("Error Power e^2[n] (dBFS)", color=color_err, fontsize=10, fontweight="semibold")
    ax1.tick_params(axis="y", labelcolor=color_err)
    ax1.grid(True, linestyle="--", alpha=0.4)

    if mu_trajectory is not None and len(mu_trajectory) == len(error_power):
        ax2 = ax1.twinx()
        color_mu = PALETTE["nlms_est"]
        ax2.plot(t, mu_trajectory, color=color_mu, linewidth=1.0, linestyle=":", label="Step-Size mu(t)")
        ax2.set_ylabel("VSS Step-Size mu(t)", color=color_mu, fontsize=10, fontweight="semibold")
        ax2.tick_params(axis="y", labelcolor=color_mu)

    plt.title("ANC X-Ray: NLMS Convergence & Adaptation Dynamics", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    return fig


def plot_filter_coefficients(
    coeff_history: np.ndarray,
    selected_taps: Optional[List[int]] = None,
    sr: int = 16000,
    hop_size: int = 128,
    out_path: Optional[str] = None,
) -> plt.Figure:
    """Plots trajectories of selected adaptive filter weights w_i(t) over time."""
    fig, ax = plt.subplots(figsize=(10, 4.5))

    if coeff_history.ndim == 1:
        coeff_history = coeff_history[:, np.newaxis]

    n_hops, filter_len = coeff_history.shape
    t = np.arange(n_hops) * (hop_size / sr)

    if selected_taps is None:
        selected_taps = [0, min(8, filter_len - 1), min(16, filter_len - 1), min(32, filter_len - 1), filter_len - 1]
    selected_taps = sorted(list(set(selected_taps)))

    for tap in selected_taps:
        if tap < filter_len:
            ax.plot(t, coeff_history[:, tap], label=f"Weight w[{tap}]", linewidth=1.2)

    ax.set_xlabel("Time (seconds)", fontsize=10, fontweight="semibold")
    ax.set_ylabel("Filter Coefficient Amplitude", fontsize=10, fontweight="semibold")
    ax.set_title("ANC X-Ray: Adaptive Filter Tap Convergence Trajectories", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.8)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    return fig


def plot_continuous_snr_improvement(
    clean: np.ndarray,
    noisy: np.ndarray,
    enhanced: np.ndarray,
    sr: int = 16000,
    window_ms: int = 100,
    out_path: Optional[str] = None,
) -> plt.Figure:
    """Computes and plots continuous Delta_SNR(t) over sliding windows."""
    fig, ax = plt.subplots(figsize=(10, 4.5))

    win_samples = int(sr * window_ms / 1000)
    hop_samples = win_samples // 2
    n_frames = (len(clean) - win_samples) // hop_samples

    times = []
    delta_snrs = []

    for i in range(max(0, n_frames)):
        idx = i * hop_samples
        c_seg = clean[idx : idx + win_samples]
        n_seg = noisy[idx : idx + win_samples]
        e_seg = enhanced[idx : idx + win_samples]

        noise_in = n_seg - c_seg
        noise_out = e_seg - c_seg

        p_s = np.mean(c_seg ** 2) + 1e-12
        p_nin = np.mean(noise_in ** 2) + 1e-12
        p_nout = np.mean(noise_out ** 2) + 1e-12

        snr_in = 10 * np.log10(p_s / p_nin)
        snr_out = 10 * np.log10(p_s / p_nout)
        delta_snrs.append(snr_out - snr_in)
        times.append((idx + win_samples / 2) / sr)

    ax.plot(times, delta_snrs, color=PALETTE["clean"], linewidth=1.5, label="Continuous Delta_SNR(t)")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("Time (seconds)", fontsize=10, fontweight="semibold")
    ax.set_ylabel("SNR Improvement Delta_SNR (dB)", fontsize=10, fontweight="semibold")
    ax.set_title("ANC X-Ray: Dynamic SNR Improvement Over Time", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    return fig


def plot_ai_mask(
    mask_matrix: np.ndarray,
    sr: int = 16000,
    hop_size: int = 128,
    out_path: Optional[str] = None,
) -> plt.Figure:
    """Plots 2D heatmap of the predicted AI spectral suppression mask M(t, f)."""
    fig, ax = plt.subplots(figsize=(10, 4.5))

    n_frames, n_bins = mask_matrix.shape
    t = np.arange(n_frames) * (hop_size / sr)
    f = np.linspace(0, sr // 2, n_bins)

    pcm = ax.pcolormesh(t, f, mask_matrix.T, shading="gouraud", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xlabel("Time (seconds)", fontsize=10, fontweight="semibold")
    ax.set_ylabel("Frequency (Hz)", fontsize=10, fontweight="semibold")
    ax.set_title("ANC X-Ray: TinyEnhancer Time-Frequency Suppression Mask M(t, f)", fontsize=12, fontweight="bold")
    cbar = fig.colorbar(pcm, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label("Spectral Gain Mask (0 = Suppress, 1 = Pass)", fontsize=9)
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    return fig


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    out_path: Optional[str] = None,
) -> plt.Figure:
    """Plots bar chart comparing speech quality metrics across pipeline configurations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    systems = list(metrics_dict.keys())
    x = np.arange(len(systems))
    width = 0.35

    # Panel 1: SNR and SI-SDR
    snrs = [metrics_dict[s].get("snr_db", 0.0) for s in systems]
    sisdrs = [metrics_dict[s].get("si_sdr_db", metrics_dict[s].get("si_snr_db", 0.0)) for s in systems]

    ax1.bar(x - width/2, snrs, width, label="SNR (dB)", color=PALETTE["nlms_est"])
    ax1.bar(x + width/2, sisdrs, width, label="SI-SDR (dB)", color=PALETTE["enhanced"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(systems, rotation=15, fontweight="semibold", fontsize=9)
    ax1.set_ylabel("Decibels (dB)", fontsize=10, fontweight="semibold")
    ax1.set_title("SNR & SI-SDR Comparison", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left")

    # Panel 2: STOI and PESQ
    stois = [metrics_dict[s].get("stoi", 0.0) for s in systems]
    pesqs = [metrics_dict[s].get("pesq", 0.0) / 4.5 for s in systems]  # Normalized PESQ for comparison

    ax2.bar(x - width/2, stois, width, label="STOI (0–1)", color=PALETTE["clean"])
    ax2.bar(x + width/2, pesqs, width, label="PESQ (norm 0–1)", color=PALETTE["residual"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(systems, rotation=15, fontweight="semibold", fontsize=9)
    ax2.set_ylabel("Normalized Score", fontsize=10, fontweight="semibold")
    ax2.set_title("Speech Intelligibility & Quality", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left")

    plt.suptitle("ANC X-Ray: Speech Quality Metrics Comparison Across Configurations", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    return fig


def plot_latency_profile(
    frame_times_ms: List[float],
    budget_ms: float = 8.0,
    out_path: Optional[str] = None,
) -> plt.Figure:
    """Plots per-hop processing time scatter with 8.0 ms real-time threshold."""
    fig, ax = plt.subplots(figsize=(10, 4.5))

    frames = np.arange(len(frame_times_ms))
    p50 = np.percentile(frame_times_ms, 50)
    p95 = np.percentile(frame_times_ms, 95)
    p99 = np.percentile(frame_times_ms, 99)
    max_lat = np.max(frame_times_ms)

    ax.plot(frames, frame_times_ms, color=PALETTE["enhanced"], linewidth=0.8, alpha=0.75, label="Frame Processing Time")
    ax.scatter(frames, frame_times_ms, color=PALETTE["enhanced"], s=8, alpha=0.6)
    ax.axhline(budget_ms, color=PALETTE["budget"], linestyle="--", linewidth=1.8, label=f"Real-Time Budget ({budget_ms:.1f} ms)")

    stats_text = (
        f"P50:  {p50:.2f} ms\n"
        f"P95:  {p95:.2f} ms\n"
        f"P99:  {p99:.2f} ms\n"
        f"MAX:  {max_lat:.2f} ms\n"
        f"Budget: {budget_ms:.1f} ms"
    )
    ax.text(
        0.02, 0.95, stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=PALETTE["grid"], alpha=0.9),
    )

    ax.set_xlabel("Hop Frame Index (128-sample hops)", fontsize=10, fontweight="semibold")
    ax.set_ylabel("Execution Time (ms)", fontsize=10, fontweight="semibold")
    ax.set_title("ANC X-Ray: Hop-by-Hop Streaming Latency Profile", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    return fig


def plot_anc_xray_full_panel(
    clean: Optional[np.ndarray],
    primary: np.ndarray,
    noise_est: np.ndarray,
    residual: np.ndarray,
    enhanced: np.ndarray,
    mask_matrix: Optional[np.ndarray],
    error_power: Optional[np.ndarray],
    frame_times_ms: Optional[List[float]],
    metrics_dict: Optional[Dict[str, Dict[str, float]]],
    sr: int = 16000,
    budget_ms: float = 8.0,
    out_path: Optional[str] = None,
) -> plt.Figure:
    """Master 8-panel diagnostic dashboard figure (anc_xray_full_panel.png)."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("ADAPTIVE-DEFENCE ANC: X-RAY DIAGNOSTIC DASHBOARD", fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.25)

    n_samples = len(primary)
    t = np.arange(n_samples) / sr

    # Panel 1: Waveforms (Top-Left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, primary, color=PALETTE["noisy"], label="Primary Mic d[n]", alpha=0.7, linewidth=0.8)
    ax1.plot(t, enhanced, color=PALETTE["enhanced"], label="AI Enhanced s_hat[n]", linewidth=1.0)
    ax1.set_title("1. Time-Domain Signal Overlay", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Amplitude", fontsize=8)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Panel 2: NLMS Residual (Top-Right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, noise_est, color=PALETTE["nlms_est"], label="NLMS Noise Estimate y[n]", alpha=0.6, linewidth=0.8)
    ax2.plot(t, residual, color=PALETTE["residual"], label="NLMS Residual e[n]", linewidth=0.8)
    ax2.set_title("2. NLMS Cancellation Residual", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Amplitude", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.4)

    # Panel 3: Noisy Spectrogram (Row 1 Left)
    ax3 = fig.add_subplot(gs[1, 0])
    f_n, t_n, s_noisy = compute_spectrogram_db(primary, sr=sr)
    pcm3 = ax3.pcolormesh(t_n, f_n, s_noisy, cmap="magma", vmin=-60, vmax=10, shading="gouraud")
    ax3.set_title("3. Primary Noisy Input Spectrogram", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Freq (Hz)", fontsize=8)

    # Panel 4: Enhanced Spectrogram (Row 1 Right)
    ax4 = fig.add_subplot(gs[1, 1], sharex=ax3)
    f_e, t_e, s_enh = compute_spectrogram_db(enhanced, sr=sr)
    ax4.pcolormesh(t_e, f_e, s_enh, cmap="magma", vmin=-60, vmax=10, shading="gouraud")
    ax4.set_title("4. AI Enhanced Output Spectrogram", fontsize=10, fontweight="bold")
    ax4.set_ylabel("Freq (Hz)", fontsize=8)

    # Panel 5: AI Mask Heatmap (Row 2 Left)
    ax5 = fig.add_subplot(gs[2, 0])
    if mask_matrix is not None:
        n_frames, n_bins = mask_matrix.shape
        t_m = np.arange(n_frames) * (128 / sr)
        f_m = np.linspace(0, sr // 2, n_bins)
        pcm5 = ax5.pcolormesh(t_m, f_m, mask_matrix.T, cmap="viridis", vmin=0.0, vmax=1.0, shading="gouraud")
        ax5.set_title("5. AI Suppression Mask Heatmap M(t, f)", fontsize=10, fontweight="bold")
        ax5.set_ylabel("Freq (Hz)", fontsize=8)
    else:
        ax5.text(0.5, 0.5, "AI Mask Data Not Available", ha="center", va="center")

    # Panel 6: NLMS Convergence (Row 2 Right)
    ax6 = fig.add_subplot(gs[2, 1])
    if error_power is not None:
        t_err = np.arange(len(error_power)) * (128 / sr)
        err_db = 10 * np.log10(np.maximum(error_power, 1e-12))
        ax6.plot(t_err, err_db, color=PALETTE["residual"], linewidth=1.2)
        ax6.set_title("6. NLMS Error Power Convergence (dBFS)", fontsize=10, fontweight="bold")
        ax6.set_ylabel("e^2[n] (dB)", fontsize=8)
        ax6.grid(True, linestyle="--", alpha=0.4)
    else:
        ax6.text(0.5, 0.5, "Error Power Data Not Available", ha="center", va="center")

    # Panel 7: Metrics Comparison (Row 3 Left)
    ax7 = fig.add_subplot(gs[3, 0])
    if metrics_dict is not None:
        systems = list(metrics_dict.keys())
        x = np.arange(len(systems))
        w = 0.35
        snrs = [metrics_dict[s].get("snr_db", 0.0) for s in systems]
        sisdrs = [metrics_dict[s].get("si_sdr_db", metrics_dict[s].get("si_snr_db", 0.0)) for s in systems]
        ax7.bar(x - w/2, snrs, w, label="SNR (dB)", color=PALETTE["nlms_est"])
        ax7.bar(x + w/2, sisdrs, w, label="SI-SDR (dB)", color=PALETTE["enhanced"])
        ax7.set_xticks(x)
        ax7.set_xticklabels(systems, fontsize=8)
        ax7.set_title("7. Speech Quality Comparison (dB)", fontsize=10, fontweight="bold")
        ax7.set_ylabel("dB", fontsize=8)
        ax7.legend(loc="upper left", fontsize=8)
        ax7.grid(True, linestyle="--", alpha=0.4)
    else:
        ax7.text(0.5, 0.5, "Metrics Data Not Available", ha="center", va="center")

    # Panel 8: Latency Profile (Row 3 Right)
    ax8 = fig.add_subplot(gs[3, 1])
    if frame_times_ms is not None and len(frame_times_ms) > 0:
        h_idx = np.arange(len(frame_times_ms))
        ax8.plot(h_idx, frame_times_ms, color=PALETTE["enhanced"], linewidth=0.8, alpha=0.8)
        ax8.axhline(budget_ms, color=PALETTE["budget"], linestyle="--", linewidth=1.5, label=f"Budget ({budget_ms} ms)")
        p95 = np.percentile(frame_times_ms, 95)
        ax8.set_title(f"8. Streaming Hop Latency (P95: {p95:.2f} ms)", fontsize=10, fontweight="bold")
        ax8.set_ylabel("ms", fontsize=8)
        ax8.set_xlabel("Hop Index", fontsize=8)
        ax8.legend(loc="upper right", fontsize=8)
        ax8.grid(True, linestyle="--", alpha=0.4)
    else:
        ax8.text(0.5, 0.5, "Latency Profile Not Available", ha="center", va="center")

    fig.subplots_adjust(top=0.93, bottom=0.06, left=0.06, right=0.96, hspace=0.38, wspace=0.25)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    return fig


def export_experiment_bundle(
    exp_dir: str,
    config: Dict[str, Any],
    clean: Optional[np.ndarray],
    primary: np.ndarray,
    reference: np.ndarray,
    noise_estimate: np.ndarray,
    residual: np.ndarray,
    enhanced: np.ndarray,
    metrics_dict: Dict[str, Dict[str, float]],
    frame_times_ms: List[float],
    mask_matrix: Optional[np.ndarray] = None,
    error_power: Optional[np.ndarray] = None,
    mu_trajectory: Optional[np.ndarray] = None,
    coeff_history: Optional[np.ndarray] = None,
    sr: int = 16000,
    budget_ms: float = 8.0,
) -> Dict[str, str]:
    """
    Exports a complete standardized experiment bundle to exp_dir:
    - config.json
    - Audio WAVs (input.wav, reference.wav, nlms_output.wav, ai_output.wav, hybrid_output.wav)
    - metrics.json
    - latency.csv
    - 8 PNG visual plots + anc_xray_full_panel.png
    """
    out_dir = Path(exp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}

    # 1. Save config.json
    config_path = out_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    manifest["config"] = str(config_path)

    # 2. Save Audio WAVs
    audio_files = {
        "input.wav": primary,
        "reference.wav": reference,
        "nlms_output.wav": residual,
        "hybrid_output.wav": enhanced,
    }
    if clean is not None:
        audio_files["clean.wav"] = clean

    for fname, arr in audio_files.items():
        p = out_dir / fname
        # Normalize and clamp safely to [-1.0, 1.0]
        safe_arr = np.clip(arr.astype(np.float32), -1.0, 1.0)
        sf.write(str(p), safe_arr, sr, subtype="PCM_16")
        manifest[fname] = str(p)

    # 3. Save metrics.json
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
    manifest["metrics"] = str(metrics_path)

    # 4. Save latency.csv
    latency_path = out_dir / "latency.csv"
    with open(latency_path, "w", encoding="utf-8") as f:
        f.write("hop_index,latency_ms,budget_exceeded\n")
        for idx, lat in enumerate(frame_times_ms):
            f.write(f"{idx},{lat:.4f},{int(lat > budget_ms)}\n")
    manifest["latency"] = str(latency_path)

    if not MATPLOTLIB_AVAILABLE:
        manifest["plots_status"] = "SKIPPED_MATPLOTLIB_UNAVAILABLE"
        return manifest

    # 5. Generate and Save Individual PNG Plots
    p_wave = out_dir / "waveform.png"
    plot_waveform_stack(clean, primary, noise_estimate, residual, enhanced, sr=sr, out_path=str(p_wave))
    manifest["waveform.png"] = str(p_wave)

    p_spec_in = out_dir / "spectrogram_input.png"
    fig_sin, ax_sin = plt.subplots(figsize=(8, 4))
    f_in, t_in, s_in = compute_spectrogram_db(primary, sr=sr)
    pcm_in = ax_sin.pcolormesh(t_in, f_in, s_in, cmap="magma", vmin=-60, vmax=10, shading="gouraud")
    ax_sin.set_title("Input Spectrogram d[n]", fontsize=11, fontweight="bold")
    ax_sin.set_ylabel("Frequency (Hz)")
    ax_sin.set_xlabel("Time (s)")
    fig_sin.colorbar(pcm_in, ax=ax_sin).set_label("dBFS")
    fig_sin.tight_layout()
    fig_sin.savefig(str(p_spec_in), dpi=150)
    plt.close(fig_sin)
    manifest["spectrogram_input.png"] = str(p_spec_in)

    p_spec_out = out_dir / "spectrogram_output.png"
    fig_sout, ax_sout = plt.subplots(figsize=(8, 4))
    f_out, t_out, s_out = compute_spectrogram_db(enhanced, sr=sr)
    pcm_out = ax_sout.pcolormesh(t_out, f_out, s_out, cmap="magma", vmin=-60, vmax=10, shading="gouraud")
    ax_sout.set_title("Enhanced Output Spectrogram s_hat[n]", fontsize=11, fontweight="bold")
    ax_sout.set_ylabel("Frequency (Hz)")
    ax_sout.set_xlabel("Time (s)")
    fig_sout.colorbar(pcm_out, ax=ax_sout).set_label("dBFS")
    fig_sout.tight_layout()
    fig_sout.savefig(str(p_spec_out), dpi=150)
    plt.close(fig_sout)
    manifest["spectrogram_output.png"] = str(p_spec_out)

    if error_power is not None:
        p_conv = out_dir / "nlms_convergence.png"
        plot_nlms_convergence(error_power, mu_trajectory, sr=sr, out_path=str(p_conv))
        manifest["nlms_convergence.png"] = str(p_conv)

    if mask_matrix is not None:
        p_mask = out_dir / "ai_mask.png"
        plot_ai_mask(mask_matrix, sr=sr, out_path=str(p_mask))
        manifest["ai_mask.png"] = str(p_mask)

    p_metrics = out_dir / "metrics_comparison.png"
    plot_metrics_comparison(metrics_dict, out_path=str(p_metrics))
    manifest["metrics_comparison.png"] = str(p_metrics)

    p_lat_plot = out_dir / "latency_profile.png"
    plot_latency_profile(frame_times_ms, budget_ms=budget_ms, out_path=str(p_lat_plot))
    manifest["latency_profile.png"] = str(p_lat_plot)

    # 6. Master Full-Panel ANC X-Ray Image
    p_master = out_dir / "anc_xray_full_panel.png"
    plot_anc_xray_full_panel(
        clean=clean,
        primary=primary,
        noise_est=noise_estimate,
        residual=residual,
        enhanced=enhanced,
        mask_matrix=mask_matrix,
        error_power=error_power,
        frame_times_ms=frame_times_ms,
        metrics_dict=metrics_dict,
        sr=sr,
        budget_ms=budget_ms,
        out_path=str(p_master),
    )
    manifest["anc_xray_full_panel.png"] = str(p_master)

    return manifest
