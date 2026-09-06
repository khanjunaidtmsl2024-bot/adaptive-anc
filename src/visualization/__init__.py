"""
ADAPTIVE-DEFENCE ANC: Visualization and Diagnostic Plotting Engine.
Provides publication-grade signal processing plots and the ANC X-Ray diagnostic dashboard.
"""

from src.visualization.anc_xray import (
    MATPLOTLIB_AVAILABLE,
    plot_waveform_stack,
    plot_spectrogram_comparison,
    plot_nlms_convergence,
    plot_filter_coefficients,
    plot_continuous_snr_improvement,
    plot_ai_mask,
    plot_metrics_comparison,
    plot_latency_profile,
    plot_anc_xray_full_panel,
    export_experiment_bundle,
)

__all__ = [
    "MATPLOTLIB_AVAILABLE",
    "plot_waveform_stack",
    "plot_spectrogram_comparison",
    "plot_nlms_convergence",
    "plot_filter_coefficients",
    "plot_continuous_snr_improvement",
    "plot_ai_mask",
    "plot_metrics_comparison",
    "plot_latency_profile",
    "plot_anc_xray_full_panel",
    "export_experiment_bundle",
]
