# Generic Noise Grid Evaluation (Cross-Repo Merge)

Protocol: **https://github.com/ichigo137/anc (generate_dataset.py + evaluate.py)**

Engine: **CausalStreamingEngine VSS-NLMS + E2_causal.onnx (256/128, center=False, strict causal)**

Clean clips (unseen speakers): data/v4/clean/SPK_009_clean.wav, data/v4/clean/SPK_010_clean.wav
Grid: 4 noise types x 6 SNRs x 3 variants = 144 evaluations

## Noise Type Breakdown (mean over SNR levels)

| Noise | Variant | SNR Δ (dB) | SI-SDR (dB) | STOI | PESQ |
|---|---|---|---|---|---|
| hum | NOISY | +0.00 | +7.50 | 0.929 | 1.58 |
| hum | NLMS_ONLY | +7.84 | +15.31 | 0.920 | 1.90 |
| hum | FULL | -9.05 | -24.34 | 0.845 | 1.84 |
| pink | NOISY | +0.00 | +7.71 | 0.777 | 1.18 |
| pink | NLMS_ONLY | +2.86 | +10.36 | 0.774 | 1.17 |
| pink | FULL | -9.40 | -23.72 | 0.729 | 1.18 |
| white | NOISY | +0.00 | +7.50 | 0.776 | 1.07 |
| white | NLMS_ONLY | -0.02 | +7.48 | 0.776 | 1.07 |
| white | FULL | -9.71 | -24.40 | 0.736 | 1.08 |
| impulsive | NOISY | +0.00 | +7.50 | 0.734 | 1.07 |
| impulsive | NLMS_ONLY | +0.12 | +7.70 | 0.734 | 1.07 |
| impulsive | FULL | -10.29 | -24.72 | 0.684 | 1.08 |

## SNR Level Breakdown (mean over noise types)

| Target SNR (dB) | Variant | SNR Δ (dB) | SI-SDR (dB) | STOI | PESQ |
|---|---|---|---|---|---|
| -5 | NOISY | +0.00 | -5.02 | 0.638 | 1.04 |
| -5 | NLMS_ONLY | +4.39 | -0.57 | 0.632 | 1.09 |
| -5 | FULL | +1.02 | -27.77 | 0.598 | 1.10 |
| +0 | NOISY | +0.00 | +0.02 | 0.724 | 1.06 |
| +0 | NLMS_ONLY | +3.84 | +3.87 | 0.719 | 1.13 |
| +0 | FULL | -2.34 | -25.55 | 0.687 | 1.15 |
| +5 | NOISY | +0.00 | +5.09 | 0.794 | 1.10 |
| +5 | NLMS_ONLY | +3.45 | +8.47 | 0.792 | 1.20 |
| +5 | FULL | -6.77 | -23.49 | 0.745 | 1.21 |
| +10 | NOISY | +0.00 | +10.10 | 0.849 | 1.18 |
| +10 | NLMS_ONLY | +2.57 | +12.56 | 0.847 | 1.29 |
| +10 | FULL | -11.55 | -23.09 | 0.795 | 1.30 |
| +15 | NOISY | +0.00 | +15.06 | 0.893 | 1.34 |
| +15 | NLMS_ONLY | +1.46 | +16.45 | 0.891 | 1.43 |
| +15 | FULL | -16.51 | -22.79 | 0.825 | 1.41 |
| +20 | NOISY | +0.00 | +20.05 | 0.927 | 1.62 |
| +20 | NLMS_ONLY | +0.48 | +20.47 | 0.925 | 1.69 |
| +20 | FULL | -21.51 | -23.10 | 0.840 | 1.58 |

## Notes

- Metrics: {'python_version': '3.11.9', 'numpy_version': '2.4.6', 'torch_version': '2.14.0+cpu', 'pesq_available': True, 'pesq_version': '0.0.4', 'pystoi_available': True, 'pystoi_version': '0.4.1'}
- FULL = deployed System A path (VSS-NLMS + E2_causal ONNX, strictly causal, 8 ms hop).
- Impulsive noise exercises the regime detector's IMPULSIVE mode and impulse protection.
