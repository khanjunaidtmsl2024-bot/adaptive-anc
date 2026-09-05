# Engineering Cautions, Failure Modes & Risk Mitigations
**Project:** ADAPTIVE-DEFENCE ANC • **DRDO PS 26052 (SIH 2026)**  
**Document Role:** System Risk Register, Boundary Limits & Scientific Defensibility  

---

## 1. Acoustic & Algorithmic Failure Modes

| Risk / Failure Mode | Acoustic Trigger | Mechanism of Failure | Engineered Mitigation |
|:---|:---|:---|:---|
| **LMS Filter Divergence** | Gunfire / artillery blast (>120 dB SPL) | Huge momentary error $e(n)$ causes gradient explosion in weight update | Huber M-estimate gradient clipping & down-scaling $\mu$ in `src/dsp/kalman.py` |
| **Speech Cancellation (Leakage)** | High vocal volume leaking into reference mic | Cross-talk between primary and reference mics causes filter to cancel desired voice | Dual-mic spacing $d \ge 8\text{ cm}$ + Voice Activity Detector (VAD) freezing adaptation during speech |
| **Musical Noise Artifacts** | Low input SNR (< -5 dB) | Uncontrolled magnitude masking with corrupted noisy phase | Spectral floor factor $\beta = 0.02$ in spectral subtraction + pre-AI NLMS noise floor reduction |
| **Buffer Underrun / Glitching** | Edge CPU overload during heavy neural pass | Audio DMA starvation causing clicks/pops | Real-time ring buffer with 4096-sample capacity (`src/streaming/ring_buffer.py`) |
| **Acoustic Feedback** | Speaker output picked up by microphone | Secondary acoustic path coupling creates high-frequency howl | FallbackController (`src/pipeline/fallback_controller.py`) detects energy ratio surge and clamps gain |

---

## 2. Hard Physical Limits & Cautions

1. **Acoustic Delay Boundary:**
   - In air at 20°C, sound travels at $343\text{ m/s}$ ($0.343\text{ mm/\mu s}$).
   - A $10\text{ cm}$ separation between primary and reference microphones introduces $\approx 0.29\text{ ms}$ of acoustic propagation delay.
   - The FIR filter length must satisfy $M \ge f_s \cdot \tau_{\text{delay}}$ to model this acoustic transit time ($M \ge 64$ taps @ 16 kHz gives $4.0\text{ ms}$ of FIR impulse response).

2. **Latency vs. Resolution Trade-Off:**
   - Frequency resolution in STFT is $\Delta f = \frac{f_s}{N}$. For $N = 512$ @ 16 kHz, $\Delta f = 31.25\text{ Hz}$.
   - Increasing $N$ to 1024 improves harmonic separation of tank engine tones, but doubles algorithmic latency from $32\text{ ms}$ to $64\text{ ms}$, violating the $<30\text{ ms}$ DRDO constraint.
   - **Frozen Decision:** Maintain $N=512$ (32 ms window) with $50\%$ overlap ($16\text{ ms}$ hop) to strictly honor the latency ceiling.
