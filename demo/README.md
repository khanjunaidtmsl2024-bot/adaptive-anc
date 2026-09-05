# DRDO PS 26052: Real-Time Tactical Audio Cockpit (Web Demo)

An interactive, browser-based real-time demonstration cockpit for **Smart India Hackathon 2026 (DRDO Problem Statement 26052)**.

---

## 1. Quick Start

Simply open [`index.html`](file:///f:/SIH%202026/demo/index.html) in any modern web browser (Google Chrome, Microsoft Edge, Firefox, or Safari). Zero external installations, servers, or npm packages required.

```powershell
# Windows Launch
Start-Process demo/index.html
```

---

## 2. Key Features

1. **Dual Real-Time Oscilloscope:** Displays the raw primary noisy waveform $x[n]$ (red/amber) against the clean hybrid-enhanced output waveform $y[n]$ (emerald glow).
2. **Dynamic 2D FFT Spectrogram:** Live waterfall showing spectral energy attenuation across the 0–8,000 Hz acoustic spectrum.
3. **Architectural Stage Comparison:**
   - **Raw Noisy:** Direct passthrough of unfiltered microphone.
   - **Spectral Subtraction:** Classical Boll 1979 baseline.
   - **Dual-Mic NLMS:** Classical pre-AI reference cancellation.
   - **TinyEnhancer AI:** 4-layer 2D ConvNet neural mask (10.4K params).
   - **Hybrid AI + DSP:** Complete Config A production chain.
   - **Hybrid + Fail-Safe:** Automatic impulse freeze & soft limiter.
4. **Defence Noise Threat Presets:**
   - T-90 Russian Diesel Tank Engine Rumble (65 Hz + track clatter).
   - ALH Dhruv Helicopter Rotor Blade-Slap (22.5 Hz BPF).
   - INSAS 5.56mm Gunfire Impulse Transients.
   - Emergency Tactical Siren Doppler Glide.
   - Combined Multi-Threat Combat Soundfield.
5. **Live Jury Telemetry HUD:**
   - ITU-T P.862 PESQ Score ($>2.50$)
   - STOI Speech Intelligibility ($>0.85$)
   - Noise Suppression $\Delta\text{SNR}$ ($>+15\text{ dB}$)
   - Scale-Invariant SNR ($>+12\text{ dB}$)
6. **Round-Trip Latency Gauge:** Visual breakdown proving the 26.8 ms round-trip allocation meets the $<30\text{ ms}$ real-time ceiling.
