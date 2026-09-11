# Raspberry Pi 4 / 5 Hardware Bring-Up & Deployment Guide
**Project:** PS 26052 — Defence-Grade Adaptive ANC  
**Causal Baseline Model:** `E2_causal.onnx` (9,569 parameters, 49.7 KB standalone FP32)  
**Target Hardware:** Raspberry Pi 4 Model B (4GB/8GB) or Raspberry Pi 5 + Waveshare WM8960 Audio HAT / ReSpeaker 2-Mics Pi HAT  

---

## 1. Hardware Architecture & Interfaces

The system operates on an 8.0 ms hop budget (128 samples @ 16 kHz sampling rate):
```
[Primary Mic (Ch0)] --+
                      |--> [WM8960 Audio HAT] --I2S--> [ALSA Ch0/1 @ 16kHz]
[Reference Mic (Ch1)] -+                                       |
                                                               v
                                                      [RingBuffer (128-hop)]
                                                               |
                                                               v
                                                     [VSS-NLMS DSP Filter]
                                                               |
                                                               v (Residual)
                                                     [Causal STFT (256/128)]
                                                               |
                                                               v
                                                     [E2_causal ONNX FP32]
                                                               |
                                                               v (Spectral Mask)
                                                     [Causal WOLA Synthesis]
                                                               |
                                                               v
[Clean Audio Output] <--3.5mm Jack/Amp<--I2S<--ALSA<-- [DAC Buffer (128-hop)]
```

---

## 2. Operating System & Driver Installation

### Step 1: Install 64-bit Raspberry Pi OS
Flash **Raspberry Pi OS Lite (64-bit)** (Debian Bookworm or Bullseye) to a high-speed MicroSD card or NVMe SSD.

### Step 2: Install WM8960 Audio HAT Drivers
```bash
sudo apt update && sudo apt install -y git python3-pip python3-venv libasound2-dev alsa-utils
git clone https://github.com/waveshare/WM8960-Audio-HAT
cd WM8960-Audio-HAT
sudo ./install.sh
sudo reboot
```

### Step 3: Verify ALSA Hardware Device Nodes
After reboot, verify that the WM8960 sound card is recognized:
```bash
arecord -l
aplay -l
```
Expected output should show `card 1: wm8960soundcard [wm8960-soundcard], device 0: ...`.

### Step 4: Configure Microphone Gains & Channel Mapping
Open `alsamixer -c 1` and configure:
- **Left Input Mixer**: Boost = 0 dB, Capture Volume = 80% (Ch0 = Primary Mic)
- **Right Input Mixer**: Boost = 0 dB, Capture Volume = 80% (Ch1 = Reference Mic)
- **Headphone / Speaker Volume**: 75%
- Save settings:
  ```bash
  sudo alsactl store
  ```

---

## 3. Software Environment Setup

On the Raspberry Pi:
```bash
# 1. Create a clean virtual environment
cd /home/pi/
mkdir -p sih_anc && cd sih_anc
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

# 2. Install minimal deployment requirements (No PyTorch needed!)
pip install --upgrade pip
pip install numpy soundfile onnxruntime

# 3. Optional: Install pyalsaaudio for direct ALSA hardware streaming
pip install pyalsaaudio

# 4. Optional: Install Numba for compiled DSP acceleration
pip install numba
```

---

## 4. Benchmarking Embedded Execution (Phase 3 Audit)

To measure actual on-device execution latencies:
```bash
python3 benchmark_embedded.py --model models/E2_causal.onnx --iters 1000 --threads 4
```

### What to Record:
- **AI Latency:** P50, P95, P99, Max (ms)
- **DSP Latency:** P50, P95, P99, Max (ms)
- **Total Pipeline Latency:** P50, P95, P99, Max (ms)
- **Real-Time Factor (RTF):** Total Latency / 8.0 ms
- Hardware condition: `vcgencmd measure_temp` and `vcgencmd measure_clock arm`

---

## 5. Live Audio Streaming Execution (Phase 5 Audit)

To run the live dual-channel ANC pipeline:
```bash
python3 deploy_e2_causal.py --model models/E2_causal.onnx --hops 1000
```

---

## 6. End-to-End Hardware Latency Measurement Protocol (Phase 6 Audit)

To claim physical real-time compliance, you **MUST NOT** rely on software timer logs alone.
Follow this physical loopback measurement protocol:

1. **Test Setup:**
   - Channel 1 (Yellow Probe of Oscilloscope): Placed on Primary Mic input analog pin / test pad.
   - Channel 2 (Blue Probe of Oscilloscope): Placed on 3.5mm Audio Output DAC pin / test pad.
2. **Signal Injection:**
   - Play a sharp 1-sample acoustic impulse or 1 kHz tone burst through an external test speaker near the microphones.
3. **Measurement:**
   - Measure time delta $\Delta t$ between Channel 1 trigger (acoustic wave at mic) and Channel 2 response (audio wave at DAC).
4. **Latency Decomposition Breakdown:**
   $$\text{Total End-to-End Latency} = \tau_{\text{ADC}} + \tau_{\text{Buffer In}} + \tau_{\text{DSP}} + \tau_{\text{AI}} + \tau_{\text{WOLA}} + \tau_{\text{Buffer Out}} + \tau_{\text{DAC}}$$
   - $\tau_{\text{ADC}}$ + $\tau_{\text{DAC}}$: Hardware converter latency (~0.5 - 1.0 ms on WM8960 @ 16 kHz)
   - $\tau_{\text{Buffer In}}$ + $\tau_{\text{Buffer Out}}$: Double buffering (128 samples = 8.0 ms)
   - $\tau_{\text{DSP}}$ + $\tau_{\text{AI}}$: Measured software computation time
   - $\tau_{\text{WOLA}}$: Algorithmic lookahead (0 ms in causal mode)
