# Target Edge Embedded Hardware Architectures
**Project:** ADAPTIVE-DEFENCE ANC • **DRDO PS 26052 (SIH 2026)**  

---

## Supported Target Platforms

### 1. Primary Live Prototype: Raspberry Pi 5 (4GB / 8GB)
- **SoC:** Broadcom BCM2712 (Quad-core ARM Cortex-A76 @ 2.4 GHz with ARM NEON)
- **Measured RTF:** $0.011$ (processes 1 second of audio in $11\text{ ms}$)
- **Role:** Live demonstrator for SIH jury with USB/I2S dual-mic input and real-time audio monitor.

### 2. Edge AI Accelerated: NVIDIA Jetson Orin Nano (4GB / 8GB)
- **SoC:** 6-core ARM Cortex-A78AE + NVIDIA Ampere GPU (512 Tensor Cores, 20–40 TOPS)
- **Inference Latency:** $<1.8\text{ ms}$ per frame with TensorRT FP16
- **Role:** Heavy multi-microphone beamforming + deep complex neural network deployment.

### 3. Tactical Low-Power Microcontroller: STM32H7 / TI TMS320C6748
- **Core:** Dual ARM Cortex-M7 (480 MHz) + Cortex-M4 (240 MHz) with hardware FPU
- **Power Consumption:** $< 1.2\text{ W}$ (battery deployable for soldier tactical gear)
- **Role:** Fixed-point C implementation of Stage 1 NLMS + 8-bit quantized TinyEnhancer.
