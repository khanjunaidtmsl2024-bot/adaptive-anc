# Hardware Specifications: Dual-Microphone Array & Edge Audio Interface
**Project:** ADAPTIVE-DEFENCE ANC • **DRDO PS 26052 (SIH 2026)**  

---

## 1. Dual-Microphone Array Geometry

```
                  +-------------------------+
                  |    TACTICAL HELMET      |
                  +-------------------------+
                    |                     |
     [Reference Mic x(n)]                 |
     Mounted on outer ear-cup             |
     (Omnidirectional MEMS)               |
     Captures ambient vehicle noise       |
                                          v
                                 [Primary Mic d(n)]
                                 Throat Contact / Directional Boom
                                 Captures near-field bone/acoustic speech
```

### Microphone Specifications:
| Parameter | Primary Microphone $d(n)$ | Reference Microphone $x(n)$ |
|:---|:---|:---|
| **Transducer Type** | Directional Differential Electret / Throat Contact | Omnidirectional High-AOP MEMS |
| **Acoustic Overload Point (AOP)** | $\ge 125\text{ dB SPL}$ | $\ge 135\text{ dB SPL}$ (Blast proof) |
| **Sensitivity** | $-38\text{ dBV/Pa} \pm 2\text{ dB}$ | $-38\text{ dBV/Pa} \pm 2\text{ dB}$ (Gain matched) |
| **Frequency Response** | $100\text{ Hz} - 8\text{ kHz}$ (Voice band optimized) | $20\text{ Hz} - 16\text{ kHz}$ (Wideband noise capture) |
| **Placement** | Near mouth ($d_1 \le 2\text{ cm}$) or laryngeal contact | Outer helmet shell ($d_2 \ge 10\text{ cm}$ from mouth) |

---

## 2. Audio Codec & ADC/DAC Interface
- **Audio Codec:** Texas Instruments TLV320AIC3254 / Cirrus Logic CS4272
- **Sampling Frequency:** $16,000\text{ Hz}$ ($16\text{ kHz}$)
- **Bit Depth:** $24\text{-bit}$ resolution ($>100\text{ dB}$ dynamic range to capture both whispered voice and tank engine roar without clipping)
- **Bus Interface:** Inter-IC Sound (I2S) / USB Audio Class 2.0
- **DMA Configuration:** Double-buffered ping-pong DMA with circular ring buffering
