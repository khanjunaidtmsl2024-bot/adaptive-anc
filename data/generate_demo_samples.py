"""
=============================================================================
PS 26052: DRDO Adaptive ANC — Demonstration Sample Audio Generator
=============================================================================
Generates standardized, lightweight (<100 KB each), high-fidelity 16 kHz 
mono audio samples stored in `data/samples/` for automated verification, 
web browser playback, and SIH jury evaluation.
"""

import math
import os
import sys
import wave
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.signal import butter, sosfilt

from src.dataset.noise_synthesizer import DefenceNoiseSynthesizer
from src.dataset.mixer import AcousticMixer
from src.pipeline.hybrid_chain import HybridEnhancementPipeline

def generate_synthetic_speech(duration: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
    """
    Synthesizes a structured speech-like signal using formant synthesis.
    Simulates vowel/consonant transitions ("Alpha-1, radio check") with 
    harmonic pitch glides and formant filtering.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = np.zeros_like(t)
    
    # Fundamental frequency envelope (F0 contour around 120-140 Hz with intonation)
    f0 = 130.0 + 15.0 * np.sin(2 * np.pi * 1.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate
    
    # Glottal pulse train with rich harmonics
    glottal = np.zeros_like(t)
    for h in range(1, 15):
        glottal += (1.0 / (h ** 0.8)) * np.sin(h * phase)
        
    # Speech cadence / syllabic amplitude envelope (bursts simulating words)
    syllables = (
        0.8 * np.sin(2 * np.pi * 2.5 * t) ** 2 * (np.sin(2 * np.pi * 0.8 * t) > -0.2) +
        0.2 * np.sin(2 * np.pi * 4.0 * t) ** 2
    )
    syllables = np.clip(syllables, 0.0, 1.0)
    
    # Apply formant resonance filters (F1=700Hz, F2=1300Hz, F3=2500Hz)
    nyq = sample_rate / 2.0
    
    # Formant 1 (Vowel body)
    sos1 = butter(2, [500 / nyq, 900 / nyq], btype="bandpass", output="sos")
    f1 = sosfilt(sos1, glottal) * 1.8
    
    # Formant 2 (Speech clarity)
    sos2 = butter(2, [1100 / nyq, 1600 / nyq], btype="bandpass", output="sos")
    f2 = sosfilt(sos2, glottal) * 1.2
    
    # Formant 3 (Consonants/Fricatives)
    sos3 = butter(2, [2200 / nyq, 3200 / nyq], btype="bandpass", output="sos")
    f3 = sosfilt(sos3, glottal) * 0.8
    
    audio = (f1 + f2 + f3) * syllables
    
    # Add subtle fricative unvoiced bursts between syllables
    noise_bursts = np.random.randn(len(t)) * 0.15 * (syllables < 0.2) * (t > 0.3) * (t < 2.7)
    sos_fric = butter(2, [3500 / nyq, 6000 / nyq], btype="bandpass", output="sos")
    fricatives = sosfilt(sos_fric, noise_bursts)
    
    speech = audio + fricatives
    peak = np.max(np.abs(speech)) + 1e-12
    return (speech / peak * 0.75).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = 16000):
    """Write 16-bit PCM WAV using Python's standard wave module (no external deps)."""
    clamped = np.clip(audio, -1.0, 1.0)
    int16_pcm = (clamped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_pcm.tobytes())


def main():
    samples_dir = Path(__file__).resolve().parent / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    sr = 16000
    duration = 3.0
    synthesizer = DefenceNoiseSynthesizer(sample_rate=sr)
    
    print(f"Generating curated demonstration audio samples in {samples_dir}...")
    
    # 1. Clean speech
    clean = generate_synthetic_speech(duration=duration, sample_rate=sr)
    clean_path = samples_dir / "clean_speech_sample.wav"
    write_wav(clean_path, clean, sr)
    print(f"  [x] {clean_path.name} ({clean_path.stat().st_size / 1024:.1f} KB)")
    
    # 2. T-90 Tank engine noise
    tank = synthesizer.generate_tank_noise(duration=duration)
    tank_path = samples_dir / "tank_t90_engine_sample.wav"
    write_wav(tank_path, tank, sr)
    print(f"  [x] {tank_path.name} ({tank_path.stat().st_size / 1024:.1f} KB)")
    
    # 3. Helicopter rotor noise
    heli = synthesizer.generate_helicopter_noise(duration=duration)
    heli_path = samples_dir / "helicopter_rotor_sample.wav"
    write_wav(heli_path, heli, sr)
    print(f"  [x] {heli_path.name} ({heli_path.stat().st_size / 1024:.1f} KB)")
    
    # 4. Gunfire impulse noise
    gun = synthesizer.generate_gunfire_transient(duration=duration, num_shots=4)
    gun_path = samples_dir / "gunfire_impulse_sample.wav"
    write_wav(gun_path, gun, sr)
    print(f"  [x] {gun_path.name} ({gun_path.stat().st_size / 1024:.1f} KB)")
    
    # 5. Mixed speech + tank at 0 dB SNR
    mixed, _, _, actual_snr = AcousticMixer.mix_at_snr(clean, tank, target_snr_db=0.0)
    mixed_path = samples_dir / "mixed_tank_0db.wav"
    write_wav(mixed_path, mixed, sr)
    print(f"  [x] {mixed_path.name} ({mixed_path.stat().st_size / 1024:.1f} KB) (SNR = {actual_snr:.2f} dB)")
    
    # 6. Hybrid enhanced output
    pipeline = HybridEnhancementPipeline(sample_rate=sr, filter_length=64, step_size=0.1)
    ref_noise = tank * 0.95
    enhanced, diagnostics = pipeline.process_signals(mixed, ref_noise)
    
    enhanced_path = samples_dir / "hybrid_enhanced_output.wav"
    write_wav(enhanced_path, enhanced, sr)
    print(f"  [x] {enhanced_path.name} ({enhanced_path.stat().st_size / 1024:.1f} KB)")
    
    print("\nAll demonstration audio samples generated successfully and ready for git tracking!")

if __name__ == "__main__":
    main()
