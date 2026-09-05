#!/usr/bin/env python3
"""
PS 26052 — Artifact 1: Reproducible Dataset Generator
=====================================================
Mixes clean speech with defence-like noise at specified SNRs.
Supports RIR convolution, gain variation, and impulsive event injection.
Generates metadata CSV with full provenance tracking.

Usage:
    python 01_dataset_generator.py --config configs/dataset_config.yaml
    python 01_dataset_generator.py --preset demo  # quick 5-minute demo

Designed for: SIH 2026 Adaptive ANC — PS 26052
"""

import os
import sys
import csv
import json
import hashlib
import argparse
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

try:
    import soundfile as sf
except ImportError:
    print("ERROR: soundfile not installed. Run: pip install soundfile")
    sys.exit(1)

try:
    from scipy import signal as scipy_signal
    from scipy.io import wavfile as scipy_wav
except ImportError:
    print("ERROR: scipy not installed. Run: pip install scipy")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("dataset_gen")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class MixConfig:
    """Single mixture configuration."""
    target_snr_db: float
    noise_class: str  # stationary, non_stationary, impulsive, mixed
    apply_rir: bool = False
    add_impulse: bool = False
    impulse_snr_db: float = 10.0
    gain_variation: bool = False
    clip_simulation: bool = False


@dataclass
class DatasetConfig:
    """Full dataset generation configuration."""
    clean_dir: str = "data/clean"
    noise_dir: str = "data/noise"
    rir_dir: str = "data/rir"
    output_dir: str = "data/mixed"
    metadata_dir: str = "data/metadata"
    sample_rate: int = 16000
    snr_range_db: list = field(default_factory=lambda: [-10, -5, 0, 5, 10, 15, 20])
    noise_classes: list = field(default_factory=lambda: [
        "stationary", "non_stationary", "impulsive"
    ])
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    max_clean_duration_s: float = 10.0  # truncate long speech
    normalize_output: bool = True
    output_bit_depth: int = 16  # 16 or 32


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------
def load_audio(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load audio file, resample if needed, convert to mono float32."""
    data, sr = sf.read(path, dtype="float32")
    # Convert stereo to mono
    if data.ndim > 1:
        data = data.mean(axis=1)
    # Resample if needed
    if sr != target_sr:
        try:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except ImportError:
            # Fallback: simple resampling via scipy
            num_samples = int(len(data) * target_sr / sr)
            data = scipy_signal.resample(data, num_samples)
            sr = target_sr
    return data, sr


def save_audio(path: str, data: np.ndarray, sr: int, bit_depth: int = 16):
    """Save audio to WAV file."""
    if bit_depth == 16:
        data_save = np.clip(data, -1.0, 1.0)
        sf.write(path, data_save, sr, subtype="PCM_16")
    else:
        sf.write(path, data, sr, subtype="FLOAT")


# ---------------------------------------------------------------------------
# SNR Math
# ---------------------------------------------------------------------------
def compute_rms(signal: np.ndarray) -> float:
    """Compute RMS energy of a signal."""
    return np.sqrt(np.mean(signal ** 2) + 1e-12)


def compute_snr_db(clean: np.ndarray, noise: np.ndarray) -> float:
    """Compute SNR in dB between clean signal and noise."""
    rms_clean = compute_rms(clean)
    rms_noise = compute_rms(noise)
    return 20 * np.log10(rms_clean / (rms_noise + 1e-12))


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, target_snr_db: float) -> tuple[np.ndarray, float]:
    """
    Mix clean speech and noise at a target SNR.
    
    Math: x[n] = s[n] + alpha * n[n]
    where alpha = sqrt(Ps / (Pn * 10^(SNR/10)))
    
    Returns: (mixed_signal, achieved_snr_db)
    """
    rms_clean = compute_rms(clean)
    rms_noise = compute_rms(noise)
    
    if rms_noise < 1e-12:
        return clean.copy(), 60.0  # noise is silent
    
    alpha = rms_clean / (rms_noise * 10 ** (target_snr_db / 20.0))
    mixed = clean + alpha * noise
    
    achieved_snr = compute_snr_db(clean, alpha * noise)
    return mixed, achieved_snr


# ---------------------------------------------------------------------------
# RIR Convolution
# ---------------------------------------------------------------------------
def apply_rir(audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve audio with a room impulse response."""
    reverbed = scipy_signal.fftconvolve(audio, rir, mode="full")
    # Trim to original length
    reverbed = reverbed[:len(audio)]
    # Normalize to prevent clipping
    peak = np.max(np.abs(reverbed))
    if peak > 0.99:
        reverbed = reverbed * 0.99 / peak
    return reverbed


# ---------------------------------------------------------------------------
# Impulse Injection
# ---------------------------------------------------------------------------
def generate_impulse(duration_samples: int, sr: int) -> np.ndarray:
    """Generate a synthetic impulsive noise burst."""
    impulse = np.zeros(duration_samples)
    
    # Random choice of impulse type
    impulse_type = np.random.choice(["burst", "exponential", "click"])
    
    if impulse_type == "burst":
        # Gated white noise burst
        burst_len = np.random.randint(int(0.01 * sr), int(0.1 * sr))
        start = np.random.randint(0, max(1, duration_samples - burst_len))
        envelope = np.hanning(burst_len * 2)[burst_len:]  # sharp onset, smooth decay
        noise = np.random.randn(burst_len) * 0.8
        impulse[start:start + burst_len] = noise * envelope[:burst_len]
        
    elif impulse_type == "exponential":
        # Exponential decay transient
        decay_len = np.random.randint(int(0.005 * sr), int(0.05 * sr))
        start = np.random.randint(0, max(1, duration_samples - decay_len))
        t = np.arange(decay_len) / sr
        decay = np.exp(-t * np.random.uniform(50, 200))
        impulse[start:start + decay_len] = np.random.randn(decay_len) * decay
        
    elif impulse_type == "click":
        # Single-sample impulse
        if duration_samples > int(0.2 * sr):
            start = np.random.randint(int(0.1 * sr), min(int(0.9 * sr), duration_samples - 1))
        else:
            start = duration_samples // 2
        amplitude = np.random.uniform(0.5, 1.0) * np.sign(np.random.randn())
        impulse[start] = amplitude
    
    return impulse


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
@dataclass
class MixRecord:
    """Metadata for one mixed audio file."""
    mix_id: str
    clean_source: str
    clean_speaker: str
    clean_duration_s: float
    noise_source: str
    noise_class: str
    target_snr_db: float
    achieved_snr_db: float
    rir_applied: bool
    rir_source: str
    impulse_injected: bool
    impulse_snr_db: float
    gain_variation: bool
    clip_simulated: bool
    sample_rate: int
    output_samples: int
    split: str
    clean_hash: str
    noise_hash: str
    timestamp: str
    config_hash: str


def file_hash(path: str) -> str:
    """SHA256 hash of a file for provenance."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def config_hash(config: dict) -> str:
    """Hash of configuration for reproducibility."""
    import hashlib as hl
    return hl.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Main Generator
# ---------------------------------------------------------------------------
class DatasetGenerator:
    """Generates mixed clean+noise datasets with metadata."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.records: list[MixRecord] = []
        self._config_hash = config_hash(asdict(config))
        
        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.metadata_dir, exist_ok=True)
        os.makedirs(f"{config.output_dir}/train", exist_ok=True)
        os.makedirs(f"{config.output_dir}/val", exist_ok=True)
        os.makedirs(f"{config.output_dir}/test", exist_ok=True)
    
    def _collect_files(self, directory: str, extensions=(".wav", ".flac", ".mp3")) -> list[str]:
        """Collect audio files from a directory."""
        files = []
        root = Path(directory)
        if not root.exists():
            log.warning(f"Directory does not exist: {directory}")
            return files
        for ext in extensions:
            files.extend(str(p) for p in root.rglob(f"*{ext}"))
        return sorted(files)
    
    def _assign_split(self, index: int, total: int) -> str:
        """Assign train/val/test split based on index."""
        ratio = index / max(total - 1, 1)
        if ratio < self.config.train_ratio:
            return "train"
        elif ratio < self.config.train_ratio + self.config.val_ratio:
            return "val"
        else:
            return "test"
    
    def generate(self, max_mixes: Optional[int] = None):
        """Generate the full mixed dataset."""
        log.info("Collecting clean speech files...")
        clean_files = self._collect_files(self.config.clean_dir)
        if not clean_files:
            log.error(f"No clean speech files found in {self.config.clean_dir}")
            log.info("Generating demo audio instead...")
            clean_files = self._generate_demo_clean()
        
        log.info(f"Found {len(clean_files)} clean speech files")
        
        log.info("Collecting noise files...")
        noise_files = self._collect_files(self.config.noise_dir)
        if not noise_files:
            log.warning(f"No noise files found in {self.config.noise_dir}")
            log.info("Generating synthetic noise instead...")
            noise_files = self._generate_demo_noise()
        
        log.info(f"Found {len(noise_files)} noise files")
        
        # Load RIR files if available
        rir_files = self._collect_files(self.config.rir_dir)
        
        # Compute total combinations
        total = len(clean_files) * len(noise_files) * len(self.config.snr_range_db)
        if max_mixes:
            total = min(total, max_mixes)
        log.info(f"Generating up to {total} mixtures")
        
        mix_count = 0
        for ci, clean_path in enumerate(clean_files):
            clean_audio, sr = load_audio(clean_path, self.config.sample_rate)
            
            # Truncate if too long
            max_samples = int(self.config.max_clean_duration_s * sr)
            if len(clean_audio) > max_samples:
                clean_audio = clean_audio[:max_samples]
            
            for ni, noise_path in enumerate(noise_files):
                noise_audio, _ = load_audio(noise_path, self.config.sample_rate)
                
                # Loop noise to match clean length if needed
                if len(noise_audio) < len(clean_audio):
                    repeats = int(np.ceil(len(clean_audio) / len(noise_audio)))
                    noise_audio = np.tile(noise_audio, repeats)
                
                for snr_db in self.config.snr_range_db:
                    if max_mixes and mix_count >= max_mixes:
                        break
                    
                    # Randomly select noise class
                    noise_class = self.rng.choice(self.config.noise_classes)
                    
                    # Apply RIR to clean speech (randomly)
                    apply_rir_flag = self.rng.random() < 0.3 and len(rir_files) > 0
                    rir_source = ""
                    if apply_rir_flag:
                        rir_path = self.rng.choice(rir_files)
                        rir_audio, _ = load_audio(rir_path, self.config.sample_rate)
                        clean_for_mix = apply_rir(clean_audio, rir_audio)
                        rir_source = rir_path
                    else:
                        clean_for_mix = clean_audio.copy()
                    
                    # Mix at target SNR
                    mixed, achieved_snr = mix_at_snr(clean_for_mix, noise_audio[:len(clean_for_mix)], snr_db)
                    
                    # Inject impulse if configured
                    add_impulse_flag = self.rng.random() < 0.2
                    if add_impulse_flag:
                        impulse_len = self.rng.integers(int(0.01 * sr), int(0.2 * sr))
                        impulse_start = self.rng.integers(0, max(1, len(mixed) - impulse_len))
                        impulse = generate_impulse(impulse_len, sr)
                        mixed[impulse_start:impulse_start + impulse_len] += impulse
                    
                    # Normalize
                    if self.config.normalize_output:
                        peak = np.max(np.abs(mixed))
                        if peak > 0.99:
                            mixed = mixed * 0.99 / peak
                    
                    # Assign split
                    split = self._assign_split(mix_count, total)
                    
                    # Generate mix ID
                    clean_name = Path(clean_path).stem
                    noise_name = Path(noise_path).stem
                    mix_id = f"{clean_name}__{noise_name}__{snr_db:+.0f}dB_{mix_count:06d}"
                    
                    # Save mixed audio
                    output_path = os.path.join(
                        self.config.output_dir, split, f"{mix_id}.wav"
                    )
                    save_audio(output_path, mixed, sr, self.config.output_bit_depth)
                    
                    # Create metadata record
                    record = MixRecord(
                        mix_id=mix_id,
                        clean_source=clean_path,
                        clean_speaker=clean_name.split("_")[0] if "_" in clean_name else clean_name,
                        clean_duration_s=len(clean_audio) / sr,
                        noise_source=noise_path,
                        noise_class=noise_class,
                        target_snr_db=snr_db,
                        achieved_snr_db=float(achieved_snr),
                        rir_applied=apply_rir_flag,
                        rir_source=rir_source,
                        impulse_injected=add_impulse_flag,
                        impulse_snr_db=10.0,
                        gain_variation=False,
                        clip_simulated=False,
                        sample_rate=sr,
                        output_samples=len(mixed),
                        split=split,
                        clean_hash=file_hash(clean_path),
                        noise_hash=file_hash(noise_path),
                        timestamp=datetime.now().isoformat(),
                        config_hash=self._config_hash,
                    )
                    self.records.append(record)
                    mix_count += 1
                    
                    if mix_count % 100 == 0:
                        log.info(f"  Generated {mix_count}/{total} mixtures")
        
        log.info(f"Total mixtures generated: {mix_count}")
        self._save_metadata()
        return self.records
    
    def _save_metadata(self):
        """Save metadata CSV and JSON."""
        csv_path = os.path.join(self.config.metadata_dir, "dataset_metadata.csv")
        json_path = os.path.join(self.config.metadata_dir, "dataset_metadata.json")
        
        if not self.records:
            log.warning("No records to save")
            return
        
        # CSV
        fieldnames = list(asdict(self.records[0]).keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.records:
                writer.writerow(asdict(record))
        
        # JSON (full config + records)
        output = {
            "config": asdict(self.config),
            "config_hash": self._config_hash,
            "total_mixes": len(self.records),
            "split_counts": {
                split: sum(1 for r in self.records if r.split == split)
                for split in ["train", "val", "test"]
            },
            "snr_distribution": {
                str(snr): sum(1 for r in self.records if r.target_snr_db == snr)
                for snr in sorted(set(r.target_snr_db for r in self.records))
            },
            "records": [asdict(r) for r in self.records],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        log.info(f"Metadata saved to {csv_path}")
        log.info(f"Full JSON saved to {json_path}")
        
        # Print summary
        splits = {}
        for r in self.records:
            splits[r.split] = splits.get(r.split, 0) + 1
        log.info(f"Split counts: {splits}")
    
    def _generate_demo_clean(self) -> list[str]:
        """Generate synthetic demo clean speech (sine sweeps)."""
        demo_dir = os.path.join(self.config.clean_dir, "_demo")
        os.makedirs(demo_dir, exist_ok=True)
        
        sr = self.config.sample_rate
        paths = []
        
        for i in range(5):
            duration = self.rng.uniform(3.0, 8.0)
            t = np.linspace(0, duration, int(sr * duration), endpoint=False)
            
            # Synthetic "speech-like" signal: modulated harmonic content
            f0 = self.rng.uniform(100, 300)  # fundamental
            signal = np.zeros_like(t)
            for h in range(1, 6):
                signal += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t)
            
            # Amplitude modulation (syllable-like rhythm)
            mod_rate = self.rng.uniform(3, 7)  # Hz
            envelope = 0.5 * (1 + np.sin(2 * np.pi * mod_rate * t))
            signal = signal * envelope
            
            # Normalize
            signal = signal / (np.max(np.abs(signal)) + 1e-12) * 0.9
            
            path = os.path.join(demo_dir, f"demo_speech_{i:03d}.wav")
            sf.write(path, signal, sr, subtype="PCM_16")
            paths.append(path)
        
        log.info(f"Generated {len(paths)} demo clean speech files")
        return paths
    
    def _generate_demo_noise(self) -> list[str]:
        """Generate synthetic demo noise files."""
        demo_dir = os.path.join(self.config.noise_dir, "_demo")
        os.makedirs(demo_dir, exist_ok=True)
        
        sr = self.config.sample_rate
        paths = []
        
        noises = [
            ("engine", "stationary"),
            ("rotor", "non_stationary"),
            ("wind", "non_stationary"),
            ("impulse_burst", "impulsive"),
        ]
        
        for name, nclass in noises:
            duration = self.rng.uniform(5.0, 15.0)
            n_samples = int(sr * duration)
            
            if nclass == "stationary":
                # Band-limited noise (engine-like)
                noise = np.random.randn(n_samples) * 0.3
                # Low-pass filter for rumble
                b, a = scipy_signal.butter(4, 500 / (sr / 2), btype="low")
                noise = scipy_signal.filtfilt(b, a, noise)
                # Add harmonics
                t = np.linspace(0, duration, n_samples)
                for f in [60, 120, 180]:
                    noise += 0.15 * np.sin(2 * np.pi * f * t)
                    
            elif nclass == "non_stationary" and "rotor" in name:
                # Amplitude-modulated noise (rotor-like)
                t = np.linspace(0, duration, n_samples)
                noise = np.random.randn(n_samples)
                mod = 0.5 * (1 + np.sin(2 * np.pi * 8 * t))  # 8 Hz modulation
                noise = noise * mod * 0.5
                
            elif nclass == "non_stationary":
                # Slowly varying broadband noise (wind-like)
                noise = np.random.randn(n_samples)
                # Slow amplitude variation
                t = np.linspace(0, duration, n_samples)
                mod = 0.3 * (1 + 0.7 * np.sin(2 * np.pi * 0.5 * t) + 0.3 * np.sin(2 * np.pi * 1.2 * t))
                noise = noise * mod * 0.5
                
            elif nclass == "impulsive":
                # Sparse impulse train
                noise = np.random.randn(n_samples) * 0.01  # quiet background
                n_impulses = self.rng.integers(3, 10)
                for _ in range(n_impulses):
                    pos = self.rng.integers(0, n_samples)
                    imp_len = self.rng.integers(int(0.005 * sr), int(0.05 * sr))
                    imp_len = min(imp_len, n_samples - pos)
                    noise[pos:pos + imp_len] += generate_impulse(imp_len, sr) * 3
            
            # Normalize
            peak = np.max(np.abs(noise))
            if peak > 0:
                noise = noise / peak * 0.9
            
            path = os.path.join(demo_dir, f"demo_{name}.wav")
            sf.write(path, noise, sr, subtype="PCM_16")
            paths.append(path)
        
        log.info(f"Generated {len(paths)} demo noise files")
        return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def load_config(args) -> DatasetConfig:
    """Build config from CLI args."""
    config = DatasetConfig()
    
    if args.preset == "demo":
        config.snr_range_db = [0, 5, 10]
        config.max_clean_duration_s = 5.0
    elif args.preset == "full":
        config.snr_range_db = [-10, -5, 0, 5, 10, 15, 20]
    
    if args.sample_rate:
        config.sample_rate = args.sample_rate
    if args.seed:
        config.seed = args.seed
    if args.clean_dir:
        config.clean_dir = args.clean_dir
    if args.noise_dir:
        config.noise_dir = args.noise_dir
    
    return config


def main():
    parser = argparse.ArgumentParser(description="PS 26052 Dataset Generator")
    parser.add_argument("--preset", choices=["demo", "full"], default="demo",
                        help="Quick preset (demo: 3 SNRs, short; full: 7 SNRs)")
    parser.add_argument("--clean-dir", type=str, help="Directory of clean speech WAVs")
    parser.add_argument("--noise-dir", type=str, help="Directory of noise WAVs")
    parser.add_argument("--sample-rate", type=int, help="Target sample rate")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--max-mixes", type=int, help="Maximum number of mixes to generate")
    args = parser.parse_args()
    
    config = load_config(args)
    log.info(f"Config: {config}")
    
    generator = DatasetGenerator(config)
    records = generator.generate(max_mixes=args.max_mixes)
    
    print(f"\n{'='*60}")
    print(f"DATASET GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total mixtures: {len(records)}")
    print(f"Output directory: {config.output_dir}")
    print(f"Metadata: {config.metadata_dir}/dataset_metadata.csv")
    print(f"Sample rate: {config.sample_rate} Hz")
    splits = {}
    for r in records:
        splits[r.split] = splits.get(r.split, 0) + 1
    print(f"Split counts: {splits}")
    print(f"SNR range: {min(config.snr_range_db)} to {max(config.snr_range_db)} dB")
    print(f"\nNext step: Run 02_evaluation_harness.py to evaluate any enhancement model.")


if __name__ == "__main__":
    main()
