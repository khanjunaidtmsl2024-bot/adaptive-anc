"""
Unit tests for classical baselines, integrations, edge export, and live stream.
DRDO SIH 2026 -- Problem Statement 26052.
"""

import pytest
import numpy as np
from src.dsp.spectral_subtraction import SpectralSubtraction
from src.dsp.wiener import WienerFilter
from src.integrations.ichigo_bridge import IchigoAncBridge
from src.streaming.live_stream_audio import LiveAudioStreamEngine
from src.ai.export_onnx import export_edge_models


def test_spectral_subtraction_execution():
    sr = 16000
    x = np.random.normal(0, 0.1, 4000).astype(np.float32)
    ss = SpectralSubtraction(sample_rate=sr)
    out = ss.process(x)
    assert len(out) == len(x)
    assert not np.isnan(out).any()


def test_wiener_filter_execution():
    sr = 16000
    x = np.random.normal(0, 0.1, 4000).astype(np.float32)
    wf = WienerFilter(sample_rate=sr)
    out = wf.process(x)
    assert len(out) == len(x)
    assert not np.isnan(out).any()


def test_ichigo_bridge_inspection():
    # Point at a guaranteed-absent checkpoint: random-init must be detectable
    # via the flag, independent of stdout parsing or machine-local files.
    bridge = IchigoAncBridge(checkpoint_path="__definitely_missing__.pt")
    info = bridge.inspect_model()
    assert info["source_repo"] == "https://github.com/ichigo137/anc"
    assert info["total_parameters"] > 0
    assert len(info["layers"]) == 8
    assert isinstance(bridge.checkpoint_loaded, bool)
    assert bridge.checkpoint_loaded is False
    assert info["checkpoint_loaded"] is bridge.checkpoint_loaded
    # Full provenance contract: random-init must be machine-detectable.
    assert bridge.weights_status == "RANDOM_INITIALIZATION"
    assert info["weights_status"] == "RANDOM_INITIALIZATION"
    assert bridge.checkpoint_path is None
    assert bridge.checkpoint_sha256 is None
    assert info["checkpoint_path"] is None
    assert info["checkpoint_sha256"] is None


def test_ichigo_bridge_benchmark():
    bridge = IchigoAncBridge()
    res = bridge.benchmark_against_defence_noise(noise_type="tank", input_snr_db=0.0, duration=1.0)
    assert "raw_noisy" in res
    assert "ichigo_ai_standalone" in res
    assert "master_hub_hybrid" in res


def test_live_stream_simulation():
    engine = LiveAudioStreamEngine(config_mode="A")
    res = engine.run_simulation(duration_sec=1.0)
    assert res["frames_processed"] > 0
    assert "p50_latency_ms" in res
    # Simulated runs must be distinguishable from real hardware runs by the
    # return dict alone (no stdout parsing needed).
    assert res["mode"] == "simulation"
    assert res["is_simulated"] is True
    # Full execution-provenance contract: simulated runs must be self-tagging.
    assert res["execution_mode"] == "SIMULATION"
    assert res["hardware_available"] is False
    assert res["hardware_target"] == "Raspberry Pi 4"
    assert res["hardware_model"] is None
    assert res["audio_interface"] is None

    # Hardware provenance schema must be distinct (unit-checked without a device).
    from src.streaming.live_stream_audio import _execution_provenance
    hw = _execution_provenance(simulated=False)
    assert hw["execution_mode"] == "HARDWARE"
    assert hw["is_simulated"] is False
    assert hw["hardware_available"] is True
    assert hw["hardware_model"] == "Raspberry Pi 4 + WM8960 CODEC"
    assert hw["audio_interface"] == "WM8960"


try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def test_edge_export():
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed in this environment")
    res = export_edge_models(output_dir="models")
    assert res["status"] == "success"
    assert "torchscript_jit" in res["artifacts"]


def test_canonical_fixtures_provenance_sha256():
    """Asserts cryptographic SHA-256 provenance of whitelisted fixtures and baselines."""
    import hashlib
    from pathlib import Path

    # 1. Canonical speech fixture
    wav_path = Path("data/v4/clean/SPK_001_clean.wav")
    assert wav_path.exists(), "Canonical test speech clip missing: data/v4/clean/SPK_001_clean.wav"
    wav_hash = hashlib.sha256(wav_path.read_bytes()).hexdigest()
    assert wav_hash == "558da0ce034cd554344709545dcef581ce7c8dcc6834771abac55e7a6a2c681a", (
        f"Canonical speech fixture modified: got {wav_hash}, expected 558da0ce034cd554344709545dcef581ce7c8dcc6834771abac55e7a6a2c681a"
    )

    # 2. Baseline checkpoint fixture
    ckpt_path = Path("checkpoints/tiny_enhancer_v3.pt")
    assert ckpt_path.exists(), "Baseline checkpoint missing: checkpoints/tiny_enhancer_v3.pt"
    ckpt_hash = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    assert ckpt_hash == "c2d2b3e59483e580d948abab1b31a091953d80346fa2d23d1c5f02f6b7fb552b", (
        f"Baseline checkpoint modified: got {ckpt_hash}, expected c2d2b3e59483e580d948abab1b31a091953d80346fa2d23d1c5f02f6b7fb552b"
    )

