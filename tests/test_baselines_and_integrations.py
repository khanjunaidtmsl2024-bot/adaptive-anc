"""
Unit tests for classical baselines, integrations, edge export, and live stream.
DRDO SIH 2026 — Problem Statement 26052.
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
    bridge = IchigoAncBridge()
    info = bridge.inspect_model()
    assert info["source_repo"] == "https://github.com/ichigo137/anc"
    assert info["total_parameters"] > 0
    assert len(info["layers"]) == 8


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
