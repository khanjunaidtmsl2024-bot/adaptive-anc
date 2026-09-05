"""Integration tests for Hybrid AI-DSP Pipeline."""

import numpy as np
from src.pipeline.hybrid_chain import HybridEnhancementPipeline
from src.pipeline.fallback_controller import FallbackController


def test_hybrid_pipeline_execution():
    """Verify full Config A hybrid pipeline processes signals without error."""
    pipeline = HybridEnhancementPipeline(config_mode="A", filter_length=32, sample_rate=16000)

    # 1 second of audio
    primary = np.random.normal(0, 0.5, 16000).astype(np.float32)
    reference = np.random.normal(0, 0.3, 16000).astype(np.float32)

    enhanced, diag = pipeline.process_signals(primary, reference)

    assert len(enhanced) == 16000
    assert np.all(np.isfinite(enhanced))
    assert "dsp_output" in diag
    assert "final_output" in diag


def test_fallback_controller():
    """Verify fallback controller detects NaN/Inf and suppresses divergence."""
    controller = FallbackController(clip_threshold=0.99)

    primary = np.ones(100, dtype=np.float32) * 0.5
    diverged = np.ones(100, dtype=np.float32) * np.nan

    safe_out, is_diverged = controller.verify_frame(primary, diverged)
    assert is_diverged is True
    assert np.all(np.isfinite(safe_out))
