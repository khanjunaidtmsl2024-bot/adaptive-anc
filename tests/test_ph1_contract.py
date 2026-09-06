"""
PH1 experiment-contract tests (configs/ph1_experiment_contract.yaml).

Asserts the model adapters (src/ai/crn.py, src/ai/dtln.py, src/ai/tiny_enhancer.py)
instantiate and process at the FROZEN contract geometry:
    sample_rate 16000, frame 256, hop 128, Hann, center=False, seed 42
i.e. 129 frequency bins (frame 256 -> rfft bins 256//2+1), with parameter
counts measured at that geometry, and a 256-sample hop runs end-to-end.

Legacy geometry (frame 512 / freq_bins 257) must still be constructible
explicitly, but must NOT be the default.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import pytest

CONTRACT_BINS = 129          # 256 // 2 + 1
CONTRACT_FRAME = 256
CONTRACT_HOP = 128
RECONCILED = {
    "TinyEnhancer": 9_569,
    "CRN_Micro": 723_801,    # freq_bins=129 (frame 256) -- NOT the 512-frame 986,457
    "DTLN": 775_939,         # frame 256 / hop 128 / enc 256 -- NOT the 512-frame 989,315
}


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_defaults_instantiate_at_contract_geometry_with_reconciled_counts():
    from src.ai.crn import CRNWrapper
    from src.ai.dtln import DTLNWrapper
    from src.ai.tiny_enhancer import TinyEnhancerWrapper

    torch.manual_seed(42)
    crn = CRNWrapper()          # no args -> must be contract geometry
    dtln = DTLNWrapper()
    tiny = TinyEnhancerWrapper()

    assert crn.net.freq_bins == CONTRACT_BINS, \
        f"CRN default freq_bins={crn.net.freq_bins}, expected {CONTRACT_BINS}"
    assert dtln.net.frame_size == CONTRACT_FRAME, \
        f"DTLN default frame_size={dtln.net.frame_size}, expected {CONTRACT_FRAME}"
    assert dtln.net.freq_bins == CONTRACT_BINS, \
        f"DTLN default freq_bins={dtln.net.freq_bins}, expected {CONTRACT_BINS}"

    assert crn.net.count_parameters() == RECONCILED["CRN_Micro"], \
        f"CRN params={crn.net.count_parameters()}, expected {RECONCILED['CRN_Micro']}"
    assert dtln.net.count_parameters() == RECONCILED["DTLN"], \
        f"DTLN params={dtln.net.count_parameters()}, expected {RECONCILED['DTLN']}"
    assert sum(p.numel() for p in tiny.net.parameters()) == RECONCILED["TinyEnhancer"], \
        f"TinyEnhancer params != {RECONCILED['TinyEnhancer']}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_legacy_geometry_still_constructible_explicitly():
    """frame-512 / 257-bin variants remain available for the offline benchmark path."""
    from src.ai.crn import CRNWrapper
    from src.ai.dtln import DTLNWrapper

    crn = CRNWrapper(freq_bins=257)
    dtln = DTLNWrapper(frame_size=512, hop_size=128)
    assert crn.net.freq_bins == 257
    assert crn.net.count_parameters() == 986_457
    assert dtln.net.frame_size == 512
    assert dtln.net.freq_bins == 257
    assert dtln.net.count_parameters() == 989_315


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_dtln_internal_framing_is_causal():
    """
    DTLN's internal framing must be causal (no future-sample padding).

    torch.stft/istft default to center=True (frame/2 future padding); DTLN must
    not call them. We assert (a) the net's forward runs with torch.stft/istft
    monkeypatched to raise, and (b) the causal helpers are bit-identical to
    torch.stft(..., center=False), which has provably no future context.
    """
    from src.ai.dtln import DTLNWrapper, _causal_stft, _causal_istft
    import src.ai.dtln as dtln_mod

    # (b) helpers == torch center=False reference
    torch.manual_seed(42)
    x = torch.randn(1, 1, CONTRACT_FRAME * 8)
    window = torch.hann_window(CONTRACT_FRAME)
    ref = torch.stft(x.squeeze(1), CONTRACT_FRAME, CONTRACT_HOP, window=window,
                     return_complex=True, center=False)
    assert torch.allclose(_causal_stft(x.squeeze(1), CONTRACT_FRAME, CONTRACT_HOP, window),
                          ref, atol=1e-6), "causal STFT must equal torch center=False reference"

    # (a) forward must not touch torch.stft/istft (whose default is center=True)
    real_stft, real_istft = torch.stft, torch.istft

    def raise_(*a, **kw):
        raise AssertionError("DTLN forward must not call torch.stft/istft (center=True default is non-causal)")

    dtln_mod.torch.stft = raise_
    dtln_mod.torch.istft = raise_
    try:
        net = DTLNWrapper().net
        with torch.no_grad():
            y = net(x)
    finally:
        dtln_mod.torch.stft = real_stft
        dtln_mod.torch.istft = real_istft

    assert y.shape == x.shape
    assert bool(torch.isfinite(y).all())

    # causal iSTFT round-trips exactly in the interior (no future context needed)
    s = _causal_stft(x.squeeze(1), CONTRACT_FRAME, CONTRACT_HOP, window)
    rec = _causal_istft(s, CONTRACT_FRAME, CONTRACT_HOP, window, length=x.shape[-1])
    interior = rec[0, CONTRACT_FRAME:-CONTRACT_FRAME] - x[0, 0, CONTRACT_FRAME:-CONTRACT_FRAME]
    assert float(interior.abs().max()) < 1e-4, "causal WOLA must reconstruct interior to float precision"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.parametrize("factory_name", ["crn", "dtln", "tiny"])
def test_256_sample_hop_runs_end_to_end(factory_name):
    """
    A 256-sample hop (129-bin frame) must run end-to-end through each adapter's
    enhance_spectrogram, matching the deployment single-frame contract.
    """
    if factory_name == "crn":
        from src.ai.crn import CRNWrapper
        wrapper = CRNWrapper()
    elif factory_name == "dtln":
        from src.ai.dtln import DTLNWrapper
        wrapper = DTLNWrapper()
    else:
        from src.ai.tiny_enhancer import TinyEnhancerWrapper
        wrapper = TinyEnhancerWrapper()

    rng = np.random.RandomState(42)
    mag = rng.rand(CONTRACT_BINS, 1).astype(np.float32) * 0.5
    phase = (rng.rand(CONTRACT_BINS, 1).astype(np.float32) * 2 - 1) * np.pi

    enh_mag, enh_phase = wrapper.enhance_spectrogram(mag, phase)

    assert enh_mag.shape == (CONTRACT_BINS, 1), \
        f"{factory_name} enhanced mag shape {enh_mag.shape}, expected ({CONTRACT_BINS}, 1)"
    assert enh_phase.shape == (CONTRACT_BINS, 1)
    assert np.all(np.isfinite(enh_mag)), f"{factory_name} produced non-finite magnitude"
    assert np.all(np.isfinite(enh_phase)), f"{factory_name} produced non-finite phase"
