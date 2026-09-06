"""
PH1 training-input contract: RAW_PRIMARY vs NLMS_RESIDUAL legs.
PS 26052 — Adaptive Defence ANC.

Covers the two new behaviors in src/ai/train.py:
1. Unknown input_mode fails loudly (no silent fallback).
2. Residual preparation is deterministic (fresh filter per clip) and
   returns float32 arrays of exactly the primary length.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ai.train import DatasetLoader, _prepare_residual_inputs, train_spectral_mask_model


class _FakeLoader:
    def __init__(self, n=1):
        self._n = n
        rng = np.random.RandomState(0)
        self._primary = rng.randn(16000).astype(np.float32)
        self._reference = rng.randn(16000).astype(np.float32)
        self._clean = rng.randn(16000).astype(np.float32)

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return self._clean, self._primary, self._reference, {"speaker_id": "SPK_TEST"}


def test_unknown_input_mode_fails_loud():
    with pytest.raises(ValueError, match="input_mode"):
        train_spectral_mask_model(model=None, input_mode="SOMETHING_ELSE")


def test_residual_preparation_deterministic_and_same_length():
    loader = _FakeLoader(n=2)
    r1 = _prepare_residual_inputs(loader)
    r2 = _prepare_residual_inputs(loader)
    assert len(r1) == 2
    for a, b in zip(r1, r2):
        assert a.dtype == np.float32
        assert a.shape == (16000,)
        np.testing.assert_array_equal(a, b)  # order-independent (fresh filter per clip)
    # Residual is the error signal: different from primary input.
    assert not np.allclose(r1[0], loader._primary)


def test_residual_preparation_pads_short_output():
    class _ShortLoader(_FakeLoader):
        def __getitem__(self, idx):
            clean, primary, reference, row = super().__getitem__(idx)
            return clean, primary, reference[:100], row  # shorter reference

    out = _prepare_residual_inputs(_ShortLoader(n=1))
    assert out[0].shape == (16000,)
    assert np.isfinite(out[0]).all()