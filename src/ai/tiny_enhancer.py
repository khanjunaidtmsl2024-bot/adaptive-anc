"""
TinyEnhancer PyTorch Architecture Adapter.
PS 26052 — Adaptive Defence ANC.

Implements the 4-layer 2D ConvNet architecture defined in ichigo137/anc
to enable local evaluation, streaming benchmarking, and supervisory auditing.
"""

from typing import Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class TinyEnhancerNet(nn.Module):
        """4-layer Conv2D mask estimator matching ichigo137/anc specification."""

        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Estimate spectral suppression mask M(f, t) in [0, 1]."""
            return self.network(x)
else:
    class TinyEnhancerNet:
        def __init__(self):
            raise ImportError("PyTorch is required to instantiate TinyEnhancerNet.")

TinyEnhancer = TinyEnhancerNet


class TinyEnhancerWrapper:
    """Wrapper that handles spectrogram conversion and inference."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self.net = None

        if TORCH_AVAILABLE:
            self.net = TinyEnhancerNet().to(self.device)
            if checkpoint_path is not None:
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                self.net.load_state_dict(state_dict)
            self.net.eval()

    def enhance_spectrogram(
        self,
        mag_spec: np.ndarray,
        phase: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Takes 2D magnitude spectrogram, predicts mask, applies mask, and returns enhanced spec.
        """
        if not TORCH_AVAILABLE or self.net is None:
            # Fallback pass-through if PyTorch is not installed
            return mag_spec, phase

        with torch.no_grad():
            tensor_in = torch.from_numpy(mag_spec).unsqueeze(0).unsqueeze(0).float().to(self.device)
            mask = self.net(tensor_in).squeeze().cpu().numpy()
            enhanced_mag = mag_spec * mask

        return enhanced_mag.astype(np.float32), phase
